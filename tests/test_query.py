"""
Contract tests for the Query resource (pay-per-response / agentic mode).

These tests mock the HTTP layer (httpx) and validate that the SDK
correctly serializes requests and deserializes responses matching
the shapes returned by POST /api/v1/query.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.types import (
    ContextError,
    QueryCost,
    QueryResult,
    QueryStreamDoneEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamToolStatusEvent,
    QueryToolUsage,
)

# ============================================================================
# Test data matching the actual endpoint response shapes
# ============================================================================

MOCK_SUCCESS_RESPONSE: dict[str, Any] = {
    "success": True,
    "response": (
        "Based on the latest data, the top whale movements on Base "
        "include a $2.3M USDC transfer from 0xabc... to Uniswap V3."
    ),
    "toolsUsed": [
        {"id": "tool-uuid-1", "name": "Whale Tracker", "skillCalls": 3},
        {"id": "tool-uuid-2", "name": "Price Feed", "skillCalls": 1},
    ],
    "cost": {
        "totalCostUsd": "0.015400",
        "toolCostUsd": "0.010000",
        "modelCostUsd": "0.005400",
    },
    "durationMs": 4200,
}

MOCK_ERROR_RESPONSE: dict[str, Any] = {
    "error": "Insufficient funds. Set a spending cap in the dashboard.",
    "code": "insufficient_allowance",
}

MOCK_SSE_LINES = [
    'data: {"type":"tool-status","status":"discovering","tool":{"id":"","name":""}}',
    'data: {"type":"tool-status","status":"discovered","tool":{"id":"tool-uuid-1","name":"Whale Tracker"}}',
    'data: {"type":"tool-status","status":"planning","tool":{"id":"","name":""}}',
    'data: {"type":"tool-status","status":"executing","tool":{"id":"","name":""}}',
    'data: {"type":"tool-status","status":"synthesizing","tool":{"id":"","name":""}}',
    'data: {"type":"text-delta","delta":"Based on "}',
    'data: {"type":"text-delta","delta":"the latest "}',
    'data: {"type":"text-delta","delta":"data, "}',
    "data: "
    + json.dumps(
        {
            "type": "done",
            "result": {
                "response": "Based on the latest data, whale activity is up 15%.",
                "toolsUsed": [
                    {"id": "tool-uuid-1", "name": "Whale Tracker", "skillCalls": 2}
                ],
                "cost": {
                    "totalCostUsd": "0.012000",
                    "toolCostUsd": "0.008000",
                    "modelCostUsd": "0.004000",
                },
                "durationMs": 3800,
            },
        }
    ),
    "data: [DONE]",
]


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_response(
    data: dict[str, Any],
    status_code: int = 200,
) -> httpx.Response:
    """Build a fake httpx.Response with the given JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://ctxprotocol.com/api/v1/query"),
    )


class _FakeStreamResponse:
    """Mimics an httpx.Response with async line iteration for SSE."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._lines = lines

    async def aiter_lines(self):  # noqa: ANN201
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    def json(self) -> dict[str, Any]:
        return {}


# ============================================================================
# Tests: query.run()
# ============================================================================


class TestQueryRun:
    """Tests for client.query.run() — non-streaming JSON queries."""

    async def test_sends_correct_request_body_string(self) -> None:
        """String shorthand sends query with stream: false."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_SUCCESS_RESPONSE
            await client.query.run("What are the top whale movements?")

        mock_fetch.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "What are the top whale movements?",
                "tools": None,
                "stream": False,
            },
        )

    async def test_sends_correct_request_body_with_tools(self) -> None:
        """Options dict with tools sends tool IDs."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_SUCCESS_RESPONSE
            await client.query.run(
                query="Analyze whale activity",
                tools=["tool-uuid-1", "tool-uuid-2"],
            )

        mock_fetch.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "tools": ["tool-uuid-1", "tool-uuid-2"],
                "stream": False,
            },
        )

    async def test_parses_success_response_into_query_result(self) -> None:
        """Successful API response is deserialized into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_SUCCESS_RESPONSE
            result = await client.query.run("test query")

        assert isinstance(result, QueryResult)
        assert "whale movements" in result.response
        assert len(result.tools_used) == 2

        whale_tool = result.tools_used[0]
        assert isinstance(whale_tool, QueryToolUsage)
        assert whale_tool.id == "tool-uuid-1"
        assert whale_tool.name == "Whale Tracker"
        assert whale_tool.skill_calls == 3

        assert isinstance(result.cost, QueryCost)
        assert result.cost.total_cost_usd == "0.015400"
        assert result.cost.tool_cost_usd == "0.010000"
        assert result.cost.model_cost_usd == "0.005400"
        assert result.duration_ms == 4200

    async def test_raises_context_error_on_insufficient_allowance(self) -> None:
        """Error response raises ContextError with correct code."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_ERROR_RESPONSE

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert "Insufficient funds" in str(exc_info.value)
        assert exc_info.value.code == "insufficient_allowance"

    async def test_raises_context_error_on_no_wallet(self) -> None:
        """no_wallet error is propagated correctly."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {
                "error": "Account not fully set up.",
                "code": "no_wallet",
            }

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert exc_info.value.code == "no_wallet"

    async def test_raises_context_error_on_query_failed(self) -> None:
        """query_failed error is propagated correctly."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {
                "error": "Query failed: Tool execution timed out",
                "code": "query_failed",
            }

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert exc_info.value.code == "query_failed"


# ============================================================================
# Tests: query.stream()
# ============================================================================


class TestQueryStream:
    """Tests for client.query.stream() — SSE streaming queries."""

    async def test_sends_correct_request_body(self) -> None:
        """Stream sends request with stream: true."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("What are whale movements?"):
                events.append(event)

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "What are whale movements?",
                "tools": None,
                "stream": True,
            },
        )

    async def test_yields_all_event_types(self) -> None:
        """All SSE events are parsed and yielded in correct order."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test query"):
                events.append(event)

        # 5 tool-status + 3 text-delta + 1 done = 9 events
        assert len(events) == 9

        # Tool status events
        status_events = [e for e in events if isinstance(e, QueryStreamToolStatusEvent)]
        assert len(status_events) == 5
        assert status_events[0].status == "discovering"
        assert status_events[1].status == "discovered"
        assert status_events[1].tool.name == "Whale Tracker"

        # Text delta events
        text_events = [e for e in events if isinstance(e, QueryStreamTextDeltaEvent)]
        assert len(text_events) == 3
        assert text_events[0].delta == "Based on "
        assert text_events[1].delta == "the latest "
        assert text_events[2].delta == "data, "

        # Done event
        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1

        result = done_events[0].result
        assert isinstance(result, QueryResult)
        assert "whale activity" in result.response
        assert len(result.tools_used) == 1
        assert result.tools_used[0].skill_calls == 2
        assert result.cost.total_cost_usd == "0.012000"
        assert result.duration_ms == 3800

    async def test_stops_on_done_sentinel(self) -> None:
        """Stream stops processing after [DONE] sentinel."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            [
                'data: {"type":"text-delta","delta":"hello "}',
                "data: [DONE]",
                'data: {"type":"text-delta","delta":"should not appear"}',
            ]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test"):
                events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], QueryStreamTextDeltaEvent)
        assert events[0].delta == "hello "

    async def test_skips_malformed_sse_events(self) -> None:
        """Malformed JSON in SSE events is skipped gracefully."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            [
                'data: {"type":"text-delta","delta":"valid "}',
                "data: {invalid json}",
                'data: {"type":"text-delta","delta":"also valid "}',
                "data: [DONE]",
            ]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test"):
                events.append(event)

        assert len(events) == 2

    async def test_supports_tools_parameter(self) -> None:
        """Stream with explicit tool IDs sends them in request."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            ['data: {"type":"text-delta","delta":"result "}', "data: [DONE]"]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query="test",
                tools=["tool-1", "tool-2"],
            ):
                events.append(event)

        call_kwargs = mock_stream.call_args
        assert call_kwargs[1]["json_body"]["tools"] == ["tool-1", "tool-2"]

    async def test_ignores_non_data_lines(self) -> None:
        """Lines not starting with 'data: ' are ignored."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            [
                ": comment line",
                "",
                'data: {"type":"text-delta","delta":"hello "}',
                "event: ping",
                "data: [DONE]",
            ]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test"):
                events.append(event)

        assert len(events) == 1
