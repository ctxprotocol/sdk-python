"""
Contract tests for the Tools resource, including execute session lifecycle APIs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.types import ContextError, ExecuteSessionResult


async def test_execute_forwards_session_fields_and_parses_response() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    mock_response = {
        "success": True,
        "mode": "execute",
        "result": {"value": 42},
        "tool": {"id": "tool-1", "name": "Market Data"},
        "method": {"name": "get_price", "executePriceUsd": "0.05"},
        "session": {
            "mode": "execute",
            "sessionId": "sess_123",
            "methodPrice": "0.05",
            "spent": "0.05",
            "remaining": "0.95",
            "maxSpend": "1",
            "status": "open",
            "expiresAt": "2026-02-22T00:00:00.000Z",
            "closeRequested": True,
            "pendingAccruedCount": 2,
            "pendingAccruedUsd": "0.10",
        },
        "durationMs": 980,
    }

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_response
        result = await client.tools.execute(
            tool_id="tool-1",
            tool_name="get_price",
            args={"symbol": "ETH"},
            idempotency_key="7b31f437-61ed-4f76-8a6e-ed2f0766ffb8",
            mode="execute",
            session_id="sess_123",
            max_spend_usd="1",
            close_session=True,
        )

    mock_fetch.assert_called_once_with(
        "/api/v1/tools/execute",
        method="POST",
        json_body={
            "toolId": "tool-1",
            "toolName": "get_price",
            "args": {"symbol": "ETH"},
            "mode": "execute",
            "sessionId": "sess_123",
            "maxSpendUsd": "1",
            "closeSession": True,
        },
        extra_headers={"Idempotency-Key": "7b31f437-61ed-4f76-8a6e-ed2f0766ffb8"},
    )

    assert result.mode == "execute"
    assert result.method.execute_price_usd == "0.05"
    assert result.session.session_id == "sess_123"
    assert result.session.pending_accrued_count == 2
    await client.close()


async def test_execute_propagates_session_budget_error() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "error": "This execute call costs $0.8, which exceeds the session maxSpend of $0.5.",
            "code": "session_budget_exceeded",
            "mode": "execute",
            "session": {
                "mode": "execute",
                "sessionId": "sess_123",
                "methodPrice": "0.8",
                "spent": "0.8",
                "remaining": "0",
                "maxSpend": "0.5",
            },
        }

        with pytest.raises(ContextError) as exc_info:
            await client.tools.execute(
                tool_id="tool-1",
                tool_name="get_expensive_data",
                max_spend_usd="0.5",
            )

    assert exc_info.value.code == "session_budget_exceeded"
    await client.close()


async def test_start_session_posts_max_spend() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "success": True,
            "mode": "execute",
            "session": {
                "mode": "execute",
                "sessionId": "sess_start",
                "methodPrice": "0",
                "spent": "0",
                "remaining": "5",
                "maxSpend": "5",
                "status": "open",
            },
        }

        result = await client.tools.start_session("5")

    mock_fetch.assert_called_once_with(
        "/api/v1/tools/execute/sessions",
        method="POST",
        json_body={"mode": "execute", "maxSpendUsd": "5"},
    )
    assert isinstance(result, ExecuteSessionResult)
    assert result.session.session_id == "sess_start"
    await client.close()


async def test_get_session_fetches_by_id() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "success": True,
            "mode": "execute",
            "session": {
                "mode": "execute",
                "sessionId": "sess_status",
                "methodPrice": "0",
                "spent": "1.2",
                "remaining": "3.8",
                "maxSpend": "5",
                "status": "open",
            },
        }

        result = await client.tools.get_session("sess_status")

    mock_fetch.assert_called_once_with("/api/v1/tools/execute/sessions/sess_status")
    assert result.session.spent == "1.2"
    await client.close()


async def test_close_session_posts_close_route() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "success": True,
            "mode": "execute",
            "session": {
                "mode": "execute",
                "sessionId": "sess_close",
                "methodPrice": "0",
                "spent": "1.2",
                "remaining": "3.8",
                "maxSpend": "5",
                "status": "closed",
            },
        }

        result = await client.tools.close_session("sess_close")

    mock_fetch.assert_called_once_with(
        "/api/v1/tools/execute/sessions/sess_close/close",
        method="POST",
        json_body={"mode": "execute"},
    )
    assert result.session.status == "closed"
    await client.close()
