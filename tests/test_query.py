"""
Contract tests for the Query resource (pay-per-response / agentic mode).

These tests mock the HTTP layer (httpx) and validate that the SDK
correctly serializes requests and deserializes responses matching
the shapes returned by POST /api/v1/query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.types import (
    ContextError,
    QueryCost,
    QueryOptions,
    QueryResult,
    QueryStreamDeveloperTraceEvent,
    QueryStreamDoneEvent,
    QueryStreamErrorEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamToolStatusEvent,
    QueryToolUsage,
)

SHARED_QUERY_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "context-sdk"
    / "fixtures"
    / "query-response"
    / "full-grounded-answer.json"
)
SHARED_QUERY_FIXTURE: dict[str, dict[str, Any]] = json.loads(
    SHARED_QUERY_FIXTURE_PATH.read_text(encoding="utf-8")
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

MOCK_EVIDENCE_RESPONSE: dict[str, Any] = {
    **MOCK_SUCCESS_RESPONSE,
    "responseShape": "evidence_only",
    "response": "Structured evidence package with 2 evidence facts and medium confidence.",
    "summary": "BTC net exchange flow is negative across the last 24h sample.",
    "evidence": {
        "facts": [
            {
                "id": "fact-1",
                "label": "Net BTC exchange flow",
                "path": "aggregateFlow.netFlowUsd",
                "relevanceScore": 0.93,
                "value": -12_500_000,
            }
        ],
        "sourceRefs": [
            {
                "id": "source-1",
                "provider": "Coinglass",
                "dataset": "exchange flows",
                "observedAt": "2026-03-23T12:00:00.000Z",
                "publishedAt": None,
                "artifactRef": "https://example.com/data.json",
                "url": "https://example.com/data.json",
                "note": "Canonical execution artifact",
            }
        ],
        "assumptions": ["Used rolling 24h window."],
        "knownUnknowns": ["No venue-specific catalyst evidence was available."],
        "retrievalPlanReasonCodes": ["bounded_retrieval_first"],
    },
    "artifacts": {
        "dataUrl": "https://example.com/data.json",
        "canonicalDataRef": {
            "datasetId": "dataset-1",
            "hash": "abc123",
            "bytes": 2048,
            "publicDataUrl": "https://example.com/data.json",
        },
        "stageArtifactKinds": ["canonical-execution-data", "completeness-evaluation"],
    },
    "view": {
        "type": "timeseries",
        "label": "Timeseries",
    },
    "freshness": {
        "asOf": "2026-03-23T12:00:00.000Z",
        "sourceTimestamps": ["2026-03-23T12:00:00.000Z"],
        "note": "Most recent evidence timestamp: 2026-03-23T12:00:00.000Z",
    },
    "confidence": {
        "level": "medium",
        "reason": "Grounded in canonical execution data with one unresolved gap.",
        "verifiedFactCount": 2,
        "inferredFactCount": 0,
        "gapCount": 1,
        "gapSignals": [
            {
                "code": "missing_catalyst",
                "severity": "medium",
                "detail": "No catalyst evidence was retrieved.",
            }
        ],
    },
    "usage": {
        "durationMs": 4200,
        "cost": {
            "totalCostUsd": "0.015400",
            "toolCostUsd": "0.010000",
            "modelCostUsd": "0.005400",
        },
        "toolsUsed": [
            {"id": "tool-uuid-1", "name": "Whale Tracker", "skillCalls": 3},
            {"id": "tool-uuid-2", "name": "Price Feed", "skillCalls": 1},
        ],
        "outcomeType": "answer",
        "orchestrationMetrics": {
            "parityStage": "candidate",
            "orchestrationMode": "agentic",
            "firstPassSuccess": True,
            "capabilityMissSignaled": False,
            "rediscoveryExecuted": False,
        },
    },
}

MOCK_DEVELOPER_TRACE: dict[str, Any] = {
    "summary": {
        "toolCalls": 4,
        "retryCount": 2,
        "selfHealCount": 1,
        "fallbackCount": 1,
        "completionChecks": 3,
        "loopCount": 2,
    },
    "timeline": [
        {
            "stepType": "tool-call",
            "timestampMs": 120,
            "tool": {
                "id": "tool-uuid-1",
                "name": "Whale Tracker",
                "method": "get_whales",
            },
        },
        {
            "stepType": "retry",
            "timestampMs": 420,
            "attempt": 2,
            "message": "Retrying after transient provider timeout",
        },
    ],
    "diagnostics": {
        "selection": {
            "selectedPolicy": "exploratory",
            "debugScoutDeepMode": "deep-heavy",
            "plannerReasoningStage": "full",
            "scoutEnabled": True,
            "oneShotBias": False,
            "candidateMethodCount": 12,
            "scoutProbeStatus": "ready",
            "scoutProbeAdequacy": "limited",
            "scoutProbeConfidence": 0.81,
            "scoutMetadataConfidence": 0.74,
            "scoutProbeQuerySafeCandidateCount": 8,
            "scoutProbeRankedMethodCount": 5,
            "scoutProbeAmbiguityPoolCount": 2,
            "scoutProbeShortlistedMethodCount": 2,
            "scoutProbeMissingCapability": None,
            "scoutPrePlanProbeCalls": 1,
            "scoutPrePlanProbeBudgetReasonCode": None,
            "scoutChangedInitialPlan": True,
            "scoutChangedPlannerReasoningStage": True,
            "scoutInitialSelectedPolicy": "exploratory",
            "scoutInitialPlannerReasoningStage": "focused",
            "scoutInitialReasonCode": "metadata_quality_deep_light",
            "scoutFinalReasonCode": "probe_detected_inadequacy",
            "scoutEvidenceAttachedToPlanning": True,
            "scoutLlmSelectionUsed": True,
            "scoutLlmSelectionFallback": False,
            "scoutLlmSelectionLatencyMs": 183.0,
            "selectedTools": [
                {
                    "toolId": "tool-uuid-1",
                    "toolName": "Whale Tracker",
                    "selectedMethodCount": 2,
                    "selectedMethods": ["get_whales", "get_whale_summary"],
                    "omittedSelectedMethodCount": 1,
                }
            ],
        },
    },
}

MOCK_ORCHESTRATION_METRICS: dict[str, Any] = {
    "parityStage": "synthesis_v2",
    "orchestrationMode": "query",
    "firstPassSuccess": True,
    "capabilityMissSignaled": False,
    "rediscoveryExecuted": False,
}

MOCK_ERROR_RESPONSE: dict[str, Any] = {
    "error": "Insufficient funds. Set a spending cap in the dashboard.",
    "code": "insufficient_allowance",
}

MOCK_CAPABILITY_MISS_RESULT: dict[str, Any] = {
    "response": (
        "I could not satisfy this request with grounded tool coverage. "
        "Try narrowing the venue or asking for supported market data instead."
    ),
    "toolsUsed": [],
    "cost": {
        "totalCostUsd": "0.000000",
        "toolCostUsd": "0.000000",
        "modelCostUsd": "0.000000",
    },
    "durationMs": 950,
    "outcomeType": "capability_miss",
    "capabilityMiss": {
        "message": (
            "I could not satisfy this request with grounded tool coverage. "
            "Try narrowing the venue or asking for supported market data instead."
        ),
        "missingCapabilities": [
            "Need venue coverage that no selected tool exposes."
        ],
        "suggestedRewrites": [
            "Ask for a supported venue instead of Bybit.",
            "Request Polymarket market liquidity rather than perpetual order-book data.",
            "Name the exact supported market you want analyzed.",
        ],
        "originalQuery": (
            "Using only Polymarket data, give me live order-book imbalance for "
            "BTC perpetuals on Bybit."
        ),
    },
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

MOCK_SSE_TRACE_LINES = [
    "data: "
    + json.dumps(
        {
            "type": "developer-trace",
            "trace": {
                "summary": {"retryCount": 2, "loopCount": 1},
                "timeline": [{"stepType": "retry", "attempt": 2}],
            },
        }
    ),
    "data: "
    + json.dumps(
        {
            "type": "developer-trace",
            "trace": {
                "summary": {"fallbackCount": 1, "completionChecks": 3},
                "timeline": [{"stepType": "fallback", "message": "Switched branch"}],
            },
        }
    ),
    "data: "
    + json.dumps(
        {
            "type": "done",
            "result": {
                "response": "Resolved with fallback branch.",
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

MOCK_SSE_ERROR_LINES = [
    'data: {"type":"tool-status","status":"executing","tool":{"id":"","name":""}}',
    (
        'data: {"type":"error","error":"Query failed before completion",'
        '"code":"query_failed","scope":"query","reasonCode":"execution_failed"}'
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
        request=httpx.Request("POST", "https://www.ctxprotocol.com/api/v1/query"),
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


def _make_done_stream_response(result: dict[str, Any]) -> _FakeStreamResponse:
    """Wrap a query result in a minimal done-event SSE response."""
    return _FakeStreamResponse(
        [
            "data: "
            + json.dumps(
                {
                    "type": "done",
                    "result": result,
                }
            ),
            "data: [DONE]",
        ]
    )


# ============================================================================
# Tests: query.run()
# ============================================================================


class TestQueryRun:
    """Tests for client.query.run() — consumes SSE and returns the final result."""

    async def test_sends_correct_request_body_string(self) -> None:
        """String shorthand sends query with stream: true."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            await client.query.run("What are the top whale movements?")

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "What are the top whale movements?",
                "stream": True,
            },
            extra_headers=None,
        )

    async def test_sends_correct_request_body_with_tools(self) -> None:
        """Options dict with tools sends tool IDs."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            await client.query.run(
                query="Analyze whale activity",
                tools=["tool-uuid-1", "tool-uuid-2"],
            )

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "tools": ["tool-uuid-1", "tool-uuid-2"],
                "stream": True,
            },
            extra_headers=None,
        )

    async def test_forwards_model_and_data_options(self) -> None:
        """Optional model and data controls are forwarded in request body."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        success_with_data = {
            **MOCK_SUCCESS_RESPONSE,
            "data": {"summary": "tool output"},
            "dataUrl": "https://example.public.blob.vercel-storage.com/data.json",
            "developerTrace": MOCK_DEVELOPER_TRACE,
            "orchestrationMetrics": MOCK_ORCHESTRATION_METRICS,
        }

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(success_with_data)
            result = await client.query.run(
                query="Analyze whale activity",
                agent_model_id="kimi-k2.6-model",
                response_shape="answer_with_evidence",
                include_data=True,
                include_data_url=True,
                include_developer_trace=True,
            )

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "stream": True,
                "agentModelId": "kimi-k2.6-model",
                "responseShape": "answer_with_evidence",
                "includeData": True,
                "includeDataUrl": True,
                "includeDeveloperTrace": True,
            },
            extra_headers=None,
        )
        assert result.data == {"summary": "tool output"}
        assert (
            result.data_url
            == "https://example.public.blob.vercel-storage.com/data.json"
        )
        assert result.developer_trace is not None
        assert result.developer_trace.summary is not None
        assert result.developer_trace.summary.retry_count == 2
        assert result.orchestration_metrics is not None
        assert result.orchestration_metrics.first_pass_success is True

    async def test_parses_developer_trace_when_present(self) -> None:
        """developerTrace payload is deserialized into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                {
                    **MOCK_SUCCESS_RESPONSE,
                    "developerTrace": MOCK_DEVELOPER_TRACE,
                }
            )
            result = await client.query.run("test query")

        assert result.developer_trace is not None
        assert result.developer_trace.summary is not None
        assert result.developer_trace.summary.retry_count == 2
        assert result.developer_trace.summary.tool_calls == 4
        assert result.developer_trace.diagnostics is not None
        assert result.developer_trace.diagnostics.selection is not None
        selection = result.developer_trace.diagnostics.selection
        assert selection.selected_policy == "exploratory"
        assert selection.one_shot_bias is False
        assert selection.scout_probe_query_safe_candidate_count == 8
        assert selection.scout_probe_ranked_method_count == 5
        assert selection.scout_probe_ambiguity_pool_count == 2
        assert selection.scout_pre_plan_probe_calls == 1
        assert selection.scout_changed_initial_plan is True
        assert selection.scout_initial_selected_policy == "exploratory"
        assert selection.scout_initial_planner_reasoning_stage == "focused"
        assert selection.scout_initial_reason_code == "metadata_quality_deep_light"
        assert selection.scout_final_reason_code == "probe_detected_inadequacy"
        assert selection.scout_llm_selection_used is True

    async def test_developer_trace_is_none_when_not_returned(self) -> None:
        """Query result keeps developer_trace unset when API omits it."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            result = await client.query.run("test query")

        assert result.developer_trace is None

    async def test_builds_synthetic_trace_when_requested_and_missing(self) -> None:
        """When include_developer_trace is set, SDK synthesizes fallback trace if API omits it."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            result = await client.query.run(
                query="test query",
                include_developer_trace=True,
            )

        assert result.developer_trace is not None
        assert result.developer_trace.summary is not None
        assert result.developer_trace.summary.tool_calls == 4
        assert result.developer_trace.summary.retry_count == 0
        assert result.developer_trace.timeline is not None
        assert len(result.developer_trace.timeline) == 2

    def test_query_options_supports_public_aliases(self) -> None:
        """QueryOptions accepts supported public aliases."""
        trace_alias = QueryOptions(query="test", includeDeveloperTrace=True)
        agent_model_alias = QueryOptions(
            query="test", agentModelId="kimi-k2.6-model"
        )
        resume_alias = QueryOptions(
            query="test",
            resumeFrom={
                "sessionId": "11111111-1111-4111-8111-111111111111",
                "attemptId": "22222222-2222-4222-8222-222222222222",
            },
        )
        fork_alias = QueryOptions(
            query="test",
            forkFrom={
                "sessionId": "11111111-1111-4111-8111-111111111111",
                "attemptId": "22222222-2222-4222-8222-222222222222",
                "reason": "manual_fork",
            },
        )
        response_shape_alias = QueryOptions(
            query="test",
            responseShape="evidence_only",
        )

        assert trace_alias.include_developer_trace is True
        assert (
            trace_alias.model_dump(by_alias=True)["includeDeveloperTrace"] is True
        )
        assert agent_model_alias.agent_model_id == "kimi-k2.6-model"
        assert (
            agent_model_alias.model_dump(by_alias=True)["agentModelId"]
            == "kimi-k2.6-model"
        )
        assert response_shape_alias.response_shape == "evidence_only"
        assert (
            response_shape_alias.model_dump(by_alias=True)["responseShape"]
            == "evidence_only"
        )
        assert resume_alias.resume_from is not None
        assert (
            resume_alias.model_dump(by_alias=True)["resumeFrom"]["sessionId"]
            == "11111111-1111-4111-8111-111111111111"
        )
        assert fork_alias.fork_from is not None
        assert fork_alias.model_dump(by_alias=True)["forkFrom"]["reason"] == "manual_fork"

    async def test_forwards_resume_from_for_run(self) -> None:
        """resume_from is forwarded to the query endpoint."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            await client.query.run(
                query="test query",
                resume_from={
                    "sessionId": "11111111-1111-4111-8111-111111111111",
                    "attemptId": "22222222-2222-4222-8222-222222222222",
                },
            )

        body = mock_stream.call_args.kwargs["json_body"]
        assert body["resumeFrom"] == {
            "sessionId": "11111111-1111-4111-8111-111111111111",
            "attemptId": "22222222-2222-4222-8222-222222222222",
        }

    async def test_includes_query_session_in_run_result(self) -> None:
        """querySession payload is deserialized into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        response = {
            **MOCK_SUCCESS_RESPONSE,
            "querySession": {
                "sessionId": "11111111-1111-4111-8111-111111111111",
                "attemptId": "22222222-2222-4222-8222-222222222222",
                "parentAttemptId": None,
                "rootAttemptId": "22222222-2222-4222-8222-222222222222",
                "mode": "initial",
                "origin": "initial_request",
                "status": "completed",
                "checkpoint": {
                    "currentStage": "synthesis",
                    "latestCheckpointArtifactId": "artifact-1",
                    "canonicalDatasetId": "dataset-1",
                    "executionProgramCurrentRevisionId": "rev-1",
                },
            },
        }

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(response)
            result = await client.query.run("test query")

        assert result.query_session is not None
        assert result.query_session.session_id == "11111111-1111-4111-8111-111111111111"
        assert result.query_session.checkpoint.current_stage == "synthesis"

    async def test_sends_idempotency_header_when_provided(self) -> None:
        """Explicit idempotency key is forwarded as request header."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            await client.query.run(
                query="Analyze whale activity",
                tools=["tool-uuid-1"],
                idempotency_key="7b31f437-61ed-4f76-8a6e-ed2f0766ffb8",
            )

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "tools": ["tool-uuid-1"],
                "stream": True,
            },
            extra_headers={"Idempotency-Key": "7b31f437-61ed-4f76-8a6e-ed2f0766ffb8"},
        )

    async def test_parses_success_response_into_query_result(self) -> None:
        """Successful API response is deserialized into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
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

    async def test_preserves_structured_evidence_envelope(self) -> None:
        """Structured evidence fields are exposed on QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_EVIDENCE_RESPONSE)
            result = await client.query.run(
                query="Where is BTC flowing today?",
                response_shape="evidence_only",
            )

        assert result.response_shape == "evidence_only"
        assert result.summary == MOCK_EVIDENCE_RESPONSE["summary"]
        assert result.evidence is not None
        assert result.evidence.facts[0].label == "Net BTC exchange flow"
        assert result.artifacts is not None
        assert result.artifacts.canonical_data_ref is not None
        assert result.artifacts.canonical_data_ref.dataset_id == "dataset-1"
        assert result.usage is not None
        assert result.usage.outcome_type == "answer"

    async def test_parses_shared_grounded_answer_fixture(self) -> None:
        """Shared SDK fixture preserves computed artifacts and grounding."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                SHARED_QUERY_FIXTURE["groundedAnswer"]
            )
            result = await client.query.run(
                query="Compare BTC and ETH returns.",
                include_developer_trace=True,
            )

        assert result.outcome_type == "answer"
        assert result.grounding is not None
        assert result.grounding.available_tool_count == 4
        assert result.grounding.selected_method_count == 3
        assert result.grounding.tool_call_count == 2
        assert result.grounding.grounded is True
        assert result.computed_artifacts is not None
        assert len(result.computed_artifacts) == 1
        chart = result.computed_artifacts[0]
        assert chart.kind == "chart"
        assert chart.title == "BTC vs ETH Cumulative Return"
        assert chart.spec.x_key == "date"
        assert chart.data[1]["btcReturn"] == 0.034
        assert result.developer_trace is not None
        assert result.developer_trace.diagnostics is not None
        assert result.developer_trace.diagnostics.execution is not None
        execution = result.developer_trace.diagnostics.execution
        assert execution.tool_registry is not None
        assert execution.tool_registry.available_tool_count == 4

    async def test_preserves_expanded_chart_artifact_specs(self) -> None:
        """Expanded chart artifacts remain structured for Python SDK consumers."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        response = {
            **MOCK_SUCCESS_RESPONSE,
            "computedArtifacts": [
                {
                    "kind": "chart",
                    "title": "Correlation Heatmap",
                    "spec": {
                        "type": "heatmap",
                        "xKey": "x",
                        "yKey": "y",
                        "valueKey": "value",
                        "expectedMeasures": ["correlation"],
                        "series": [
                            {
                                "key": "value",
                                "label": "Correlation",
                                "satisfies": "correlation",
                            }
                        ],
                        "yAxis": {"label": "Asset"},
                    },
                    "data": [
                        {"x": "BTC", "y": "BTC", "value": 1},
                        {"x": "BTC", "y": "ETH", "value": 0.82},
                    ],
                },
                {
                    "kind": "chart",
                    "title": "BTC Daily Candles",
                    "spec": {
                        "type": "candlestick",
                        "xKey": "time",
                        "series": [{"key": "close", "label": "Close"}],
                        "xAxis": {"type": "time", "label": "Date"},
                        "yAxis": {"label": "Price", "format": "currency"},
                        "ohlc": {
                            "openKey": "open",
                            "highKey": "high",
                            "lowKey": "low",
                            "closeKey": "close",
                        },
                    },
                    "data": [
                        {
                            "time": "2026-04-01",
                            "open": 100,
                            "high": 104,
                            "low": 98,
                            "close": 102,
                        }
                    ],
                },
                {
                    "kind": "chart",
                    "title": "Probability and Volume",
                    "spec": {
                        "type": "composed",
                        "xKey": "market",
                        "expectedMeasures": ["probability", "volume"],
                        "series": [
                            {
                                "key": "probability",
                                "label": "Probability",
                                "satisfies": "probability",
                                "yAxis": "left",
                            },
                            {
                                "key": "volumeUsd",
                                "label": "Volume",
                                "satisfies": "volume",
                                "yAxis": "right",
                            },
                        ],
                        "yAxis": {"format": "percent", "valueScale": "fraction"},
                        "yAxisRight": {"format": "currency"},
                    },
                    "data": [
                        {"market": "A", "probability": 0.42, "volumeUsd": 1_500_000},
                        {"market": "B", "probability": 0.31, "volumeUsd": 900_000},
                    ],
                },
            ],
        }

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(response)
            result = await client.query.run("Render richer chart artifacts")

        assert result.computed_artifacts is not None
        assert len(result.computed_artifacts) == 3
        heatmap = result.computed_artifacts[0]
        assert heatmap.kind == "chart"
        assert heatmap.spec.type == "heatmap"
        assert heatmap.spec.expected_measures == ["correlation"]
        assert heatmap.spec.series[0].satisfies == "correlation"
        assert heatmap.spec.value_key == "value"
        assert heatmap.data[1]["value"] == 0.82
        candlestick = result.computed_artifacts[1]
        assert candlestick.kind == "chart"
        assert candlestick.spec.type == "candlestick"
        assert candlestick.spec.ohlc is not None
        assert candlestick.spec.ohlc.close_key == "close"
        assert candlestick.spec.x_axis is not None
        assert candlestick.spec.x_axis.label == "Date"
        mixed_axis = result.computed_artifacts[2]
        assert mixed_axis.kind == "chart"
        assert mixed_axis.spec.y_axis is not None
        assert mixed_axis.spec.y_axis.value_scale == "fraction"
        assert mixed_axis.spec.y_axis_right is not None
        assert mixed_axis.spec.y_axis_right.format == "currency"
        assert mixed_axis.spec.series[1].y_axis == "right"
        assert mixed_axis.spec.series[1].satisfies == "volume"

    async def test_parses_rendered_image_artifact(self) -> None:
        """Rendered image artifacts (kind: image) deserialize without error."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        response = {
            **MOCK_SUCCESS_RESPONSE,
            "computedArtifacts": [
                {
                    "kind": "image",
                    "url": "https://blob.example.com/charts/abc123.png",
                    "alt": "Sector ETF YTD normalized performance chart",
                    "title": "Sector ETF YTD Performance",
                    "contentHash": "abc123",
                    "bytes": 320749,
                    "width": 1926,
                    "height": 1030,
                },
            ],
        }
        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(response)
            result = await client.query.run("Render a chart image")

        assert result.computed_artifacts is not None
        assert len(result.computed_artifacts) == 1
        image = result.computed_artifacts[0]
        assert image.kind == "image"
        assert image.url == "https://blob.example.com/charts/abc123.png"
        assert image.title == "Sector ETF YTD Performance"
        assert image.content_hash == "abc123"
        assert image.bytes == 320749
        assert image.width == 1926
        assert image.height == 1030

    async def test_parses_shared_ungrounded_capability_miss_fixture(self) -> None:
        """Ungrounded runtime outcomes deserialize as capability_miss."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                SHARED_QUERY_FIXTURE["ungroundedCapabilityMiss"]
            )
            result = await client.query.run(
                query="Compare BTC and ETH returns.",
            )

        assert result.outcome_type == "capability_miss"
        assert result.grounding is not None
        assert result.grounding.available_tool_count == 3
        assert result.grounding.grounded is False
        assert result.capability_miss is not None
        assert result.capability_miss.missing_capabilities == [
            "runtime_did_not_invoke_selected_tools"
        ]

    async def test_returns_structured_capability_miss_result(self) -> None:
        """Structured capability miss results deserialize into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                MOCK_CAPABILITY_MISS_RESULT
            )
            result = await client.query.run(
                query=(
                    "Using only Polymarket data, give me live order-book imbalance "
                    "for BTC perpetuals on Bybit."
                ),
            )

        assert result.outcome_type == "capability_miss"
        assert result.capability_miss is not None
        assert result.capability_miss.missing_capabilities == [
            "Need venue coverage that no selected tool exposes."
        ]
        assert len(result.capability_miss.suggested_rewrites) == 3

    async def test_raises_if_run_stream_ends_before_done_event(self) -> None:
        """Run raises when the stream ends without a done event."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _FakeStreamResponse(["data: [DONE]"])

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert "Streaming query ended before done event" in str(exc_info.value)

    async def test_raises_context_error_on_stream_error_event(self) -> None:
        """Run raises the terminal SSE error when no done event arrives."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _FakeStreamResponse(MOCK_SSE_ERROR_LINES)

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert "Query failed before completion" in str(exc_info.value)
        assert exc_info.value.code == "query_failed"

    async def test_raises_context_error_on_no_wallet(self) -> None:
        """no_wallet error is propagated correctly."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.side_effect = ContextError(
                message="Account not fully set up.",
                code="no_wallet",
                status_code=400,
            )

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert exc_info.value.code == "no_wallet"

    async def test_raises_context_error_on_query_failed(self) -> None:
        """query_failed error is propagated correctly."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.side_effect = ContextError(
                message="Query failed: Tool execution timed out",
                code="query_failed",
                status_code=422,
            )

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert exc_info.value.code == "query_failed"

    async def test_raises_context_error_on_insufficient_allowance(self) -> None:
        """HTTP errors before stream setup still propagate correctly."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.side_effect = ContextError(
                message="Insufficient funds. Set a spending cap in the dashboard.",
                code="insufficient_allowance",
                status_code=402,
            )

            with pytest.raises(ContextError) as exc_info:
                await client.query.run("test query")

        assert "Insufficient funds" in str(exc_info.value)
        assert exc_info.value.code == "insufficient_allowance"


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
                "stream": True,
            },
            extra_headers=None,
        )

    async def test_forwards_fork_from_for_stream(self) -> None:
        """fork_from is forwarded to the query endpoint."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query="What are whale movements?",
                fork_from={
                    "sessionId": "11111111-1111-4111-8111-111111111111",
                    "attemptId": "22222222-2222-4222-8222-222222222222",
                    "reason": "manual_fork",
                },
            ):
                events.append(event)

        body = mock_stream.call_args.kwargs["json_body"]
        assert body["forkFrom"] == {
            "sessionId": "11111111-1111-4111-8111-111111111111",
            "attemptId": "22222222-2222-4222-8222-222222222222",
            "reason": "manual_fork",
        }

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
        assert call_kwargs[1]["extra_headers"] is None

    async def test_stream_forwards_model_and_data_options(self) -> None:
        """Streaming request forwards model and data options."""
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
                agent_model_id="claude-sonnet-model",
                include_data=True,
                include_data_url=True,
                include_developer_trace=True,
            ):
                events.append(event)

        call_kwargs = mock_stream.call_args
        assert (
            call_kwargs[1]["json_body"]["agentModelId"]
            == "claude-sonnet-model"
        )
        assert call_kwargs[1]["json_body"]["includeData"] is True
        assert call_kwargs[1]["json_body"]["includeDataUrl"] is True
        assert call_kwargs[1]["json_body"]["includeDeveloperTrace"] is True

    async def test_stream_yields_structured_capability_miss_done_events(self) -> None:
        """Structured capability misses are preserved in stream mode."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _make_done_stream_response(MOCK_CAPABILITY_MISS_RESULT)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query=(
                    "Using only Polymarket data, give me live order-book imbalance "
                    "for BTC perpetuals on Bybit."
                ),
            ):
                events.append(event)

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        assert done_events[0].result.outcome_type == "capability_miss"
        assert done_events[0].result.capability_miss is not None

    async def test_stream_handles_trace_events_and_aggregates_done_trace(self) -> None:
        """Trace events are emitted and merged into done.result.developer_trace."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_TRACE_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test"):
                events.append(event)

        trace_events = [
            e for e in events if isinstance(e, QueryStreamDeveloperTraceEvent)
        ]
        assert len(trace_events) == 2
        assert trace_events[0].trace.summary is not None
        assert trace_events[0].trace.summary.retry_count == 2
        assert trace_events[1].trace.summary is not None
        assert trace_events[1].trace.summary.fallback_count == 1

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        done_trace = done_events[0].result.developer_trace
        assert done_trace is not None
        assert done_trace.summary is not None
        assert done_trace.summary.retry_count == 2
        assert done_trace.summary.fallback_count == 1
        assert done_trace.summary.completion_checks == 3
        assert done_trace.timeline is not None
        assert len(done_trace.timeline) == 2

    async def test_stream_accepts_non_dict_completeness_evaluations(self) -> None:
        """Developer trace completeness diagnostics tolerate mixed evaluation payloads."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            [
                "data: "
                + json.dumps(
                    {
                        "type": "developer-trace",
                        "trace": {
                            "diagnostics": {
                                "completeness": {
                                    "evaluations": ["[OMITTED:CIRCULAR_REFERENCE]"],
                                    "repairEvents": [
                                        {
                                            "attempt": 2,
                                            "outcome": "patched",
                                            "semanticRetryCount": 1,
                                            "maxSemanticRetries": 2,
                                            "strategy": "patch",
                                            "summary": "Added one extra market snapshot call.",
                                            "failReason": None,
                                            "requestedReplan": False,
                                            "hadSyntaxFix": False,
                                            "editCount": 1,
                                            "skipReason": None,
                                            "boundedAnswerReason": None,
                                            "blockingDiagnostics": [],
                                        }
                                    ],
                                    "triggerNeedsDifferentTools": False,
                                    "triggerMissingCapability": None,
                                }
                            }
                        },
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "result": MOCK_SUCCESS_RESPONSE,
                    }
                ),
                "data: [DONE]",
            ]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                "test",
                include_developer_trace=True,
            ):
                events.append(event)

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        done_trace = done_events[0].result.developer_trace
        assert done_trace is not None
        assert done_trace.diagnostics is not None
        assert done_trace.diagnostics.completeness is not None
        assert done_trace.diagnostics.completeness.evaluations == [
            "[OMITTED:CIRCULAR_REFERENCE]"
        ]
        assert done_trace.diagnostics.completeness.repair_events is not None
        assert done_trace.diagnostics.completeness.repair_events[0].outcome == "patched"

    async def test_stream_yields_structured_error_events(self) -> None:
        """Structured stream error payloads are surfaced as typed events."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_ERROR_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream("test"):
                events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], QueryStreamToolStatusEvent)
        assert isinstance(events[1], QueryStreamErrorEvent)
        assert events[1].error == "Query failed before completion"
        assert events[1].code == "query_failed"
        assert events[1].scope == "query"
        assert events[1].reason_code == "execution_failed"

    async def test_stream_builds_synthetic_done_trace_when_requested_and_missing(self) -> None:
        """When include_developer_trace is set, stream done result gets fallback trace if backend omits it."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(MOCK_SSE_LINES)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query="test query",
                include_developer_trace=True,
            ):
                events.append(event)

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        done_trace = done_events[0].result.developer_trace
        assert done_trace is not None
        assert done_trace.summary is not None
        assert done_trace.summary.tool_calls == 2
        assert done_trace.timeline is not None
        assert len(done_trace.timeline) > 0

    async def test_stream_forwards_idempotency_header(self) -> None:
        """Streaming query forwards explicit idempotency key."""
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
                tools=["tool-1"],
                idempotency_key="9131f6f5-cc3e-4b61-97e5-e850f36eff5d",
            ):
                events.append(event)

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "test",
                "tools": ["tool-1"],
                "stream": True,
            },
            extra_headers={"Idempotency-Key": "9131f6f5-cc3e-4b61-97e5-e850f36eff5d"},
        )

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


class TestQueryJobs:
    """Tests for durable async query job helpers."""

    async def test_start_creates_query_job(self) -> None:
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        payload = {
            "status": "running",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "pollingTool": "context_query_status",
            "message": "running",
            "progress": None,
            "querySession": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:00:00.000Z",
        }

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = payload
            job = await client.query.start(
                query="long query",
                response_shape="evidence_only",
                include_data_url=True,
                idempotency_key="21118fda-33be-4d66-8df5-0e50b3371f54",
            )

        assert job.job_id == "11111111-1111-4111-8111-111111111111"
        mock_fetch.assert_called_once_with(
            "/api/v1/query/jobs",
            method="POST",
            json_body={
                "query": "long query",
                "stream": False,
                "responseShape": "evidence_only",
                "includeDataUrl": True,
            },
            extra_headers={
                "Idempotency-Key": "21118fda-33be-4d66-8df5-0e50b3371f54",
            },
        )

    async def test_start_omits_null_optional_fields(self) -> None:
        """Regression: optional fields must be absent (not JSON null) in the job body.

        The /api/v1/query/jobs endpoint validates the body with a Zod schema whose
        optional fields accept `undefined`/absent but reject `null`. Sending
        `"tools": null` (as the SDK previously did) made query.start() 400 with
        "Invalid query job request". Optional fields must be omitted when unset,
        matching the TS SDK (JSON.stringify drops undefined keys).
        """
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        payload = {
            "status": "running",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "pollingTool": "context_query_status",
            "message": "running",
            "progress": None,
            "querySession": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:00:00.000Z",
        }

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = payload
            await client.query.start(query="long query")

        body = mock_fetch.call_args.kwargs["json_body"]
        # No optional field may serialize to JSON null.
        assert all(value is not None for value in body.values()), body
        # tools specifically must be omitted when not supplied (was the 400 trigger).
        assert "tools" not in body, body
        # query is always present and non-null.
        assert body["query"] == "long query"

    async def test_get_status_fetches_query_job(self) -> None:
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        payload = {
            "status": "completed",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "progress": None,
            "querySession": None,
            "result": MOCK_SUCCESS_RESPONSE,
            "error": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:01:00.000Z",
            "completedAt": "2026-06-14T00:01:00.000Z",
        }

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = payload
            status = await client.query.get_status(
                "11111111-1111-4111-8111-111111111111"
            )

        assert status.status == "completed"
        assert status.result is not None
        assert status.result.response == MOCK_SUCCESS_RESPONSE["response"]
        mock_fetch.assert_called_once_with(
            "/api/v1/query/jobs/11111111-1111-4111-8111-111111111111"
        )

    async def test_poll_waits_until_query_job_completes(self) -> None:
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        running_payload = {
            "status": "running",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "progress": None,
            "querySession": None,
            "result": None,
            "error": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:00:30.000Z",
            "completedAt": None,
        }
        completed_payload = {
            **running_payload,
            "status": "completed",
            "result": MOCK_SUCCESS_RESPONSE,
            "updatedAt": "2026-06-14T00:01:00.000Z",
            "completedAt": "2026-06-14T00:01:00.000Z",
        }

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [running_payload, completed_payload]
            status = await client.query.poll(
                "11111111-1111-4111-8111-111111111111",
                interval_ms=1,
                timeout_ms=1000,
            )

        assert status.status == "completed"
        assert mock_fetch.await_count == 2

    async def test_run_or_poll_returns_completed_query_result(self) -> None:
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        start_payload = {
            "status": "running",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "pollingTool": "context_query_poll",
            "message": "running",
            "progress": None,
            "querySession": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:00:00.000Z",
        }
        completed_payload = {
            "status": "completed",
            "jobId": "11111111-1111-4111-8111-111111111111",
            "progress": None,
            "querySession": None,
            "result": MOCK_SUCCESS_RESPONSE,
            "error": None,
            "createdAt": "2026-06-14T00:00:00.000Z",
            "updatedAt": "2026-06-14T00:01:00.000Z",
            "completedAt": "2026-06-14T00:01:00.000Z",
        }

        with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [start_payload, completed_payload]
            result = await client.query.run_or_poll(
                "Analyze whale activity",
                interval_ms=1,
                timeout_ms=1000,
            )

        assert result.response == MOCK_SUCCESS_RESPONSE["response"]
        assert mock_fetch.await_count == 2
        assert mock_fetch.await_args_list[0].args[0] == "/api/v1/query/jobs"
        assert (
            mock_fetch.await_args_list[1].args[0]
            == "/api/v1/query/jobs/11111111-1111-4111-8111-111111111111"
        )
