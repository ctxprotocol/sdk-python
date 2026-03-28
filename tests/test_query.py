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
    QueryOptions,
    QueryResult,
    QueryStreamDeveloperTraceEvent,
    QueryStreamDoneEvent,
    QueryStreamErrorEvent,
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
            "selectedDepth": "deep",
            "deepMode": "deep-heavy",
            "debugScoutDeepMode": "deep-heavy",
            "plannerReasoningStage": "full",
            "scoutEnabled": True,
            "preserveFastOneShot": False,
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
            "scoutInitialSelectedDepth": "deep",
            "scoutInitialDeepMode": "deep-light",
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
        "clarification": {
            "orchestrationMode": "query",
            "rolloutStage": "candidate",
            "shadowMode": False,
            "policy": "return",
            "outcomeType": "clarification_required",
            "triggered": True,
            "optionCount": 2,
            "candidateCount": 3,
            "viableCandidateCount": 2,
            "recommendedOptionId": "tool-1:analyze_event_outcome_liquidity",
            "recommendedOptionReason": "Event-level interpretation stays broadest.",
            "autoResolved": False,
            "autoSelectEnabled": False,
            "assumptionMade": None,
            "missingCapability": None,
            "decisionReasonCode": "semantic_scope_ambiguity",
            "decisionSignals": ["multi_outcome_market_scope"],
            "evidenceSources": {
                "usesMethodSchemas": True,
                "usesProbeArgs": True,
                "usesMethodMetadata": True,
                "usesToolSelectionContext": True,
                "usesLlmSelection": True,
            },
            "comparedOptionIds": [
                "tool-1:analyze_event_outcome_liquidity",
                "tool-1:analyze_market_liquidity",
            ],
            "decisionStrategy": "llm_primary",
            "judgeAttempted": True,
            "judgeApplied": True,
            "judgeOutcomeType": "clarification_required",
            "judgeConfidence": 0.84,
            "judgeReason": "Need the user to choose event-wide or single-outcome scope.",
            "judgeError": None,
            "validatorReason": None,
            "fallbackReason": None,
            "copyStrategy": "deterministic",
            "rewriteAttempted": False,
            "rewriteApplied": False,
            "rewriteError": None,
            "candidateSummaries": [],
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

MOCK_CLARIFICATION_RESULT: dict[str, Any] = {
    "response": (
        "I found multiple plausible ways to interpret this request. "
        "Which direction should I take?"
    ),
    "toolsUsed": [],
    "cost": {
        "totalCostUsd": "0.000000",
        "toolCostUsd": "0.000000",
        "modelCostUsd": "0.000000",
    },
    "durationMs": 1100,
    "outcomeType": "clarification_required",
    "clarification": {
        "question": "Which direction should I take?",
        "options": [
            {
                "id": "tool-1:analyze_event_outcome_liquidity",
                "toolId": "tool-1",
                "toolName": "Polymarket",
                "methodName": "analyze_event_outcome_liquidity",
                "label": "Compare event-level liquidity",
                "description": "Polymarket -> analyze_event_outcome_liquidity",
                "fitScore": 9,
                "recommended": True,
            },
            {
                "id": "tool-1:analyze_market_liquidity",
                "toolId": "tool-1",
                "toolName": "Polymarket",
                "methodName": "analyze_market_liquidity",
                "label": "Analyze one specific outcome",
                "description": "Polymarket -> analyze_market_liquidity",
                "fitScore": 5,
                "recommended": False,
            },
        ],
        "allowFreeform": True,
        "recommendedOptionId": "tool-1:analyze_event_outcome_liquidity",
        "originalQuery": "Analyze liquidity for the World Cup winner market",
    },
}

MOCK_AUTO_SELECTED_RESULT: dict[str, Any] = {
    **MOCK_SUCCESS_RESPONSE,
    "outcomeType": "answer",
    "assumptionMade": {
        "mode": "auto",
        "optionId": "tool-1:analyze_event_outcome_liquidity",
        "label": "Compare event-level liquidity",
        "reason": (
            "Recommended because Polymarket.analyze_event_outcome_liquidity "
            "ranked highest after comparing probe fit, method contract details, "
            "and grounded query eligibility."
        ),
    },
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
                "tools": None,
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
                answer_model_id="glm-model",
                response_shape="answer_with_evidence",
                include_data=True,
                include_data_url=True,
                include_developer_trace=True,
                query_depth="auto",
                debug_scout_deep_mode="deep-light",
            )

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "tools": None,
                "stream": True,
                "answerModelId": "glm-model",
                "responseShape": "answer_with_evidence",
                "includeData": True,
                "includeDataUrl": True,
                "includeDeveloperTrace": True,
                "queryDepth": "auto",
                "debugScoutDeepMode": "deep-light",
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
        assert selection.scout_probe_query_safe_candidate_count == 8
        assert selection.scout_probe_ranked_method_count == 5
        assert selection.scout_probe_ambiguity_pool_count == 2
        assert selection.scout_pre_plan_probe_calls == 1
        assert selection.scout_changed_initial_plan is True
        assert selection.scout_initial_selected_depth == "deep"
        assert selection.scout_initial_deep_mode == "deep-light"
        assert selection.scout_initial_planner_reasoning_stage == "focused"
        assert selection.scout_initial_reason_code == "metadata_quality_deep_light"
        assert selection.scout_final_reason_code == "probe_detected_inadequacy"
        assert selection.scout_llm_selection_used is True
        assert result.developer_trace.diagnostics.clarification is not None
        clarification = result.developer_trace.diagnostics.clarification
        assert clarification.decision_strategy == "llm_primary"
        assert clarification.judge_confidence == 0.84

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

    def test_query_options_supports_query_depth_and_alias(self) -> None:
        """QueryOptions accepts query_depth and queryDepth aliases."""
        snake_case = QueryOptions(query="test", query_depth="fast")
        camel_case = QueryOptions(query="test", queryDepth="deep")
        trace_alias = QueryOptions(query="test", includeDeveloperTrace=True)
        deep_mode_alias = QueryOptions(query="test", debugScoutDeepMode="deep-heavy")
        clarification_alias = QueryOptions(query="test", clarificationPolicy="auto")
        answer_model_alias = QueryOptions(query="test", answerModelId="glm-model")
        response_shape_alias = QueryOptions(
            query="test",
            responseShape="evidence_only",
        )

        assert snake_case.query_depth == "fast"
        assert camel_case.query_depth == "deep"
        assert snake_case.model_dump(by_alias=True)["queryDepth"] == "fast"
        assert trace_alias.include_developer_trace is True
        assert (
            trace_alias.model_dump(by_alias=True)["includeDeveloperTrace"] is True
        )
        assert deep_mode_alias.debug_scout_deep_mode == "deep-heavy"
        assert (
            deep_mode_alias.model_dump(by_alias=True)["debugScoutDeepMode"]
            == "deep-heavy"
        )
        assert clarification_alias.clarification_policy == "auto"
        assert (
            clarification_alias.model_dump(by_alias=True)["clarificationPolicy"]
            == "auto"
        )
        assert answer_model_alias.answer_model_id == "glm-model"
        assert (
            answer_model_alias.model_dump(by_alias=True)["answerModelId"]
            == "glm-model"
        )
        assert response_shape_alias.response_shape == "evidence_only"
        assert (
            response_shape_alias.model_dump(by_alias=True)["responseShape"]
            == "evidence_only"
        )

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

    async def test_forwards_clarification_policy(self) -> None:
        """clarificationPolicy is forwarded to the server payload."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(MOCK_SUCCESS_RESPONSE)
            await client.query.run(
                query="Analyze whale activity",
                clarification_policy="auto",
            )

        mock_stream.assert_called_once_with(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": "Analyze whale activity",
                "tools": None,
                "stream": True,
                "clarificationPolicy": "auto",
            },
            extra_headers=None,
        )

    async def test_returns_structured_clarification_result(self) -> None:
        """Structured clarification results deserialize into QueryResult."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                MOCK_CLARIFICATION_RESULT
            )
            result = await client.query.run(
                query="Analyze liquidity for the World Cup winner market",
                clarification_policy="return",
            )

        assert result.outcome_type == "clarification_required"
        assert result.clarification is not None
        assert len(result.clarification.options) == 2
        assert (
            result.clarification.recommended_option_id
            == "tool-1:analyze_event_outcome_liquidity"
        )

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
                clarification_policy="return",
            )

        assert result.outcome_type == "capability_miss"
        assert result.capability_miss is not None
        assert result.capability_miss.missing_capabilities == [
            "Need venue coverage that no selected tool exposes."
        ]
        assert len(result.capability_miss.suggested_rewrites) == 3

    async def test_preserves_auto_assumption_metadata(self) -> None:
        """Auto clarification decisions keep assumption metadata on answer results."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                MOCK_AUTO_SELECTED_RESULT
            )
            result = await client.query.run(
                query="Analyze liquidity for the World Cup winner market",
                clarification_policy="auto",
            )

        assert result.outcome_type == "answer"
        assert result.assumption_made is not None
        assert result.assumption_made.mode == "auto"
        assert (
            result.assumption_made.option_id
            == "tool-1:analyze_event_outcome_liquidity"
        )

    async def test_raises_when_error_policy_receives_clarification(self) -> None:
        """error clarificationPolicy turns clarification outcomes into ContextError."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                MOCK_CLARIFICATION_RESULT
            )

            with pytest.raises(ContextError) as exc_info:
                await client.query.run(
                    query="Analyze liquidity for the World Cup winner market",
                    clarification_policy="error",
                )

        assert exc_info.value.code == "clarification_required"

    async def test_raises_when_error_policy_receives_capability_miss(self) -> None:
        """error clarificationPolicy turns capability misses into ContextError."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = _make_done_stream_response(
                MOCK_CAPABILITY_MISS_RESULT
            )

            with pytest.raises(ContextError) as exc_info:
                await client.query.run(
                    query=(
                        "Using only Polymarket data, give me live order-book "
                        "imbalance for BTC perpetuals on Bybit."
                    ),
                    clarification_policy="error",
                )

        assert exc_info.value.code == "capability_miss"

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
                "tools": None,
                "stream": True,
            },
            extra_headers=None,
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
                answer_model_id="claude-sonnet-model",
                include_data=True,
                include_data_url=True,
                include_developer_trace=True,
                query_depth="deep",
                debug_scout_deep_mode="deep-heavy",
            ):
                events.append(event)

        call_kwargs = mock_stream.call_args
        assert (
            call_kwargs[1]["json_body"]["answerModelId"]
            == "claude-sonnet-model"
        )
        assert call_kwargs[1]["json_body"]["includeData"] is True
        assert call_kwargs[1]["json_body"]["includeDataUrl"] is True
        assert call_kwargs[1]["json_body"]["includeDeveloperTrace"] is True
        assert call_kwargs[1]["json_body"]["queryDepth"] == "deep"
        assert call_kwargs[1]["json_body"]["debugScoutDeepMode"] == "deep-heavy"

    async def test_stream_forwards_clarification_policy(self) -> None:
        """Streaming request forwards clarificationPolicy."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _FakeStreamResponse(
            ['data: {"type":"text-delta","delta":"result "}', "data: [DONE]"]
        )

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            async for _event in client.query.stream(
                query="test",
                clarification_policy="auto",
            ):
                pass

        call_kwargs = mock_stream.call_args
        assert call_kwargs[1]["json_body"]["clarificationPolicy"] == "auto"

    async def test_stream_yields_structured_clarification_done_events(self) -> None:
        """Structured clarification done payloads are preserved in stream mode."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _make_done_stream_response(MOCK_CLARIFICATION_RESULT)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query="test",
                clarification_policy="return",
            ):
                events.append(event)

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        assert done_events[0].result.outcome_type == "clarification_required"
        assert done_events[0].result.clarification is not None

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
                clarification_policy="return",
            ):
                events.append(event)

        done_events = [e for e in events if isinstance(e, QueryStreamDoneEvent)]
        assert len(done_events) == 1
        assert done_events[0].result.outcome_type == "capability_miss"
        assert done_events[0].result.capability_miss is not None

    async def test_stream_turns_structured_outcomes_into_error_events_for_error_policy(
        self,
    ) -> None:
        """error clarificationPolicy yields terminal error events instead of done."""
        client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
        mock_response = _make_done_stream_response(MOCK_CLARIFICATION_RESULT)

        with patch.object(
            client, "fetch_stream", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = mock_response

            events = []
            async for event in client.query.stream(
                query="test",
                clarification_policy="error",
            ):
                events.append(event)

        error_events = [e for e in events if isinstance(e, QueryStreamErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].outcome_type == "clarification_required"
        assert error_events[0].clarification is not None
        assert not any(isinstance(event, QueryStreamDoneEvent) for event in events)

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
