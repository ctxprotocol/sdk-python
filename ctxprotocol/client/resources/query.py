"""
Query resource for pay-per-response agentic queries.

Unlike ``tools.execute()`` which calls a single tool once (pay-per-request),
the Query resource sends a natural-language question and lets the server
handle the live librarian pipeline (discover -> select -> iterative
execute (with in-loop clarification if needed) -> synthesize -> settle)
and AI synthesis — all for one flat fee.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator

from ctxprotocol.client.types import (
    ContextError,
    QueryClarificationPolicy,
    QueryDeveloperTrace,
    QueryResponseShape,
    QueryResult,
    QueryStreamDeveloperTraceEvent,
    QueryStreamDoneEvent,
    QueryStreamErrorEvent,
    QueryStreamEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamToolStatusEvent,
)

if TYPE_CHECKING:
    from ctxprotocol.client.client import ContextClient


class Query:
    """Query resource for pay-per-response agentic queries."""

    def __init__(self, client: ContextClient) -> None:
        """Initialize the Query resource.

        Args:
            client: The parent ContextClient instance
        """
        self._client = client

    @staticmethod
    def _merge_developer_trace(
        first: QueryDeveloperTrace | None,
        second: QueryDeveloperTrace | None,
    ) -> QueryDeveloperTrace | None:
        if first is None:
            return second
        if second is None:
            return first

        first_data = first.model_dump(by_alias=True, exclude_none=True)
        second_data = second.model_dump(by_alias=True, exclude_none=True)
        merged_data: dict[str, Any] = {**first_data, **second_data}

        first_timeline = first_data.get("timeline")
        second_timeline = second_data.get("timeline")
        merged_timeline: list[Any] = []
        if isinstance(first_timeline, list):
            merged_timeline.extend(first_timeline)
        if isinstance(second_timeline, list):
            merged_timeline.extend(second_timeline)
        if merged_timeline:
            merged_data["timeline"] = merged_timeline

        first_summary = first_data.get("summary")
        second_summary = second_data.get("summary")
        if isinstance(first_summary, dict) or isinstance(second_summary, dict):
            merged_data["summary"] = {
                **(first_summary if isinstance(first_summary, dict) else {}),
                **(second_summary if isinstance(second_summary, dict) else {}),
            }

        return QueryDeveloperTrace.model_validate(merged_data)

    @staticmethod
    def _build_synthetic_trace_from_run_result(
        tools_used: list[Any],
        duration_ms: int,
    ) -> QueryDeveloperTrace:
        timeline: list[dict[str, Any]] = []
        tool_calls = 0
        for index, tool in enumerate(tools_used):
            skill_calls = int(getattr(tool, "skill_calls", 0) or 0)
            tool_calls += max(skill_calls, 0)
            timeline.append(
                {
                    "stepType": "tool-call",
                    "event": "tool-call",
                    "status": "success",
                    "timestampMs": index,
                    "tool": {
                        "id": getattr(tool, "id", None),
                        "name": getattr(tool, "name", None),
                    },
                    "metadata": {
                        "skillCalls": skill_calls,
                        "synthetic": True,
                    },
                }
            )

        return QueryDeveloperTrace.model_validate(
            {
                "summary": {
                    "toolCalls": tool_calls,
                    "retryCount": 0,
                    "selfHealCount": 0,
                    "fallbackCount": 0,
                    "failureCount": 0,
                    "recoveryCount": 0,
                    "completionChecks": 0,
                    "loopCount": 0,
                },
                "timeline": timeline,
                "source": "sdk-fallback",
                "synthetic": True,
                "reason": "backend_trace_missing",
                "durationMs": duration_ms,
            }
        )

    @staticmethod
    def _build_synthetic_trace_from_stream_status(
        status_timeline: list[dict[str, Any]],
        tools_used: list[Any],
        duration_ms: int,
    ) -> QueryDeveloperTrace:
        timeline: list[dict[str, Any]] = []
        for index, status_entry in enumerate(status_timeline):
            timeline.append(
                {
                    "stepType": "tool-status",
                    "event": "tool-status",
                    "status": status_entry.get("status", ""),
                    "timestampMs": index,
                    "tool": status_entry.get("tool"),
                    "metadata": {"synthetic": True},
                }
            )

        tool_calls_from_usage = sum(
            max(int(getattr(tool, "skill_calls", 0) or 0), 0) for tool in tools_used
        )
        tool_calls_from_status = sum(
            1 for status_entry in status_timeline if status_entry.get("status") == "tool-complete"
        )
        tool_calls = (
            tool_calls_from_usage
            if tool_calls_from_usage > 0
            else tool_calls_from_status
        )

        retry_count = sum(
            1
            for status_entry in status_timeline
            if any(
                token in str(status_entry.get("status", "")).lower()
                for token in ("retry", "fix", "reflect", "recover")
            )
        )
        completion_checks = sum(
            1
            for status_entry in status_timeline
            if "complet" in str(status_entry.get("status", "")).lower()
        )

        return QueryDeveloperTrace.model_validate(
            {
                "summary": {
                    "toolCalls": tool_calls,
                    "retryCount": retry_count,
                    "selfHealCount": retry_count,
                    "fallbackCount": 0,
                    "failureCount": 0,
                    "recoveryCount": 0,
                    "completionChecks": completion_checks,
                    "loopCount": retry_count,
                },
                "timeline": timeline,
                "source": "sdk-fallback",
                "synthetic": True,
                "reason": "backend_trace_missing",
                "durationMs": duration_ms,
            }
        )

    @staticmethod
    def _build_policy_error_event(
        result: QueryResult,
        clarification_policy: QueryClarificationPolicy | None,
    ) -> QueryStreamErrorEvent | None:
        if clarification_policy != "error":
            return None

        if (
            result.outcome_type == "clarification_required"
            and result.clarification is not None
        ):
            return QueryStreamErrorEvent.model_validate(
                {
                    "type": "error",
                    "error": result.response,
                    "code": "clarification_required",
                    "reasonCode": "clarification_required",
                    "outcomeType": "clarification_required",
                    "clarification": result.clarification.model_dump(
                        by_alias=True,
                        exclude_none=True,
                    ),
                }
            )

        if (
            result.outcome_type == "capability_miss"
            and result.capability_miss is not None
        ):
            return QueryStreamErrorEvent.model_validate(
                {
                    "type": "error",
                    "error": result.response,
                    "code": "capability_miss",
                    "reasonCode": "capability_miss",
                    "outcomeType": "capability_miss",
                    "capabilityMiss": result.capability_miss.model_dump(
                        by_alias=True,
                        exclude_none=True,
                    ),
                }
            )

        return None

    async def run(
        self,
        query: str,
        clarification_policy: QueryClarificationPolicy | None = None,
        tools: list[str] | None = None,
        favorites_only: bool | None = None,
        answer_model_id: str | None = None,
        response_shape: QueryResponseShape | None = None,
        include_data: bool | None = None,
        include_data_url: bool | None = None,
        include_developer_trace: bool | None = None,
        idempotency_key: str | None = None,
    ) -> QueryResult:
        """Run an agentic query and wait for the full response.

        The server discovers relevant tools (or uses the ones you specify),
        executes the discovery-first pipeline (up to 100 MCP calls per tool),
        and returns an AI-synthesized answer. Payment is settled after
        successful execution via deferred settlement.
        Internally this follows the same SSE `done` path as `query.stream()`
        so Python and TypeScript observe the same query runtime behavior.

        Args:
            query: The natural-language question to answer
            tools: Optional tool IDs to use (auto-discover if not provided)
            favorites_only: Restrict auto-discovery to favorite tools for this request
            answer_model_id: Optional answer model ID for final synthesis
            response_shape: Structured response mode for query answers
            include_data: Include execution data inline in the query response
            include_data_url: Persist execution data to blob and return URL
            include_developer_trace: Include machine-readable Developer Mode traces
            idempotency_key: Optional idempotency key (UUID recommended) for safe retries

        Returns:
            The complete query result with response text, tools used, and cost

        Raises:
            ContextError: With code ``no_wallet`` if wallet not set up
            ContextError: With code ``insufficient_allowance`` if spending cap not set
            ContextError: With code ``payment_failed`` if payment settlement fails
            ContextError: With code ``execution_failed`` if the agentic pipeline fails

        Example:
            >>> # Simple question — server discovers tools automatically
            >>> answer = await client.query.run("What are the top whale movements on Base?")
            >>> print(answer.response)       # AI-synthesized answer
            >>> print(answer.tools_used)     # Which tools were used
            >>> print(answer.cost)           # Cost breakdown
            >>>
            >>> # With specific tools (Manual Mode)
            >>> answer = await client.query.run(
            ...     query="Analyze whale activity",
            ...     tools=["tool-uuid-1", "tool-uuid-2"],
            ... )
        """
        terminal_error: QueryStreamErrorEvent | None = None

        async for event in self.stream(
            query=query,
            clarification_policy=clarification_policy,
            tools=tools,
            favorites_only=favorites_only,
            answer_model_id=answer_model_id,
            response_shape=response_shape,
            include_data=include_data,
            include_data_url=include_data_url,
            include_developer_trace=include_developer_trace,
            idempotency_key=idempotency_key,
        ):
            if event.type == "error":
                terminal_error = event
                continue

            if event.type == "done":
                return event.result

        if terminal_error is not None:
            raise ContextError(
                message=terminal_error.error,
                code=terminal_error.code,
            )

        raise ContextError("Streaming query ended before done event")

    async def stream(
        self,
        query: str,
        clarification_policy: QueryClarificationPolicy | None = None,
        tools: list[str] | None = None,
        favorites_only: bool | None = None,
        answer_model_id: str | None = None,
        response_shape: QueryResponseShape | None = None,
        include_data: bool | None = None,
        include_data_url: bool | None = None,
        include_developer_trace: bool | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[QueryStreamEvent, None]:
        """Run an agentic query with streaming via SSE.

        Yields events as the server processes the query in real-time:
        - ``tool-status`` — A tool started executing or changed status
        - ``text-delta`` — A chunk of the AI response text
        - ``developer-trace`` — Runtime trace metadata (when enabled)
        - ``error`` — A structured query/runtime error emitted before completion
        - ``done`` — The full response is complete (includes final QueryResult)

        Args:
            query: The natural-language question to answer
            tools: Optional tool IDs to use (auto-discover if not provided)
            favorites_only: Restrict auto-discovery to favorite tools for this request
            answer_model_id: Optional answer model ID for final synthesis
            response_shape: Structured response mode for query answers
            include_data: Include execution data inline in the query response
            include_data_url: Persist execution data to blob and return URL
            include_developer_trace: Include machine-readable Developer Mode traces
            idempotency_key: Optional idempotency key (UUID recommended) for safe retries

        Yields:
            Stream events as the query is processed

        Example:
            >>> async for event in client.query.stream("What are the top whale movements?"):
            ...     if event.type == "text-delta":
            ...         print(event.delta, end="")
            ...     elif event.type == "error":
            ...         print(f"\\nStream error: {event.error}")
            ...     elif event.type == "done":
            ...         print(f"\\nCost: {event.result.cost.total_cost_usd}")
        """
        request_body: dict[str, Any] = {
            "query": query,
            "tools": tools,
            "stream": True,
        }
        if favorites_only is not None:
            request_body["favoritesOnly"] = favorites_only
        if clarification_policy is not None:
            request_body["clarificationPolicy"] = clarification_policy
        if answer_model_id is not None:
            request_body["answerModelId"] = answer_model_id
        if response_shape is not None:
            request_body["responseShape"] = response_shape
        if include_data is not None:
            request_body["includeData"] = include_data
        if include_data_url is not None:
            request_body["includeDataUrl"] = include_data_url
        if include_developer_trace is not None:
            request_body["includeDeveloperTrace"] = include_developer_trace

        response = await self._client.fetch_stream(
            "/api/v1/query",
            method="POST",
            json_body=request_body,
            extra_headers=(
                {"Idempotency-Key": idempotency_key}
                if idempotency_key
                else None
            ),
        )

        aggregated_trace: QueryDeveloperTrace | None = None
        status_timeline: list[dict[str, Any]] = []

        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped.startswith("data: "):
                continue

            data = stripped[6:]
            if data == "[DONE]":
                return

            try:
                parsed: dict[str, Any] = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = parsed.get("type")
            if event_type == "tool-status":
                status_event = QueryStreamToolStatusEvent.model_validate(parsed)
                status_timeline.append(
                    {
                        "status": status_event.status,
                        "tool": {
                            "id": status_event.tool.id,
                            "name": status_event.tool.name,
                        },
                    }
                )
                yield status_event
            elif event_type == "text-delta":
                yield QueryStreamTextDeltaEvent.model_validate(parsed)
            elif event_type == "developer-trace":
                trace_event = QueryStreamDeveloperTraceEvent.model_validate(parsed)
                aggregated_trace = self._merge_developer_trace(
                    aggregated_trace,
                    trace_event.trace,
                )
                yield trace_event
            elif event_type == "error":
                yield QueryStreamErrorEvent.model_validate(parsed)
            elif event_type == "done":
                done_event = QueryStreamDoneEvent.model_validate(parsed)
                done_trace = self._merge_developer_trace(
                    aggregated_trace,
                    done_event.result.developer_trace,
                )
                if done_trace is None and include_developer_trace:
                    if status_timeline:
                        done_trace = self._build_synthetic_trace_from_stream_status(
                            status_timeline=status_timeline,
                            tools_used=done_event.result.tools_used,
                            duration_ms=done_event.result.duration_ms,
                        )
                    else:
                        done_trace = self._build_synthetic_trace_from_run_result(
                            done_event.result.tools_used,
                            done_event.result.duration_ms,
                        )
                done_event.result.developer_trace = done_trace
                policy_error_event = self._build_policy_error_event(
                    done_event.result,
                    clarification_policy,
                )
                if policy_error_event is not None:
                    yield policy_error_event
                else:
                    yield done_event
