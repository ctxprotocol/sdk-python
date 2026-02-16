"""
Query resource for pay-per-response agentic queries.

Unlike ``tools.execute()`` which calls a single tool once (pay-per-request),
the Query resource sends a natural-language question and lets the server
handle tool discovery, multi-tool orchestration, self-healing retries,
completeness checks, and AI synthesis — all for one flat fee.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator

from ctxprotocol.client.types import (
    ContextError,
    ExecuteApiErrorResponse,
    QueryApiSuccessResponse,
    QueryCost,
    QueryResult,
    QueryStreamDoneEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamToolStatusEvent,
    QueryToolUsage,
    ToolInfo,
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

    async def run(
        self,
        query: str,
        tools: list[str] | None = None,
        model_id: str | None = None,
        include_data: bool | None = None,
        include_data_url: bool | None = None,
        idempotency_key: str | None = None,
    ) -> QueryResult:
        """Run an agentic query and wait for the full response.

        The server discovers relevant tools (or uses the ones you specify),
        executes the full agentic pipeline (up to 100 MCP calls per tool),
        and returns an AI-synthesized answer. Payment is settled after
        successful execution via deferred settlement.

        Args:
            query: The natural-language question to answer
            tools: Optional tool IDs to use (auto-discover if not provided)
            model_id: Optional model ID for query orchestration/synthesis
            include_data: Include execution data inline in the query response
            include_data_url: Persist execution data to blob and return URL
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
        request_body: dict[str, Any] = {
            "query": query,
            "tools": tools,
            "stream": False,
        }
        if model_id is not None:
            request_body["modelId"] = model_id
        if include_data is not None:
            request_body["includeData"] = include_data
        if include_data_url is not None:
            request_body["includeDataUrl"] = include_data_url

        response = await self._client.fetch(
            "/api/v1/query",
            method="POST",
            json_body=request_body,
            extra_headers=(
                {"Idempotency-Key": idempotency_key}
                if idempotency_key
                else None
            ),
        )

        # Handle error response
        if "error" in response:
            error_response = ExecuteApiErrorResponse.model_validate(response)
            raise ContextError(
                message=error_response.error,
                code=error_response.code,
                status_code=None,
                help_url=error_response.help_url,
            )

        # Handle success response
        if response.get("success"):
            success_response = QueryApiSuccessResponse.model_validate(response)
            return QueryResult(
                response=success_response.response,
                tools_used=success_response.tools_used,
                cost=success_response.cost,
                duration_ms=success_response.duration_ms,
                data=success_response.data,
                data_url=success_response.data_url,
            )

        raise ContextError("Unexpected response format from query API")

    async def stream(
        self,
        query: str,
        tools: list[str] | None = None,
        model_id: str | None = None,
        include_data: bool | None = None,
        include_data_url: bool | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[
        QueryStreamToolStatusEvent | QueryStreamTextDeltaEvent | QueryStreamDoneEvent,
        None,
    ]:
        """Run an agentic query with streaming via SSE.

        Yields events as the server processes the query in real-time:
        - ``tool-status`` — A tool started executing or changed status
        - ``text-delta`` — A chunk of the AI response text
        - ``done`` — The full response is complete (includes final QueryResult)

        Args:
            query: The natural-language question to answer
            tools: Optional tool IDs to use (auto-discover if not provided)
            model_id: Optional model ID for query orchestration/synthesis
            include_data: Include execution data inline in the query response
            include_data_url: Persist execution data to blob and return URL
            idempotency_key: Optional idempotency key (UUID recommended) for safe retries

        Yields:
            Stream events as the query is processed

        Example:
            >>> async for event in client.query.stream("What are the top whale movements?"):
            ...     if event.type == "text-delta":
            ...         print(event.delta, end="")
            ...     elif event.type == "done":
            ...         print(f"\\nCost: {event.result.cost.total_cost_usd}")
        """
        request_body: dict[str, Any] = {
            "query": query,
            "tools": tools,
            "stream": True,
        }
        if model_id is not None:
            request_body["modelId"] = model_id
        if include_data is not None:
            request_body["includeData"] = include_data
        if include_data_url is not None:
            request_body["includeDataUrl"] = include_data_url

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
                yield QueryStreamToolStatusEvent.model_validate(parsed)
            elif event_type == "text-delta":
                yield QueryStreamTextDeltaEvent.model_validate(parsed)
            elif event_type == "done":
                yield QueryStreamDoneEvent.model_validate(parsed)
