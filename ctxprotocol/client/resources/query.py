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
    ) -> QueryResult:
        """Run an agentic query and wait for the full response.

        The server discovers relevant tools (or uses the ones you specify),
        executes the full agentic pipeline (up to 100 MCP calls per tool),
        and returns an AI-synthesized answer. Payment is settled after
        successful execution via deferred settlement.

        Args:
            query: The natural-language question to answer
            tools: Optional tool IDs to use (auto-discover if not provided)

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
        response = await self._client.fetch(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": query,
                "tools": tools,
                "stream": False,
            },
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
            )

        raise ContextError("Unexpected response format from query API")

    async def stream(
        self,
        query: str,
        tools: list[str] | None = None,
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

        Yields:
            Stream events as the query is processed

        Example:
            >>> async for event in client.query.stream("What are the top whale movements?"):
            ...     if event.type == "text-delta":
            ...         print(event.delta, end="")
            ...     elif event.type == "done":
            ...         print(f"\\nCost: {event.result.cost.total_cost_usd}")
        """
        response = await self._client.fetch_stream(
            "/api/v1/query",
            method="POST",
            json_body={
                "query": query,
                "tools": tools,
                "stream": True,
            },
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
