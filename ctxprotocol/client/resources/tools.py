"""
Tools resource for executing tools on the Context Protocol marketplace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

from ctxprotocol.client.types import (
    ContextError,
    ExecuteApiErrorResponse,
    ExecuteApiSuccessResponse,
    ExecuteSessionApiSuccessResponse,
    ExecuteSessionResult,
    ExecutionResult,
    ToolInfo,
)

if TYPE_CHECKING:
    from ctxprotocol.client.client import ContextClient


class Tools:
    """Tools resource for executing tools on the Context Protocol marketplace."""

    def __init__(self, client: ContextClient) -> None:
        """Initialize the Tools resource.

        Args:
            client: The parent ContextClient instance
        """
        self._client = client

    async def execute(
        self,
        tool_id: str,
        tool_name: str,
        args: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        mode: Literal["execute"] | None = None,
        session_id: str | None = None,
        max_spend_usd: str | None = None,
        close_session: bool | None = None,
    ) -> ExecutionResult:
        """Execute a tool with the provided arguments.

        Args:
            tool_id: The UUID of the tool (from search results)
            tool_name: The specific MCP tool method to call (from tool's mcp_tools array)
            args: Arguments to pass to the tool
            idempotency_key: Optional idempotency key (UUID recommended) for safe retries
            mode: Explicit execute mode label for request clarity
            session_id: Optional execute session identifier
            max_spend_usd: Optional per-session spend budget envelope (USD)
            close_session: Request session closure after this execute call settles

        Returns:
            The execution result with the tool's output data

        Raises:
            ContextError: With code `no_wallet` if wallet not set up
            ContextError: With code `insufficient_allowance` if spending cap not set
            ContextError: With code `payment_failed` if payment settlement fails
            ContextError: With code `execution_failed` if tool execution fails

        Example:
            >>> # First, search for a tool
            >>> tools = await client.discovery.search("gas prices")
            >>> tool = tools[0]
            >>>
            >>> # Execute a specific method from the tool's mcp_tools
            >>> result = await client.tools.execute(
            ...     tool_id=tool.id,
            ...     tool_name=tool.mcp_tools[0].name,  # e.g., "get_gas_prices"
            ...     args={"chainId": 1}
            ... )
            >>>
            >>> print(result.result)  # The tool's output
            >>> print(result.duration_ms)  # Execution time
        """
        payload: dict[str, Any] = {
            "toolId": tool_id,
            "toolName": tool_name,
            "args": args or {},
            "mode": mode or "execute",
        }
        if session_id is not None:
            payload["sessionId"] = session_id
        if max_spend_usd is not None:
            payload["maxSpendUsd"] = max_spend_usd
        if close_session is not None:
            payload["closeSession"] = close_session

        response = await self._client.fetch(
            "/api/v1/tools/execute",
            method="POST",
            json_body=payload,
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
                status_code=None,  # Don't hardcode - this was a 200 OK with error body
                help_url=error_response.help_url,
            )

        # Handle success response
        if response.get("success"):
            success_response = ExecuteApiSuccessResponse.model_validate(response)
            return ExecutionResult(
                mode=success_response.mode,
                result=success_response.result,
                tool=ToolInfo(
                    id=success_response.tool.id,
                    name=success_response.tool.name,
                ),
                method=success_response.method,
                session=success_response.session,
                duration_ms=success_response.duration_ms,
            )

        # Fallback - shouldn't reach here with valid API responses
        raise ContextError("Unexpected response format from API")

    async def start_session(self, max_spend_usd: str) -> ExecuteSessionResult:
        """Start an execute session with a max spend budget."""
        response = await self._client.fetch(
            "/api/v1/tools/execute/sessions",
            method="POST",
            json_body={"mode": "execute", "maxSpendUsd": max_spend_usd},
        )
        return self._resolve_session_lifecycle_response(response)

    async def get_session(self, session_id: str) -> ExecuteSessionResult:
        """Fetch current execute session status by ID."""
        if not session_id:
            raise ContextError("session_id is required")

        encoded_session_id = quote(session_id, safe="")
        response = await self._client.fetch(
            f"/api/v1/tools/execute/sessions/{encoded_session_id}"
        )
        return self._resolve_session_lifecycle_response(response)

    async def close_session(self, session_id: str) -> ExecuteSessionResult:
        """Close an execute session by ID."""
        if not session_id:
            raise ContextError("session_id is required")

        encoded_session_id = quote(session_id, safe="")
        response = await self._client.fetch(
            f"/api/v1/tools/execute/sessions/{encoded_session_id}/close",
            method="POST",
            json_body={"mode": "execute"},
        )
        return self._resolve_session_lifecycle_response(response)

    def _resolve_session_lifecycle_response(
        self, response: dict[str, Any]
    ) -> ExecuteSessionResult:
        if "error" in response:
            error_response = ExecuteApiErrorResponse.model_validate(response)
            raise ContextError(
                message=error_response.error,
                code=error_response.code,
                status_code=None,
                help_url=error_response.help_url,
            )

        if response.get("success"):
            success_response = ExecuteSessionApiSuccessResponse.model_validate(response)
            return ExecuteSessionResult(
                mode=success_response.mode,
                session=success_response.session,
            )

        raise ContextError("Unexpected response format from API")

