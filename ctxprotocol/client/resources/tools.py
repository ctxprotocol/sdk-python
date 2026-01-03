"""
Tools resource for executing tools on the Context Protocol marketplace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ctxprotocol.client.types import (
    ContextError,
    ExecuteApiErrorResponse,
    ExecuteApiSuccessResponse,
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
    ) -> ExecutionResult:
        """Execute a tool with the provided arguments.

        Args:
            tool_id: The UUID of the tool (from search results)
            tool_name: The specific MCP tool method to call (from tool's mcp_tools array)
            args: Arguments to pass to the tool

        Returns:
            The execution result with the tool's output data

        Raises:
            ContextError: With code `no_wallet` if wallet not set up
            ContextError: With code `insufficient_allowance` if Auto Pay not enabled
            ContextError: With code `payment_failed` if on-chain payment fails
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
        response = await self._client.fetch(
            "/api/v1/tools/execute",
            method="POST",
            json_body={
                "toolId": tool_id,
                "toolName": tool_name,
                "args": args or {},
            },
        )

        # Handle error response
        if "error" in response:
            error_response = ExecuteApiErrorResponse.model_validate(response)
            raise ContextError(
                message=error_response.error,
                code=error_response.code,
                status_code=400,
                help_url=error_response.help_url,
            )

        # Handle success response
        if response.get("success"):
            success_response = ExecuteApiSuccessResponse.model_validate(response)
            return ExecutionResult(
                result=success_response.result,
                tool=ToolInfo(
                    id=success_response.tool.id,
                    name=success_response.tool.name,
                ),
                duration_ms=success_response.duration_ms,
            )

        # Fallback - shouldn't reach here with valid API responses
        raise ContextError("Unexpected response format from API")

