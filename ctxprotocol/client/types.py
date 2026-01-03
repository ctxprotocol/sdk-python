"""
Type definitions for the Context Protocol SDK.

This module contains all Pydantic models and type definitions used by the client.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextClientOptions(BaseModel):
    """Configuration options for initializing the ContextClient.

    Attributes:
        api_key: Your Context Protocol API key (e.g., "sk_live_abc123...")
        base_url: Base URL for the Context Protocol API. Defaults to "https://ctxprotocol.com"
    """

    api_key: str = Field(..., description="Your Context Protocol API key")
    base_url: str = Field(
        default="https://ctxprotocol.com",
        description="Base URL for the Context Protocol API",
    )


class McpTool(BaseModel):
    """An individual MCP tool exposed by a tool listing.

    Attributes:
        name: Name of the MCP tool method
        description: Description of what this method does
        input_schema: JSON Schema for the input arguments this tool accepts
        output_schema: JSON Schema for the output this tool returns
    """

    name: str = Field(..., description="Name of the MCP tool method")
    description: str = Field(..., description="Description of what this method does")
    input_schema: dict[str, Any] | None = Field(
        default=None,
        alias="inputSchema",
        description="JSON Schema for the input arguments this tool accepts",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        alias="outputSchema",
        description="JSON Schema for the output this tool returns",
    )

    model_config = {"populate_by_name": True}


class Tool(BaseModel):
    """Represents a tool available on the Context Protocol marketplace.

    Attributes:
        id: Unique identifier for the tool (UUID)
        name: Human-readable name of the tool
        description: Description of what the tool does
        price: Price per execution in USDC
        category: Tool category (e.g., "defi", "nft")
        is_verified: Whether the tool is verified by Context Protocol
        mcp_tools: Available MCP tool methods
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: str = Field(..., description="Unique identifier for the tool (UUID)")
    name: str = Field(..., description="Human-readable name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    price: str = Field(..., description="Price per execution in USDC")
    category: str | None = Field(default=None, description="Tool category")
    is_verified: bool | None = Field(
        default=None,
        alias="isVerified",
        description="Whether the tool is verified by Context Protocol",
    )
    mcp_tools: list[McpTool] | None = Field(
        default=None,
        alias="mcpTools",
        description="Available MCP tool methods",
    )
    created_at: str | None = Field(
        default=None,
        alias="createdAt",
        description="Creation timestamp",
    )
    updated_at: str | None = Field(
        default=None,
        alias="updatedAt",
        description="Last update timestamp",
    )

    model_config = {"populate_by_name": True}


class SearchResponse(BaseModel):
    """Response from the tools search endpoint.

    Attributes:
        tools: Array of matching tools
        query: The search query that was used
        count: Total number of results
    """

    tools: list[Tool] = Field(..., description="Array of matching tools")
    query: str = Field(..., description="The search query that was used")
    count: int = Field(..., description="Total number of results")


class SearchOptions(BaseModel):
    """Options for searching tools.

    Attributes:
        query: Search query (semantic search)
        limit: Maximum number of results (1-50, default 10)
    """

    query: str | None = Field(default=None, description="Search query (semantic search)")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of results (1-50, default 10)",
    )


class ExecuteOptions(BaseModel):
    """Options for executing a tool.

    Attributes:
        tool_id: The UUID of the tool to execute (from search results)
        tool_name: The specific MCP tool name to call (from tool's mcp_tools array)
        args: Arguments to pass to the tool
    """

    tool_id: str = Field(
        ...,
        alias="toolId",
        description="The UUID of the tool to execute (from search results)",
    )
    tool_name: str = Field(
        ...,
        alias="toolName",
        description="The specific MCP tool name to call (from tool's mcp_tools array)",
    )
    args: dict[str, Any] | None = Field(
        default=None,
        description="Arguments to pass to the tool",
    )

    model_config = {"populate_by_name": True}


class ToolInfo(BaseModel):
    """Information about an executed tool."""

    id: str
    name: str


class ExecuteApiSuccessResponse(BaseModel):
    """Successful execution response from the API.

    Attributes:
        success: Always True for success responses
        result: The result data from the tool execution
        tool: Information about the executed tool
        duration_ms: Execution duration in milliseconds
    """

    success: Literal[True] = Field(..., description="Always True for success responses")
    result: Any = Field(..., description="The result data from the tool execution")
    tool: ToolInfo = Field(..., description="Information about the executed tool")
    duration_ms: int = Field(
        ...,
        alias="durationMs",
        description="Execution duration in milliseconds",
    )

    model_config = {"populate_by_name": True}


class ExecuteApiErrorResponse(BaseModel):
    """Error response from the API.

    Attributes:
        error: Human-readable error message
        code: Error code for programmatic handling
        help_url: URL to help resolve the issue
    """

    error: str = Field(..., description="Human-readable error message")
    code: str | None = Field(
        default=None,
        description="Error code for programmatic handling",
    )
    help_url: str | None = Field(
        default=None,
        alias="helpUrl",
        description="URL to help resolve the issue",
    )

    model_config = {"populate_by_name": True}


class ExecutionResult(BaseModel):
    """The resolved result returned to the user after SDK processing.

    Attributes:
        result: The data returned by the tool
        tool: Information about the executed tool
        duration_ms: Execution duration in milliseconds
    """

    result: Any = Field(..., description="The data returned by the tool")
    tool: ToolInfo = Field(..., description="Information about the executed tool")
    duration_ms: int = Field(..., description="Execution duration in milliseconds")


# Type alias for specific error codes returned by the Context Protocol API
ContextErrorCode = Literal[
    "unauthorized",
    "no_wallet",
    "insufficient_allowance",
    "payment_failed",
    "execution_failed",
]


class ContextError(Exception):
    """Error thrown by the Context Protocol client.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        status_code: HTTP status code
        help_url: URL to help resolve the issue
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
        help_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.help_url = help_url

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"ContextError(message={self.message!r}, code={self.code!r}, "
            f"status_code={self.status_code!r}, help_url={self.help_url!r})"
        )

