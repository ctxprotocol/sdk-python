"""
Type definitions for the Context Protocol SDK.

This module contains all Pydantic models and type definitions used by the client.
"""

from __future__ import annotations

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


DiscoveryMode = Literal["query", "execute"]
McpToolSurface = Literal["answer", "execute", "both"]
McpToolLatencyClass = Literal["instant", "fast", "slow", "streaming"]
ExecuteSessionStatus = Literal["open", "closed", "expired"]


class McpToolRateLimitHints(BaseModel):
    """Optional planner/runtime pacing hints for MCP methods."""

    max_requests_per_minute: int | None = Field(
        default=None,
        alias="maxRequestsPerMinute",
        description="Suggested request budget for this method",
    )
    max_concurrency: int | None = Field(
        default=None,
        alias="maxConcurrency",
        description="Suggested parallel call ceiling for this method",
    )
    cooldown_ms: int | None = Field(
        default=None,
        alias="cooldownMs",
        description="Suggested minimum delay between sequential calls",
    )
    supports_bulk: bool | None = Field(
        default=None,
        alias="supportsBulk",
        description="Whether this method supports bulk retrieval",
    )
    recommended_batch_tools: list[str] | None = Field(
        default=None,
        alias="recommendedBatchTools",
        description="Preferred batch methods to call instead of fan-out loops",
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable notes for planning",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class McpToolPricingMeta(BaseModel):
    """Method-level pricing metadata."""

    execute_usd: str | None = Field(
        default=None,
        alias="executeUsd",
        description="Execute price in USD",
    )
    query_usd: str | None = Field(
        default=None,
        alias="queryUsd",
        description="Reserved query price metadata",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class McpToolMeta(BaseModel):
    """Typed MCP method metadata used for discovery and execution routing."""

    surface: McpToolSurface | None = Field(
        default=None,
        description="Declared method surface",
    )
    query_eligible: bool | None = Field(
        default=None,
        alias="queryEligible",
        description="Whether this method can be selected in query mode",
    )
    latency_class: McpToolLatencyClass | None = Field(
        default=None,
        alias="latencyClass",
        description="Declared latency class for planner/runtime gating",
    )
    pricing: McpToolPricingMeta | None = Field(
        default=None,
        description="Method-level pricing metadata",
    )
    execute_eligible: bool | None = Field(
        default=None,
        alias="executeEligible",
        description="Derived discovery flag for execute eligibility",
    )
    execute_price_usd: str | None = Field(
        default=None,
        alias="executePriceUsd",
        description="Derived discovery field for execute pricing visibility",
    )
    context_requirements: list[str] | None = Field(
        default=None,
        alias="contextRequirements",
        description="Context injection requirements handled by Context runtime",
    )
    rate_limit: McpToolRateLimitHints | None = Field(
        default=None,
        alias="rateLimit",
        description="Planner/runtime pacing hints",
    )
    rate_limit_hints: McpToolRateLimitHints | None = Field(
        default=None,
        alias="rateLimitHints",
        description="Planner/runtime pacing hints (alternate key)",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class StructuredMethodGuidanceHints(BaseModel):
    """Normalized guidance hints extracted from contributor method descriptions."""

    call_order_hints: list[str] | None = Field(
        default=None,
        alias="callOrderHints",
        description="Suggested call-order sequence extracted from descriptions",
    )
    parameter_caveats: list[str] | None = Field(
        default=None,
        alias="parameterCaveats",
        description="Parameter usage caveats extracted from descriptions",
    )
    edge_case_notes: list[str] | None = Field(
        default=None,
        alias="edgeCaseNotes",
        description="Edge-case behavior notes extracted from descriptions",
    )

    model_config = {"populate_by_name": True}


class McpTool(BaseModel):
    """An individual MCP tool exposed by a tool listing.

    Attributes:
        name: Name of the MCP tool method
        description: Description of what this method does
        input_schema: JSON Schema for the input arguments this tool accepts
        output_schema: JSON Schema for the output this tool returns
        meta: MCP metadata extensions (context requirements, rate-limit hints)
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
    meta: McpToolMeta | None = Field(
        default=None,
        alias="_meta",
        description="MCP metadata extensions",
    )
    execute_eligible: bool | None = Field(
        default=None,
        alias="executeEligible",
        description="Whether this method is execute-eligible",
    )
    execute_price_usd: str | None = Field(
        default=None,
        alias="executePriceUsd",
        description="Explicit execute price visibility for this method",
    )
    has_structured_guidance: bool | None = Field(
        default=None,
        alias="hasStructuredGuidance",
        description="Whether this method has normalized guidance hints",
    )
    structured_guidance: StructuredMethodGuidanceHints | None = Field(
        default=None,
        alias="structuredGuidance",
        description="Structured guidance hints derived from method descriptions",
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
        kind: Tool type - currently always "mcp"
        mcp_tools: Available MCP tool methods
        total_queries: Total number of queries processed
        success_rate: Success rate percentage (0-100)
        uptime_percent: Uptime percentage (0-100)
        total_staked: Total USDC staked by the developer
        is_proven: Whether the tool has "Proven" status
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
    kind: str | None = Field(
        default=None,
        description="Tool type - currently always 'mcp'",
    )
    mcp_tools: list[McpTool] | None = Field(
        default=None,
        alias="mcpTools",
        description="Available MCP tool methods",
    )
    total_queries: int | None = Field(
        default=None,
        alias="totalQueries",
        description="Total number of queries processed",
    )
    success_rate: str | None = Field(
        default=None,
        alias="successRate",
        description="Success rate percentage",
    )
    uptime_percent: str | None = Field(
        default=None,
        alias="uptimePercent",
        description="Uptime percentage",
    )
    total_staked: str | None = Field(
        default=None,
        alias="totalStaked",
        description="Total USDC staked by developer",
    )
    is_proven: bool | None = Field(
        default=None,
        alias="isProven",
        description="Whether tool has Proven status",
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
    mode: DiscoveryMode | None = Field(
        default=None,
        description="Discovery mode used by the server",
    )
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
    mode: DiscoveryMode | None = Field(
        default=None,
        description="Discovery mode with billing semantics",
    )
    surface: McpToolSurface | None = Field(
        default=None,
        description="Optional explicit method surface filter",
    )
    query_eligible: bool | None = Field(
        default=None,
        alias="queryEligible",
        description="Require methods marked query eligible",
    )
    require_execute_pricing: bool | None = Field(
        default=None,
        alias="requireExecutePricing",
        description="Require explicit method execute pricing",
    )
    exclude_latency_classes: list[McpToolLatencyClass] | None = Field(
        default=None,
        alias="excludeLatencyClasses",
        description="Exclude methods by latency class",
    )
    exclude_slow: bool | None = Field(
        default=None,
        alias="excludeSlow",
        description="Exclude slow methods in query mode",
    )

    model_config = {"populate_by_name": True}


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
    idempotency_key: str | None = Field(
        default=None,
        alias="idempotencyKey",
        description="Optional idempotency key (UUID recommended) for safe retries",
    )
    mode: Literal["execute"] | None = Field(
        default=None,
        description="Explicit execute mode label for request clarity",
    )
    session_id: str | None = Field(
        default=None,
        alias="sessionId",
        description="Optional execute session identifier",
    )
    max_spend_usd: str | None = Field(
        default=None,
        alias="maxSpendUsd",
        description="Optional per-session spend budget envelope (USD)",
    )
    close_session: bool | None = Field(
        default=None,
        alias="closeSession",
        description="Request session closure after this execute call settles",
    )

    model_config = {"populate_by_name": True}


class ToolInfo(BaseModel):
    """Information about an executed tool."""

    id: str
    name: str


class ExecuteMethodInfo(BaseModel):
    """Method-level execute pricing details for a call."""

    name: str
    execute_price_usd: str = Field(..., alias="executePriceUsd")

    model_config = {"populate_by_name": True}


class ExecuteSessionSpend(BaseModel):
    """Spend envelope visibility for execute calls and sessions."""

    mode: Literal["execute"] = "execute"
    session_id: str | None = Field(default=None, alias="sessionId")
    method_price: str = Field(..., alias="methodPrice")
    spent: str
    remaining: str | None = None
    max_spend: str | None = Field(default=None, alias="maxSpend")
    status: ExecuteSessionStatus | None = None
    expires_at: str | None = Field(default=None, alias="expiresAt")
    close_requested: bool | None = Field(default=None, alias="closeRequested")
    pending_accrued_count: int | None = Field(default=None, alias="pendingAccruedCount")
    pending_accrued_usd: str | None = Field(default=None, alias="pendingAccruedUsd")

    model_config = {"populate_by_name": True}


class ExecuteApiSuccessResponse(BaseModel):
    """Successful execution response from the API.

    Attributes:
        success: Always True for success responses
        result: The result data from the tool execution
        tool: Information about the executed tool
        duration_ms: Execution duration in milliseconds
    """

    success: Literal[True] = Field(..., description="Always True for success responses")
    mode: Literal["execute"] = Field(
        default="execute",
        description="Explicit mode label for clarity",
    )
    result: Any = Field(..., description="The result data from the tool execution")
    tool: ToolInfo = Field(..., description="Information about the executed tool")
    method: ExecuteMethodInfo = Field(
        ...,
        description="Method-level execute pricing used for this call",
    )
    session: ExecuteSessionSpend = Field(
        ...,
        description="Spend envelope visibility for execute sessions",
    )
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
    mode: Literal["execute"] | None = Field(
        default=None,
        description="Explicit mode label for clarity",
    )
    code: str | None = Field(
        default=None,
        description="Error code for programmatic handling",
    )
    help_url: str | None = Field(
        default=None,
        alias="helpUrl",
        description="URL to help resolve the issue",
    )
    session: ExecuteSessionSpend | None = Field(
        default=None,
        description="Optional spend envelope context when available",
    )

    model_config = {"populate_by_name": True}


class ExecutionResult(BaseModel):
    """The resolved result returned to the user after SDK processing.

    Attributes:
        result: The data returned by the tool
        tool: Information about the executed tool
        duration_ms: Execution duration in milliseconds
    """

    mode: Literal["execute"] = Field(
        default="execute",
        description="Explicit mode label for clarity",
    )
    result: Any = Field(..., description="The data returned by the tool")
    tool: ToolInfo = Field(..., description="Information about the executed tool")
    method: ExecuteMethodInfo = Field(
        ...,
        description="Method-level execute pricing used for this call",
    )
    session: ExecuteSessionSpend = Field(
        ...,
        description="Spend envelope visibility for execute calls",
    )
    duration_ms: int = Field(..., description="Execution duration in milliseconds")


class ExecuteSessionStartOptions(BaseModel):
    """Options for starting an execute session."""

    max_spend_usd: str = Field(
        ...,
        alias="maxSpendUsd",
        description="Maximum spend budget for the session (USD string)",
    )

    model_config = {"populate_by_name": True}


class ExecuteSessionApiSuccessResponse(BaseModel):
    """Successful execute-session lifecycle response."""

    success: Literal[True]
    mode: Literal["execute"] = "execute"
    session: ExecuteSessionSpend

    model_config = {"populate_by_name": True}


class ExecuteSessionResult(BaseModel):
    """Resolved execute-session lifecycle result returned by the SDK."""

    mode: Literal["execute"] = "execute"
    session: ExecuteSessionSpend

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Query types (pay-per-response / agentic mode)
# ---------------------------------------------------------------------------


class QueryOptions(BaseModel):
    """Options for the agentic query endpoint (pay-per-response).

    Unlike ``execute()`` which calls a single tool once, ``query()`` sends a
    natural-language question and lets the server handle tool discovery,
    multi-tool orchestration, self-healing retries, and AI synthesis.
    One flat fee covers up to 100 MCP skill calls per tool.

    Attributes:
        query: The natural-language question to answer
        tools: Optional tool IDs to use (auto-discover if not provided)
        model_id: Optional model ID for query orchestration/synthesis
        include_data: Include execution data inline in the query response
        include_data_url: Persist execution data to blob and return URL
    """

    query: str = Field(..., description="The natural-language question to answer")
    tools: list[str] | None = Field(
        default=None,
        description="Optional tool IDs to use (auto-discover if not provided)",
    )
    model_id: str | None = Field(
        default=None,
        alias="modelId",
        description="Optional model ID for query orchestration/synthesis",
    )
    include_data: bool | None = Field(
        default=None,
        alias="includeData",
        description="Include execution data inline in the query response",
    )
    include_data_url: bool | None = Field(
        default=None,
        alias="includeDataUrl",
        description="Persist execution data to blob and return URL",
    )
    idempotency_key: str | None = Field(
        default=None,
        alias="idempotencyKey",
        description="Optional idempotency key (UUID recommended) for safe retries",
    )

    model_config = {"populate_by_name": True}


class QueryToolUsage(BaseModel):
    """Information about a tool that was used during a query response.

    Attributes:
        id: Tool ID
        name: Tool name
        skill_calls: Number of MCP skill calls made for this tool
    """

    id: str = Field(..., description="Tool ID")
    name: str = Field(..., description="Tool name")
    skill_calls: int = Field(
        ...,
        alias="skillCalls",
        description="Number of MCP skill calls made for this tool",
    )

    model_config = {"populate_by_name": True}


class QueryCost(BaseModel):
    """Cost breakdown for a query response.

    Attributes:
        model_cost_usd: AI model inference cost in USD
        tool_cost_usd: Sum of all tool fees in USD
        total_cost_usd: Total cost in USD
    """

    model_cost_usd: str = Field(
        ..., alias="modelCostUsd", description="AI model inference cost"
    )
    tool_cost_usd: str = Field(
        ..., alias="toolCostUsd", description="Sum of all tool fees"
    )
    total_cost_usd: str = Field(
        ..., alias="totalCostUsd", description="Total cost (model + tools)"
    )

    model_config = {"populate_by_name": True}


class QueryResult(BaseModel):
    """The resolved result of a pay-per-response query.

    Attributes:
        response: The AI-synthesized response text
        tools_used: Tools that were used to answer the query
        cost: Cost breakdown
        duration_ms: Total duration in milliseconds
        data: Optional execution data (when include_data is enabled)
        data_url: Optional blob URL for execution data (when include_data_url is enabled)
    """

    response: str = Field(..., description="The AI-synthesized response text")
    tools_used: list[QueryToolUsage] = Field(
        ..., alias="toolsUsed", description="Tools that were used"
    )
    cost: QueryCost = Field(..., description="Cost breakdown")
    duration_ms: int = Field(
        ..., alias="durationMs", description="Total duration in milliseconds"
    )
    data: Any | None = Field(
        default=None,
        description="Optional execution data from tools",
    )
    data_url: str | None = Field(
        default=None,
        alias="dataUrl",
        description="Optional blob URL for persisted execution data",
    )

    model_config = {"populate_by_name": True}


class QueryApiSuccessResponse(BaseModel):
    """Successful response from the /api/v1/query endpoint."""

    success: Literal[True]
    response: str
    tools_used: list[QueryToolUsage] = Field(..., alias="toolsUsed")
    cost: QueryCost
    duration_ms: int = Field(..., alias="durationMs")
    data: Any | None = None
    data_url: str | None = Field(default=None, alias="dataUrl")

    model_config = {"populate_by_name": True}


class QueryStreamToolStatusEvent(BaseModel):
    """Emitted when a tool starts or changes execution status."""

    type: Literal["tool-status"]
    tool: ToolInfo
    status: str


class QueryStreamTextDeltaEvent(BaseModel):
    """Emitted for each chunk of the AI response text."""

    type: Literal["text-delta"]
    delta: str


class QueryStreamDoneEvent(BaseModel):
    """Emitted when the full response is complete."""

    type: Literal["done"]
    result: QueryResult


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

# Type alias for specific error codes returned by the Context Protocol API
ContextErrorCode = Literal[
    "unauthorized",
    "no_wallet",
    "insufficient_allowance",
    "payment_failed",
    "execution_failed",
    "query_failed",
    "invalid_tool_method",
    "method_not_execute_eligible",
    "invalid_max_spend",
    "session_not_found",
    "session_forbidden",
    "session_closed",
    "session_expired",
    "max_spend_mismatch",
    "session_budget_exceeded",
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

