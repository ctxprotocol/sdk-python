"""
Type definitions for the Context Protocol SDK.

This module contains all Pydantic models and type definitions used by the client.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from ctxprotocol.contrib.search.types import ContributorSearchTraceRecord


class ContextClientOptions(BaseModel):
    """Configuration options for initializing the ContextClient.

    Attributes:
        api_key: Your Context Protocol API key (e.g., "sk_live_abc123...")
        base_url: Base URL for the Context Protocol API. Defaults to "https://www.ctxprotocol.com"
        request_timeout_seconds: Timeout for non-streaming JSON requests in seconds
        stream_timeout_seconds: Timeout for streaming requests in seconds; also used by query.run()
    """

    api_key: str = Field(..., description="Your Context Protocol API key")
    base_url: str = Field(
        default="https://www.ctxprotocol.com",
        description="Base URL for the Context Protocol API",
    )
    request_timeout_seconds: float = Field(
        default=300.0,
        description="Timeout for non-streaming JSON requests in seconds",
    )
    stream_timeout_seconds: float = Field(
        default=600.0,
        description="Timeout for streaming requests in seconds; also used by query.run()",
    )


DiscoveryMode = Literal["query", "execute"]
McpToolSurface = Literal["answer", "execute", "both"]
McpToolLatencyClass = Literal["instant", "fast", "slow", "streaming"]
ExecuteSessionStatus = Literal["open", "closed", "expired"]
SuggestedPromptSource = Literal["contributor", "platform", "sdk"]
DEFAULT_AGENT_MODEL_ID = "kimi-k2.6-model"
AGENT_MODEL_IDS = (
    "kimi-k2.6-model",
    "kimi-k3-model",
    "glm-5.2-model",
    "grok-4.5-model",
    "deepseek-v4-pro-model",
    "deepseek-v4-flash-model",
    "qwen-3.7-plus-model",
    "qwen-3.7-max-model",
    "gpt-5.5-model",
    "claude-opus-model",
)
AgentModelId = Literal[
    "kimi-k2.6-model",
    "kimi-k3-model",
    "glm-5.2-model",
    "grok-4.5-model",
    "deepseek-v4-pro-model",
    "deepseek-v4-flash-model",
    "qwen-3.7-plus-model",
    "qwen-3.7-max-model",
    "gpt-5.5-model",
    "claude-opus-model",
]


class McpToolRateLimitHints(BaseModel):
    """Optional runtime pacing hints for MCP methods."""

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
        description="Optional human-readable notes for execution behavior",
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
        description="Declared latency class for runtime gating",
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


class SuggestedPrompt(BaseModel):
    """Clickable example prompt shown on a marketplace listing."""

    text: str = Field(..., description="Prompt text shown to users")
    source: SuggestedPromptSource = Field(
        ...,
        description="Where this prompt came from",
    )
    price_hint: str | None = Field(
        default=None,
        alias="priceHint",
        description="Optional listing price display hint",
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
        suggested_prompts: Clickable example prompts shown in the Context app
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
    suggested_prompts: list[SuggestedPrompt] | None = Field(
        default=None,
        alias="suggestedPrompts",
        description="Clickable example prompts shown in the Context app",
    )
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
    favorites_only: bool | None = Field(
        default=None,
        alias="favoritesOnly",
        description=(
            "Restrict discovery to favorite tools for this request. "
            "Omit to use the account-level default."
        ),
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


# External orchestration depth hint. The server decides the effective depth
# from the query; this is a soft hint rather than a hard switch.
QueryDepth = Literal["fast", "auto", "deep"]
QueryOutcomeType = Literal["answer", "capability_miss"]
QueryResponseShape = Literal["answer_with_evidence", "evidence_only"]
QueryAttemptForkReason = Literal[
    "manual_fork",
    "bounded_rediscovery",
    "resume_replay",
    "patch_retry",
    "unknown",
]
QueryResponseEnvelopeViewType = Literal[
    "table",
    "leaderboard",
    "heatmap",
    "timeseries",
]
class QueryAttemptReference(BaseModel):
    """Public handle for resuming a prior query attempt."""

    session_id: str = Field(..., alias="sessionId")
    attempt_id: str = Field(..., alias="attemptId")

    model_config = {"populate_by_name": True}


class QueryForkReference(QueryAttemptReference):
    """Public handle for forking a new query attempt from a prior attempt."""

    reason: QueryAttemptForkReason | None = None


class QuerySessionCheckpoint(BaseModel):
    """Public durable checkpoint summary returned by Query responses."""

    current_stage: str | None = Field(default=None, alias="currentStage")
    latest_checkpoint_artifact_id: str | None = Field(
        default=None,
        alias="latestCheckpointArtifactId",
    )
    canonical_dataset_id: str | None = Field(default=None, alias="canonicalDatasetId")
    execution_program_current_revision_id: str | None = Field(
        default=None,
        alias="executionProgramCurrentRevisionId",
    )

    model_config = {"populate_by_name": True}


class QuerySessionState(BaseModel):
    """Public durable continuation state returned by Query responses."""

    session_id: str = Field(..., alias="sessionId")
    attempt_id: str = Field(..., alias="attemptId")
    parent_attempt_id: str | None = Field(default=None, alias="parentAttemptId")
    root_attempt_id: str = Field(..., alias="rootAttemptId")
    mode: Literal["initial", "resume", "fork"]
    origin: Literal["initial_request", "resume", "fork"]
    status: Literal["active", "completed", "failed", "aborted"]
    checkpoint: QuerySessionCheckpoint

    model_config = {"populate_by_name": True}


class QueryOptions(BaseModel):
    """Options for the agentic query endpoint (pay-per-response).

    Unlike ``execute()`` which calls a single tool once, ``query()`` sends a
    natural-language question and lets the server handle the live librarian
    pipeline (discover -> select -> iterative execute ->
    synthesize -> settle).
    One flat fee covers up to 100 MCP skill calls per tool.

    Attributes:
        query: The natural-language question to answer
        tools: Optional tool IDs to use (auto-discover if not provided)
        agent_model_id: Optional model ID for the main librarian agent loop
        include_data: Include execution data inline in the query response
        include_data_url: Persist execution data to blob and return URL
        include_developer_trace: Include machine-readable developer runtime traces
    """

    query: str = Field(..., description="The natural-language question to answer")
    tools: list[str] | None = Field(
        default=None,
        description="Optional tool IDs to use (auto-discover if not provided)",
    )
    resume_from: QueryAttemptReference | None = Field(
        default=None,
        alias="resumeFrom",
        description="Resume a prior durable query attempt from its latest checkpoint",
    )
    fork_from: QueryForkReference | None = Field(
        default=None,
        alias="forkFrom",
        description="Fork a new durable query attempt from a previous attempt",
    )
    favorites_only: bool | None = Field(
        default=None,
        alias="favoritesOnly",
        description=(
            "Restrict auto-discovery to favorite tools for this request. "
            "Ignored when explicit tools are provided."
        ),
    )
    agent_model_id: AgentModelId | str | None = Field(
        default=None,
        alias="agentModelId",
        description=(
            "Optional model ID for the main librarian agent loop. Controls the "
            "merged iterative execution + final response stage; internal tool "
            "selection remains managed by the server."
        ),
    )
    response_shape: QueryResponseShape | None = Field(
        default=None,
        alias="responseShape",
        description=(
            "Structured response mode. Defaults to `answer_with_evidence` on the "
            "server when omitted. The runtime always produces a grounded result "
            "(bounded evidence + computed artifacts + full-data references); "
            "this controls whether a prose synthesis layer is added on top. "
            "`answer_with_evidence` returns prose plus the structured grounding "
            "(chat parity); `evidence_only` returns grounding only with no prose "
            "(the agent-harness shape, with bounded evidence, computed_artifacts, "
            "and data_url/canonical_data_ref references)."
        ),
    )
    include_data: bool | None = Field(
        default=None,
        alias="includeData",
        description=(
            "Include bounded execution data inline. Defaults to false for every "
            "response_shape. Large payloads are returned as a preview object "
            "with fullData.dataUrl/canonicalDataRef instead of unbounded raw rows."
        ),
    )
    include_data_url: bool | None = Field(
        default=None,
        alias="includeDataUrl",
        description="Persist execution data to blob and return URL",
    )
    include_developer_trace: bool | None = Field(
        default=None,
        alias="includeDeveloperTrace",
        description=(
            "Include machine-readable developer trace output with runtime details "
            "(tool timeline, retries, fallback branches, loop checks)"
        ),
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


QueryChartType = Literal[
    "line",
    "bar",
    "area",
    "scatter",
    "composed",
    "histogram",
    "heatmap",
    "candlestick",
]
QueryChartSeriesType = Literal["line", "bar", "area", "scatter"]
QueryChartAxisType = Literal["time", "category", "number"]
QueryChartValueFormat = Literal["number", "percent", "currency", "compact"]
QueryChartDataValue = str | int | float | None


class QueryChartSeries(BaseModel):
    """Single series entry inside a structured chart artifact."""

    key: str
    label: str | None = None
    type: QueryChartSeriesType | None = None
    error_key: str | None = Field(default=None, alias="errorKey")
    y_axis: Literal["left", "right"] | None = Field(default=None, alias="yAxis")
    satisfies: str | None = None

    model_config = {"populate_by_name": True}


class QueryChartAxis(BaseModel):
    """Optional axis configuration for a structured chart artifact."""

    type: QueryChartAxisType | None = None
    label: str | None = None
    format: QueryChartValueFormat | None = None
    value_scale: Literal["fraction", "percent_points"] | None = Field(
        default=None,
        alias="valueScale",
    )

    model_config = {"populate_by_name": True}


class QueryChartOhlcKeys(BaseModel):
    """OHLC field mapping for candlestick chart artifacts."""

    open_key: str = Field(..., alias="openKey")
    high_key: str = Field(..., alias="highKey")
    low_key: str = Field(..., alias="lowKey")
    close_key: str = Field(..., alias="closeKey")

    model_config = {"populate_by_name": True}


class QueryChartReferenceLine(BaseModel):
    """Reference threshold or event marker for a structured chart artifact."""

    axis: Literal["x", "y"]
    value: str | int | float
    label: str | None = None


class QueryChartReferenceArea(BaseModel):
    """Highlighted x/y region for a structured chart artifact."""

    x1: str | int | float | None = None
    x2: str | int | float | None = None
    y1: int | float | None = None
    y2: int | float | None = None
    label: str | None = None


class QueryChartSpec(BaseModel):
    """Structured chart spec emitted by the code interpreter."""

    type: QueryChartType
    x_key: str = Field(..., alias="xKey")
    series: list[QueryChartSeries]
    expected_measures: list[str] | None = Field(default=None, alias="expectedMeasures")
    x_axis: QueryChartAxis | None = Field(default=None, alias="xAxis")
    y_axis: QueryChartAxis | None = Field(default=None, alias="yAxis")
    y_axis_right: QueryChartAxis | None = Field(default=None, alias="yAxisRight")
    legend: bool | None = None
    stacked: bool | None = None
    brush: bool | None = None
    reference_lines: list[QueryChartReferenceLine] | None = Field(
        default=None,
        alias="referenceLines",
    )
    reference_areas: list[QueryChartReferenceArea] | None = Field(
        default=None,
        alias="referenceAreas",
    )
    y_key: str | None = Field(default=None, alias="yKey")
    value_key: str | None = Field(default=None, alias="valueKey")
    ohlc: QueryChartOhlcKeys | None = None

    model_config = {"populate_by_name": True}


class QueryChartArtifact(BaseModel):
    """Structured chart artifact returned as spec + compact data rows."""

    kind: Literal["chart"]
    spec: QueryChartSpec
    data: list[dict[str, QueryChartDataValue]]
    title: str | None = None


class QueryImageArtifact(BaseModel):
    """Rendered image artifact (e.g. a server-rendered chart PNG).

    Emitted by the code interpreter alongside or instead of the structured
    ``chart`` spec. ``url`` points at a hosted, already-rendered image so
    consumers that cannot render a chart spec (image-first surfaces such as
    social posting) can attach it directly.
    """

    kind: Literal["image"]
    url: str
    alt: str | None = None
    title: str | None = None
    content_hash: str | None = Field(default=None, alias="contentHash")
    bytes: int | None = None
    width: int | None = None
    height: int | None = None

    model_config = {"populate_by_name": True}


QueryComputedArtifact = Annotated[
    Union[QueryChartArtifact, QueryImageArtifact],
    Field(discriminator="kind"),
]


class QueryToolCallFailureSample(BaseModel):
    """Capped sample of a marketplace tool call that failed before returning data."""

    tool_name: str = Field(..., alias="toolName")
    method_name: str = Field(..., alias="methodName")
    reason: str

    model_config = {"populate_by_name": True}


class QueryGroundingSummary(BaseModel):
    """Public grounding summary for marketplace tool execution."""

    available_tool_count: int = Field(
        ...,
        alias="availableToolCount",
        description="Marketplace methods registered in the iterative runtime",
    )
    available_method_names_sample: list[str] = Field(
        default_factory=list,
        alias="availableMethodNamesSample",
        description="Capped sample of method names available to the model",
    )
    selected_method_count: int = Field(
        ...,
        alias="selectedMethodCount",
        description="Methods selected before runtime filtering",
    )
    selected_but_filtered_out: list[str] = Field(
        default_factory=list,
        alias="selectedButFilteredOut",
        description="Selected methods that did not survive runtime filtering",
    )
    tool_call_count: int = Field(
        ...,
        alias="toolCallCount",
        description="Grounded marketplace tool calls actually executed (successes only)",
    )
    tool_call_attempt_count: int = Field(
        default=0,
        alias="toolCallAttemptCount",
        description="Total marketplace method invocations attempted (success + failure)",
    )
    tool_call_success_count: int = Field(
        default=0,
        alias="toolCallSuccessCount",
        description="Marketplace method invocations that completed without throwing",
    )
    tool_call_failure_count: int = Field(
        default=0,
        alias="toolCallFailureCount",
        description="Marketplace method invocations that threw before returning data",
    )
    tool_call_failure_samples: list[QueryToolCallFailureSample] = Field(
        default_factory=list,
        alias="toolCallFailureSamples",
        description="Capped sample of recent failed marketplace invocations with reasons",
    )
    grounded: bool = Field(
        ...,
        description="True when at least one marketplace tool call grounded the answer",
    )

    model_config = {"populate_by_name": True}


class QueryToolRegistryDiagnostics(BaseModel):
    """Runtime tool registry diagnostics exposed in developer traces."""

    available_tool_count: int = Field(..., alias="availableToolCount")
    available_method_names_sample: list[str] = Field(
        default_factory=list,
        alias="availableMethodNamesSample",
    )
    selected_method_count: int = Field(..., alias="selectedMethodCount")
    selected_but_filtered_out: list[str] = Field(
        default_factory=list,
        alias="selectedButFilteredOut",
    )
    tool_call_attempt_count: int = Field(
        default=0,
        alias="toolCallAttemptCount",
    )
    tool_call_success_count: int = Field(
        default=0,
        alias="toolCallSuccessCount",
    )
    tool_call_failure_count: int = Field(
        default=0,
        alias="toolCallFailureCount",
    )
    tool_call_failure_samples: list[QueryToolCallFailureSample] = Field(
        default_factory=list,
        alias="toolCallFailureSamples",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryExecutionDiagnostics(BaseModel):
    """Iterative execution diagnostics attached to developer traces."""

    reasoning_enabled: bool | None = Field(default=None, alias="reasoningEnabled")
    received_reasoning: bool | None = Field(default=None, alias="receivedReasoning")
    reasoning_chars: int | None = Field(default=None, alias="reasoningChars")
    step_budget: int | None = Field(default=None, alias="stepBudget")
    completed_step_count: int | None = Field(default=None, alias="completedStepCount")
    tool_call_count: int | None = Field(default=None, alias="toolCallCount")
    tool_registry: QueryToolRegistryDiagnostics | None = Field(
        default=None,
        alias="toolRegistry",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryCapabilityMissPayload(BaseModel):
    """Structured capability miss payload surfaced when no grounded path remains."""

    message: str
    missing_capabilities: list[str] = Field(..., alias="missingCapabilities")
    suggested_rewrites: list[str] = Field(..., alias="suggestedRewrites")
    original_query: str = Field(..., alias="originalQuery")

    model_config = {"populate_by_name": True}


class QueryAssumptionMetadata(BaseModel):
    """Auto-resolution metadata attached when the server continues with an assumption."""

    mode: Literal["auto"]
    option_id: str = Field(..., alias="optionId")
    label: str
    reason: str

    model_config = {"populate_by_name": True}


class QueryDeveloperTraceToolRef(BaseModel):
    """Tool reference attached to developer trace timeline steps."""

    id: str | None = None
    name: str | None = None
    method: str | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceLoopInfo(BaseModel):
    """Loop metadata attached to developer trace timeline steps."""

    name: str | None = None
    iteration: int | None = None
    max_iterations: int | None = Field(default=None, alias="maxIterations")

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceToolSelection(BaseModel):
    """Tool-selection metadata attached to discovery diagnostics."""

    tool_id: str | None = Field(default=None, alias="toolId")
    tool_name: str | None = Field(default=None, alias="toolName")
    selected_method_count: int | None = Field(default=None, alias="selectedMethodCount")
    selected_methods: list[str] | None = Field(default=None, alias="selectedMethods")
    omitted_selected_method_count: int | None = Field(
        default=None,
        alias="omittedSelectedMethodCount",
    )
    price_usd: str | None = Field(default=None, alias="priceUsd")

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryPlanningTraceDiagnostic(BaseModel):
    """Execution-contract diagnostics handed to the iterative runtime."""

    planner_query: str | None = Field(default=None, alias="plannerQuery")
    scout_evidence_attached: bool | None = Field(default=None, alias="scoutEvidenceAttached")
    scout_evidence_prompt_block: str | None = Field(
        default=None,
        alias="scoutEvidencePromptBlock",
    )
    allowed_modules: list[str] | None = Field(default=None, alias="allowedModules")

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryRediscoveryTraceDiagnostic(BaseModel):
    """Rediscovery/bounded-fallback diagnostics for query orchestration."""

    considered: bool | None = None
    executed: bool | None = None
    skip_reason: str | None = Field(default=None, alias="skipReason")
    missing_capability: str | None = Field(default=None, alias="missingCapability")
    rediscovery_query: str | None = Field(default=None, alias="rediscoveryQuery")
    capability_looks_like_search_need: bool | None = Field(
        default=None,
        alias="capabilityLooksLikeSearchNeed",
    )
    allow_search_fallback_on_elapsed_cap: bool | None = Field(
        default=None,
        alias="allowSearchFallbackOnElapsedCap",
    )
    search_fallback_used: bool | None = Field(default=None, alias="searchFallbackUsed")
    pre_rediscovery_budget_reason_code: str | None = Field(
        default=None,
        alias="preRediscoveryBudgetReasonCode",
    )
    candidate_search_results: list[QueryDeveloperTraceToolSelection] | None = Field(
        default=None,
        alias="candidateSearchResults",
    )
    selected_alternatives: list[QueryDeveloperTraceToolSelection] | None = Field(
        default=None,
        alias="selectedAlternatives",
    )
    merged_tools: list[QueryDeveloperTraceToolSelection] | None = Field(
        default=None,
        alias="mergedTools",
    )
    using_paid_fallback: bool | None = Field(default=None, alias="usingPaidFallback")
    branch_plan: QueryPlanningTraceDiagnostic | None = Field(
        default=None,
        alias="branchPlan",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceSelectionDiagnostics(BaseModel):
    """Tool-selection and policy diagnostics for managed query-runtime internals."""

    selected_policy: str | None = Field(default=None, alias="selectedPolicy")
    debug_scout_deep_mode: str | None = Field(default=None, alias="debugScoutDeepMode")
    planner_reasoning_stage: str | None = Field(default=None, alias="plannerReasoningStage")
    scout_enabled: bool | None = Field(default=None, alias="scoutEnabled")
    one_shot_bias: bool | None = Field(default=None, alias="oneShotBias")
    candidate_method_count: int | None = Field(default=None, alias="candidateMethodCount")
    scout_probe_status: str | None = Field(default=None, alias="scoutProbeStatus")
    scout_probe_adequacy: str | None = Field(default=None, alias="scoutProbeAdequacy")
    scout_probe_confidence: float | None = Field(default=None, alias="scoutProbeConfidence")
    scout_metadata_confidence: float | None = Field(
        default=None,
        alias="scoutMetadataConfidence",
    )
    scout_probe_query_safe_candidate_count: int | None = Field(
        default=None,
        alias="scoutProbeQuerySafeCandidateCount",
    )
    scout_probe_ranked_method_count: int | None = Field(
        default=None,
        alias="scoutProbeRankedMethodCount",
    )
    scout_probe_ambiguity_pool_count: int | None = Field(
        default=None,
        alias="scoutProbeAmbiguityPoolCount",
    )
    scout_probe_shortlisted_method_count: int | None = Field(
        default=None,
        alias="scoutProbeShortlistedMethodCount",
    )
    scout_probe_missing_capability: str | None = Field(
        default=None,
        alias="scoutProbeMissingCapability",
    )
    scout_pre_plan_probe_calls: int | None = Field(
        default=None,
        alias="scoutPrePlanProbeCalls",
    )
    scout_pre_plan_probe_budget_reason_code: str | None = Field(
        default=None,
        alias="scoutPrePlanProbeBudgetReasonCode",
    )
    scout_changed_initial_plan: bool | None = Field(
        default=None,
        alias="scoutChangedInitialPlan",
    )
    scout_changed_planner_reasoning_stage: bool | None = Field(
        default=None,
        alias="scoutChangedPlannerReasoningStage",
    )
    scout_initial_selected_policy: str | None = Field(
        default=None,
        alias="scoutInitialSelectedPolicy",
    )
    scout_initial_planner_reasoning_stage: str | None = Field(
        default=None,
        alias="scoutInitialPlannerReasoningStage",
    )
    scout_initial_reason_code: str | None = Field(
        default=None,
        alias="scoutInitialReasonCode",
    )
    scout_final_reason_code: str | None = Field(
        default=None,
        alias="scoutFinalReasonCode",
    )
    scout_evidence_attached_to_planning: bool | None = Field(
        default=None,
        alias="scoutEvidenceAttachedToPlanning",
    )
    scout_llm_selection_used: bool | None = Field(
        default=None,
        alias="scoutLlmSelectionUsed",
    )
    scout_llm_selection_fallback: bool | None = Field(
        default=None,
        alias="scoutLlmSelectionFallback",
    )
    scout_llm_selection_latency_ms: float | None = Field(
        default=None,
        alias="scoutLlmSelectionLatencyMs",
    )
    selected_tools: list[QueryDeveloperTraceToolSelection] | None = Field(
        default=None,
        alias="selectedTools",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTracePlanningDiagnostics(BaseModel):
    """Planning diagnostics payload."""

    initial: QueryPlanningTraceDiagnostic | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceCostDiagnostics(BaseModel):
    """Cost diagnostics payload."""

    planning_cost_usd: float | None = Field(default=None, alias="planningCostUsd")
    initial_execution_cost_usd: float | None = Field(
        default=None,
        alias="initialExecutionCostUsd",
    )
    rediscovery_additional_cost_usd: float | None = Field(
        default=None,
        alias="rediscoveryAdditionalCostUsd",
    )
    synthesis_cost_usd: float | None = Field(default=None, alias="synthesisCostUsd")
    total_model_cost_usd: float | None = Field(default=None, alias="totalModelCostUsd")
    tool_cost_usd: float | None = Field(default=None, alias="toolCostUsd")
    total_charged_usd: float | None = Field(default=None, alias="totalChargedUsd")

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryCompletenessRepairEvent(BaseModel):
    """Explicit completeness-repair outcome emitted by the execution loop."""

    attempt: int
    outcome: str
    semantic_retry_count: int = Field(alias="semanticRetryCount")
    max_semantic_retries: int = Field(alias="maxSemanticRetries")
    strategy: Literal["patch", "replan"] | None = None
    summary: str | None = None
    fail_reason: str | None = Field(default=None, alias="failReason")
    requested_replan: bool = Field(default=False, alias="requestedReplan")
    had_syntax_fix: bool = Field(default=False, alias="hadSyntaxFix")
    edit_count: int | None = Field(default=None, alias="editCount")
    skip_reason: str | None = Field(default=None, alias="skipReason")
    bounded_answer_reason: str | None = Field(
        default=None,
        alias="boundedAnswerReason",
    )
    blocking_diagnostics: list[dict[str, Any]] | None = Field(
        default=None,
        alias="blockingDiagnostics",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceCompletenessDiagnostics(BaseModel):
    """Completeness diagnostics payload."""

    evaluations: list[Any] | None = None
    repair_events: list[QueryCompletenessRepairEvent] | None = Field(
        default=None,
        alias="repairEvents",
    )
    trigger_needs_different_tools: bool | None = Field(
        default=None,
        alias="triggerNeedsDifferentTools",
    )
    trigger_missing_capability: str | None = Field(
        default=None,
        alias="triggerMissingCapability",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceDiagnostics(BaseModel):
    """Rich developer-trace diagnostics for managed query-runtime internals."""

    selection: QueryDeveloperTraceSelectionDiagnostics | None = None
    execution_contract: QueryPlanningTraceDiagnostic | None = Field(
        default=None,
        alias="executionContract",
    )
    cost: QueryDeveloperTraceCostDiagnostics | None = None
    verification: QueryDeveloperTraceCompletenessDiagnostics | None = None
    completeness: QueryDeveloperTraceCompletenessDiagnostics | None = None
    execution: QueryExecutionDiagnostics | None = None
    contributor_searches: list[ContributorSearchTraceRecord] | None = Field(
        default=None,
        alias="contributorSearches",
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceStep(BaseModel):
    """Single timeline step in a query developer trace payload."""

    step_type: str | None = Field(default=None, alias="stepType")
    event: str | None = None
    status: str | None = None
    message: str | None = None
    timestamp_ms: int | None = Field(default=None, alias="timestampMs")
    tool: QueryDeveloperTraceToolRef | None = None
    attempt: int | None = None
    loop: QueryDeveloperTraceLoopInfo | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTraceSummary(BaseModel):
    """Aggregate counters for query developer trace behavior."""

    tool_calls: int | None = Field(default=None, alias="toolCalls")
    retry_count: int | None = Field(default=None, alias="retryCount")
    self_heal_count: int | None = Field(default=None, alias="selfHealCount")
    fallback_count: int | None = Field(default=None, alias="fallbackCount")
    failure_count: int | None = Field(default=None, alias="failureCount")
    recovery_count: int | None = Field(default=None, alias="recoveryCount")
    completion_checks: int | None = Field(default=None, alias="completionChecks")
    loop_count: int | None = Field(default=None, alias="loopCount")

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryDeveloperTrace(BaseModel):
    """Developer Mode trace payload returned per query response (opt-in)."""

    summary: QueryDeveloperTraceSummary | None = None
    timeline: list[QueryDeveloperTraceStep] | None = None
    request_id: str | None = Field(default=None, alias="requestId")
    query: str | None = None
    source: str | None = None
    diagnostics: QueryDeveloperTraceDiagnostics | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class QueryOrchestrationMetrics(BaseModel):
    """High-level orchestration outcome metrics returned by the query API."""

    parity_stage: str = Field(..., alias="parityStage")
    orchestration_mode: str = Field(..., alias="orchestrationMode")
    first_pass_success: bool = Field(..., alias="firstPassSuccess")
    capability_miss_signaled: bool = Field(..., alias="capabilityMissSignaled")
    rediscovery_executed: bool = Field(..., alias="rediscoveryExecuted")

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeFact(BaseModel):
    """A single structured evidence fact attached to a query answer."""

    id: str
    label: str
    path: str | None = None
    relevance_score: float | None = Field(default=None, alias="relevanceScore")
    value: Any

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeSourceRef(BaseModel):
    """A source reference extracted from canonical execution evidence."""

    id: str
    provider: str | None = None
    dataset: str | None = None
    observed_at: str | None = Field(default=None, alias="observedAt")
    published_at: str | None = Field(default=None, alias="publishedAt")
    artifact_ref: str | None = Field(default=None, alias="artifactRef")
    url: str | None = None
    note: str | None = None

    model_config = {"populate_by_name": True}


QueryResponseEnvelopeTone = Literal["positive", "negative", "neutral", "caution"]
QueryControllerStopReason = Literal[
    "complete_answer",
    "bounded_runtime_budget",
    "bounded_same_endpoint_guardrail",
    "bounded_upstream_abort_guardrail",
    "bounded_explicit_empty_result_guardrail",
    "capability_miss",
]
QueryControllerIssueClass = Literal[
    "missing_evidence",
    "missing_capability",
    "stale_data",
    "wrong_tool_path",
]
QueryControllerAction = Literal[
    "inspect_current_grounding",
    "patch_current_program",
    "bounded_rediscovery",
    "return_capability_miss",
    "return_bounded_answer",
    "return_complete_answer",
]


class QueryResponseEnvelopeMarketAggregateFlow(BaseModel):
    net_flow_usd: float | None = Field(default=None, alias="netFlowUsd")
    gross_inflow_usd: float | None = Field(default=None, alias="grossInflowUsd")
    gross_outflow_usd: float | None = Field(default=None, alias="grossOutflowUsd")
    native_net_flow: float | None = Field(default=None, alias="nativeNetFlow")
    native_unit: str | None = Field(default=None, alias="nativeUnit")
    direction: Literal["inflow", "outflow", "flat", "mixed"]

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeMarketVenueBreakdown(BaseModel):
    venue: str
    asset: str | None = None
    net_flow_usd: float | None = Field(default=None, alias="netFlowUsd")
    gross_inflow_usd: float | None = Field(default=None, alias="grossInflowUsd")
    gross_outflow_usd: float | None = Field(default=None, alias="grossOutflowUsd")
    native_net_flow: float | None = Field(default=None, alias="nativeNetFlow")
    native_unit: str | None = Field(default=None, alias="nativeUnit")
    share_of_total: float | None = Field(default=None, alias="shareOfTotal")
    rank: int | None = None

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeCatalystRef(BaseModel):
    source: str
    published_at: str | None = Field(default=None, alias="publishedAt")
    claim: str | None = None
    relation_to_flow: str | None = Field(default=None, alias="relationToFlow")
    url: str | None = None

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeDerivativesContext(BaseModel):
    open_interest_direction: str | None = Field(
        default=None, alias="openInterestDirection"
    )
    open_interest_change_pct: float | None = Field(
        default=None, alias="openInterestChangePct"
    )
    liquidation_bias: str | None = Field(default=None, alias="liquidationBias")
    venues: list[str] = Field(default_factory=list)
    relationship_to_spot_flows: str | None = Field(
        default=None, alias="relationshipToSpotFlows"
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeMarketIntelligence(BaseModel):
    asset: str | None = None
    assets: list[str] | None = None
    time_window: str | None = Field(default=None, alias="timeWindow")
    as_of: str | None = Field(default=None, alias="asOf")
    aggregate_flow: QueryResponseEnvelopeMarketAggregateFlow | None = Field(
        default=None, alias="aggregateFlow"
    )
    venue_breakdown: list[QueryResponseEnvelopeMarketVenueBreakdown] = Field(
        default_factory=list,
        alias="venueBreakdown",
    )
    catalyst_refs: list[QueryResponseEnvelopeCatalystRef] = Field(
        default_factory=list,
        alias="catalystRefs",
    )
    derivatives_context: QueryResponseEnvelopeDerivativesContext | None = Field(
        default=None,
        alias="derivativesContext",
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeViewMetric(BaseModel):
    label: str
    value: str
    tone: QueryResponseEnvelopeTone | None = None


class QueryResponseEnvelopeViewRow(BaseModel):
    key: str
    cells: list[str]
    tone: QueryResponseEnvelopeTone | None = None
    source_ref_ids: list[str] | None = Field(default=None, alias="sourceRefIds")

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeOutcome(BaseModel):
    label: str
    tone: QueryResponseEnvelopeTone
    stop_reason: QueryControllerStopReason = Field(..., alias="stopReason")
    issue_class: QueryControllerIssueClass | None = Field(
        default=None,
        alias="issueClass",
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeController(BaseModel):
    scope: Literal["wedge", "standard"]
    next_action: QueryControllerAction = Field(..., alias="nextAction")
    actions_taken: list[QueryControllerAction] = Field(..., alias="actionsTaken")
    patch_first_program_preserved: bool = Field(
        ..., alias="patchFirstProgramPreserved"
    )
    execution_program_revision_id: str | None = Field(
        default=None,
        alias="executionProgramRevisionId",
    )
    hard_budget_applied: bool = Field(..., alias="hardBudgetApplied")

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeCanonicalDataRef(BaseModel):
    """Canonical dataset metadata for a structured query answer."""

    dataset_id: str = Field(..., alias="datasetId")
    hash: str
    bytes: int
    public_data_url: str | None = Field(default=None, alias="publicDataUrl")

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeEvidence(BaseModel):
    """Structured evidence payload attached to a query answer."""

    facts: list[QueryResponseEnvelopeFact]
    source_refs: list[QueryResponseEnvelopeSourceRef] = Field(
        default_factory=list,
        alias="sourceRefs",
    )
    assumptions: list[str] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list, alias="knownUnknowns")
    retrieval_plan_reason_codes: list[str] = Field(
        default_factory=list,
        alias="retrievalPlanReasonCodes",
    )
    market_intelligence: QueryResponseEnvelopeMarketIntelligence | None = Field(
        default=None,
        alias="marketIntelligence",
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeArtifacts(BaseModel):
    """Artifact references attached to a structured query answer."""

    data_url: str | None = Field(default=None, alias="dataUrl")
    canonical_data_ref: QueryResponseEnvelopeCanonicalDataRef | None = Field(
        default=None,
        alias="canonicalDataRef",
    )
    stage_artifact_kinds: list[str] = Field(
        default_factory=list,
        alias="stageArtifactKinds",
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeView(BaseModel):
    """Optional render hint for first-party or external clients."""

    type: QueryResponseEnvelopeViewType
    label: str
    title: str | None = None
    metrics: list[QueryResponseEnvelopeViewMetric] | None = None
    columns: list[str] | None = None
    rows: list[QueryResponseEnvelopeViewRow] | None = None

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeFreshness(BaseModel):
    """Freshness metadata derived from structured evidence."""

    as_of: str | None = Field(default=None, alias="asOf")
    source_timestamps: list[str] = Field(
        default_factory=list,
        alias="sourceTimestamps",
    )
    note: str

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeGapSignal(BaseModel):
    """Gap signal attached to structured query confidence metadata."""

    code: str
    severity: str
    detail: str

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeConfidence(BaseModel):
    """Confidence metadata for a structured query answer."""

    level: Literal["high", "medium", "low"]
    reason: str
    verified_fact_count: int = Field(..., alias="verifiedFactCount")
    inferred_fact_count: int = Field(..., alias="inferredFactCount")
    gap_count: int = Field(..., alias="gapCount")
    gap_signals: list[QueryResponseEnvelopeGapSignal] = Field(
        default_factory=list,
        alias="gapSignals",
    )

    model_config = {"populate_by_name": True}


class QueryResponseEnvelopeUsage(BaseModel):
    """Usage metadata surfaced with structured query answers."""

    duration_ms: int = Field(..., alias="durationMs")
    cost: QueryCost
    tools_used: list[QueryToolUsage] = Field(..., alias="toolsUsed")
    outcome_type: QueryOutcomeType = Field(..., alias="outcomeType")
    orchestration_metrics: QueryOrchestrationMetrics | None = Field(
        default=None,
        alias="orchestrationMetrics",
    )

    model_config = {"populate_by_name": True}


class QueryResult(BaseModel):
    """The resolved result of a pay-per-response query.

    Attributes:
        response: The AI-synthesized response text
        tools_used: Tools that were used to answer the query
        cost: Cost breakdown
        duration_ms: Total duration in milliseconds
        data: Bounded execution data when include_data is enabled
        data_url: Optional blob URL for execution data (when include_data_url is enabled)
        computed_artifacts: Optional chart artifacts emitted by code interpreter
        developer_trace: Optional machine-readable Developer Mode trace
        orchestration_metrics: Optional high-level first-pass/bounded-fallback metrics
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
        description=(
            "Bounded execution data from tools. Returned only when include_data "
            "is true. Small payloads may be direct data; large payloads are a "
            "truncation object with preview and fullData references."
        ),
    )
    data_url: str | None = Field(
        default=None,
        alias="dataUrl",
        description="Optional blob URL for persisted execution data",
    )
    computed_artifacts: list[QueryComputedArtifact] | None = Field(
        default=None,
        alias="computedArtifacts",
        description="Optional chart artifacts emitted by code interpreter",
    )
    grounding: QueryGroundingSummary | None = Field(
        default=None,
        description="Public grounding summary for marketplace tool execution",
    )
    developer_trace: QueryDeveloperTrace | None = Field(
        default=None,
        alias="developerTrace",
        description="Optional machine-readable Developer Mode trace payload",
    )
    orchestration_metrics: QueryOrchestrationMetrics | None = Field(
        default=None,
        alias="orchestrationMetrics",
        description="Optional orchestration outcome metrics for rollout diagnostics",
    )
    response_shape: QueryResponseShape | None = Field(
        default=None,
        alias="responseShape",
        description="Structured response mode returned by the query API",
    )
    outcome: QueryResponseEnvelopeOutcome | None = Field(
        default=None,
        description="Public outcome label and stop reason for structured query answers",
    )
    controller: QueryResponseEnvelopeController | None = Field(
        default=None,
        description="Public bounded-controller summary for structured query answers",
    )
    summary: str | None = Field(
        default=None,
        description="Machine-friendly summary attached to structured query responses",
    )
    evidence: QueryResponseEnvelopeEvidence | None = Field(
        default=None,
        description="Structured evidence package for query answers",
    )
    artifacts: QueryResponseEnvelopeArtifacts | None = Field(
        default=None,
        description="Artifact references attached to structured query answers",
    )
    view: QueryResponseEnvelopeView | None = Field(
        default=None,
        description="Optional render hint for structured query answers",
    )
    freshness: QueryResponseEnvelopeFreshness | None = Field(
        default=None,
        description="Freshness metadata derived from structured evidence",
    )
    confidence: QueryResponseEnvelopeConfidence | None = Field(
        default=None,
        description="Confidence metadata for structured query answers",
    )
    usage: QueryResponseEnvelopeUsage | None = Field(
        default=None,
        description="Usage metadata surfaced for structured query answers",
    )
    outcome_type: QueryOutcomeType = Field(
        default="answer",
        alias="outcomeType",
        description="Structured query outcome classification",
    )
    capability_miss: QueryCapabilityMissPayload | None = Field(
        default=None,
        alias="capabilityMiss",
        description="Structured capability miss payload when no grounded path remains",
    )
    assumption_made: QueryAssumptionMetadata | None = Field(
        default=None,
        alias="assumptionMade",
        description="Auto-resolution metadata when the server proceeded with an assumption",
    )
    stop_reason: QueryControllerStopReason | None = Field(
        default=None,
        alias="stopReason",
        description="Typed stop reason for the final outcome",
    )
    issue_class: QueryControllerIssueClass | None = Field(
        default=None,
        alias="issueClass",
        description="Typed controller issue class for the final outcome",
    )
    actions_taken: list[QueryControllerAction] | None = Field(
        default=None,
        alias="actionsTaken",
        description="Ordered public controller actions taken before the final outcome",
    )
    query_session: QuerySessionState | None = Field(
        default=None,
        alias="querySession",
        description="Public durable continuation handles for resume/fork flows",
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
    computed_artifacts: list[QueryComputedArtifact] | None = Field(
        default=None,
        alias="computedArtifacts",
    )
    grounding: QueryGroundingSummary | None = None
    developer_trace: QueryDeveloperTrace | None = Field(
        default=None,
        alias="developerTrace",
    )
    orchestration_metrics: QueryOrchestrationMetrics | None = Field(
        default=None,
        alias="orchestrationMetrics",
    )
    response_shape: QueryResponseShape | None = Field(
        default=None,
        alias="responseShape",
    )
    outcome: QueryResponseEnvelopeOutcome | None = None
    controller: QueryResponseEnvelopeController | None = None
    summary: str | None = None
    evidence: QueryResponseEnvelopeEvidence | None = None
    artifacts: QueryResponseEnvelopeArtifacts | None = None
    view: QueryResponseEnvelopeView | None = None
    freshness: QueryResponseEnvelopeFreshness | None = None
    confidence: QueryResponseEnvelopeConfidence | None = None
    usage: QueryResponseEnvelopeUsage | None = None
    outcome_type: QueryOutcomeType = Field(default="answer", alias="outcomeType")
    capability_miss: QueryCapabilityMissPayload | None = Field(
        default=None,
        alias="capabilityMiss",
    )
    assumption_made: QueryAssumptionMetadata | None = Field(
        default=None,
        alias="assumptionMade",
    )
    stop_reason: QueryControllerStopReason | None = Field(
        default=None,
        alias="stopReason",
    )
    issue_class: QueryControllerIssueClass | None = Field(
        default=None,
        alias="issueClass",
    )
    actions_taken: list[QueryControllerAction] | None = Field(
        default=None,
        alias="actionsTaken",
    )
    query_session: QuerySessionState | None = Field(default=None, alias="querySession")

    model_config = {"populate_by_name": True}


QueryJobStatus = Literal["queued", "running", "completed", "failed"]


class QueryJobStartResult(BaseModel):
    """Response from starting a durable async query job."""

    status: QueryJobStatus
    job_id: str = Field(..., alias="jobId")
    polling_tool: str | None = Field(default=None, alias="pollingTool")
    message: str | None = None
    progress: Any | None = None
    query_session: Any | None = Field(default=None, alias="querySession")
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")

    model_config = {"populate_by_name": True}


class QueryJobStatusResult(BaseModel):
    """Current status for a durable async query job."""

    status: QueryJobStatus
    job_id: str = Field(..., alias="jobId")
    progress: Any | None = None
    query_session: Any | None = Field(default=None, alias="querySession")
    result: QueryResult | None = None
    error: str | None = None
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")

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


class QueryStreamDeveloperTraceEvent(BaseModel):
    """Emitted when the server streams developer trace updates/chunks."""

    type: Literal["developer-trace"]
    trace: QueryDeveloperTrace


class QueryStreamDoneEvent(BaseModel):
    """Emitted when the full response is complete."""

    type: Literal["done"]
    result: QueryResult


class QueryStreamErrorEvent(BaseModel):
    """Emitted when the server reports a structured query/stream error."""

    type: Literal["error"]
    error: str
    code: ContextErrorCode | str | None = None
    scope: str | None = None
    reason_code: str | None = Field(default=None, alias="reasonCode")
    outcome_type: Literal["capability_miss"] | None = Field(
        default=None,
        alias="outcomeType",
    )
    capability_miss: QueryCapabilityMissPayload | None = Field(
        default=None,
        alias="capabilityMiss",
    )
    query_session: QuerySessionState | None = Field(default=None, alias="querySession")

    model_config = {"populate_by_name": True}


QueryStreamEvent = Union[
    QueryStreamToolStatusEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamDeveloperTraceEvent,
    QueryStreamDoneEvent,
    QueryStreamErrorEvent,
]


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

