"""
ctxprotocol.client

Client module for AI Agents to query marketplace and execute tools.
"""

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.resources.discovery import Discovery
from ctxprotocol.client.resources.query import Query
from ctxprotocol.client.resources.tools import Tools
from ctxprotocol.client.types import (
    ContextClientOptions,
    ContextError,
    ContextErrorCode,
    ExecuteApiErrorResponse,
    ExecuteApiSuccessResponse,
    ExecuteMethodInfo,
    ExecuteOptions,
    ExecuteSessionApiSuccessResponse,
    ExecuteSessionResult,
    ExecuteSessionSpend,
    ExecuteSessionStartOptions,
    ExecuteSessionStatus,
    ExecutionResult,
    McpToolMeta,
    McpToolPricingMeta,
    McpToolRateLimitHints,
    McpTool,
    # Query types (pay-per-response)
    QueryApiSuccessResponse,
    QueryCost,
    QueryDeveloperTrace,
    QueryDeveloperTraceSummary,
    QueryDeveloperTraceStep,
    QueryDeveloperTraceToolRef,
    QueryDeveloperTraceLoopInfo,
    QueryStreamDeveloperTraceEvent,
    QueryStreamEvent,
    QueryOptions,
    QueryResult,
    QueryStreamDoneEvent,
    QueryStreamTextDeltaEvent,
    QueryStreamToolStatusEvent,
    QueryToolUsage,
    SearchOptions,
    SearchResponse,
    Tool,
    ToolInfo,
)

__all__ = [
    # Main client
    "ContextClient",
    # Resources
    "Discovery",
    "Tools",
    "Query",
    # Types
    "ContextClientOptions",
    "Tool",
    "McpTool",
    "McpToolMeta",
    "McpToolPricingMeta",
    "McpToolRateLimitHints",
    "SearchResponse",
    "SearchOptions",
    "ExecuteOptions",
    "ExecuteSessionStartOptions",
    "ExecuteSessionStatus",
    "ExecuteSessionSpend",
    "ExecuteSessionResult",
    "ExecutionResult",
    "ExecuteMethodInfo",
    "ExecuteApiSuccessResponse",
    "ExecuteApiErrorResponse",
    "ExecuteSessionApiSuccessResponse",
    "ToolInfo",
    # Query types (pay-per-response)
    "QueryOptions",
    "QueryResult",
    "QueryToolUsage",
    "QueryCost",
    "QueryDeveloperTrace",
    "QueryDeveloperTraceSummary",
    "QueryDeveloperTraceStep",
    "QueryDeveloperTraceToolRef",
    "QueryDeveloperTraceLoopInfo",
    "QueryApiSuccessResponse",
    "QueryStreamToolStatusEvent",
    "QueryStreamTextDeltaEvent",
    "QueryStreamDeveloperTraceEvent",
    "QueryStreamDoneEvent",
    "QueryStreamEvent",
    "ContextErrorCode",
    # Errors
    "ContextError",
]
