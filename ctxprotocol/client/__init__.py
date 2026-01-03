"""
ctxprotocol.client

Client module for AI Agents to query marketplace and execute tools.
"""

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.resources.discovery import Discovery
from ctxprotocol.client.resources.tools import Tools
from ctxprotocol.client.types import (
    ContextClientOptions,
    ContextError,
    ContextErrorCode,
    ExecuteApiErrorResponse,
    ExecuteApiSuccessResponse,
    ExecuteOptions,
    ExecutionResult,
    McpTool,
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
    # Types
    "ContextClientOptions",
    "Tool",
    "McpTool",
    "SearchResponse",
    "SearchOptions",
    "ExecuteOptions",
    "ExecutionResult",
    "ExecuteApiSuccessResponse",
    "ExecuteApiErrorResponse",
    "ToolInfo",
    "ContextErrorCode",
    # Errors
    "ContextError",
]

