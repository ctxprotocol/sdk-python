"""
ctxprotocol - Official Python SDK for the Context Protocol

The Universal Adapter for AI Agents.

Connect your AI to the real world without managing API keys, hosting servers,
or reading documentation.

Context Protocol is **pip for AI capabilities**. Just as you install packages
to add functionality to your code, use the Context SDK to give your Agent
instant access to thousands of live data sources and actions—from DeFi and
Gas Oracles to Weather and Search.

Example:
    >>> from ctxprotocol import ContextClient
    >>>
    >>> async with ContextClient(api_key="sk_live_...") as client:
    ...     # Search for tools
    ...     tools = await client.discovery.search("gas price")
    ...
    ...     # Execute a tool
    ...     result = await client.tools.execute(
    ...         tool_id=tools[0].id,
    ...         tool_name=tools[0].mcp_tools[0].name,
    ...         args={"chainId": 8453},
    ...     )
    ...     print(result.result)

For more information, visit: https://ctxprotocol.com
"""

__version__ = "0.5.5"

# Re-export everything from client module
from ctxprotocol.client import (
    ContextClient,
    ContextError,
    Discovery,
    Tools,
)
from ctxprotocol.client.types import (
    ContextClientOptions,
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

# Context types for portfolio injection
from ctxprotocol.context import (
    # Constants
    CONTEXT_REQUIREMENTS_KEY,
    # Type aliases
    ContextRequirementType,
    # Wallet types
    WalletContext,
    ERC20Context,
    ERC20TokenBalance,
    # Polymarket types
    PolymarketContext,
    PolymarketPosition,
    PolymarketOrder,
    # Hyperliquid types
    HyperliquidContext,
    HyperliquidPerpPosition,
    HyperliquidOrder,
    HyperliquidSpotBalance,
    HyperliquidAccountSummary,
    CrossMarginSummary,
    LeverageInfo,
    CumFunding,
    # Composite types
    UserContext,
    ToolRequirements,
)

# Auth utilities for verifying platform requests
from ctxprotocol.auth import (
    verify_context_request,
    is_protected_mcp_method,
    is_open_mcp_method,
    create_context_middleware,
    ContextMiddleware,
    VerifyRequestOptions,
    CreateContextMiddlewareOptions,
)

# Handshake types and helpers for tools that need user interaction
# (signatures, transactions, OAuth)
from ctxprotocol.handshake import (
    # Types
    HandshakeMeta,
    EIP712Domain,
    EIP712TypeField,
    SignatureRequest,
    TransactionProposalMeta,
    TransactionProposal,
    AuthRequiredMeta,
    AuthRequired,
    HandshakeAction,
    # Type guards
    is_handshake_action,
    is_signature_request,
    is_transaction_proposal,
    is_auth_required,
    # Helper functions
    create_signature_request,
    create_transaction_proposal,
    create_auth_required,
    wrap_handshake_response,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "ContextClient",
    "Discovery",
    "Tools",
    # Client types
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
    # Context constants
    "CONTEXT_REQUIREMENTS_KEY",
    # Context type aliases
    "ContextRequirementType",
    # Wallet context types
    "WalletContext",
    "ERC20Context",
    "ERC20TokenBalance",
    # Polymarket context types
    "PolymarketContext",
    "PolymarketPosition",
    "PolymarketOrder",
    # Hyperliquid context types
    "HyperliquidContext",
    "HyperliquidPerpPosition",
    "HyperliquidOrder",
    "HyperliquidSpotBalance",
    "HyperliquidAccountSummary",
    "CrossMarginSummary",
    "LeverageInfo",
    "CumFunding",
    # Composite context types
    "UserContext",
    "ToolRequirements",
    # Auth utilities
    "verify_context_request",
    "is_protected_mcp_method",
    "is_open_mcp_method",
    "create_context_middleware",
    "ContextMiddleware",
    "VerifyRequestOptions",
    "CreateContextMiddlewareOptions",
    # Handshake types
    "HandshakeMeta",
    "EIP712Domain",
    "EIP712TypeField",
    "SignatureRequest",
    "TransactionProposalMeta",
    "TransactionProposal",
    "AuthRequiredMeta",
    "AuthRequired",
    "HandshakeAction",
    # Handshake type guards
    "is_handshake_action",
    "is_signature_request",
    "is_transaction_proposal",
    "is_auth_required",
    # Handshake helper functions
    "create_signature_request",
    "create_transaction_proposal",
    "create_auth_required",
    "wrap_handshake_response",
]

