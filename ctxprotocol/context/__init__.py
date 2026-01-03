"""
Context types for portfolio and protocol data injection.

These types allow MCP tools to receive personalized user context
(wallet addresses, positions, balances) for analysis.

=============================================================================
DECLARING CONTEXT REQUIREMENTS
=============================================================================

Since the MCP protocol only transmits standard fields (name, description,
inputSchema, outputSchema), context requirements MUST be embedded in the
inputSchema using the "x-context-requirements" JSON Schema extension.

Example:
    >>> from ctxprotocol import CONTEXT_REQUIREMENTS_KEY, ContextRequirementType
    >>> from ctxprotocol.context import HyperliquidContext
    >>>
    >>> tool = {
    ...     "name": "analyze_my_positions",
    ...     "inputSchema": {
    ...         "type": "object",
    ...         CONTEXT_REQUIREMENTS_KEY: ["hyperliquid"],
    ...         "properties": {
    ...             "portfolio": {"type": "object"}
    ...         },
    ...         "required": ["portfolio"]
    ...     }
    ... }
    >>>
    >>> # Your handler receives the injected context:
    >>> def handle_analyze_my_positions(portfolio: HyperliquidContext):
    ...     positions = portfolio.perp_positions
    ...     account = portfolio.account_summary
    ...     # ... analyze and return insights
"""

from typing import Literal

from pydantic import BaseModel, Field

# Wallet context types
from ctxprotocol.context.wallet import ERC20Context, ERC20TokenBalance, WalletContext

# Protocol-specific context types
from ctxprotocol.context.polymarket import (
    PolymarketContext,
    PolymarketOrder,
    PolymarketPosition,
)
from ctxprotocol.context.hyperliquid import (
    CrossMarginSummary,
    CumFunding,
    HyperliquidAccountSummary,
    HyperliquidContext,
    HyperliquidOrder,
    HyperliquidPerpPosition,
    HyperliquidSpotBalance,
    LeverageInfo,
)

# ============================================================================
# CONTEXT REQUIREMENTS
#
# MCP tools that need user portfolio data MUST declare this in inputSchema.
# The MCP protocol only transmits standard fields (name, description,
# inputSchema, outputSchema). Custom fields get stripped by the MCP SDK.
# ============================================================================

CONTEXT_REQUIREMENTS_KEY = "x-context-requirements"
"""
JSON Schema extension key for declaring context requirements.

WHY THIS APPROACH?
- MCP protocol only transmits: name, description, inputSchema, outputSchema
- Custom fields like `requirements` get stripped by MCP SDK during transport
- JSON Schema allows custom "x-" prefixed extension properties
- inputSchema is preserved end-to-end through MCP transport

Example:
    >>> tool = {
    ...     "name": "analyze_my_positions",
    ...     "inputSchema": {
    ...         "type": "object",
    ...         CONTEXT_REQUIREMENTS_KEY: ["hyperliquid"],
    ...         "properties": {"portfolio": {"type": "object"}},
    ...         "required": ["portfolio"]
    ...     }
    ... }
"""

# Context requirement types supported by the Context marketplace
ContextRequirementType = Literal["polymarket", "hyperliquid", "wallet"]
"""
Context requirement types supported by the Context marketplace.
Maps to protocol-specific context builders on the platform.

Example:
    >>> input_schema = {
    ...     "type": "object",
    ...     "x-context-requirements": ["hyperliquid"],  # type: ContextRequirementType
    ...     "properties": {"portfolio": {"type": "object"}},
    ...     "required": ["portfolio"]
    ... }
"""


class ToolRequirements(BaseModel):
    """
    DEPRECATED: The `requirements` field at tool level gets stripped by MCP SDK.
    Use `x-context-requirements` inside `inputSchema` instead.

    Example:
        >>> # ❌ OLD (doesn't work - stripped by MCP SDK)
        >>> {"requirements": {"context": ["hyperliquid"]}}
        >>>
        >>> # ✅ NEW (works - preserved through MCP transport)
        >>> {"inputSchema": {"x-context-requirements": ["hyperliquid"], ...}}
    """

    context: list[ContextRequirementType] | None = Field(
        default=None,
        description="DEPRECATED: Use x-context-requirements in inputSchema instead.",
    )


class UserContext(BaseModel):
    """Composite context for tools that need multiple data sources.

    This is the unified structure that can be passed to MCP tools
    to provide comprehensive user context.

    Attributes:
        wallet: Base wallet information
        erc20: ERC20 token holdings
        polymarket: Polymarket positions and orders
        hyperliquid: Hyperliquid perpetual positions and account data
    """

    wallet: WalletContext | None = Field(
        default=None,
        description="Base wallet information",
    )
    erc20: ERC20Context | None = Field(
        default=None,
        description="ERC20 token holdings",
    )
    polymarket: PolymarketContext | None = Field(
        default=None,
        description="Polymarket positions and orders",
    )
    hyperliquid: HyperliquidContext | None = Field(
        default=None,
        description="Hyperliquid perpetual positions and account data",
    )


__all__ = [
    # Constants
    "CONTEXT_REQUIREMENTS_KEY",
    # Type aliases
    "ContextRequirementType",
    # Wallet types
    "WalletContext",
    "ERC20Context",
    "ERC20TokenBalance",
    # Polymarket types
    "PolymarketContext",
    "PolymarketPosition",
    "PolymarketOrder",
    # Hyperliquid types
    "HyperliquidContext",
    "HyperliquidPerpPosition",
    "HyperliquidOrder",
    "HyperliquidSpotBalance",
    "HyperliquidAccountSummary",
    "CrossMarginSummary",
    "LeverageInfo",
    "CumFunding",
    # Composite types
    "UserContext",
    "ToolRequirements",
]

