"""
Hyperliquid context types for portfolio tracking.

These types represent Hyperliquid perpetual positions, orders, and account
data that can be injected into MCP tools for personalized portfolio analysis.
"""

from typing import Literal

from pydantic import BaseModel, Field


class LeverageInfo(BaseModel):
    """Leverage information for a position."""

    type: Literal["cross", "isolated"] = Field(..., description="Leverage type")
    value: float = Field(..., description="Leverage value")


class CumFunding(BaseModel):
    """Cumulative funding information."""

    all_time: float = Field(..., alias="allTime", description="All-time cumulative funding")
    since_open: float = Field(
        ...,
        alias="sinceOpen",
        description="Cumulative funding since position opened",
    )

    model_config = {"populate_by_name": True}


class HyperliquidPerpPosition(BaseModel):
    """Hyperliquid Perpetual Position.

    Attributes:
        coin: Asset symbol (e.g., "ETH", "BTC")
        size: Position size (positive = long, negative = short)
        entry_price: Entry price
        mark_price: Current mark price
        unrealized_pnl: Unrealized PnL in USD
        liquidation_price: Liquidation price
        position_value: Position value in USD
        leverage: Leverage info
        margin_used: Margin used for this position
        return_on_equity: Return on equity percentage
        cum_funding: Cumulative funding paid/received
    """

    coin: str = Field(..., description="Asset symbol (e.g., 'ETH', 'BTC')")
    size: float = Field(..., description="Position size (positive = long, negative = short)")
    entry_price: float = Field(..., alias="entryPrice", description="Entry price")
    mark_price: float | None = Field(
        default=None,
        alias="markPrice",
        description="Current mark price",
    )
    unrealized_pnl: float = Field(
        ...,
        alias="unrealizedPnl",
        description="Unrealized PnL in USD",
    )
    liquidation_price: float = Field(
        ...,
        alias="liquidationPrice",
        description="Liquidation price",
    )
    position_value: float = Field(
        ...,
        alias="positionValue",
        description="Position value in USD",
    )
    leverage: LeverageInfo = Field(..., description="Leverage info")
    margin_used: float = Field(
        ...,
        alias="marginUsed",
        description="Margin used for this position",
    )
    return_on_equity: float = Field(
        ...,
        alias="returnOnEquity",
        description="Return on equity percentage",
    )
    cum_funding: CumFunding = Field(
        ...,
        alias="cumFunding",
        description="Cumulative funding paid/received",
    )

    model_config = {"populate_by_name": True}


class HyperliquidOrder(BaseModel):
    """Hyperliquid Open Order.

    Attributes:
        oid: Order ID
        coin: Asset symbol
        side: Order side: "B" = Buy, "A" = Ask/Sell
        limit_price: Limit price
        size: Order size
        original_size: Original order size
        order_type: Order type
        reduce_only: Is reduce-only order
        is_trigger: Is trigger order
        trigger_price: Trigger price (if trigger order)
        timestamp: Order timestamp
    """

    oid: int = Field(..., description="Order ID")
    coin: str = Field(..., description="Asset symbol")
    side: Literal["B", "A"] = Field(..., description="Order side: 'B' = Buy, 'A' = Ask/Sell")
    limit_price: float = Field(..., alias="limitPrice", description="Limit price")
    size: float = Field(..., description="Order size")
    original_size: float = Field(..., alias="originalSize", description="Original order size")
    order_type: Literal["Limit", "Market", "Stop", "TakeProfit"] = Field(
        ...,
        alias="orderType",
        description="Order type",
    )
    reduce_only: bool = Field(..., alias="reduceOnly", description="Is reduce-only order")
    is_trigger: bool = Field(..., alias="isTrigger", description="Is trigger order")
    trigger_price: float | None = Field(
        default=None,
        alias="triggerPrice",
        description="Trigger price (if trigger order)",
    )
    timestamp: int = Field(..., description="Order timestamp")

    model_config = {"populate_by_name": True}


class HyperliquidSpotBalance(BaseModel):
    """Hyperliquid Spot Balance.

    Attributes:
        token: Token symbol
        balance: Token balance
        usd_value: USD value
    """

    token: str = Field(..., description="Token symbol")
    balance: float = Field(..., description="Token balance")
    usd_value: float | None = Field(
        default=None,
        alias="usdValue",
        description="USD value",
    )

    model_config = {"populate_by_name": True}


class CrossMarginSummary(BaseModel):
    """Cross margin summary information."""

    account_value: float = Field(
        ...,
        alias="accountValue",
        description="Cross margin account value",
    )
    total_margin_used: float = Field(
        ...,
        alias="totalMarginUsed",
        description="Total margin used in cross margin",
    )

    model_config = {"populate_by_name": True}


class HyperliquidAccountSummary(BaseModel):
    """Hyperliquid Account Summary.

    Attributes:
        account_value: Total account value in USD
        total_margin_used: Total margin used
        total_notional_position: Total notional position value
        withdrawable: Withdrawable amount
        cross_margin: Cross margin summary
    """

    account_value: float = Field(
        ...,
        alias="accountValue",
        description="Total account value in USD",
    )
    total_margin_used: float = Field(
        ...,
        alias="totalMarginUsed",
        description="Total margin used",
    )
    total_notional_position: float = Field(
        ...,
        alias="totalNotionalPosition",
        description="Total notional position value",
    )
    withdrawable: float = Field(..., description="Withdrawable amount")
    cross_margin: CrossMarginSummary = Field(
        ...,
        alias="crossMargin",
        description="Cross margin summary",
    )

    model_config = {"populate_by_name": True}


class HyperliquidContext(BaseModel):
    """Complete Hyperliquid portfolio context.

    This is what gets passed to MCP tools for personalized analysis.

    Attributes:
        wallet_address: The wallet address this context is for
        perp_positions: Perpetual positions
        open_orders: Open orders
        spot_balances: Spot balances
        account_summary: Account summary
        fetched_at: When this context was fetched (ISO 8601 string)
    """

    wallet_address: str = Field(
        ...,
        alias="walletAddress",
        description="The wallet address this context is for",
    )
    perp_positions: list[HyperliquidPerpPosition] = Field(
        ...,
        alias="perpPositions",
        description="Perpetual positions",
    )
    open_orders: list[HyperliquidOrder] = Field(
        ...,
        alias="openOrders",
        description="Open orders",
    )
    spot_balances: list[HyperliquidSpotBalance] = Field(
        ...,
        alias="spotBalances",
        description="Spot balances",
    )
    account_summary: HyperliquidAccountSummary = Field(
        ...,
        alias="accountSummary",
        description="Account summary",
    )
    fetched_at: str = Field(
        ...,
        alias="fetchedAt",
        description="When this context was fetched (ISO 8601 string)",
    )

    model_config = {"populate_by_name": True}

