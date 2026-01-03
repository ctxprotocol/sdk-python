"""
Polymarket context types for portfolio tracking.

These types represent Polymarket positions and orders that can be
injected into MCP tools for personalized portfolio analysis.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PolymarketPosition(BaseModel):
    """A single Polymarket position.

    Attributes:
        condition_id: The market's condition ID
        token_id: The specific outcome token ID
        outcome: Which outcome this position is for
        shares: Number of shares held
        avg_entry_price: Average entry price (0-1 scale)
        market_title: Market question/title for display
    """

    condition_id: str = Field(..., alias="conditionId", description="The market's condition ID")
    token_id: str = Field(..., alias="tokenId", description="The specific outcome token ID")
    outcome: Literal["YES", "NO"] = Field(..., description="Which outcome this position is for")
    shares: float = Field(..., description="Number of shares held")
    avg_entry_price: float = Field(
        ...,
        alias="avgEntryPrice",
        description="Average entry price (0-1 scale)",
    )
    market_title: str | None = Field(
        default=None,
        alias="marketTitle",
        description="Market question/title for display",
    )

    model_config = {"populate_by_name": True}


class PolymarketOrder(BaseModel):
    """An open order on Polymarket.

    Attributes:
        order_id: Order ID
        condition_id: The market's condition ID
        side: Order side
        outcome: Which outcome this order is for
        price: Limit price (0-1 scale)
        size: Order size in shares
        filled: Amount already filled
    """

    order_id: str = Field(..., alias="orderId", description="Order ID")
    condition_id: str = Field(..., alias="conditionId", description="The market's condition ID")
    side: Literal["BUY", "SELL"] = Field(..., description="Order side")
    outcome: Literal["YES", "NO"] = Field(..., description="Which outcome this order is for")
    price: float = Field(..., description="Limit price (0-1 scale)")
    size: float = Field(..., description="Order size in shares")
    filled: float = Field(..., description="Amount already filled")

    model_config = {"populate_by_name": True}


class PolymarketContext(BaseModel):
    """Complete Polymarket portfolio context.

    This is what gets passed to MCP tools for personalized analysis.

    Attributes:
        wallet_address: The wallet address this context is for
        positions: All open positions
        open_orders: All open orders
        total_value: Total portfolio value in USD (sum of position values)
        fetched_at: When this context was fetched (ISO 8601 string)
    """

    wallet_address: str = Field(
        ...,
        alias="walletAddress",
        description="The wallet address this context is for",
    )
    positions: list[PolymarketPosition] = Field(..., description="All open positions")
    open_orders: list[PolymarketOrder] = Field(
        ...,
        alias="openOrders",
        description="All open orders",
    )
    total_value: float | None = Field(
        default=None,
        alias="totalValue",
        description="Total portfolio value in USD (sum of position values)",
    )
    fetched_at: str = Field(
        ...,
        alias="fetchedAt",
        description="When this context was fetched (ISO 8601 string)",
    )

    model_config = {"populate_by_name": True}

