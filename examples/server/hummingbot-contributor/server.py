"""
Hummingbot Market Intelligence MCP Server (Python + FastMCP)

A PUBLIC MARKET DATA MCP server powered by Hummingbot API.
Provides access to real-time market data, liquidity analysis, and DEX quotes.

SCOPE: Public market data only - NO user accounts, NO trading operations

Features:
- Multi-exchange price data (40+ CEX/DEX connectors)
- Order book analysis with VWAP and slippage estimation
- Funding rate analysis for perpetuals
- Trade impact analysis

Architecture:
- Built with FastMCP (MCP 2025-06-18 spec compliant)
- Runs on the SAME server as Hummingbot API (localhost:8000)
- Uses Official hummingbot-api-client by Fede @ Hummingbot
- Integrates ctxprotocol SDK for payment verification
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
import uvicorn

# Official Hummingbot API Client (by Fede @ Hummingbot)
# https://github.com/hummingbot/hummingbot-api-client
from hummingbot_api_client import HummingbotAPIClient

from ctxprotocol import verify_context_request, ContextError

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

HUMMINGBOT_API_URL = os.getenv("HUMMINGBOT_API_URL", "http://localhost:8000")
HB_USERNAME = os.getenv("HB_USERNAME", "admin")
HB_PASSWORD = os.getenv("HB_PASSWORD", "admin")
PORT = int(os.getenv("PORT", "4010"))

# Exchange lists for enums
TOP_SPOT_EXCHANGES = ["binance", "bybit", "okx", "kucoin", "gate_io"]
TOP_PERP_EXCHANGES = [
    "binance_perpetual",
    "bybit_perpetual",
    "hyperliquid_perpetual",
    "okx_perpetual",
    "gate_io_perpetual",
]
ALL_EXCHANGES = TOP_SPOT_EXCHANGES + TOP_PERP_EXCHANGES

# ============================================================================
# HUMMINGBOT API CLIENT (Official SDK)
# ============================================================================

# Global client instance - initialized on startup
hb_client: HummingbotAPIClient | None = None


async def get_hb_client() -> HummingbotAPIClient:
    """Get the initialized Hummingbot API client."""
    global hb_client
    if hb_client is None:
        raise ToolError("Hummingbot client not initialized")
    return hb_client


# ============================================================================
# PYDANTIC MODELS (auto-generate outputSchema + structuredContent)
# ============================================================================


class PriceInfo(BaseModel):
    trading_pair: str
    price: float


class PricesResult(BaseModel):
    connector: str
    prices: list[PriceInfo]
    timestamp: str


class SpreadInfo(BaseModel):
    absolute: float
    percentage: float


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookResult(BaseModel):
    connector: str
    trading_pair: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    spread: SpreadInfo
    timestamp: str


class CandleData(BaseModel):
    timestamp: str | None
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesResult(BaseModel):
    connector: str
    trading_pair: str
    interval: str
    candles: list[CandleData]
    count: int


class FundingRateResult(BaseModel):
    connector: str
    trading_pair: str
    funding_rate: float
    funding_rate_pct: str
    annualized_rate_pct: str
    mark_price: float
    index_price: float
    next_funding_time: str
    timestamp: str


class TradeImpactResult(BaseModel):
    trading_pair: str
    side: str
    requested_amount: float
    vwap: float
    price_impact_pct: float
    total_quote_volume: float
    mid_price: float
    spread: SpreadInfo
    sufficient_liquidity: bool
    timestamp: str


class ConnectorsResult(BaseModel):
    spot_exchanges: list[str]
    perpetual_exchanges: list[str]
    dex_connectors: list[str]
    total_count: int


# ============================================================================
# CONTEXT PROTOCOL MIDDLEWARE (only runs on tools/call)
# ============================================================================


class ContextProtocolAuthMiddleware(Middleware):
    """
    Middleware that verifies Context Protocol JWT for tool calls.
    
    This only intercepts `on_call_tool` - initialize and tools/list
    remain open for discovery (no auth required).
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Verify Context Protocol payment JWT before executing tool."""
        headers = get_http_headers()
        auth_header = headers.get("authorization", "")

        try:
            await verify_context_request(authorization_header=auth_header)
        except ContextError as e:
            raise ToolError(f"Unauthorized: {e.message}")

        # Auth passed, execute the tool
        return await call_next(context)


# ============================================================================
# FASTMCP SERVER
# ============================================================================

mcp = FastMCP(
    name="hummingbot-market-intel",
    instructions="""Hummingbot Market Intelligence MCP Server.
    
Provides real-time market data from 40+ CEX and DEX connectors:
- Price data across exchanges
- Order book analysis with VWAP and slippage
- Funding rates for perpetual futures
- Trade impact analysis

All data is PUBLIC market data - no user accounts or trading.""",
)

# Add Context Protocol payment verification middleware
mcp.add_middleware(ContextProtocolAuthMiddleware())


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================


@mcp.tool(
    name="get_prices",
    description="""📊 Get real-time prices for trading pairs across exchanges.

Fetches current mid prices for one or more trading pairs from any supported exchange.

Example: Get BTC and ETH prices from Binance
- connector_name: "binance"
- trading_pairs: ["BTC-USDT", "ETH-USDT"]

Supported exchanges: binance, bybit, okx, kucoin, gate_io, hyperliquid_perpetual, and 40+ more.""",
)
async def get_prices(
    connector_name: Annotated[str, Field(description="Exchange connector name")],
    trading_pairs: Annotated[list[str], Field(description="Trading pairs (e.g., ['BTC-USDT', 'ETH-USDT'])")],
) -> PricesResult:
    """Get real-time prices for trading pairs."""
    client = await get_hb_client()

    result = await client.market_data.get_prices(
        connector_name=connector_name,
        trading_pairs=trading_pairs,
    )

    prices = []
    prices_data = result.get("prices", {}) if isinstance(result, dict) else {}
    for pair, price in prices_data.items():
        prices.append(PriceInfo(
            trading_pair=pair,
            price=float(price) if price else 0,
        ))

    return PricesResult(
        connector=connector_name,
        prices=prices,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@mcp.tool(
    name="get_order_book",
    description="""📊 Get order book snapshot for a trading pair.

Returns top bids and asks from the order book with price and quantity.

Example: Get BTC-USDT order book from Binance
- connector_name: "binance"
- trading_pair: "BTC-USDT"
- depth: 10

Supported exchanges: All CEX connectors.""",
)
async def get_order_book(
    connector_name: Annotated[str, Field(description="Exchange connector name")],
    trading_pair: Annotated[str, Field(description="Trading pair (e.g., 'BTC-USDT')")],
    depth: Annotated[int, Field(description="Number of levels to fetch", default=10)] = 10,
) -> OrderBookResult:
    """Get order book snapshot for a trading pair."""
    client = await get_hb_client()

    result = await client.market_data.get_order_book(
        connector_name=connector_name,
        trading_pair=trading_pair,
        depth=depth,
    )

    bids = result.get("bids", [])[:depth] if isinstance(result, dict) else []
    asks = result.get("asks", [])[:depth] if isinstance(result, dict) else []

    # Handle both array and dict formats
    if bids and isinstance(bids[0], dict):
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0)) if asks else 0
        bids_formatted = [OrderBookLevel(price=float(b.get("price", 0)), quantity=float(b.get("amount", b.get("quantity", 0)))) for b in bids]
        asks_formatted = [OrderBookLevel(price=float(a.get("price", 0)), quantity=float(a.get("amount", a.get("quantity", 0)))) for a in asks]
    else:
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        bids_formatted = [OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in bids]
        asks_formatted = [OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in asks]

    spread_abs = best_ask - best_bid if best_bid and best_ask else 0
    spread_pct = (spread_abs / best_bid * 100) if best_bid else 0

    return OrderBookResult(
        connector=connector_name,
        trading_pair=trading_pair,
        bids=bids_formatted,
        asks=asks_formatted,
        spread=SpreadInfo(absolute=spread_abs, percentage=round(spread_pct, 4)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@mcp.tool(
    name="get_candles",
    description="""📊 Get OHLCV candlestick data for technical analysis.

Returns historical candlestick data with open, high, low, close, volume.

Example: Get 1-hour BTC candles from Binance
- connector_name: "binance"
- trading_pair: "BTC-USDT"
- interval: "1h"
- limit: 100

Intervals: 1m, 5m, 15m, 1h, 4h, 1d""",
)
async def get_candles(
    connector_name: Annotated[str, Field(description="Exchange connector name")],
    trading_pair: Annotated[str, Field(description="Trading pair (e.g., 'BTC-USDT')")],
    interval: Annotated[str, Field(description="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d")],
    limit: Annotated[int, Field(description="Number of candles (default: 100, max: 500)", default=100)] = 100,
) -> CandlesResult:
    """Get OHLCV candlestick data."""
    client = await get_hb_client()
    limit = min(limit, 500)

    result = await client.market_data.get_candles(
        connector_name=connector_name,
        trading_pair=trading_pair,
        interval=interval,
        max_records=limit,
    )

    # Handle both list response and dict with candles key
    candles_data = result if isinstance(result, list) else result.get("candles", result) if isinstance(result, dict) else []

    candles = []
    for candle in candles_data:
        if isinstance(candle, dict):
            candles.append(CandleData(
                timestamp=candle.get("timestamp"),
                open=float(candle.get("open", 0)),
                high=float(candle.get("high", 0)),
                low=float(candle.get("low", 0)),
                close=float(candle.get("close", 0)),
                volume=float(candle.get("volume", 0)),
            ))

    return CandlesResult(
        connector=connector_name,
        trading_pair=trading_pair,
        interval=interval,
        candles=candles,
        count=len(candles),
    )


@mcp.tool(
    name="get_funding_rates",
    description="""📊 Get funding rate for perpetual futures.

Returns current funding rate, next funding time, and mark/index prices.

Example: Get BTC funding rate from Binance Perpetual
- connector_name: "binance_perpetual"
- trading_pair: "BTC-USDT"

Supported: binance_perpetual, bybit_perpetual, hyperliquid_perpetual, okx_perpetual""",
)
async def get_funding_rates(
    connector_name: Annotated[str, Field(description="Perpetual exchange connector")],
    trading_pair: Annotated[str, Field(description="Trading pair (e.g., 'BTC-USDT')")],
) -> FundingRateResult:
    """Get funding rate for perpetual futures."""
    client = await get_hb_client()

    result = await client.market_data.get_funding_info(
        connector_name=connector_name,
        trading_pair=trading_pair,
    )

    result_dict = result if isinstance(result, dict) else {}
    funding_rate = float(result_dict.get("funding_rate") or 0)
    annualized = funding_rate * 3 * 365 * 100  # 8h funding, 3x per day, annualized

    return FundingRateResult(
        connector=connector_name,
        trading_pair=trading_pair,
        funding_rate=funding_rate,
        funding_rate_pct=f"{funding_rate * 100:.4f}%",
        annualized_rate_pct=f"{annualized:.2f}%",
        mark_price=float(result_dict.get("mark_price") or 0),
        index_price=float(result_dict.get("index_price") or 0),
        next_funding_time=result_dict.get("next_funding_time", ""),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@mcp.tool(
    name="analyze_trade_impact",
    description="""🧠 Calculate exact price impact and VWAP for a trade.

Uses real order book data to compute:
- Exact execution price for your trade size
- VWAP (Volume Weighted Average Price)
- Price impact / slippage percentage
- Whether sufficient liquidity exists

Perfect for: Pre-trade analysis, optimal execution planning, large order sizing.""",
)
async def analyze_trade_impact(
    connector_name: Annotated[str, Field(description="Exchange connector")],
    trading_pair: Annotated[str, Field(description="Trading pair (e.g., 'BTC-USDT')")],
    side: Annotated[str, Field(description="Trade side - BUY walks the asks, SELL walks the bids")],
    amount: Annotated[float, Field(description="Trade amount in BASE token (e.g., 1.5 for 1.5 BTC)")],
) -> TradeImpactResult:
    """Calculate exact price impact and VWAP for a trade."""
    client = await get_hb_client()

    # Get order book
    ob_result = await client.market_data.get_order_book(
        connector_name=connector_name,
        trading_pair=trading_pair,
        depth=100,
    )

    ob_dict = ob_result if isinstance(ob_result, dict) else {}
    bids = ob_dict.get("bids", [])
    asks = ob_dict.get("asks", [])

    # Handle both array and dict formats
    def get_price_qty(level: Any) -> tuple[float, float]:
        if isinstance(level, dict):
            return float(level.get("price", 0)), float(level.get("amount", level.get("quantity", 0)))
        return float(level[0]), float(level[1])

    # Calculate mid price
    best_bid = get_price_qty(bids[0])[0] if bids else 0
    best_ask = get_price_qty(asks[0])[0] if asks else 0
    mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

    # Walk the book to calculate VWAP
    book = asks if side.upper() == "BUY" else bids
    remaining = amount
    total_quote = 0.0
    total_base = 0.0

    for level in book:
        price, qty = get_price_qty(level)
        fill_qty = min(remaining, qty)
        total_quote += fill_qty * price
        total_base += fill_qty
        remaining -= fill_qty
        if remaining <= 0:
            break

    sufficient_liquidity = remaining <= 0
    vwap = total_quote / total_base if total_base > 0 else 0
    price_impact = ((vwap - mid_price) / mid_price * 100) if mid_price else 0
    if side.upper() == "SELL":
        price_impact = -price_impact

    spread_abs = best_ask - best_bid if best_bid and best_ask else 0
    spread_pct = (spread_abs / best_bid * 100) if best_bid else 0

    return TradeImpactResult(
        trading_pair=trading_pair,
        side=side.upper(),
        requested_amount=amount,
        vwap=round(vwap, 8),
        price_impact_pct=round(abs(price_impact), 4),
        total_quote_volume=round(total_quote, 2),
        mid_price=round(mid_price, 8),
        spread=SpreadInfo(absolute=round(spread_abs, 8), percentage=round(spread_pct, 4)),
        sufficient_liquidity=sufficient_liquidity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@mcp.tool(
    name="get_connectors",
    description="""📋 List all supported exchange connectors.

Returns the full list of 40+ CEX and DEX connectors available in Hummingbot.

No arguments required.""",
)
async def get_connectors() -> ConnectorsResult:
    """List all supported exchange connectors."""
    client = await get_hb_client()

    try:
        connectors = await client.connectors.list_connectors()
        connector_list = connectors if isinstance(connectors, list) else []
    except Exception:
        connector_list = []

    # Categorize connectors
    spot = [c for c in connector_list if not c.endswith("_perpetual") and c not in ["jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex"]]
    perps = [c for c in connector_list if c.endswith("_perpetual")]
    dex = [c for c in connector_list if c in ["jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex"]]

    # Use hardcoded if API returned empty
    if not connector_list:
        return ConnectorsResult(
            spot_exchanges=[
                "binance", "bybit", "okx", "kucoin", "gate_io", "coinbase_advanced_trade",
                "kraken", "bitfinex", "mexc", "bitget", "htx", "crypto_com",
            ],
            perpetual_exchanges=[
                "binance_perpetual", "bybit_perpetual", "okx_perpetual", "gate_io_perpetual",
                "kucoin_perpetual", "hyperliquid_perpetual", "dydx_v4_perpetual",
            ],
            dex_connectors=[
                "jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex",
            ],
            total_count=25,
        )

    return ConnectorsResult(
        spot_exchanges=spot[:15],
        perpetual_exchanges=perps,
        dex_connectors=dex,
        total_count=len(connector_list),
    )


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "hummingbot-mcp-python-fastmcp",
        "version": "2.0.0",
        "framework": "FastMCP",
        "mcp_spec": "2025-06-18",
        "tools": [
            "get_prices",
            "get_order_book",
            "get_candles",
            "get_funding_rates",
            "analyze_trade_impact",
            "get_connectors",
        ],
        "tool_count": 6,
        "auth": "Context Protocol JWT on tools/call",
        "hummingbot_api": HUMMINGBOT_API_URL,
    })


# ============================================================================
# APPLICATION LIFESPAN (combines FastMCP + Hummingbot client)
# ============================================================================

# Create the MCP ASGI app first (needed for lifespan)
mcp_app = mcp.http_app(path="/mcp")


def compose_lifespans(*lifespans):
    """Compose multiple lifespans into a single lifespan context manager."""
    from contextlib import AsyncExitStack

    @asynccontextmanager
    async def composed_lifespan(app):
        async with AsyncExitStack() as stack:
            for lifespan in lifespans:
                await stack.enter_async_context(lifespan(app))
            yield

    return composed_lifespan


@asynccontextmanager
async def hummingbot_lifespan(app):
    """Manage Hummingbot client lifecycle."""
    global hb_client

    # Startup: Initialize Hummingbot API client
    print(f"🔌 Connecting to Hummingbot API at {HUMMINGBOT_API_URL}")
    hb_client = HummingbotAPIClient(
        base_url=HUMMINGBOT_API_URL,
        username=HB_USERNAME,
        password=HB_PASSWORD,
    )
    await hb_client.init()
    print("✅ Hummingbot API client connected")

    yield

    # Shutdown: Close client
    if hb_client:
        await hb_client.close()
        print("🔌 Hummingbot API client disconnected")


# ============================================================================
# STARLETTE APP (combines FastMCP + health check + composed lifespan)
# ============================================================================

# Compose both lifespans: FastMCP's session manager + Hummingbot client
combined_lifespan = compose_lifespans(mcp_app.lifespan, hummingbot_lifespan)

# Create Starlette app with health check and MCP mounted
app = Starlette(
    routes=[
        Route("/health", health_check),
        Mount("/", app=mcp_app),
    ],
    lifespan=combined_lifespan,
)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Starting Hummingbot MCP Server (FastMCP) on port {PORT}")
    print(f"📡 Hummingbot API: {HUMMINGBOT_API_URL}")
    print(f"🔧 Framework: FastMCP (MCP 2025-06-18 spec)")
    print(f"")
    print(f"📊 Tools (6 total):")
    print(f"   - get_prices: Multi-pair price fetching")
    print(f"   - get_order_book: Order book with spread analysis")
    print(f"   - get_candles: OHLCV candlestick data")
    print(f"   - get_funding_rates: Perpetual funding rates")
    print(f"   - analyze_trade_impact: VWAP and slippage analysis")
    print(f"   - get_connectors: List 40+ supported exchanges")
    print(f"")
    print(f"🔒 Auth: Context Protocol JWT on tools/call only")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
