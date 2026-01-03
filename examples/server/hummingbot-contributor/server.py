"""
Hummingbot Market Intelligence MCP Server (Python)

A PUBLIC MARKET DATA MCP server powered by Hummingbot API.
Provides access to real-time market data, liquidity analysis, and DEX quotes.

SCOPE: Public market data only - NO user account data, NO trading operations

Features:
- Multi-exchange price data (40+ CEX/DEX connectors)
- Order book analysis with VWAP and slippage estimation
- Funding rate analysis for perpetuals
- DEX swap quotes (Jupiter, 0x, etc.)

Architecture:
- Runs on the SAME server as Hummingbot API (localhost:8000)
- Uses Official hummingbot-api-client by Fede @ Hummingbot
- Integrates ctxprotocol SDK for payment verification
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Official Hummingbot API Client (by Fede @ Hummingbot)
# https://github.com/hummingbot/hummingbot-api-client
from hummingbot_api_client import HummingbotAPIClient

from ctxprotocol import (
    verify_context_request,
    is_protected_mcp_method,
    ContextError,
)

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
        raise HTTPException(status_code=503, detail="Hummingbot client not initialized")
    return hb_client


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

TOOLS = [
    # =========================================================================
    # RAW DATA TOOLS - Direct market data access
    # =========================================================================
    {
        "name": "get_prices",
        "description": """📊 Get real-time prices for trading pairs across exchanges.

Fetches current mid prices for one or more trading pairs from any supported exchange.

Example: Get BTC and ETH prices from Binance
- connector_name: "binance"
- trading_pairs: ["BTC-USDT", "ETH-USDT"]

Supported exchanges: binance, bybit, okx, kucoin, gate_io, hyperliquid_perpetual, and 40+ more.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connector_name": {
                    "type": "string",
                    "enum": ALL_EXCHANGES,
                    "description": "Exchange connector name",
                },
                "trading_pairs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Trading pairs (e.g., ['BTC-USDT', 'ETH-USDT'])",
                },
            },
            "required": ["connector_name", "trading_pairs"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "connector": {"type": "string"},
                "prices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trading_pair": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
                "timestamp": {"type": "string"},
            },
        },
    },
    {
        "name": "get_order_book",
        "description": """📊 Get order book snapshot for a trading pair.

Returns top bids and asks from the order book with price and quantity.

Example: Get BTC-USDT order book from Binance
- connector_name: "binance"
- trading_pair: "BTC-USDT"
- depth: 10

Supported exchanges: All CEX connectors.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connector_name": {
                    "type": "string",
                    "enum": ALL_EXCHANGES,
                    "description": "Exchange connector name",
                },
                "trading_pair": {
                    "type": "string",
                    "description": "Trading pair (e.g., 'BTC-USDT')",
                },
                "depth": {
                    "type": "integer",
                    "description": "Number of levels to fetch (default: 10)",
                    "default": 10,
                },
            },
            "required": ["connector_name", "trading_pair"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "connector": {"type": "string"},
                "trading_pair": {"type": "string"},
                "bids": {"type": "array", "items": {"type": "object"}},
                "asks": {"type": "array", "items": {"type": "object"}},
                "spread": {"type": "object"},
                "timestamp": {"type": "string"},
            },
        },
    },
    {
        "name": "get_candles",
        "description": """📊 Get OHLCV candlestick data for technical analysis.

Returns historical candlestick data with open, high, low, close, volume.

Example: Get 1-hour BTC candles from Binance
- connector_name: "binance"
- trading_pair: "BTC-USDT"
- interval: "1h"
- limit: 100

Intervals: 1m, 5m, 15m, 1h, 4h, 1d""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connector_name": {
                    "type": "string",
                    "enum": ALL_EXCHANGES,
                    "description": "Exchange connector name",
                },
                "trading_pair": {
                    "type": "string",
                    "description": "Trading pair (e.g., 'BTC-USDT')",
                },
                "interval": {
                    "type": "string",
                    "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    "description": "Candle interval",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of candles (default: 100, max: 500)",
                    "default": 100,
                },
            },
            "required": ["connector_name", "trading_pair", "interval"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "connector": {"type": "string"},
                "trading_pair": {"type": "string"},
                "interval": {"type": "string"},
                "candles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string"},
                            "open": {"type": "number"},
                            "high": {"type": "number"},
                            "low": {"type": "number"},
                            "close": {"type": "number"},
                            "volume": {"type": "number"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_funding_rates",
        "description": """📊 Get funding rate for perpetual futures.

Returns current funding rate, next funding time, and mark/index prices.

Example: Get BTC funding rate from Binance Perpetual
- connector_name: "binance_perpetual"
- trading_pair: "BTC-USDT"

Supported: binance_perpetual, bybit_perpetual, hyperliquid_perpetual, okx_perpetual""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connector_name": {
                    "type": "string",
                    "enum": TOP_PERP_EXCHANGES,
                    "description": "Perpetual exchange connector",
                },
                "trading_pair": {
                    "type": "string",
                    "description": "Trading pair (e.g., 'BTC-USDT')",
                },
            },
            "required": ["connector_name", "trading_pair"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "connector": {"type": "string"},
                "trading_pair": {"type": "string"},
                "funding_rate": {"type": "number"},
                "funding_rate_pct": {"type": "string"},
                "annualized_rate_pct": {"type": "string"},
                "mark_price": {"type": "number"},
                "index_price": {"type": "number"},
                "next_funding_time": {"type": "string"},
                "timestamp": {"type": "string"},
            },
        },
    },
    # =========================================================================
    # INTELLIGENCE TOOLS - Analysis and computation
    # =========================================================================
    {
        "name": "analyze_trade_impact",
        "description": """🧠 Calculate exact price impact and VWAP for a trade.

Uses real order book data to compute:
- Exact execution price for your trade size
- VWAP (Volume Weighted Average Price)
- Price impact / slippage percentage
- Whether sufficient liquidity exists

Perfect for: Pre-trade analysis, optimal execution planning, large order sizing.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connector_name": {
                    "type": "string",
                    "enum": ALL_EXCHANGES,
                    "description": "Exchange connector",
                },
                "trading_pair": {
                    "type": "string",
                    "description": "Trading pair (e.g., 'BTC-USDT')",
                },
                "side": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Trade side - BUY walks the asks, SELL walks the bids",
                },
                "amount": {
                    "type": "number",
                    "description": "Trade amount in BASE token (e.g., 1.5 for 1.5 BTC)",
                },
            },
            "required": ["connector_name", "trading_pair", "side", "amount"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "trading_pair": {"type": "string"},
                "side": {"type": "string"},
                "requested_amount": {"type": "number"},
                "vwap": {"type": "number"},
                "price_impact_pct": {"type": "number"},
                "total_quote_volume": {"type": "number"},
                "mid_price": {"type": "number"},
                "spread": {"type": "object"},
                "sufficient_liquidity": {"type": "boolean"},
                "timestamp": {"type": "string"},
            },
        },
    },
    {
        "name": "get_connectors",
        "description": """📋 List all supported exchange connectors.

Returns the full list of 40+ CEX and DEX connectors available in Hummingbot.

No arguments required.""",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "spot_exchanges": {"type": "array", "items": {"type": "string"}},
                "perpetual_exchanges": {"type": "array", "items": {"type": "string"}},
                "dex_connectors": {"type": "array", "items": {"type": "string"}},
                "total_count": {"type": "integer"},
            },
        },
    },
]


# ============================================================================
# TOOL HANDLERS (using official hummingbot-api-client)
# ============================================================================


async def handle_get_prices(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_prices tool call using official Hummingbot API client."""
    client = await get_hb_client()
    connector_name = args["connector_name"]
    trading_pairs = args["trading_pairs"]

    # Use official client method (note: param is connector_name, not connector)
    result = await client.market_data.get_prices(
        connector_name=connector_name,
        trading_pairs=trading_pairs,
    )

    prices = []
    prices_data = result.get("prices", {}) if isinstance(result, dict) else {}
    for pair, price in prices_data.items():
        prices.append({
            "trading_pair": pair,
            "price": float(price) if price else 0,
        })

    return {
        "connector": connector_name,
        "prices": prices,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def handle_get_order_book(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_order_book tool call using official Hummingbot API client."""
    client = await get_hb_client()
    connector_name = args["connector_name"]
    trading_pair = args["trading_pair"]
    depth = args.get("depth", 10)

    # Use official client method (note: param is connector_name, not connector)
    result = await client.market_data.get_order_book(
        connector_name=connector_name,
        trading_pair=trading_pair,
        depth=depth,
    )

    bids = result.get("bids", [])[:depth] if isinstance(result, dict) else []
    asks = result.get("asks", [])[:depth] if isinstance(result, dict) else []

    # Calculate spread - handle both array and dict formats
    if bids and isinstance(bids[0], dict):
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0)) if asks else 0
        bids_formatted = bids
        asks_formatted = asks
    else:
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        bids_formatted = [{"price": float(b[0]), "quantity": float(b[1])} for b in bids]
        asks_formatted = [{"price": float(a[0]), "quantity": float(a[1])} for a in asks]

    spread_abs = best_ask - best_bid if best_bid and best_ask else 0
    spread_pct = (spread_abs / best_bid * 100) if best_bid else 0

    return {
        "connector": connector_name,
        "trading_pair": trading_pair,
        "bids": bids_formatted,
        "asks": asks_formatted,
        "spread": {
            "absolute": spread_abs,
            "percentage": round(spread_pct, 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def handle_get_candles(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_candles tool call using official Hummingbot API client."""
    client = await get_hb_client()
    connector_name = args["connector_name"]
    trading_pair = args["trading_pair"]
    interval = args["interval"]
    limit = min(args.get("limit", 100), 500)

    # Use official client method
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
            candles.append({
                "timestamp": candle.get("timestamp"),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0)),
            })

    return {
        "connector": connector_name,
        "trading_pair": trading_pair,
        "interval": interval,
        "candles": candles,
        "count": len(candles),
    }


async def handle_get_funding_rates(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_funding_rates tool call using official Hummingbot API client."""
    client = await get_hb_client()
    connector_name = args["connector_name"]
    trading_pair = args["trading_pair"]

    # Use official client method (note: param is connector_name, not connector)
    result = await client.market_data.get_funding_info(
        connector_name=connector_name,
        trading_pair=trading_pair,
    )

    result_dict = result if isinstance(result, dict) else {}
    funding_rate = float(result_dict.get("funding_rate") or 0)
    annualized = funding_rate * 3 * 365 * 100  # 8h funding, 3x per day, annualized

    return {
        "connector": connector_name,
        "trading_pair": trading_pair,
        "funding_rate": funding_rate,
        "funding_rate_pct": f"{funding_rate * 100:.4f}%",
        "annualized_rate_pct": f"{annualized:.2f}%",
        "mark_price": float(result_dict.get("mark_price") or 0),
        "index_price": float(result_dict.get("index_price") or 0),
        "next_funding_time": result_dict.get("next_funding_time", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def handle_analyze_trade_impact(args: dict[str, Any]) -> dict[str, Any]:
    """Handle analyze_trade_impact tool call using official Hummingbot API client."""
    client = await get_hb_client()
    connector_name = args["connector_name"]
    trading_pair = args["trading_pair"]
    side = args["side"]
    amount = float(args["amount"])

    # Use official client method to get order book (note: param is connector_name, not connector)
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
    if bids:
        best_bid, _ = get_price_qty(bids[0])
    else:
        best_bid = 0
    if asks:
        best_ask, _ = get_price_qty(asks[0])
    else:
        best_ask = 0
    mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

    # Walk the book to calculate VWAP
    book = asks if side == "BUY" else bids
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
    if side == "SELL":
        price_impact = -price_impact

    spread_abs = best_ask - best_bid if best_bid and best_ask else 0
    spread_pct = (spread_abs / best_bid * 100) if best_bid else 0

    return {
        "trading_pair": trading_pair,
        "side": side,
        "requested_amount": amount,
        "vwap": round(vwap, 8),
        "price_impact_pct": round(abs(price_impact), 4),
        "total_quote_volume": round(total_quote, 2),
        "mid_price": round(mid_price, 8),
        "spread": {
            "absolute": round(spread_abs, 8),
            "percentage": round(spread_pct, 4),
        },
        "sufficient_liquidity": sufficient_liquidity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def handle_get_connectors(args: dict[str, Any]) -> dict[str, Any]:
    """Handle get_connectors tool call using official Hummingbot API client."""
    client = await get_hb_client()
    
    # Use official client method
    try:
        connectors = await client.connectors.list_connectors()
        connector_list = connectors if isinstance(connectors, list) else []
    except Exception:
        # Fallback to hardcoded list if API fails
        connector_list = []

    # Categorize connectors
    spot = [c for c in connector_list if not c.endswith("_perpetual") and c not in ["jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex"]]
    perps = [c for c in connector_list if c.endswith("_perpetual")]
    dex = [c for c in connector_list if c in ["jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex"]]
    
    # Use hardcoded if API returned empty
    if not connector_list:
        return {
            "spot_exchanges": [
                "binance", "bybit", "okx", "kucoin", "gate_io", "coinbase_advanced_trade",
                "kraken", "bitfinex", "mexc", "bitget", "htx", "crypto_com",
            ],
            "perpetual_exchanges": [
                "binance_perpetual", "bybit_perpetual", "okx_perpetual", "gate_io_perpetual",
                "kucoin_perpetual", "hyperliquid_perpetual", "dydx_v4_perpetual",
            ],
            "dex_connectors": [
                "jupiter", "uniswap", "pancakeswap", "raydium", "meteora", "vertex",
            ],
            "total_count": 25,
        }

    return {
        "spot_exchanges": spot[:15],  # Limit for readability
        "perpetual_exchanges": perps,
        "dex_connectors": dex,
        "total_count": len(connector_list),
    }


# Tool handler dispatch
TOOL_HANDLERS = {
    "get_prices": handle_get_prices,
    "get_order_book": handle_get_order_book,
    "get_candles": handle_get_candles,
    "get_funding_rates": handle_get_funding_rates,
    "analyze_trade_impact": handle_analyze_trade_impact,
    "get_connectors": handle_get_connectors,
}


# ============================================================================
# FASTAPI APP WITH LIFESPAN (for client lifecycle)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(
    title="Hummingbot Market Intelligence MCP Server",
    description="Public market data MCP server powered by official hummingbot-api-client",
    version="1.1.0",
    lifespan=lifespan,
)


# ============================================================================
# CONTEXT PROTOCOL AUTHENTICATION
# ============================================================================


async def verify_context_auth(request: Request) -> dict[str, Any] | None:
    """Verify Context Protocol JWT for protected methods."""
    try:
        body = await request.json()
    except Exception:
        return None

    method = body.get("method", "")

    # Allow discovery methods without authentication
    if not method or not is_protected_mcp_method(method):
        return None

    # Protected method - require authentication
    authorization = request.headers.get("authorization")

    try:
        payload = await verify_context_request(
            authorization_header=authorization,
        )
        return payload
    except ContextError as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {e.message}")


# ============================================================================
# MCP ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hummingbot-mcp-python",
        "tools": [t["name"] for t in TOOLS],
        "hummingbot_api": HUMMINGBOT_API_URL,
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP JSON-RPC endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
        )

    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id")

    # Verify authentication for protected methods
    if is_protected_mcp_method(method):
        authorization = request.headers.get("authorization")
        try:
            await verify_context_request(authorization_header=authorization)
        except ContextError as e:
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": f"Unauthorized: {e.message}"},
                    "id": request_id,
                },
            )

    # Handle MCP methods
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "hummingbot-market-intel",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                },
            },
            "id": request_id,
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": TOOLS},
            "id": request_id,
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                "id": request_id,
            }

        try:
            result = await handler(arguments)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": str(result)}],
                    "structuredContent": result,
                },
                "id": request_id,
            }
        except HTTPException as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": e.detail},
                "id": request_id,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": request_id,
            }

    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": request_id,
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting Hummingbot MCP Server on port {PORT}")
    print(f"📡 Hummingbot API: {HUMMINGBOT_API_URL}")
    print(f"🔧 Tools: {[t['name'] for t in TOOLS]}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)

