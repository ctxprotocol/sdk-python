# Hummingbot Market Intelligence MCP Server (Python)

A **public market data** MCP server powered by the Hummingbot API. Built with **FastMCP** (MCP 2025-06-18 spec) and the `ctxprotocol` SDK for payment verification.

## Scope

✅ **Public Market Data**
- Price data, order books, candles
- Liquidity analysis, trade impact estimation
- Funding rates for perpetuals

✅ **Gateway Integration**
- Real blockchain wallet balances via Hummingbot Gateway
- Portfolio tracking across CEX and DEX positions

❌ **Excluded (User-Specific Operations)**
- Trading execution
- Account management
- Bot orchestration

## Tools Overview

| Tool | Description |
|------|-------------|
| `get_prices` | Batch price lookup for multiple pairs |
| `get_order_book` | Order book snapshot with spread |
| `get_candles` | OHLCV candlestick data |
| `get_funding_rates` | Perpetual funding rate data |
| `analyze_trade_impact` | VWAP and price impact calculation |
| `get_connectors` | List all supported exchanges |
| `get_gateway_portfolio` | **Real** blockchain wallet balances from Gateway |

## Supported Exchanges

**CEX (Spot):** Binance, Bybit, OKX, KuCoin, Gate.io, Coinbase, Kraken, and more

**CEX (Perpetuals):** Binance Perpetual, Bybit Perpetual, Hyperliquid, OKX Perpetual, dYdX v4

**DEX:** Jupiter (Solana), Uniswap, PancakeSwap, Raydium, Meteora

## Setup

### 1. Install Dependencies

```bash
cd examples/server/hummingbot-contributor
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp env.example .env
```

Edit `.env`:
```bash
# Hummingbot API connection
HUMMINGBOT_API_URL=http://localhost:8000
HB_USERNAME=admin
HB_PASSWORD=admin

# Server port
PORT=4010
```

### 3. Run the Server

```bash
python server.py
```

Or with uvicorn directly:
```bash
uvicorn server:app --host 0.0.0.0 --port 4010 --reload
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with tool list |
| `POST /mcp` | MCP JSON-RPC endpoint (streamable HTTP) |
| `GET /mcp` | MCP session polling (streamable HTTP) |

## Example Usage

### Get Prices

```bash
curl -X POST http://localhost:4010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_prices",
      "arguments": {
        "connector_name": "binance",
        "trading_pairs": ["BTC-USDT", "ETH-USDT"]
      }
    },
    "id": 1
  }'
```

### Analyze Trade Impact

```bash
curl -X POST http://localhost:4010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "analyze_trade_impact",
      "arguments": {
        "connector_name": "binance",
        "trading_pair": "BTC-USDT",
        "side": "BUY",
        "amount": 1.0
      }
    },
    "id": 1
  }'
```

### Get Gateway Portfolio (Real Wallet Balances)

```bash
curl -X POST http://localhost:4010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_gateway_portfolio",
      "arguments": {
        "refresh": true
      }
    },
    "id": 1
  }'
```

## Context Protocol Integration

This server uses `ctxprotocol` for payment verification via FastMCP middleware:

```python
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from ctxprotocol import verify_context_request, ContextError

class ContextProtocolAuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        auth_header = headers.get("authorization", "")
        
        try:
            await verify_context_request(authorization_header=auth_header)
        except ContextError as e:
            raise ToolError(f"Unauthorized: {e.message}")
        
        return await call_next(context)

mcp = FastMCP(name="hummingbot-market-intel")
mcp.add_middleware(ContextProtocolAuthMiddleware())
```

### Security Model

| MCP Method | Auth Required | Reason |
|------------|---------------|--------|
| `initialize` | ❌ No | Session setup |
| `tools/list` | ❌ No | Discovery - returns tool schemas |
| `tools/call` | ✅ Yes | Execution - runs code, costs money |

## Deployment

### Deploy to Server

```bash
./deploy-hummingbot.sh
```

### On the Server

```bash
cd ~/hummingbot-mcp-python
./setup-server.sh      # Start with systemd
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Server                                   │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  Hummingbot API     │    │  Market Intel MCP (Python)  │ │
│  │  (localhost:8000)   │◄───│  (localhost:4010)           │ │
│  │                     │    │                             │ │
│  │  • Market Data      │    │  • FastMCP framework        │ │
│  │  • Order Books      │    │  • ctxprotocol auth         │ │
│  │  • Gateway (DEX)    │    │  • MCP 2025-06-18 spec      │ │
│  │  • Portfolio State  │    │  • Streamable HTTP          │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### FastMCP Benefits
- **Auto-generated schemas**: Pydantic models → `inputSchema` + `outputSchema`
- **Structured content**: Automatic `structuredContent` in responses
- **MCP 2025-06-18 compliant**: Streamable HTTP transport
- **Middleware support**: Custom auth via `on_call_tool` hook

### Gateway Integration
- Reads **real** blockchain wallet balances
- Uses `client.portfolio.get_state(skip_gateway=False)`
- Tracks holdings across CEX and DEX positions

## Comparison: TypeScript vs Python Implementation

| Aspect | TypeScript | Python |
|--------|------------|--------|
| Framework | Express + MCP SDK | **FastMCP** |
| Auth SDK | `@ctxprotocol/sdk` | `ctxprotocol` |
| Port | 4009 | 4010 |
| API Client | Raw fetch | `hummingbot-api-client` |
| Schema Gen | Manual | Auto (Pydantic) |
| Same functionality | ✅ | ✅ |

## Dependencies

Key packages:
- `fastmcp>=2.14.2` - MCP server framework
- `ctxprotocol>=0.5.6` - Context Protocol SDK
- `hummingbot-api-client>=1.2.6` - Official Hummingbot API client
- `pydantic>=2.0` - Data validation and schema generation

## License

MIT
