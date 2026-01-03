# Hummingbot Market Intelligence MCP Server (Python)

A **public market data** MCP server powered by the Hummingbot API. This is a Python implementation using FastAPI and the `ctxprotocol` SDK for payment verification.

## Scope

✅ **Public Market Data Only**
- Price data, order books, candles
- Liquidity analysis, trade impact estimation
- Funding rates for perpetuals

❌ **Excluded (User-Specific Data)**
- Portfolio balances
- Trading positions/orders
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
| `POST /mcp` | MCP JSON-RPC endpoint |

## Example Usage

### Get Prices

```bash
curl -X POST http://localhost:4010/mcp \
  -H "Content-Type: application/json" \
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

### Get Funding Rates

```bash
curl -X POST http://localhost:4010/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_funding_rates",
      "arguments": {
        "connector_name": "binance_perpetual",
        "trading_pair": "BTC-USDT"
      }
    },
    "id": 1
  }'
```

## Context Protocol Integration

This server uses `ctxprotocol` for payment verification:

```python
from ctxprotocol import verify_context_request, is_protected_mcp_method

# In your endpoint handler:
if is_protected_mcp_method(method):
    payload = await verify_context_request(
        authorization_header=request.headers.get("authorization"),
    )
```

### Security Model

| MCP Method | Auth Required | Reason |
|------------|---------------|--------|
| `tools/list` | ❌ No | Discovery - returns tool schemas |
| `tools/call` | ✅ Yes | Execution - runs code, costs money |
| `initialize` | ❌ No | Session setup |

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
│  │  • Market Data      │    │  • FastAPI                  │ │
│  │  • Order Books      │    │  • ctxprotocol auth         │ │
│  │  • Gateway (DEX)    │    │  • MCP Protocol             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Comparison: TypeScript vs Python Implementation

| Aspect | TypeScript | Python |
|--------|------------|--------|
| Framework | Express + MCP SDK | FastAPI |
| Auth SDK | `@ctxprotocol/sdk` | `ctxprotocol` |
| Port | 4009 | 4010 |
| Same functionality | ✅ | ✅ |

## License

MIT

