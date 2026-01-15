# ctxprotocol

**The Universal Adapter for AI Agents.**

Connect your AI to the real world without managing API keys, hosting servers, or reading documentation.

Context Protocol is **pip for AI capabilities**. Just as you install packages to add functionality to your code, use the Context SDK to give your Agent instant access to thousands of live data sources and actions—from DeFi and Gas Oracles to Weather and Search.

[![PyPI version](https://img.shields.io/pypi/v/ctxprotocol.svg)](https://pypi.org/project/ctxprotocol/)
[![Python versions](https://img.shields.io/pypi/pyversions/ctxprotocol.svg)](https://pypi.org/project/ctxprotocol/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 💰 $10,000 Developer Grant Program

We're funding the initial supply of MCP Tools for the Context Marketplace. **Become a Data Broker.**

- **🛠️ Build:** Create an MCP Server using this SDK (Solana data, Trading tools, Scrapers, etc.)
- **📦 List:** Publish it to the Context Registry
- **💵 Earn:** Get a **$250–$1,000 Grant** + earn USDC every time an agent queries your tool

👉 [**View Open Bounties & Apply Here**](https://docs.ctxprotocol.com/grants)

---

## Why use Context?

- **🔌 One Interface, Everything:** Stop integrating APIs one by one. Use a single SDK to access any tool in the marketplace.
- **🧠 Zero-Ops:** We're a gateway to the best MCP tools. Just send the JSON and get the result.
- **⚡️ Agentic Discovery:** Your Agent can search the marketplace at runtime to find tools it didn't know it needed.
- **💸 Pay-Per-Response:** The $500/year subscription? Now $0.01/response. No monthly fees, just results.

## Who Is This SDK For?

| Role | What You Use |
|------|--------------|
| **AI Agent Developer** | `ctxprotocol` — Query marketplace, execute tools, handle payments |
| **Tool Contributor (Data Broker)** | `mcp` + `ctxprotocol` — Standard MCP server + security middleware |

## Installation

```bash
pip install ctxprotocol
```

Or with optional FastAPI support:

```bash
pip install ctxprotocol[fastapi]
```

## Prerequisites

Before using the API, complete setup at [ctxprotocol.com](https://ctxprotocol.com):

1. **Sign in** — Creates your embedded wallet
2. **Enable Auto Pay** — Approve USDC spending for tool payments
3. **Fund wallet** — Add USDC for tool execution fees
4. **Generate API key** — In Settings page

## Quick Start

```python
import asyncio
from ctxprotocol import ContextClient

async def main():
    async with ContextClient(api_key="sk_live_...") as client:
        # Discover tools
        tools = await client.discovery.search("gas prices")
        
        # Execute a tool
        result = await client.tools.execute(
            tool_id=tools[0].id,
            tool_name=tools[0].mcp_tools[0].name,
            args={"chainId": 1},
        )
        
        print(result.result)

asyncio.run(main())
```

## Configuration

### Client Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | — | Your Context Protocol API key |
| `base_url` | `str` | No | `https://ctxprotocol.com` | API base URL (for development) |

```python
# Production
client = ContextClient(api_key=os.environ["CONTEXT_API_KEY"])

# Local development
client = ContextClient(
    api_key="sk_test_...",
    base_url="http://localhost:3000",
)
```

## API Reference

### Discovery

#### `client.discovery.search(query, limit?)`

Search for tools matching a query string.

```python
tools = await client.discovery.search("ethereum gas", limit=10)
```

#### `client.discovery.get_featured(limit?)`

Get featured/popular tools.

```python
featured = await client.discovery.get_featured(limit=5)
```

### Tools

#### `client.tools.execute(tool_id, tool_name, args?)`

Execute a tool method.

```python
result = await client.tools.execute(
    tool_id="uuid-of-tool",
    tool_name="get_gas_prices",
    args={"chainId": 1},
)
```

## Types

```python
from ctxprotocol import (
    # Auth utilities for tool contributors
    verify_context_request,
    is_protected_mcp_method,
    is_open_mcp_method,
    
    # Client types
    ContextClientOptions,
    Tool,
    McpTool,
    ExecuteOptions,
    ExecutionResult,
    ContextErrorCode,
    
    # Auth types (for MCP server contributors)
    VerifyRequestOptions,
    
    # Context types (for MCP server contributors receiving injected data)
    ContextRequirementType,
    HyperliquidContext,
    PolymarketContext,
    WalletContext,
    UserContext,
)
```

## Error Handling

The SDK raises `ContextError` with specific error codes:

```python
from ctxprotocol import ContextClient, ContextError

try:
    result = await client.tools.execute(...)
except ContextError as e:
    match e.code:
        case "no_wallet":
            # User needs to set up wallet
            print(f"Setup required: {e.help_url}")
        case "insufficient_allowance":
            # User needs to enable Auto Pay
            print(f"Enable Auto Pay: {e.help_url}")
        case "payment_failed":
            # Insufficient USDC balance
            pass
        case "execution_failed":
            # Tool execution error
            pass
```

### Error Codes

| Code | Description | Handling |
|------|-------------|----------|
| `unauthorized` | Invalid API key | Check configuration |
| `no_wallet` | Wallet not set up | Direct user to `help_url` |
| `insufficient_allowance` | Auto Pay not enabled | Direct user to `help_url` |
| `payment_failed` | USDC payment failed | Check balance |
| `execution_failed` | Tool error | Retry with different args |

## 🔒 Securing Your Tool (MCP Contributors)

If you're building an MCP server (tool contributor), verify incoming requests:

### Quick Implementation with FastAPI

```python
from fastapi import FastAPI, Request, Depends, HTTPException
from ctxprotocol import create_context_middleware, ContextError

app = FastAPI()
verify_context = create_context_middleware(audience="https://your-tool.com/mcp")

@app.post("/mcp")
async def handle_mcp(request: Request, context: dict = Depends(verify_context)):
    # context contains verified JWT payload (on protected methods)
    # None for open methods like tools/list
    body = await request.json()
    # Handle MCP request...
```

### Manual Verification

```python
from ctxprotocol import verify_context_request, is_protected_mcp_method, ContextError

# Check if a method requires auth
if is_protected_mcp_method(body["method"]):
    try:
        payload = await verify_context_request(
            authorization_header=request.headers.get("authorization"),
            audience="https://your-tool.com/mcp",  # optional
        )
        # payload contains verified JWT claims
    except ContextError as e:
        # Handle authentication error
        raise HTTPException(status_code=401, detail="Unauthorized")
```

### MCP Security Model

The SDK implements a **selective authentication** model — discovery is open, execution is protected:

| MCP Method | Auth Required | Why |
|------------|---------------|-----|
| `initialize` | ❌ No | Session setup |
| `tools/list` | ❌ No | Discovery - agents need to see your schemas |
| `resources/list` | ❌ No | Discovery |
| `prompts/list` | ❌ No | Discovery |
| `tools/call` | ✅ **Yes** | **Execution - costs money, runs your code** |

**What this means in practice:**
- ✅ `https://your-mcp.com/mcp` + `initialize` → Works without auth
- ✅ `https://your-mcp.com/mcp` + `tools/list` → Works without auth  
- ❌ `https://your-mcp.com/mcp` + `tools/call` → **Requires Context Protocol JWT**

This matches standard API patterns (OpenAPI schemas are public, GraphQL introspection is open).

## Execution Timeout & Product Design

⚠️ **Important**: MCP tool execution has a **~60 second timeout** (enforced at the platform/client level, not by MCP itself). This is intentional—it encourages building pre-computed insight products rather than raw data access.

**Best practice**: Run heavy queries offline (via cron jobs), store results in your database, and serve instant results via MCP. This is how Bloomberg, Nansen, and Arkham work.

```python
# ❌ BAD: Raw access (timeout-prone, no moat)
{"name": "run_sql", "description": "Run any SQL against blockchain data"}

# ✅ GOOD: Pre-computed product (instant, defensible)
{"name": "get_smart_money_wallets", "description": "Top 100 wallets that timed market tops"}
```

See the [full documentation](https://docs.ctxprotocol.com/guides/build-tools#execution-limits--product-design) for detailed guidance.

## Context Injection (Personalized Tools)

For tools that analyze user data, Context automatically injects user context:

```python
from ctxprotocol import CONTEXT_REQUIREMENTS_KEY, HyperliquidContext

# Define tool with context requirements
TOOLS = [{
    "name": "analyze_my_positions",
    "description": "Analyze your positions with personalized insights",
    "_meta": {
        "contextRequirements": ["hyperliquid"],
    },
    "inputSchema": {
        "type": "object",
        "properties": {
            "portfolio": {
                "type": "object",
                "description": "Portfolio context (injected by platform)",
            },
        },
        "required": ["portfolio"],
    },
}]

# Your handler receives typed context
async def handle_analyze_positions(portfolio: HyperliquidContext):
    positions = portfolio.perp_positions
    account = portfolio.account_summary
    # ... analyze and return insights
```

## Links

- [Context Protocol](https://ctxprotocol.com) — Main website
- [Documentation](https://docs.ctxprotocol.com)
- [GitHub](https://github.com/ctxprotocol/sdk-python) — This SDK
- [TypeScript SDK](https://github.com/ctxprotocol/sdk) — For Node.js
- [PyPI Package](https://pypi.org/project/ctxprotocol/)

## Requirements

- Python 3.10+
- httpx
- pydantic
- pyjwt[crypto]

## License

MIT
