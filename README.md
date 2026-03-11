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
- **💸 Dual-Surface Economics:** Use Query for pay-per-response intelligence or Execute for session-budgeted method calls.

## Who Is This SDK For?

| Role | What You Use |
|------|--------------|
| **AI Agent Developer** | `ctxprotocol` — Query curated answers or Execute with explicit method pricing + sessions |
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
2. **Set spending cap** — Approve USDC spending on the ContextRouter (one-time setup)
3. **Fund wallet** — Add USDC for tool execution fees
4. **Generate API key** — In Settings page

## Two Modes: Precision vs Intelligence

The SDK offers two payment models to serve different use cases:

| Mode | Method | Payment Model | Settlement Shape | Use Case |
|------|--------|---------------|------------------|----------|
| **Execute** | `client.tools.execute()` | Per execute call | Session accrual + deferred batch flush | Deterministic pipelines, raw outputs, explicit spend envelopes |
| **Query** | `client.query.run()` | Pay-per-response | Deferred post-response | Complex questions, multi-tool synthesis, curated intelligence |

**Execute mode** gives you raw data and full control with explicit method pricing and session budgets:
```python
session = await client.tools.start_session(max_spend_usd="2.00")
execute_tools = await client.discovery.search(
    "whale transactions",
    mode="execute",
    surface="execute",
    require_execute_pricing=True,
)

result = await client.tools.execute(
    tool_id=execute_tools[0].id,
    tool_name=execute_tools[0].mcp_tools[0].name,
    args={"chain": "base", "limit": 20},
    session_id=session.session.session_id,
)
print(result.session)  # method_price, spent, remaining, max_spend, ...
```

**Query mode** gives you curated answers — the server runs a discovery-first planner contract (`discover/probe -> plan-from-evidence -> execute -> bounded fallback`) with model-aware context budgeting and AI synthesis for one flat fee:
```python
answer = await client.query.run(
    query="What are the top whale movements on Base?",
    model_id="glm-model",      # optional: choose a supported model
    query_depth="deep",        # optional: fast | auto | deep
    include_data_url=True,     # optional: persist full execution data to blob
    include_developer_trace=True,  # optional: include runtime developer trace
)
print(answer.response)    # AI-synthesized answer
print(answer.tools_used)  # Which tools were used
print(answer.cost)        # Cost breakdown
print(answer.data_url)    # Optional blob URL with full data
print(answer.developer_trace.summary if answer.developer_trace else None)
print(
    answer.developer_trace.diagnostics.selection
    if answer.developer_trace and answer.developer_trace.diagnostics
    else None
)
print(answer.orchestration_metrics)  # Optional first-pass / rediscovery metrics
```

> Mixed listings are first-class: one listing can expose methods to both surfaces. Methods without `_meta.pricing.executeUsd` remain query-only until priced.
>
> Compatibility: SDK/API payload fields such as `price` and `price_per_query` are retained for backward compatibility. In Query mode, they represent listing-level **price per response turn**.
> A future major release can add response-named aliases (for example, `price_per_response`) before deprecating legacy names.

## Quick Start

```python
import asyncio
from ctxprotocol import ContextClient

async def main():
    async with ContextClient(api_key="sk_live_...") as client:
        # Pay-per-response: Ask a question, get a curated answer
        answer = await client.query.run("What are the top whale movements on Base?")
        print(answer.response)

        # Execute surface: require explicit execute pricing
        tools = await client.discovery.search(
            "gas prices",
            mode="execute",
            surface="execute",
            require_execute_pricing=True,
        )
        session = await client.tools.start_session(max_spend_usd="1.00")
        result = await client.tools.execute(
            tool_id=tools[0].id,
            tool_name=tools[0].mcp_tools[0].name,
            args={"chainId": 1},
            session_id=session.session.session_id,
        )
        print(result.result)

asyncio.run(main())
```

See a full dual-surface client script in [`examples/two-surfaces-client.py`](./examples/two-surfaces-client.py).

## Configuration

### Client Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | — | Your Context Protocol API key |
| `base_url` | `str` | No | `https://www.ctxprotocol.com` | API base URL (for development) |
| `request_timeout_seconds` | `float` | No | `300.0` | Timeout for non-streaming JSON API calls |
| `stream_timeout_seconds` | `float` | No | `600.0` | Timeout for streaming API calls; also used by `client.query.run()` |

```python
# Production
client = ContextClient(api_key=os.environ["CONTEXT_API_KEY"])

# Local development
client = ContextClient(
    api_key="sk_test_...",
    base_url="http://localhost:3000",
    request_timeout_seconds=420.0,
    stream_timeout_seconds=840.0,
)
```

## API Reference

### Discovery

#### `client.discovery.search(query, limit?)`

Search for tools with optional surface-aware filters.

```python
tools = await client.discovery.search("ethereum gas", limit=10)

execute_tools = await client.discovery.search(
    "ethereum gas",
    mode="execute",
    surface="execute",
    require_execute_pricing=True,
)
```

#### `client.discovery.get_featured(limit?, ...)`

Get featured/popular tools.

```python
featured = await client.discovery.get_featured(limit=5)
featured_execute = await client.discovery.get_featured(
    limit=5,
    mode="execute",
    require_execute_pricing=True,
)
```

### Tools (Execute Surface)

Session lifecycle helpers use the canonical execute-scoped API contract:
`/api/v1/tools/execute/sessions...`

#### `client.tools.execute(tool_id, tool_name, args?)`

Execute a single tool method. Execute calls can run inside session budgets.

```python
session = await client.tools.start_session(max_spend_usd="2.50")

result = await client.tools.execute(
    tool_id="uuid-of-tool",
    tool_name="get_gas_prices",
    args={"chainId": 1},
    session_id=session.session.session_id,
)
print(result.method.execute_price_usd)
print(result.session)
```

#### `client.tools.start_session(max_spend_usd)`

```python
started = await client.tools.start_session(max_spend_usd="5.00")
```

#### `client.tools.get_session(session_id)`

```python
status = await client.tools.get_session("sess_123")
```

#### `client.tools.close_session(session_id)`

```python
closed = await client.tools.close_session("sess_123")
```

### Query (Pay-Per-Response)

#### `client.query.run(query, tools?, model_id?, include_data?, include_data_url?, include_developer_trace?, query_depth?, debug_scout_deep_mode?, idempotency_key?)`

Run an agentic query. The server applies discovery-first orchestration (`discover/probe -> plan-from-evidence -> execute -> bounded fallback`) with up to 100 MCP calls per response turn, then returns an AI-synthesized answer.

`client.query.run()` buffers the same SSE transport used by `client.query.stream()` and returns the final `done` result. This keeps Python aligned with the TypeScript SDK and the live query runtime.

`query_depth` controls orchestration depth:
- `fast`: lower-latency path for simple lookups.
- `auto`: server routes to either `fast` or `deep` from query intent + selected tool complexity.
- `deep`: completeness-oriented path (default when omitted).

`include_developer_trace` and `orchestration_metrics` are optional diagnostics.
`debug_scout_deep_mode` remains test-only and should not be used in production flows.

```python
# Simple string
answer = await client.query.run("What are the top whale movements on Base?")

# With specific tools
answer = await client.query.run(
    query="Analyze whale activity on Base",
    tools=["tool-uuid-1", "tool-uuid-2"],  # optional — auto-discover if omitted
    model_id="kimi-model-thinking",          # optional
    query_depth="auto",                      # optional: fast | auto | deep
    include_data=True,                       # optional: include execution data inline
    include_data_url=True,                   # optional: include blob URL for full data
    include_developer_trace=True,            # optional: include Developer Mode trace
)

print(answer.response)      # AI-synthesized text
print(answer.tools_used)    # [QueryToolUsage(id, name, skill_calls)]
print(answer.cost)          # QueryCost(model_cost_usd, tool_cost_usd, total_cost_usd)
print(answer.duration_ms)   # Total time
print(answer.data)          # Optional execution data (when include_data=True)
print(answer.data_url)      # Optional blob URL (when include_data_url=True)
print(answer.developer_trace.summary if answer.developer_trace else None)
print(
    answer.developer_trace.diagnostics.selection
    if answer.developer_trace and answer.developer_trace.diagnostics
    else None
)
print(answer.orchestration_metrics)  # Optional first-pass / rediscovery metrics
```

When retrieval-first synthesis rollout is enabled server-side, full-data or truncation-sensitive query requests can switch to retrieval-first context assembly using private stage artifacts and canonical execution data slices. `include_data` and `include_data_url` continue to reference the same canonical dataset used for synthesis.

#### `client.query.stream(query, tools?, model_id?, include_data?, include_data_url?, include_developer_trace?, query_depth?, debug_scout_deep_mode?, idempotency_key?)`

Same as `run()` but streams events in real-time via SSE.

Event types:
- `tool-status`
- `text-delta`
- `developer-trace` (when `include_developer_trace=True`)
- `error`
- `done`

```python
async for event in client.query.stream(
    query="What are the top whale movements?",
    query_depth="fast",
):
    if event.type == "tool-status":
        print(f"Tool {event.tool.name}: {event.status}")
    elif event.type == "text-delta":
        print(event.delta, end="")
    elif event.type == "error":
        print(f"\nStream error: {event.error}")
    elif event.type == "done":
        print(f"\nTotal cost: {event.result.cost.total_cost_usd}")
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
            # User needs to set a spending cap
            print(f"Set spending cap: {e.help_url}")
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
| `insufficient_allowance` | Spending cap not set | Direct user to `help_url` |
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
        "rateLimit": {
            "maxRequestsPerMinute": 30,
            "cooldownMs": 2000,
            "maxConcurrency": 1,
            "supportsBulk": True,
            "recommendedBatchTools": ["get_portfolio_snapshot"],
            "notes": "Hobby tier: use snapshot endpoints before fan-out loops.",
        },
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
