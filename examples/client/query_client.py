"""Minimal Query-mode client example."""

from __future__ import annotations

import asyncio
import os

from ctxprotocol import ContextClient, ContextError


async def main() -> None:
    try:
        async with ContextClient(api_key=os.environ["CONTEXT_API_KEY"]) as client:
            # tools omitted/None => auto-discovery; tools=["id"] => manual mode;
            # tools=[] => direct synthesis (no tool execution).
            answer = await client.query.run("What are the top whale movements on Base?")
            print("Response:", answer.response)
            print("Total cost (USD):", answer.cost.total_cost_usd)
            print("Duration (ms):", answer.duration_ms)
            print("Tools used:", [tool.name for tool in answer.tools_used])

            # Manual selected-tools mode example:
            # await client.query.run("Analyze whale activity", tools=["tool-uuid"])
    except ContextError as error:
        if error.code == "no_wallet":
            print("Wallet setup is required before running paid queries.")
        elif error.code == "insufficient_allowance":
            print("Increase your spending allowance to run queries.")
        elif error.code == "payment_failed":
            print("Payment settlement failed. Try again shortly.")
        elif error.code == "query_failed":
            print("Query orchestration failed. Try a narrower prompt.")
        else:
            print(f"Context API error: {error}")


if __name__ == "__main__":
    asyncio.run(main())
