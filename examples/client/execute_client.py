"""Execute-mode client example with session spend tracking."""

from __future__ import annotations

import asyncio
import os

from ctxprotocol import ContextClient, ContextError


async def main() -> None:
    try:
        async with ContextClient(api_key=os.environ["CONTEXT_API_KEY"]) as client:
            tools = await client.discovery.search(
                "crypto prices",
                mode="execute",
                surface="execute",
                require_execute_pricing=True,
            )
            if not tools or not tools[0].mcp_tools:
                print("No execute-eligible methods found.")
                return

            tool = tools[0]
            method = tool.mcp_tools[0]

            session = await client.tools.start_session(max_spend_usd="2.00")
            session_id = session.session.session_id
            if not session_id:
                raise RuntimeError("Expected non-null execute session_id")

            for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
                result = await client.tools.execute(
                    tool_id=tool.id,
                    tool_name=method.name,
                    args={"symbol": symbol},
                    session_id=session_id,
                )
                print(f"{symbol}: {result.result}")
                print(
                    "price=",
                    result.method.execute_price_usd,
                    "spent=",
                    result.session.spent,
                    "remaining=",
                    result.session.remaining,
                )

            closed = await client.tools.close_session(session_id)
            print("Closed session:", closed.session.model_dump())

            final_session = await client.tools.get_session(session_id)
            print("Final session:", final_session.session.model_dump())
    except ContextError as error:
        if error.code == "method_not_execute_eligible":
            print("Selected method is not execute-eligible.")
        elif error.code == "session_budget_exceeded":
            print("Session budget exceeded. Start a new session with a higher limit.")
        elif error.code == "session_closed":
            print("Session is already closed. Start a new session.")
        elif error.code == "session_expired":
            print("Session expired. Start a new session.")
        else:
            print(f"Context API error: {error}")


if __name__ == "__main__":
    asyncio.run(main())
