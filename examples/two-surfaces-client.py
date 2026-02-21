"""Example client using both Query and Execute surfaces."""

from __future__ import annotations

import asyncio
import os

from ctxprotocol import ContextClient


async def main() -> None:
    api_key = os.environ.get("CONTEXT_API_KEY", "sk_live_your_api_key_here")

    async with ContextClient(api_key=api_key) as client:
        # Query surface: curated, pay-per-response.
        answer = await client.query.run("What are the top whale movements on Base?")
        print("Query response:")
        print(answer.response)

        # Execute surface: explicit method pricing + session budget envelope.
        execute_tools = await client.discovery.search(
            "gas prices",
            mode="execute",
            surface="execute",
            require_execute_pricing=True,
        )

        if not execute_tools or not execute_tools[0].mcp_tools:
            print("No execute-eligible methods found.")
            return

        method = execute_tools[0].mcp_tools[0]
        session = await client.tools.start_session(max_spend_usd="1.00")
        session_id = session.session.session_id
        if not session_id:
            raise RuntimeError("Expected non-null execute session_id")

        result = await client.tools.execute(
            tool_id=execute_tools[0].id,
            tool_name=method.name,
            args={"chainId": 1},
            session_id=session_id,
            close_session=True,
        )

        print("\nExecute result:")
        print(result.result)
        print("\nExecute spend envelope:")
        print(result.session.model_dump())

        # Optional: confirm closed lifecycle state.
        closed = await client.tools.get_session(session_id)
        print("\nSession lifecycle:")
        print(closed.session.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
