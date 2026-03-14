"""Live verification for discovery.get() against the Context API."""

from __future__ import annotations

import asyncio
import os

from ctxprotocol import ContextClient


async def main() -> None:
    client_kwargs = {"api_key": os.environ["CONTEXT_API_KEY"]}
    base_url = os.environ.get("CONTEXT_BASE_URL")
    explicit_tool_id = os.environ.get("CONTEXT_TOOL_ID")
    if base_url:
        client_kwargs["base_url"] = base_url

    async with ContextClient(**client_kwargs) as client:
        if explicit_tool_id:
            source_tool_id = explicit_tool_id
            print(f"Selected tool for verification: {source_tool_id} (explicit tool ID)")
            source_tool_name = None
            source_tool_mcp_count = None
        else:
            search_results = await client.discovery.search("kalshi", limit=5)
            featured_results = await client.discovery.get_featured(limit=1)
            source_tool = search_results[0] if search_results else (
                featured_results[0] if featured_results else None
            )

            if source_tool is None:
                raise RuntimeError("Unable to find a marketplace tool for verification.")

            source_tool_id = source_tool.id
            source_tool_name = source_tool.name
            source_tool_mcp_count = len(source_tool.mcp_tools or [])
            print(f"Selected tool for verification: {source_tool_id} {source_tool_name}")

        tool = await client.discovery.get(source_tool_id)

        print(f"Fetched by ID: {tool.id} {tool.name}")
        print(f"Description length: {len(tool.description)}")
        print(f"MCP tool count: {len(tool.mcp_tools or [])}")

        assert tool.id == source_tool_id
        if source_tool_name is not None:
            assert tool.name == source_tool_name
        assert tool.description.strip()
        if source_tool_mcp_count is not None:
            assert len(tool.mcp_tools or []) == source_tool_mcp_count

        print("PASS: discovery.get() returned the expected tool payload.")


if __name__ == "__main__":
    asyncio.run(main())
