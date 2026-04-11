"""Live integration check for favorites-only discovery and query."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ctxprotocol import ContextClient, ContextError


def read_env_value(file_path: Path, key: str) -> str | None:
    if not file_path.exists():
        return None

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        current_key, raw_value = line.split("=", 1)
        if current_key.strip() != key:
            continue

        return raw_value.strip().strip("'\"")

    return None


def get_context_api_key() -> str:
    env_key = os.environ.get("CONTEXT_API_KEY", "").strip()
    if env_key:
        return env_key

    env_path = Path(__file__).resolve().parents[3] / "context-sdk" / ".env.local"
    file_key = read_env_value(env_path, "CONTEXT_API_KEY")
    if file_key:
        return file_key

    raise RuntimeError(
        "Set CONTEXT_API_KEY or provide it in context-sdk/.env.local."
    )


def get_context_base_url() -> str | None:
    base_url = os.environ.get("CONTEXT_BASE_URL", "").strip()
    return base_url or None


def summarize_tools(tools: list[object]) -> list[str]:
    summary: list[str] = []
    for tool in tools[:5]:
        tool_id = getattr(tool, "id", "unknown-id")
        tool_name = getattr(tool, "name", "Unnamed tool")
        summary.append(f"{tool_name} ({tool_id})")
    return summary


async def main() -> None:
    query = "crypto"
    base_url = get_context_base_url()

    try:
        print("Base URL:", base_url or "https://www.ctxprotocol.com")
        async with ContextClient(
            api_key=get_context_api_key(),
            **({"base_url": base_url} if base_url else {}),
        ) as client:
            default_results, unrestricted_results, favorites_results = (
                await asyncio.gather(
                    client.discovery.search(
                        query,
                        limit=10,
                        mode="query",
                        surface="answer",
                    ),
                    client.discovery.search(
                        query,
                        limit=10,
                        mode="query",
                        surface="answer",
                        favorites_only=False,
                    ),
                    client.discovery.search(
                        query,
                        limit=10,
                        mode="query",
                        surface="answer",
                        favorites_only=True,
                    ),
                )
            )

            print("Default discovery count:", len(default_results))
            print("Explicit unrestricted count:", len(unrestricted_results))
            print("Favorites-only count:", len(favorites_results))
            print("Default discovery sample:", summarize_tools(default_results))
            print(
                "Explicit unrestricted sample:",
                summarize_tools(unrestricted_results),
            )
            print("Favorites-only sample:", summarize_tools(favorites_results))

            if len(favorites_results) > len(unrestricted_results):
                raise RuntimeError(
                    "favorites_only=True returned more tools than explicit unrestricted discovery."
                )

            unrestricted_ids = {tool.id for tool in unrestricted_results}
            outside_unrestricted = [
                tool for tool in favorites_results if tool.id not in unrestricted_ids
            ]
            if outside_unrestricted:
                print(
                    "Favorites-only returned tools outside the unrestricted top 10 window:",
                    summarize_tools(outside_unrestricted),
                )

            query_result = await client.query.run(
                query="What are the top whale movements on Base?",
                favorites_only=True,
                include_developer_trace=True,
            )

            if not query_result.response.strip():
                raise RuntimeError("Query response was empty.")

            print("Query response length:", len(query_result.response))
            print(
                "Query tools used:",
                [f"{tool.name} ({tool.id})" for tool in query_result.tools_used],
            )
            print("Query total cost (USD):", query_result.cost.total_cost_usd)
            print(
                "Developer trace present:",
                bool(
                    query_result.developer_trace
                    and query_result.developer_trace.timeline
                ),
            )
            print("Python SDK favorites-only validation passed.")
    except ContextError as error:
        print("Context Protocol error:", error)
        print("Error code:", error.code)
        raise SystemExit(1) from error


if __name__ == "__main__":
    asyncio.run(main())
