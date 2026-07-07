#!/usr/bin/env python3
"""Throwaway Hermes-parity sanity check for ctxprotocol 0.21.0.

Mirrors ~/.hermes/workspace/ctx-gtm/scripts/ctxprotocol-query.py:
  - same ContextClient timeouts
  - same query.run() kwargs (no agent_model_id)
  - same output contract (JSON + CHART_URL + COST_USD)

Reads CONTEXT_API_KEY from context-sdk/.env.local (or env).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

SDK_QUERY_TIMEOUT = 1860  # Hermes: 1800s ceiling + 60s grace
FAVORITES_ONLY = os.environ.get("CTX_FAVORITES_ONLY", "true").strip().lower() in (
    "true",
    "1",
    "yes",
)

DEFAULT_QUERY = (
    "What is the current price of Bitcoin? One sentence answer, no charts needed."
)


def load_api_key() -> str:
    candidates = [
        os.environ.get("CONTEXT_API_KEY"),
        _read_env_file(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "context-sdk",
                ".env.local",
            )
        ),
        _read_env_file(
            os.path.expanduser("~/.hermes/.env"),
        ),
    ]
    for value in candidates:
        if value:
            return value
    print(
        "ERROR: CONTEXT_API_KEY not found in env, context-sdk/.env.local, or ~/.hermes/.env",
        file=sys.stderr,
    )
    sys.exit(1)


def _read_env_file(path: str) -> str | None:
    try:
        with open(os.path.abspath(path)) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("CONTEXT_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _chart_urls_from_result(ans) -> list[str]:
    urls: list[str] = []
    for art in getattr(ans, "computed_artifacts", None) or []:
        if getattr(art, "kind", None) == "image":
            url = getattr(art, "url", None)
            if url:
                urls.append(url)
    if not urls:
        for text in (getattr(ans, "response", None), getattr(ans, "summary", None)):
            if not text:
                continue
            for match in re.finditer(r"!\[.*?\]\((https?://[^)\s]+)\)", text):
                urls.append(match.group(1))
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def emit(ans) -> None:
    cost = getattr(ans, "cost", None)
    total = getattr(cost, "total_cost_usd", None) if cost else None
    payload = {
        "response": getattr(ans, "response", None),
        "summary": getattr(ans, "summary", None),
        "cost_usd": total,
        "data_url": getattr(ans, "data_url", None),
        "tools_used": [
            {"name": getattr(tool, "name", None), "id": getattr(tool, "id", None)}
            for tool in (getattr(ans, "tools_used", None) or [])
        ],
        "outcome_type": getattr(ans, "outcome_type", None),
        "stop_reason": getattr(ans, "stop_reason", None),
        "duration_ms": getattr(ans, "duration_ms", None),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    chart_urls = _chart_urls_from_result(ans)
    if chart_urls:
        for index, url in enumerate(chart_urls, start=1):
            print(f"CHART_URL_{index}: {url}")
    else:
        print("CHART_URL_1: none")
    print(f"COST_USD: {total if total is not None else 'unknown'}")


async def run_hermes_style_query(question: str) -> None:
    from ctxprotocol import ContextClient, __version__

    print(f"ctxprotocol version: {__version__}", file=sys.stderr)
    print(f"favorites_only: {FAVORITES_ONLY}", file=sys.stderr)
    print(f"query: {question[:120]}{'...' if len(question) > 120 else ''}", file=sys.stderr)

    async with ContextClient(
        api_key=load_api_key(),
        request_timeout_seconds=float(SDK_QUERY_TIMEOUT),
        stream_timeout_seconds=float(SDK_QUERY_TIMEOUT),
    ) as client:
        answer = await client.query.run(
            query=question,
            response_shape="answer_with_evidence",
            include_data=False,
            include_data_url=True,
            favorites_only=FAVORITES_ONLY,
        )
    emit(answer)


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    try:
        asyncio.run(run_hermes_style_query(question))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        print("CHART_URL_1: none")
        print("COST_USD: unknown")
        sys.exit(1)


if __name__ == "__main__":
    main()
