#!/usr/bin/env python3
"""Throwaway: same Hermes kwargs but via stream() (direct /api/v1/query SSE).

Use to compare job-backed run() vs held-open sync path on long queries.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

SDK_QUERY_TIMEOUT = 1860
FAVORITES_ONLY = True


def load_api_key() -> str:
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "context-sdk", ".env.local")
    )
    with open(path) as handle:
        for line in handle:
            if line.startswith("CONTEXT_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("CONTEXT_API_KEY missing")


async def main() -> None:
    query_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "saudi-oil-query.txt"
    )
    with open(query_path) as handle:
        query = handle.read().strip()

    from ctxprotocol import ContextClient, __version__

    print(f"stream path | ctxprotocol {__version__}", file=sys.stderr)
    started = time.time()
    async with ContextClient(
        api_key=load_api_key(),
        request_timeout_seconds=float(SDK_QUERY_TIMEOUT),
        stream_timeout_seconds=float(SDK_QUERY_TIMEOUT),
    ) as client:
        result = None
        async for event in client.query.stream(
            query=query,
            response_shape="answer_with_evidence",
            include_data=False,
            include_data_url=True,
            favorites_only=FAVORITES_ONLY,
        ):
            if event.type == "done":
                result = event.result
            elif event.type == "error":
                raise RuntimeError(getattr(event, "message", str(event)))

    elapsed = time.time() - started
    cost = getattr(getattr(result, "cost", None), "total_cost_usd", None)
    charts = [
        getattr(art, "url", None)
        for art in (getattr(result, "computed_artifacts", None) or [])
        if getattr(art, "kind", None) == "image" and getattr(art, "url", None)
    ]
    print(
        json.dumps(
            {
                "elapsed_s": round(elapsed, 1),
                "stop_reason": getattr(result, "stop_reason", None),
                "cost_usd": cost,
                "chart_count": len(charts),
                "tools_used": [
                    getattr(t, "name", None)
                    for t in (getattr(result, "tools_used", None) or [])
                ],
                "response_preview": (getattr(result, "response", None) or "")[:800],
            },
            indent=2,
        )
    )
    for index, url in enumerate(charts, start=1):
        print(f"CHART_URL_{index}: {url}")
    if not charts:
        print("CHART_URL_1: none")
    print(f"COST_USD: {cost}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        print("CHART_URL_1: none")
        print("COST_USD: unknown")
        sys.exit(1)
