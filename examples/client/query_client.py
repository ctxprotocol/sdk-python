"""Minimal Query-mode client example."""

from __future__ import annotations

import asyncio
import os

from ctxprotocol import ContextClient, ContextError, QueryDeveloperTrace


def summarize_developer_trace(trace: QueryDeveloperTrace | None) -> dict[str, int]:
    """Return key health counters from a developer trace payload."""
    timeline = (trace.timeline or []) if trace else []

    def count_step(step_type: str) -> int:
        return sum(
            1
            for step in timeline
            if step.step_type == step_type or step.event == step_type
        )

    return {
        "retries": (
            trace.summary.retry_count
            if trace and trace.summary and trace.summary.retry_count is not None
            else count_step("retry")
        ),
        "tool_calls": (
            trace.summary.tool_calls
            if trace and trace.summary and trace.summary.tool_calls is not None
            else count_step("tool-call")
        ),
        "loops": (
            trace.summary.loop_count
            if trace and trace.summary and trace.summary.loop_count is not None
            else count_step("loop")
        ),
    }


async def main() -> None:
    try:
        async with ContextClient(api_key=os.environ["CONTEXT_API_KEY"]) as client:
            # tools omitted/None => auto-discovery; tools=["id"] => manual mode;
            # tools=[] => direct synthesis (no tool execution).
            answer = await client.query.run(
                query=(
                    "What are the top whale movements on Base, and what confidence "
                    "checks did you run?"
                ),
                query_depth="deep",
                include_developer_trace=True,
            )
            print("Response:", answer.response)
            print("Total cost (USD):", answer.cost.total_cost_usd)
            print("Duration (ms):", answer.duration_ms)
            print("Tools used:", [tool.name for tool in answer.tools_used])
            print(
                "Developer trace summary:",
                summarize_developer_trace(answer.developer_trace),
            )

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
