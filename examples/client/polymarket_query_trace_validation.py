"""Deep Polymarket query validation with developer trace capture."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctxprotocol import ContextClient, ContextError, QueryDeveloperTrace
from ctxprotocol.client.types import QueryStreamDoneEvent

POLYMARKET_TOOL_ID = "294100e8-c648-4e5f-a254-95a14b56e398"
OUTPUT_PATH = Path("polymarket-query-trace-results-py.json")

PROMPTS = [
    "What are the top 5 prediction markets by volume right now?",
    "Search for markets related to the 2028 presidential election",
    "How efficient is the pricing on the Trump vs Biden market? Is there arbitrage?",
    "Analyze whale flow on the Fed rate decision event who are the top holders and what is their cost basis?",
    "Show me markets between 30-70% probability with high volume that might be mispriced",
    "Analyze my Polymarket positions which ones have poor exit liquidity?",
    "Compare the Polymarket odds vs Kalshi odds on the next Fed meeting outcome",
    "Find correlated markets where a move in one should predict a move in another",
    "What are the orderbook depth and spread for all outcomes on event slug 'democratic-presidential-nominee-2028'?",
    "Browse all markets in the Politics category sorted by recent activity",
    "Build a high-conviction workflow: find mispriced markets, check whale positioning, then verify liquidity",
]


def _count_trace_steps(trace: QueryDeveloperTrace | None, key: str) -> int:
    if not trace or not trace.timeline:
        return 0
    return sum(
        1
        for step in trace.timeline
        if step.step_type == key or step.event == key
    )


def _trace_summary(trace: QueryDeveloperTrace | None) -> dict[str, int]:
    summary = trace.summary if trace else None
    timeline = trace.timeline if trace and trace.timeline else []
    return {
        "toolCalls": (
            summary.tool_calls
            if summary and summary.tool_calls is not None
            else _count_trace_steps(trace, "tool-call")
        ),
        "retryCount": (
            summary.retry_count
            if summary and summary.retry_count is not None
            else _count_trace_steps(trace, "retry")
        ),
        "selfHealCount": (
            summary.self_heal_count
            if summary and summary.self_heal_count is not None
            else _count_trace_steps(trace, "self-heal")
        ),
        "fallbackCount": (
            summary.fallback_count
            if summary and summary.fallback_count is not None
            else _count_trace_steps(trace, "fallback")
        ),
        "failureCount": (
            summary.failure_count
            if summary and summary.failure_count is not None
            else _count_trace_steps(trace, "failure")
        ),
        "recoveryCount": (
            summary.recovery_count
            if summary and summary.recovery_count is not None
            else _count_trace_steps(trace, "recovery")
        ),
        "completionChecks": (
            summary.completion_checks
            if summary and summary.completion_checks is not None
            else _count_trace_steps(trace, "completion-check")
        ),
        "loopCount": (
            summary.loop_count
            if summary and summary.loop_count is not None
            else _count_trace_steps(trace, "loop")
        ),
        "timelineLength": len(timeline),
    }


def _top_timeline_events(
    trace: QueryDeveloperTrace | None,
) -> list[dict[str, int | str]]:
    counts: dict[str, int] = {}
    for step in trace.timeline if trace and trace.timeline else []:
        key = step.step_type or step.event or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return [
        {"key": key, "count": count}
        for key, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
    ]


def _inefficiency_signals(summary: dict[str, int]) -> list[str]:
    signals: list[str] = []
    if summary["retryCount"] > 0:
        signals.append(f"retry:{summary['retryCount']}")
    if summary["selfHealCount"] > 0:
        signals.append(f"self-heal:{summary['selfHealCount']}")
    if summary["fallbackCount"] > 0:
        signals.append(f"fallback:{summary['fallbackCount']}")
    if summary["failureCount"] > 0:
        signals.append(f"failure:{summary['failureCount']}")
    if summary["loopCount"] > 0:
        signals.append(f"loop:{summary['loopCount']}")
    if summary["completionChecks"] > 2:
        signals.append(f"completion-checks:{summary['completionChecks']}")
    return signals


async def run_stream_probe(client: ContextClient) -> dict[str, Any]:
    event_counts = {"tool-status": 0, "text-delta": 0, "developer-trace": 0, "done": 0}
    done_trace_present = False

    async for event in client.query.stream(
        query="Top 3 politics markets by recent volume",
        tools=[POLYMARKET_TOOL_ID],
        query_depth="deep",
        include_developer_trace=True,
    ):
        if event.type in event_counts:
            event_counts[event.type] += 1
        if isinstance(event, QueryStreamDoneEvent):
            done_trace_present = event.result.developer_trace is not None

    return {
        "eventCounts": event_counts,
        "doneTracePresent": done_trace_present,
    }


async def main() -> None:
    api_key = os.environ.get("CONTEXT_API_KEY")
    if not api_key:
        raise RuntimeError("Set CONTEXT_API_KEY before running this script.")

    reports: list[dict[str, Any]] = []

    async with ContextClient(api_key=api_key) as client:
        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\n[{index}/{len(PROMPTS)}] {prompt}")
            try:
                result = await client.query.run(
                    query=prompt,
                    tools=[POLYMARKET_TOOL_ID],
                    query_depth="deep",
                    include_data=True,
                    include_developer_trace=True,
                )

                trace_summary = _trace_summary(result.developer_trace)
                signals = _inefficiency_signals(trace_summary)
                tools_used = [
                    {"id": tool.id, "name": tool.name, "skillCalls": tool.skill_calls}
                    for tool in result.tools_used
                ]
                total_skill_calls = sum(tool["skillCalls"] for tool in tools_used)

                reports.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "ok": True,
                        "durationMs": result.duration_ms,
                        "costUsd": result.cost.total_cost_usd,
                        "responsePreview": result.response[:320],
                        "toolsUsed": tools_used,
                        "totalSkillCalls": total_skill_calls,
                        "developerTracePresent": result.developer_trace is not None,
                        "traceSummary": trace_summary,
                        "topTimelineEvents": _top_timeline_events(result.developer_trace),
                        "inefficiencySignals": signals,
                    }
                )

                print(
                    f"  duration={result.duration_ms}ms cost={result.cost.total_cost_usd} skillCalls={total_skill_calls}"
                )
                print(
                    f"  trace retries={trace_summary['retryCount']} selfHeal={trace_summary['selfHealCount']} fallback={trace_summary['fallbackCount']} loop={trace_summary['loopCount']}"
                )
                if signals:
                    print(f"  signals: {', '.join(signals)}")
            except ContextError as error:
                reports.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "ok": False,
                        "error": {
                            "name": type(error).__name__,
                            "message": error.message,
                            "code": error.code,
                            "statusCode": error.status_code,
                        },
                    }
                )
                print(
                    f"  failed: {type(error).__name__} "
                    f"{f'[{error.code}] ' if error.code else ''}{error.message}"
                )
            except Exception as error:  # pragma: no cover - safety for runtime runs
                reports.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "ok": False,
                        "error": {
                            "name": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                print(f"  failed: {type(error).__name__} {error}")

            await asyncio.sleep(0.5)

        stream_probe = await run_stream_probe(client)

    ok_reports = [report for report in reports if report.get("ok")]
    totals = {
        "costUsd": 0.0,
        "durationMs": 0,
        "skillCalls": 0,
        "retryCount": 0,
        "selfHealCount": 0,
        "fallbackCount": 0,
        "failureCount": 0,
        "recoveryCount": 0,
        "completionChecks": 0,
        "loopCount": 0,
    }

    for report in ok_reports:
        totals["costUsd"] += float(report.get("costUsd", "0"))
        totals["durationMs"] += int(report.get("durationMs", 0))
        totals["skillCalls"] += int(report.get("totalSkillCalls", 0))
        summary = report.get("traceSummary", {})
        totals["retryCount"] += int(summary.get("retryCount", 0))
        totals["selfHealCount"] += int(summary.get("selfHealCount", 0))
        totals["fallbackCount"] += int(summary.get("fallbackCount", 0))
        totals["failureCount"] += int(summary.get("failureCount", 0))
        totals["recoveryCount"] += int(summary.get("recoveryCount", 0))
        totals["completionChecks"] += int(summary.get("completionChecks", 0))
        totals["loopCount"] += int(summary.get("loopCount", 0))

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "toolId": POLYMARKET_TOOL_ID,
        "promptCount": len(PROMPTS),
        "successCount": len(ok_reports),
        "failureCount": len(reports) - len(ok_reports),
        "totals": {
            **totals,
            "averageDurationMs": round(totals["durationMs"] / len(ok_reports))
            if ok_reports
            else 0,
            "averageCostUsd": round(totals["costUsd"] / len(ok_reports), 6)
            if ok_reports
            else 0,
        },
        "streamProbe": stream_probe,
        "reports": reports,
    }

    OUTPUT_PATH.write_text(f"{json.dumps(output, indent=2)}\n", encoding="utf-8")
    print("\nRun complete.")
    print(
        f"Saved {OUTPUT_PATH} with {len(ok_reports)}/{len(PROMPTS)} successful prompts."
    )


if __name__ == "__main__":
    asyncio.run(main())
