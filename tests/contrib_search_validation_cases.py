from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctxprotocol.contrib.search import (
    ContributorSearchConfig,
    ResolveContributorSearchParams,
    SearchCandidate,
    SearchCandidateProvenance,
    SearchIntent,
    build_contributor_search_validation_artifact,
    create_search_intent,
    resolve_contributor_search,
)
from ctxprotocol.contrib.search.types import (
    CandidateValidator,
    ContributorSearchJudgeResult,
    ContributorSearchResolution,
    ContributorSearchValidationArtifact,
    ContributorSearchValidationCaseKind,
    ContributorSearchValidationExpectation,
)

GENERATED_AT = "2026-03-28T12:00:00.000Z"
REPO_ROOT = Path(__file__).resolve().parents[1]

IRAN_PROMPT = (
    "What are Polymarket's implied odds on US or allied boots on the ground in Iran"
    "—which specific markets, resolution dates, and price levels matter, and how"
    " should I think about that for broader risk-off positioning?"
)
TARIFFS_PROMPT = (
    "What does Kalshi imply about the Supreme Court tariffs case, and which exact"
    " market should I inspect?"
)
GENERIC_OVERLAP_PROMPT = (
    "Which exact tariffs market should I inspect for the Supreme Court case?"
)
STILL_AMBIGUOUS_PROMPT = (
    "Find the best trade-policy market to inspect without assuming the scope."
)
CAPABILITY_MISS_PROMPT = "Find a Bybit perpetual order-book market on Kalshi."


@dataclass(slots=True, frozen=True)
class ValidationCase:
    artifact_path: str
    case_id: str
    case_kind: ContributorSearchValidationCaseKind
    raw_request: str
    intents: list[SearchIntent]
    candidates: list[SearchCandidate]
    expectation: ContributorSearchValidationExpectation
    helper_config: ContributorSearchConfig | None = None
    contributor_config: ContributorSearchConfig | None = None
    overrides: ContributorSearchConfig | None = None
    trace_label: str | None = None
    judge_result: ContributorSearchJudgeResult | None = None
    is_candidate_valid: CandidateValidator | None = None


class StaticJudge:
    def __init__(self, result: ContributorSearchJudgeResult) -> None:
        self.result = result

    async def evaluate(self, input: Any, context: Any) -> ContributorSearchJudgeResult:
        del input, context
        return self.result


def build_candidate(
    *,
    candidate_id: str,
    title: str,
    query: str,
    rank: int,
    description: str | None = None,
    source: str = "website-search-v2",
    raw_ids: dict[str, str] | None = None,
    rank_features: dict[str, bool | int | float | str | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SearchCandidate:
    candidate_metadata = metadata or {}
    return SearchCandidate(
        candidate_id=candidate_id,
        title=title,
        description=description,
        raw_ids=raw_ids or {},
        rank_features=rank_features or {},
        provenance=[
            SearchCandidateProvenance(
                source=source,
                query=query,
                rank=rank,
                fetched_at=GENERATED_AT,
                metadata={
                    "fixture": True,
                    **candidate_metadata,
                },
            )
        ],
        metadata=candidate_metadata,
    )


def read_validation_artifact(relative_path: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


async def resolve_validation_case(
    validation_case: ValidationCase,
) -> ContributorSearchResolution:
    judge = (
        StaticJudge(validation_case.judge_result)
        if validation_case.judge_result is not None
        else None
    )
    return await resolve_contributor_search(
        ResolveContributorSearchParams(
            raw_request=validation_case.raw_request,
            intents=validation_case.intents,
            candidates=validation_case.candidates,
            judge=judge,
            helper_config=validation_case.helper_config,
            contributor_config=validation_case.contributor_config,
            overrides=validation_case.overrides,
            is_candidate_valid=validation_case.is_candidate_valid,
            trace_label=validation_case.trace_label,
        )
    )


async def build_validation_artifact_for_case(
    validation_case: ValidationCase,
) -> tuple[ContributorSearchValidationArtifact, ContributorSearchResolution]:
    resolution = await resolve_validation_case(validation_case)
    artifact = build_contributor_search_validation_artifact(
        case_id=validation_case.case_id,
        case_kind=validation_case.case_kind,
        raw_request=validation_case.raw_request,
        intents=validation_case.intents,
        candidates=validation_case.candidates,
        resolution=resolution,
        expectation=validation_case.expectation,
        generated_at=GENERATED_AT,
    )
    return artifact, resolution


def _is_bybit_perpetual_orderbook(candidate: SearchCandidate) -> bool:
    metadata = candidate.metadata or {}
    return (
        metadata.get("venue") == "bybit"
        and metadata.get("capability") == "perpetual_orderbook"
    )


IRAN_INTENTS = [
    create_search_intent(
        intent_id="iran-ground-entry",
        raw_request=IRAN_PROMPT,
        query="boots on the ground iran polymarket",
        clause="exact boots-on-the-ground contract selection",
    ),
    create_search_intent(
        intent_id="iran-allied-escalation",
        raw_request=IRAN_PROMPT,
        query="us allied enter invade iran",
        clause="related allied escalation markets",
    ),
]

TARIFFS_INTENTS = [
    create_search_intent(
        intent_id="kalshi-tariffs-case",
        raw_request=TARIFFS_PROMPT,
        query="supreme court tariffs case",
        clause="exact Kalshi market selection",
    ),
    create_search_intent(
        intent_id="kalshi-tariffs-ticker",
        raw_request=TARIFFS_PROMPT,
        query="kxdjtvostariffs trump tariffs supreme court",
        clause="ticker-aware validation",
    ),
]

OVERLAP_INTENTS = [
    create_search_intent(
        raw_request=GENERIC_OVERLAP_PROMPT,
        query="supreme court tariffs case",
        clause="exact market selection",
    )
]

AMBIGUOUS_INTENTS = [
    create_search_intent(
        raw_request=STILL_AMBIGUOUS_PROMPT,
        query="trade policy outlook",
        clause="scope remains ambiguous across candidate families",
    )
]

CAPABILITY_MISS_INTENTS = [
    create_search_intent(
        raw_request=CAPABILITY_MISS_PROMPT,
        query="bybit perpetual order book",
        clause="unsupported venue and capability request",
    )
]

IRAN_CANDIDATES = [
    build_candidate(
        candidate_id="pm-us-forces-enter-iran-mar-31",
        title="US forces enter Iran by March 31?",
        query="boots on the ground iran polymarket",
        rank=1,
        description=(
            "Active US military personnel physically enter Iranian territory by March"
            " 31, 2026."
        ),
        raw_ids={
            "conditionId": "0x306d10d4a4d51b41910dbc779ca00908bd917c131541c5c42bbbc736258d2d56",
            "venue": "polymarket",
        },
        rank_features={
            "exactPhraseMatch": True,
            "semanticScore": 0.99,
            "yesPrice": 0.155,
            "resolutionDate": "2026-03-31",
        },
        metadata={
            "url": "https://polymarket.com/event/158299",
            "eventSlug": "us-forces-enter-iran-by",
            "resolutionDate": "2026-03-31T23:59:59Z",
            "yesPrice": 0.155,
        },
    ),
    build_candidate(
        candidate_id="pm-us-forces-enter-iran-apr-30",
        title="US forces enter Iran by April 30?",
        query="us allied enter invade iran",
        rank=2,
        description=(
            "The same event family with a later resolution window for direct US entry."
        ),
        raw_ids={
            "conditionId": "0x6d0e09d0f04572d9b1adad84703458b0297bc5603b69dccbde93147ee4443246",
            "venue": "polymarket",
        },
        rank_features={
            "exactPhraseMatch": True,
            "semanticScore": 0.95,
            "yesPrice": 0.585,
            "resolutionDate": "2026-04-30",
        },
        metadata={
            "url": "https://polymarket.com/event/158299",
            "eventSlug": "us-forces-enter-iran-by",
            "resolutionDate": "2026-04-30T23:59:59Z",
            "yesPrice": 0.585,
        },
    ),
    build_candidate(
        candidate_id="pm-us-invade-iran-before-2027",
        title="Will the U.S. invade Iran before 2027?",
        query="us allied enter invade iran",
        rank=3,
        description=(
            "A stricter invasion framing for longer-dated US boots-on-the-ground risk."
        ),
        raw_ids={
            "conditionId": "0x5db999fad322cea2914535aae5517060c3f80ad6d8c0231cde2124a434d16846",
            "venue": "polymarket",
        },
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.93,
            "yesPrice": 0.605,
            "resolutionDate": "2026-12-31",
        },
        metadata={
            "url": "https://polymarket.com/event/73130",
            "eventSlug": "will-the-us-invade-iran-before-2027",
            "resolutionDate": "2026-12-31T23:59:59Z",
            "yesPrice": 0.605,
        },
    ),
    build_candidate(
        candidate_id="pm-netanyahu-enters-iran-jun-30",
        title="Will Benjamin Netanyahu enter Iran by June 30?",
        query="us allied enter invade iran",
        rank=4,
        description=(
            "An allied-personnel proxy that still belongs to the broader Iran-entry"
            " family."
        ),
        raw_ids={
            "conditionId": "0x83f38b0110a93a4e68d2391dc70868ab1f8a9a074de58b0ef50d5312e3fcfcf7",
            "venue": "polymarket",
        },
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.77,
            "yesPrice": 0.055,
            "resolutionDate": "2026-06-30",
        },
        metadata={
            "url": "https://polymarket.com/event/239820",
            "eventSlug": "who-will-enter-iran-by-june-30",
            "resolutionDate": "2026-06-30T23:59:59Z",
            "yesPrice": 0.055,
        },
    ),
    build_candidate(
        candidate_id="pm-us-iran-ceasefire-mar-31",
        title="Will the U.S. and Iran reach a ceasefire by March 31?",
        query="boots on the ground iran polymarket",
        rank=5,
        description=(
            "A neighboring macro conflict contract that should stay visible but"
            " rejected for direct boots-on-the-ground resolution."
        ),
        raw_ids={"venue": "polymarket"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.4,
            "yesPrice": 0.043,
            "resolutionDate": "2026-03-31",
        },
        metadata={
            "url": "https://polymarket.com/event/us-iran-ceasefire-by-march-31",
            "eventSlug": "us-iran-ceasefire-by-march-31",
            "resolutionDate": "2026-03-31T23:59:59Z",
            "yesPrice": 0.043,
        },
    ),
]

TARIFFS_CANDIDATES = [
    build_candidate(
        candidate_id="kalshi-kxdjtvostariffs",
        title=(
            "Will the Supreme Court rule in favor of Trump in V.O.S. Selections,"
            " Inc. v. Trump?"
        ),
        query="supreme court tariffs case",
        rank=1,
        source="search_markets",
        description=(
            "The exact Kalshi tariff-case contract surfaced by the contributor"
            " search examples."
        ),
        raw_ids={
            "ticker": "KXDJTVOSTARIFFS",
            "eventTicker": "KXDJTVOSTARIFFS",
            "slug": "kxdjtvostariffs",
        },
        rank_features={
            "exactPhraseMatch": True,
            "semanticScore": 0.98,
            "yesPrice": 0.32,
            "closeDate": "2026-06-30",
        },
        metadata={
            "url": "https://kalshi.com/markets/kxdjtvostariffs/tariffs-case",
            "yesPrice": 0.32,
            "closeTime": "2026-06-30T23:59:59Z",
            "eventTitle": "Will the Supreme Court rule on the tariffs case?",
        },
    ),
    build_candidate(
        candidate_id="kalshi-scotus-trade-powers",
        title="Will the Supreme Court limit executive trade powers in 2026?",
        query="kxdjtvostariffs trump tariffs supreme court",
        rank=2,
        source="search_markets",
        description=(
            "A close neighboring court-case contract that is related but broader"
            " than the exact tariffs case."
        ),
        raw_ids={
            "ticker": "KXSCOTUSTRADE",
            "eventTicker": "KXSCOTUSTRADE",
        },
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.83,
            "yesPrice": 0.41,
            "closeDate": "2026-12-31",
        },
        metadata={
            "url": "https://kalshi.com/markets/kxscotustrade/executive-trade-powers",
            "yesPrice": 0.41,
            "closeTime": "2026-12-31T23:59:59Z",
        },
    ),
    build_candidate(
        candidate_id="kalshi-tariff-revenue-2026",
        title="Will tariff revenue exceed $300B in 2026?",
        query="supreme court tariffs case",
        rank=3,
        source="search_markets",
        description=(
            "A tariff-adjacent macro contract that should be rejected for the exact"
            " court-case ask."
        ),
        raw_ids={
            "ticker": "KXTARIFFREV",
            "eventTicker": "KXTARIFFREV",
        },
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.52,
            "yesPrice": 0.28,
            "closeDate": "2026-12-31",
        },
        metadata={
            "url": "https://kalshi.com/markets/kxtariffrev/tariff-revenue-2026",
            "yesPrice": 0.28,
            "closeTime": "2026-12-31T23:59:59Z",
        },
    ),
]

GENERIC_OVERLAP_CANDIDATES = [
    build_candidate(
        candidate_id="generic-scotus-tariffs",
        title="Supreme Court tariff ruling in 2026",
        query="supreme court tariffs case",
        rank=1,
        description="The exact court-case market family.",
        raw_ids={"marketId": "market-1"},
        rank_features={
            "exactPhraseMatch": True,
            "semanticScore": 0.94,
        },
    ),
    build_candidate(
        candidate_id="generic-trade-policy-outlook",
        title="Broader tariff policy outlook",
        query="supreme court tariffs case",
        rank=2,
        description="A broader policy proxy with overlapping language.",
        raw_ids={"marketId": "market-2"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.78,
        },
    ),
    build_candidate(
        candidate_id="generic-tariff-revenue-benchmark",
        title="Tariff revenue benchmark in 2026",
        query="supreme court tariffs case",
        rank=3,
        description="Tariff-adjacent but not the exact court-case resolution.",
        raw_ids={"marketId": "market-3"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.51,
        },
    ),
]

AMBIGUOUS_CANDIDATES = [
    build_candidate(
        candidate_id="ambiguous-scotus-case",
        title="Supreme Court tariff ruling in 2026",
        query="trade policy outlook",
        rank=1,
        description="A court-case contract.",
        raw_ids={"marketId": "ambiguous-1"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.72,
        },
    ),
    build_candidate(
        candidate_id="ambiguous-policy-basket",
        title="Broader trade policy outlook in 2026",
        query="trade policy outlook",
        rank=2,
        description="A macro policy basket.",
        raw_ids={"marketId": "ambiguous-2"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.71,
        },
    ),
    build_candidate(
        candidate_id="ambiguous-tariff-revenue",
        title="Tariff revenue benchmark in 2026",
        query="trade policy outlook",
        rank=3,
        description="A metric-style contract instead of the user-facing market choice.",
        raw_ids={"marketId": "ambiguous-3"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.55,
        },
    ),
]

UNSUPPORTED_VENUE_CANDIDATES = [
    build_candidate(
        candidate_id="kalshi-weather-high-ny",
        title="NYC high temperature above 47.5F?",
        query="bybit perpetual order book",
        rank=1,
        source="search_markets",
        description=(
            "A valid Kalshi contract that cannot satisfy the requested venue or"
            " capability."
        ),
        raw_ids={"ticker": "KXHIGHNY-26MAR19-B47.5"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.08,
        },
        metadata={
            "venue": "kalshi",
            "capability": "weather",
        },
    ),
    build_candidate(
        candidate_id="kalshi-bitcoin-spot-range",
        title="Will Bitcoin settle above $120k this week?",
        query="bybit perpetual order book",
        rank=2,
        source="search_markets",
        description="A venue-matched but capability-mismatched contract.",
        raw_ids={"ticker": "KXBTC-120K"},
        rank_features={
            "exactPhraseMatch": False,
            "semanticScore": 0.12,
        },
        metadata={
            "venue": "kalshi",
            "capability": "event_contract",
        },
    ),
]

VALIDATION_CASES = [
    ValidationCase(
        artifact_path="examples/client/validation/named-regression-iran-boots-on-ground.json",
        case_id="polymarket-iran-boots-on-ground",
        case_kind="named_regression",
        raw_request=IRAN_PROMPT,
        intents=IRAN_INTENTS,
        candidates=IRAN_CANDIDATES,
        expectation=ContributorSearchValidationExpectation(
            outcome="selected",
            selected_candidate_id="pm-us-forces-enter-iran-mar-31",
        ),
        helper_config=ContributorSearchConfig(
            provider="openrouter",
            model="openai/gpt-5.4-nano",
            timeout_ms=5000,
            budget_usd="0.020000",
            max_shortlist_size=5,
        ),
        contributor_config=ContributorSearchConfig(
            model="anthropic/claude-sonnet-4.5",
            budget_usd="0.015000",
        ),
        overrides=ContributorSearchConfig(
            model="glm-turbo-model",
            timeout_ms=1500,
            budget_usd="0.005000",
        ),
        trace_label="polymarket-iran-helper-pilot",
        judge_result=ContributorSearchJudgeResult(
            primary_candidate_id="pm-us-forces-enter-iran-mar-31",
            related_candidate_ids=[
                "pm-us-forces-enter-iran-apr-30",
                "pm-us-invade-iran-before-2027",
                "pm-netanyahu-enters-iran-jun-30",
            ],
            rejected_candidate_ids=["pm-us-iran-ceasefire-mar-31"],
            confidence="high",
            reason=(
                "The March 31 US-forces-enter contract is the tightest direct"
                " boots-on-the-ground market, while the April 30, before-2027,"
                " and Netanyahu-in-Iran contracts remain related escalation"
                " follow-ups."
            ),
            usage={
                "promptTokens": 420,
                "completionTokens": 118,
                "totalTokens": 538,
                "costUsd": "0.001230",
                "latencyMs": 312,
            },
        ),
    ),
    ValidationCase(
        artifact_path="examples/client/validation/named-regression-supreme-court-tariffs.json",
        case_id="kalshi-supreme-court-tariffs",
        case_kind="named_regression",
        raw_request=TARIFFS_PROMPT,
        intents=TARIFFS_INTENTS,
        candidates=TARIFFS_CANDIDATES,
        expectation=ContributorSearchValidationExpectation(
            outcome="selected",
            selected_candidate_id="kalshi-kxdjtvostariffs",
        ),
        helper_config=ContributorSearchConfig(
            provider="openrouter",
            model="openai/gpt-5.4-nano",
            timeout_ms=4000,
            budget_usd="0.010000",
            max_shortlist_size=4,
        ),
        overrides=ContributorSearchConfig(
            model="glm-turbo-model",
            timeout_ms=1200,
            budget_usd="0.003000",
        ),
        trace_label="kalshi-tariffs-helper-pilot",
        judge_result=ContributorSearchJudgeResult(
            primary_candidate_id="kalshi-kxdjtvostariffs",
            related_candidate_ids=["kalshi-scotus-trade-powers"],
            rejected_candidate_ids=["kalshi-tariff-revenue-2026"],
            confidence="high",
            reason=(
                "The KXDJTVOSTARIFFS contract is the exact Supreme Court tariffs"
                " case, while the other candidates are broader or only"
                " tariff-adjacent."
            ),
            usage={
                "promptTokens": 310,
                "completionTokens": 92,
                "totalTokens": 402,
                "costUsd": "0.000980",
                "latencyMs": 244,
            },
        ),
    ),
    ValidationCase(
        artifact_path="examples/client/validation/shared-generic-overlap-best-match.json",
        case_id="generic-overlap-best-match",
        case_kind="generic_overlap",
        raw_request=GENERIC_OVERLAP_PROMPT,
        intents=OVERLAP_INTENTS,
        candidates=GENERIC_OVERLAP_CANDIDATES,
        expectation=ContributorSearchValidationExpectation(
            outcome="selected",
            selected_candidate_id="generic-scotus-tariffs",
        ),
        helper_config=ContributorSearchConfig(
            provider="openrouter",
            model="openai/gpt-5.4-nano",
            timeout_ms=3500,
            budget_usd="0.009000",
            max_shortlist_size=3,
        ),
        contributor_config=ContributorSearchConfig(
            model="anthropic/claude-sonnet-4.5",
            budget_usd="0.008000",
        ),
        overrides=ContributorSearchConfig(
            model="glm-turbo-model",
            timeout_ms=900,
            budget_usd="0.002500",
        ),
        trace_label="generic-overlap-parity",
        judge_result=ContributorSearchJudgeResult(
            primary_candidate_id="generic-scotus-tariffs",
            related_candidate_ids=["generic-trade-policy-outlook"],
            rejected_candidate_ids=["generic-tariff-revenue-benchmark"],
            confidence="high",
            reason=(
                "The court-case candidate is the exact clause-level match, while"
                " the broader policy and revenue contracts should stay secondary."
            ),
            usage={
                "promptTokens": 180,
                "completionTokens": 48,
                "totalTokens": 228,
                "costUsd": "0.000410",
                "latencyMs": 133,
            },
        ),
    ),
    ValidationCase(
        artifact_path="examples/client/validation/shared-still-ambiguous-shortlist.json",
        case_id="still-ambiguous-shortlist",
        case_kind="still_ambiguous",
        raw_request=STILL_AMBIGUOUS_PROMPT,
        intents=AMBIGUOUS_INTENTS,
        candidates=AMBIGUOUS_CANDIDATES,
        expectation=ContributorSearchValidationExpectation(
            outcome="shortlist_only",
            degraded_reason_code="ambiguous_shortlist",
        ),
        helper_config=ContributorSearchConfig(
            provider="openrouter",
            model="glm-turbo-model",
            timeout_ms=1000,
            budget_usd="0.002000",
            max_shortlist_size=3,
        ),
        trace_label="still-ambiguous-parity",
        judge_result=ContributorSearchJudgeResult(
            primary_candidate_id=None,
            related_candidate_ids=["ambiguous-scotus-case", "ambiguous-policy-basket"],
            rejected_candidate_ids=["ambiguous-tariff-revenue"],
            confidence="low",
            reason=(
                "The shortlist still splits between an exact court-case scope and"
                " a broader policy scope, so the helper should preserve ambiguity"
                " instead of inventing certainty."
            ),
            usage={
                "promptTokens": 165,
                "completionTokens": 61,
                "totalTokens": 226,
                "costUsd": "0.000390",
                "latencyMs": 149,
            },
        ),
    ),
    ValidationCase(
        artifact_path="examples/client/validation/shared-capability-miss-unsupported-venue.json",
        case_id="capability-miss-unsupported-venue",
        case_kind="capability_miss",
        raw_request=CAPABILITY_MISS_PROMPT,
        intents=CAPABILITY_MISS_INTENTS,
        candidates=UNSUPPORTED_VENUE_CANDIDATES,
        expectation=ContributorSearchValidationExpectation(
            outcome="capability_miss",
            degraded_reason_code="no_viable_candidates",
        ),
        helper_config=ContributorSearchConfig(
            provider="openrouter",
            model="glm-turbo-model",
            timeout_ms=1000,
            budget_usd="0.001500",
            max_shortlist_size=2,
        ),
        trace_label="capability-miss-parity",
        is_candidate_valid=_is_bybit_perpetual_orderbook,
    ),
]
