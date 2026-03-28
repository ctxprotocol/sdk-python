from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ctxprotocol.contrib.search.types import (
    CONTRIBUTOR_SEARCH_METADATA_VERSION,
    CandidateValidator,
    ContributorSearchBudgetExceededError,
    ContributorSearchConfidence,
    ContributorSearchConfig,
    ContributorSearchDegradedOutcome,
    ContributorSearchDegradedOutcomePolicy,
    ContributorSearchDegradedReasonCode,
    ContributorSearchJudge,
    ContributorSearchJudgeContext,
    ContributorSearchJudgeInput,
    ContributorSearchJudgeResult,
    ContributorSearchJudgeSnapshot,
    ContributorSearchJudgeUsage,
    ContributorSearchMetadata,
    ContributorSearchMetadataSource,
    ContributorSearchOutcome,
    ContributorSearchResolution,
    ContributorSearchResolvedConfig,
    ContributorSearchTraceSummary,
    ContributorSearchValidatorStatus,
    ResolveContributorSearchParams,
    SearchCandidate,
    SearchCandidateProvenance,
    SearchIntent,
    SearchShortlist,
)

DEFAULT_MAX_SHORTLIST_SIZE = 8
DEFAULT_DEGRADED_OUTCOME_POLICY: ContributorSearchDegradedOutcomePolicy = (
    "return_shortlist"
)
MAX_METADATA_PROVENANCE_ENTRIES = 8
MAX_METADATA_INTENT_QUERIES = 6
MAX_REASON_LENGTH = 240


class ContributorSearchTimeoutError(Exception):
    def __init__(self, message: str = "Contributor search judge timed out") -> None:
        super().__init__(message)


LiteralJudgeValidationReason: TypeAlias = Literal[
    "judge_invalid_output",
    "validator_rejected",
]


@dataclass(slots=True)
class JudgeValidationResult:
    selected_candidate: SearchCandidate | None
    related_candidates: list[SearchCandidate]
    rejected_candidates: list[SearchCandidate]
    reason_code: LiteralJudgeValidationReason | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason_code is None and self.reason is None


def _normalize_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _unique_strings(values: Sequence[str]) -> list[str]:
    deduped: dict[str, None] = {}
    for value in values:
        normalized = _normalize_string(value)
        if normalized is None:
            continue
        deduped[normalized] = None
    return list(deduped.keys())


def _truncate_reason(reason: str) -> str:
    if len(reason) <= MAX_REASON_LENGTH:
        return reason
    return f"{reason[:MAX_REASON_LENGTH]}..."


def _candidate_provenance_key(provenance: SearchCandidateProvenance) -> str:
    return "::".join(
        [
            provenance.source,
            provenance.query,
            "" if provenance.rank is None else str(provenance.rank),
            provenance.fetched_at or "",
            json.dumps(provenance.metadata or {}, sort_keys=True),
        ]
    )


def _merge_provenance(
    first: Sequence[SearchCandidateProvenance],
    second: Sequence[SearchCandidateProvenance],
) -> list[SearchCandidateProvenance]:
    merged: dict[str, SearchCandidateProvenance] = {}
    for provenance in [*first, *second]:
        merged[_candidate_provenance_key(provenance)] = provenance
    return list(merged.values())


def _merge_candidates(first: SearchCandidate, second: SearchCandidate) -> SearchCandidate:
    return SearchCandidate(
        candidateId=first.candidate_id,
        title=first.title,
        description=first.description if first.description is not None else second.description,
        rawIds={**(second.raw_ids or {}), **(first.raw_ids or {})},
        rankFeatures={**(second.rank_features or {}), **(first.rank_features or {})},
        provenance=_merge_provenance(first.provenance, second.provenance),
        metadata={**(second.metadata or {}), **(first.metadata or {})},
    )


def _is_candidate_selectable(
    candidate: SearchCandidate,
    validate_candidate: CandidateValidator,
) -> bool:
    return validate_candidate(candidate)


def _summarize_provenance(
    candidates: Sequence[SearchCandidate],
) -> list[ContributorSearchMetadataSource]:
    grouped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for provenance in candidate.provenance:
            key = f"{provenance.source}::{provenance.query}"
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "source": provenance.source,
                    "query": provenance.query,
                    "candidate_ids": {candidate.candidate_id},
                }
                continue
            existing["candidate_ids"].add(candidate.candidate_id)

    summarized = [
        ContributorSearchMetadataSource(
            source=entry["source"],
            query=entry["query"],
            candidateCount=len(entry["candidate_ids"]),
        )
        for entry in grouped.values()
    ]
    return summarized[:MAX_METADATA_PROVENANCE_ENTRIES]


def _build_judge_snapshot(
    *,
    config: ContributorSearchResolvedConfig,
    applied: bool,
    usage: ContributorSearchJudgeUsage | None,
) -> ContributorSearchJudgeSnapshot:
    return ContributorSearchJudgeSnapshot(
        provider=config.provider,
        model=config.model,
        timeoutMs=config.timeout_ms,
        budgetUsd=config.budget_usd,
        disabled=config.disable_judge,
        applied=applied,
        usage=usage,
    )


def _build_trace_summary(
    *,
    used_deterministic_fallback: bool,
    validator_status: ContributorSearchValidatorStatus,
    validator_reason_code: str | None,
    validator_reason: str | None,
) -> ContributorSearchTraceSummary:
    return ContributorSearchTraceSummary(
        usedDeterministicFallback=used_deterministic_fallback,
        validatorStatus=validator_status,
        validatorReasonCode=validator_reason_code,
        validatorReason=validator_reason,
    )


def _build_search_metadata(
    *,
    intents: Sequence[SearchIntent],
    candidates: Sequence[SearchCandidate],
    shortlist: Sequence[SearchCandidate],
    selected_candidate: SearchCandidate | None,
    related_candidates: Sequence[SearchCandidate],
    rejected_candidates: Sequence[SearchCandidate],
    outcome: ContributorSearchOutcome,
    confidence: ContributorSearchConfidence,
    degraded: ContributorSearchDegradedOutcome | None,
    judge_snapshot: ContributorSearchJudgeSnapshot,
    trace: ContributorSearchTraceSummary,
) -> ContributorSearchMetadata:
    intent_queries = _unique_strings([intent.query for intent in intents])[
        :MAX_METADATA_INTENT_QUERIES
    ]
    return ContributorSearchMetadata(
        version=CONTRIBUTOR_SEARCH_METADATA_VERSION,
        outcome=outcome,
        confidence=confidence,
        selectedCandidateId=(
            selected_candidate.candidate_id if selected_candidate is not None else None
        ),
        shortlistCandidateIds=[candidate.candidate_id for candidate in shortlist],
        relatedCandidateIds=[candidate.candidate_id for candidate in related_candidates],
        rejectedCandidateIds=[candidate.candidate_id for candidate in rejected_candidates],
        candidateCount=len(candidates),
        shortlistCount=len(shortlist),
        intentQueries=intent_queries,
        degraded=degraded,
        judge=judge_snapshot,
        provenance=_summarize_provenance(candidates),
        trace=trace,
    )


def _build_resolution(
    *,
    intents: Sequence[SearchIntent],
    candidates: Sequence[SearchCandidate],
    shortlist: Sequence[SearchCandidate],
    selected_candidate: SearchCandidate | None,
    related_candidates: Sequence[SearchCandidate],
    rejected_candidates: Sequence[SearchCandidate],
    outcome: ContributorSearchOutcome,
    confidence: ContributorSearchConfidence,
    reason: str,
    degraded: ContributorSearchDegradedOutcome | None,
    judge_snapshot: ContributorSearchJudgeSnapshot,
    trace: ContributorSearchTraceSummary,
) -> ContributorSearchResolution:
    search_metadata = _build_search_metadata(
        intents=intents,
        candidates=candidates,
        shortlist=shortlist,
        selected_candidate=selected_candidate,
        related_candidates=related_candidates,
        rejected_candidates=rejected_candidates,
        outcome=outcome,
        confidence=confidence,
        degraded=degraded,
        judge_snapshot=judge_snapshot,
        trace=trace,
    )
    return ContributorSearchResolution(
        outcome=outcome,
        selectedCandidate=selected_candidate,
        shortlist=list(shortlist),
        relatedCandidates=list(related_candidates),
        rejectedCandidates=list(rejected_candidates),
        confidence=confidence,
        reason=_truncate_reason(reason),
        degraded=degraded,
        searchMetadata=search_metadata,
    )


@dataclass(slots=True)
class DedupeCandidateIdsResult:
    ids: list[str]
    had_duplicates: bool


def _dedupe_candidate_ids(ids: Sequence[str]) -> DedupeCandidateIdsResult:
    deduped: list[str] = []
    seen: set[str] = set()
    had_duplicates = False

    for candidate_id in ids:
        normalized = _normalize_string(candidate_id)
        if normalized is None:
            continue
        if normalized in seen:
            had_duplicates = True
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return DedupeCandidateIdsResult(ids=deduped, had_duplicates=had_duplicates)


def _validate_judge_selection(
    *,
    shortlist: Sequence[SearchCandidate],
    primary_candidate_id: str | None,
    related_candidate_ids: Sequence[str],
    rejected_candidate_ids: Sequence[str],
    validate_candidate: CandidateValidator,
) -> JudgeValidationResult:
    shortlist_by_id = {candidate.candidate_id: candidate for candidate in shortlist}
    normalized_primary_candidate_id = _normalize_string(primary_candidate_id)
    related_ids = _dedupe_candidate_ids(related_candidate_ids)
    rejected_ids = _dedupe_candidate_ids(rejected_candidate_ids)

    if related_ids.had_duplicates or rejected_ids.had_duplicates:
        return JudgeValidationResult(
            selected_candidate=None,
            related_candidates=[],
            rejected_candidates=[],
            reason_code="judge_invalid_output",
            reason="Judge returned duplicate candidate ids within a bucket.",
        )

    referenced_ids: set[str] = set()
    if normalized_primary_candidate_id is not None:
        referenced_ids.add(normalized_primary_candidate_id)
    for candidate_id in [*related_ids.ids, *rejected_ids.ids]:
        if candidate_id in referenced_ids:
            return JudgeValidationResult(
                selected_candidate=None,
                related_candidates=[],
                rejected_candidates=[],
                reason_code="validator_rejected",
                reason="Judge referenced the same candidate across multiple buckets.",
            )
        referenced_ids.add(candidate_id)

    if (
        normalized_primary_candidate_id is not None
        and normalized_primary_candidate_id not in shortlist_by_id
    ):
        return JudgeValidationResult(
            selected_candidate=None,
            related_candidates=[],
            rejected_candidates=[],
            reason_code="validator_rejected",
            reason="Judge selected a candidate outside the bounded shortlist.",
        )

    for candidate_id in [*related_ids.ids, *rejected_ids.ids]:
        if candidate_id not in shortlist_by_id:
            return JudgeValidationResult(
                selected_candidate=None,
                related_candidates=[],
                rejected_candidates=[],
                reason_code="validator_rejected",
                reason="Judge referenced a candidate outside the bounded shortlist.",
            )

    selected_candidate = (
        shortlist_by_id.get(normalized_primary_candidate_id)
        if normalized_primary_candidate_id is not None
        else None
    )
    if selected_candidate is not None and not _is_candidate_selectable(
        selected_candidate,
        validate_candidate,
    ):
        return JudgeValidationResult(
            selected_candidate=None,
            related_candidates=[],
            rejected_candidates=[],
            reason_code="validator_rejected",
            reason=(
                "Judge selected a candidate that failed deterministic contributor validation."
            ),
        )

    return JudgeValidationResult(
        selected_candidate=selected_candidate,
        related_candidates=[
            shortlist_by_id[candidate_id]
            for candidate_id in related_ids.ids
            if candidate_id in shortlist_by_id
        ],
        rejected_candidates=[
            shortlist_by_id[candidate_id]
            for candidate_id in rejected_ids.ids
            if candidate_id in shortlist_by_id
        ],
    )


async def _evaluate_judge(
    *,
    judge: ContributorSearchJudge,
    input: ContributorSearchJudgeInput,
    context: ContributorSearchJudgeContext,
    timeout_ms: int | None,
) -> ContributorSearchJudgeResult:
    if timeout_ms is None or timeout_ms <= 0:
        return await judge.evaluate(input, context)

    try:
        return await asyncio.wait_for(
            judge.evaluate(input, context),
            timeout=timeout_ms / 1000,
        )
    except asyncio.TimeoutError as error:
        raise ContributorSearchTimeoutError() from error


def _build_fallback_resolution(
    *,
    intents: Sequence[SearchIntent],
    candidates: Sequence[SearchCandidate],
    valid_shortlist: Sequence[SearchCandidate],
    reason_code: ContributorSearchDegradedReasonCode,
    reason: str,
    config: ContributorSearchResolvedConfig,
    judge_applied: bool,
    judge_usage: ContributorSearchJudgeUsage | None,
    validator_status: ContributorSearchValidatorStatus,
    validator_reason_code: str | None,
    validator_reason: str | None,
) -> ContributorSearchResolution:
    if len(valid_shortlist) == 0:
        return _build_resolution(
            intents=intents,
            candidates=candidates,
            shortlist=[],
            selected_candidate=None,
            related_candidates=[],
            rejected_candidates=[],
            outcome="capability_miss",
            confidence="low",
            reason=reason,
            degraded=ContributorSearchDegradedOutcome(
                reasonCode="no_viable_candidates",
                message=_truncate_reason(reason),
            ),
            judge_snapshot=_build_judge_snapshot(
                config=config,
                applied=judge_applied,
                usage=judge_usage,
            ),
            trace=_build_trace_summary(
                used_deterministic_fallback=True,
                validator_status=validator_status,
                validator_reason_code=validator_reason_code,
                validator_reason=validator_reason,
            ),
        )

    if len(valid_shortlist) == 1 and config.degraded_outcome_policy == "allow_low_confidence_selected":
        return _build_resolution(
            intents=intents,
            candidates=candidates,
            shortlist=valid_shortlist,
            selected_candidate=valid_shortlist[0],
            related_candidates=[],
            rejected_candidates=[],
            outcome="selected",
            confidence="low",
            reason=reason,
            degraded=ContributorSearchDegradedOutcome(
                reasonCode=reason_code,
                message=_truncate_reason(reason),
            ),
            judge_snapshot=_build_judge_snapshot(
                config=config,
                applied=judge_applied,
                usage=judge_usage,
            ),
            trace=_build_trace_summary(
                used_deterministic_fallback=True,
                validator_status=validator_status,
                validator_reason_code=validator_reason_code,
                validator_reason=validator_reason,
            ),
        )

    return _build_resolution(
        intents=intents,
        candidates=candidates,
        shortlist=valid_shortlist,
        selected_candidate=None,
        related_candidates=[],
        rejected_candidates=[],
        outcome="shortlist_only",
        confidence="low",
        reason=reason,
        degraded=ContributorSearchDegradedOutcome(
            reasonCode=reason_code,
            message=_truncate_reason(reason),
        ),
        judge_snapshot=_build_judge_snapshot(
            config=config,
            applied=judge_applied,
            usage=judge_usage,
        ),
        trace=_build_trace_summary(
            used_deterministic_fallback=True,
            validator_status=validator_status,
            validator_reason_code=validator_reason_code,
            validator_reason=validator_reason,
        ),
    )


def create_search_intent(
    *,
    raw_request: str,
    query: str,
    intent_id: str | None = None,
    clause: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SearchIntent:
    trimmed_raw_request = raw_request.strip()
    normalized_query = _normalize_string(query) or trimmed_raw_request
    fallback_intent_id = re.sub(r"[^a-z0-9]+", "-", normalized_query.lower()).strip("-")[:80]
    return SearchIntent(
        intentId=intent_id or fallback_intent_id or "search-intent",
        rawRequest=trimmed_raw_request,
        query=normalized_query,
        clause=_normalize_string(clause),
        metadata=metadata,
    )


def dedupe_search_candidates(candidates: Sequence[SearchCandidate]) -> list[SearchCandidate]:
    deduped: dict[str, SearchCandidate] = {}
    for candidate in candidates:
        normalized_candidate_id = _normalize_string(candidate.candidate_id)
        if normalized_candidate_id is None:
            continue

        normalized_candidate = SearchCandidate(
            candidateId=normalized_candidate_id,
            title=candidate.title.strip(),
            description=_normalize_string(candidate.description),
            rawIds=candidate.raw_ids or {},
            rankFeatures=candidate.rank_features or {},
            provenance=list(candidate.provenance or []),
            metadata=candidate.metadata or {},
        )
        existing = deduped.get(normalized_candidate_id)
        deduped[normalized_candidate_id] = (
            _merge_candidates(existing, normalized_candidate)
            if existing is not None
            else normalized_candidate
        )
    return list(deduped.values())


def build_search_shortlist(
    candidates: Sequence[SearchCandidate],
    max_shortlist_size: int = DEFAULT_MAX_SHORTLIST_SIZE,
) -> SearchShortlist:
    capped_size = max(1, int(max_shortlist_size))
    return SearchShortlist(
        maxSize=capped_size,
        candidates=dedupe_search_candidates(candidates)[:capped_size],
    )


def merge_contributor_search_config(
    *configs: ContributorSearchConfig | None,
) -> ContributorSearchResolvedConfig:
    resolved = ContributorSearchResolvedConfig(
        provider=None,
        model=None,
        timeoutMs=None,
        budgetUsd=None,
        disableJudge=False,
        degradedOutcomePolicy=DEFAULT_DEGRADED_OUTCOME_POLICY,
        maxShortlistSize=DEFAULT_MAX_SHORTLIST_SIZE,
    )

    for config in configs:
        if config is None:
            continue

        if "provider" in config.model_fields_set:
            resolved.provider = _normalize_string(config.provider)
        if "model" in config.model_fields_set:
            resolved.model = _normalize_string(config.model)
        if "timeout_ms" in config.model_fields_set:
            resolved.timeout_ms = (
                max(0, int(config.timeout_ms))
                if isinstance(config.timeout_ms, int) and not isinstance(config.timeout_ms, bool)
                else None
            )
        if "budget_usd" in config.model_fields_set:
            resolved.budget_usd = _normalize_string(config.budget_usd)
        if "disable_judge" in config.model_fields_set and isinstance(
            config.disable_judge,
            bool,
        ):
            resolved.disable_judge = config.disable_judge
        if config.degraded_outcome_policy in (
            "return_shortlist",
            "allow_low_confidence_selected",
        ):
            resolved.degraded_outcome_policy = config.degraded_outcome_policy
        if isinstance(config.max_shortlist_size, int) and not isinstance(
            config.max_shortlist_size,
            bool,
        ):
            resolved.max_shortlist_size = max(1, int(config.max_shortlist_size))

    return resolved


def attach_contributor_search_metadata(
    data: dict[str, Any],
    resolution: ContributorSearchResolution,
) -> dict[str, Any]:
    return {
        **data,
        "searchMetadata": resolution.search_metadata.model_dump(by_alias=True),
    }


def _allow_all_candidates(_candidate: SearchCandidate) -> bool:
    return True


async def resolve_contributor_search(
    params: ResolveContributorSearchParams,
) -> ContributorSearchResolution:
    config = merge_contributor_search_config(
        params.helper_config,
        params.contributor_config,
        params.overrides,
    )
    candidates = dedupe_search_candidates(params.candidates)
    shortlist = build_search_shortlist(
        candidates,
        config.max_shortlist_size,
    ).candidates
    validate_candidate: CandidateValidator = (
        params.is_candidate_valid or _allow_all_candidates
    )
    valid_shortlist = [
        candidate
        for candidate in shortlist
        if _is_candidate_selectable(candidate, validate_candidate)
    ]
    judge = params.judge

    if len(shortlist) == 0:
        return _build_fallback_resolution(
            intents=params.intents,
            candidates=candidates,
            valid_shortlist=valid_shortlist,
            reason_code="no_viable_candidates",
            reason="No viable candidates survived deterministic gathering before judging.",
            config=config,
            judge_applied=False,
            judge_usage=None,
            validator_status="not_run",
            validator_reason_code=None,
            validator_reason=None,
        )

    if config.disable_judge or judge is None:
        return _build_fallback_resolution(
            intents=params.intents,
            candidates=candidates,
            valid_shortlist=valid_shortlist,
            reason_code="judge_missing" if judge is None else "judge_disabled",
            reason=(
                "No contributor search judge was configured for this resolution."
                if judge is None
                else "Contributor search judge was disabled for this resolution."
            ),
            config=config,
            judge_applied=False,
            judge_usage=None,
            validator_status="not_run",
            validator_reason_code=None,
            validator_reason=None,
        )

    judge_input = ContributorSearchJudgeInput(
        rawRequest=params.raw_request,
        intents=params.intents,
        shortlist=SearchShortlist(
            maxSize=config.max_shortlist_size,
            candidates=shortlist,
        ),
        instructions=params.instructions,
        policy=config,
    )
    judge_context = ContributorSearchJudgeContext(
        provider=config.provider,
        model=config.model,
        timeoutMs=config.timeout_ms,
        budgetUsd=config.budget_usd,
        traceLabel=params.trace_label,
    )

    judge_usage: ContributorSearchJudgeUsage | None = None
    try:
        judge_result = await _evaluate_judge(
            judge=judge,
            input=judge_input,
            context=judge_context,
            timeout_ms=config.timeout_ms,
        )
        judge_usage = judge_result.usage
        validation = _validate_judge_selection(
            shortlist=shortlist,
            primary_candidate_id=judge_result.primary_candidate_id,
            related_candidate_ids=judge_result.related_candidate_ids,
            rejected_candidate_ids=judge_result.rejected_candidate_ids,
            validate_candidate=validate_candidate,
        )

        if not validation.ok:
            return _build_fallback_resolution(
                intents=params.intents,
                candidates=candidates,
                valid_shortlist=valid_shortlist,
                reason_code=validation.reason_code or "validator_rejected",
                reason=validation.reason or "Judge validation rejected the candidate set.",
                config=config,
                judge_applied=True,
                judge_usage=judge_usage,
                validator_status="rejected",
                validator_reason_code=validation.reason_code,
                validator_reason=validation.reason,
            )

        if validation.selected_candidate is None:
            return _build_fallback_resolution(
                intents=params.intents,
                candidates=candidates,
                valid_shortlist=valid_shortlist,
                reason_code="ambiguous_shortlist",
                reason=_normalize_string(judge_result.reason)
                or "Judge declined to select a single grounded candidate.",
                config=config,
                judge_applied=True,
                judge_usage=judge_usage,
                validator_status="accepted",
                validator_reason_code=None,
                validator_reason=None,
            )

        return _build_resolution(
            intents=params.intents,
            candidates=candidates,
            shortlist=shortlist,
            selected_candidate=validation.selected_candidate,
            related_candidates=validation.related_candidates,
            rejected_candidates=validation.rejected_candidates,
            outcome="selected",
            confidence=judge_result.confidence,
            reason=_normalize_string(judge_result.reason) or "Judge selected a candidate.",
            degraded=None,
            judge_snapshot=_build_judge_snapshot(
                config=config,
                applied=True,
                usage=judge_usage,
            ),
            trace=_build_trace_summary(
                used_deterministic_fallback=False,
                validator_status="accepted",
                validator_reason_code=None,
                validator_reason=None,
            ),
        )
    except Exception as error:
        if isinstance(error, ContributorSearchBudgetExceededError):
            reason_code: ContributorSearchDegradedReasonCode = "judge_budget_exceeded"
        elif isinstance(error, ContributorSearchTimeoutError):
            reason_code = "judge_timeout"
        else:
            reason_code = "judge_error"
        reason = str(error) or "Contributor search judge failed with a non-Error value."
        return _build_fallback_resolution(
            intents=params.intents,
            candidates=candidates,
            valid_shortlist=valid_shortlist,
            reason_code=reason_code,
            reason=reason,
            config=config,
            judge_applied=True,
            judge_usage=judge_usage,
            validator_status="not_run",
            validator_reason_code=None,
            validator_reason=None,
        )
