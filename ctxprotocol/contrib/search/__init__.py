from __future__ import annotations

from typing import Any

from ctxprotocol.contrib.search.core import (
    attach_contributor_search_metadata,
    build_search_shortlist,
    create_search_intent,
    dedupe_search_candidates,
    merge_contributor_search_config,
    resolve_contributor_search,
)
from ctxprotocol.contrib.search.types import (
    CONTRIBUTOR_SEARCH_METADATA_VERSION,
    CONTRIBUTOR_SEARCH_VALIDATION_VERSION,
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
    ContributorSearchTraceRecord,
    ContributorSearchTraceSummary,
    ContributorSearchValidationArtifact,
    ContributorSearchValidationCaseKind,
    ContributorSearchValidationExpectation,
    ContributorSearchValidationResolution,
    ContributorSearchValidatorStatus,
    ResolveContributorSearchParams,
    SearchCandidate,
    SearchCandidateProvenance,
    SearchIntent,
    SearchShortlist,
)
from ctxprotocol.contrib.search.validation import (
    build_contributor_search_validation_artifact,
)


def extract_contributor_search_metadata(result: Any) -> ContributorSearchMetadata | None:
    from ctxprotocol.contrib.search.trace import (
        extract_contributor_search_metadata as _extract_contributor_search_metadata,
    )

    return _extract_contributor_search_metadata(result)


def extract_contributor_searches_from_developer_trace(
    trace: Any,
) -> list[ContributorSearchTraceRecord]:
    from ctxprotocol.contrib.search.trace import (
        extract_contributor_searches_from_developer_trace as _extract_contributor_searches_from_developer_trace,
    )

    return _extract_contributor_searches_from_developer_trace(trace)

__all__ = [
    "attach_contributor_search_metadata",
    "build_contributor_search_validation_artifact",
    "build_search_shortlist",
    "create_search_intent",
    "dedupe_search_candidates",
    "extract_contributor_search_metadata",
    "extract_contributor_searches_from_developer_trace",
    "merge_contributor_search_config",
    "resolve_contributor_search",
    "ContributorSearchBudgetExceededError",
    "CONTRIBUTOR_SEARCH_METADATA_VERSION",
    "CONTRIBUTOR_SEARCH_VALIDATION_VERSION",
    "ContributorSearchConfig",
    "ContributorSearchConfidence",
    "ContributorSearchDegradedOutcome",
    "ContributorSearchDegradedOutcomePolicy",
    "ContributorSearchDegradedReasonCode",
    "ContributorSearchJudge",
    "ContributorSearchJudgeContext",
    "ContributorSearchJudgeInput",
    "ContributorSearchJudgeResult",
    "ContributorSearchJudgeSnapshot",
    "ContributorSearchJudgeUsage",
    "ContributorSearchMetadata",
    "ContributorSearchMetadataSource",
    "ContributorSearchOutcome",
    "ContributorSearchResolution",
    "ContributorSearchResolvedConfig",
    "ContributorSearchTraceRecord",
    "ContributorSearchTraceSummary",
    "ContributorSearchValidationArtifact",
    "ContributorSearchValidationCaseKind",
    "ContributorSearchValidationExpectation",
    "ContributorSearchValidationResolution",
    "ContributorSearchValidatorStatus",
    "ResolveContributorSearchParams",
    "SearchCandidate",
    "SearchCandidateProvenance",
    "SearchIntent",
    "SearchShortlist",
]
