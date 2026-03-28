from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

CONTRIBUTOR_SEARCH_METADATA_VERSION = "ctx-contributor-search/v1"
CONTRIBUTOR_SEARCH_VALIDATION_VERSION = "ctx-contributor-search-validation/v1"

ContributorSearchOutcome: TypeAlias = Literal[
    "selected",
    "shortlist_only",
    "capability_miss",
]
ContributorSearchConfidence: TypeAlias = Literal["high", "medium", "low"]
ContributorSearchDegradedOutcomePolicy: TypeAlias = Literal[
    "return_shortlist",
    "allow_low_confidence_selected",
]
ContributorSearchDegradedReasonCode: TypeAlias = Literal[
    "judge_disabled",
    "judge_missing",
    "judge_timeout",
    "judge_budget_exceeded",
    "judge_invalid_output",
    "judge_error",
    "validator_rejected",
    "ambiguous_shortlist",
    "no_viable_candidates",
]
ContributorSearchValidationCaseKind: TypeAlias = Literal[
    "named_regression",
    "generic_overlap",
    "still_ambiguous",
    "capability_miss",
]
ContributorSearchValidatorStatus: TypeAlias = Literal[
    "accepted",
    "rejected",
    "not_run",
]
SearchRankFeatureValue: TypeAlias = bool | int | float | str | None
CandidateValidator: TypeAlias = Callable[["SearchCandidate"], bool]


class ContributorSearchModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class SearchIntent(ContributorSearchModel):
    intent_id: str = Field(..., alias="intentId")
    raw_request: str = Field(..., alias="rawRequest")
    query: str
    clause: str | None = None
    metadata: dict[str, Any] | None = None


class SearchCandidateProvenance(ContributorSearchModel):
    source: str
    query: str
    rank: int | None = None
    fetched_at: str | None = Field(default=None, alias="fetchedAt")
    metadata: dict[str, Any] | None = None


class SearchCandidate(ContributorSearchModel):
    candidate_id: str = Field(..., alias="candidateId")
    title: str
    description: str | None = None
    raw_ids: dict[str, str] | None = Field(default=None, alias="rawIds")
    rank_features: dict[str, SearchRankFeatureValue] | None = Field(
        default=None,
        alias="rankFeatures",
    )
    provenance: list[SearchCandidateProvenance]
    metadata: dict[str, Any] | None = None


class SearchShortlist(ContributorSearchModel):
    max_size: int = Field(..., alias="maxSize")
    candidates: list[SearchCandidate]


class ContributorSearchConfig(ContributorSearchModel):
    provider: str | None = None
    model: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    budget_usd: str | None = Field(default=None, alias="budgetUsd")
    disable_judge: bool | None = Field(default=None, alias="disableJudge")
    degraded_outcome_policy: ContributorSearchDegradedOutcomePolicy | None = Field(
        default=None,
        alias="degradedOutcomePolicy",
    )
    max_shortlist_size: int | None = Field(default=None, alias="maxShortlistSize")


class ContributorSearchResolvedConfig(ContributorSearchModel):
    provider: str | None = None
    model: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    budget_usd: str | None = Field(default=None, alias="budgetUsd")
    disable_judge: bool = Field(..., alias="disableJudge")
    degraded_outcome_policy: ContributorSearchDegradedOutcomePolicy = Field(
        ...,
        alias="degradedOutcomePolicy",
    )
    max_shortlist_size: int = Field(..., alias="maxShortlistSize")


class ContributorSearchJudgeUsage(ContributorSearchModel):
    prompt_tokens: int | None = Field(default=None, alias="promptTokens")
    completion_tokens: int | None = Field(default=None, alias="completionTokens")
    total_tokens: int | None = Field(default=None, alias="totalTokens")
    cost_usd: str | None = Field(default=None, alias="costUsd")
    latency_ms: int | None = Field(default=None, alias="latencyMs")


class ContributorSearchJudgeInput(ContributorSearchModel):
    raw_request: str = Field(..., alias="rawRequest")
    intents: list[SearchIntent]
    shortlist: SearchShortlist
    instructions: str | None = None
    policy: ContributorSearchResolvedConfig


class ContributorSearchJudgeContext(ContributorSearchModel):
    provider: str | None = None
    model: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    budget_usd: str | None = Field(default=None, alias="budgetUsd")
    trace_label: str | None = Field(default=None, alias="traceLabel")


class ContributorSearchJudgeResult(ContributorSearchModel):
    primary_candidate_id: str | None = Field(default=None, alias="primaryCandidateId")
    related_candidate_ids: list[str] = Field(default_factory=list, alias="relatedCandidateIds")
    rejected_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="rejectedCandidateIds",
    )
    confidence: ContributorSearchConfidence
    reason: str
    usage: ContributorSearchJudgeUsage | None = None


class ContributorSearchJudge(Protocol):
    async def evaluate(
        self,
        input: ContributorSearchJudgeInput,
        context: ContributorSearchJudgeContext,
    ) -> ContributorSearchJudgeResult: ...


class ContributorSearchDegradedOutcome(ContributorSearchModel):
    reason_code: ContributorSearchDegradedReasonCode = Field(..., alias="reasonCode")
    message: str


class ContributorSearchMetadataSource(ContributorSearchModel):
    source: str
    query: str
    candidate_count: int = Field(..., alias="candidateCount")


class ContributorSearchJudgeSnapshot(ContributorSearchModel):
    provider: str | None = None
    model: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    budget_usd: str | None = Field(default=None, alias="budgetUsd")
    disabled: bool
    applied: bool
    usage: ContributorSearchJudgeUsage | None = None


class ContributorSearchTraceSummary(ContributorSearchModel):
    used_deterministic_fallback: bool = Field(
        ...,
        alias="usedDeterministicFallback",
    )
    validator_status: ContributorSearchValidatorStatus = Field(
        ...,
        alias="validatorStatus",
    )
    validator_reason_code: str | None = Field(default=None, alias="validatorReasonCode")
    validator_reason: str | None = Field(default=None, alias="validatorReason")


class ContributorSearchMetadata(ContributorSearchModel):
    version: str
    outcome: ContributorSearchOutcome
    confidence: ContributorSearchConfidence
    selected_candidate_id: str | None = Field(default=None, alias="selectedCandidateId")
    shortlist_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="shortlistCandidateIds",
    )
    related_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="relatedCandidateIds",
    )
    rejected_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="rejectedCandidateIds",
    )
    candidate_count: int = Field(..., alias="candidateCount")
    shortlist_count: int = Field(..., alias="shortlistCount")
    intent_queries: list[str] = Field(default_factory=list, alias="intentQueries")
    degraded: ContributorSearchDegradedOutcome | None = None
    judge: ContributorSearchJudgeSnapshot
    provenance: list[ContributorSearchMetadataSource]
    trace: ContributorSearchTraceSummary


class ContributorSearchTraceRecord(ContributorSearchModel):
    tool_id: str | None = Field(default=None, alias="toolId")
    tool_name: str | None = Field(default=None, alias="toolName")
    timestamp_ms: int | None = Field(default=None, alias="timestampMs")
    search_metadata: ContributorSearchMetadata = Field(..., alias="searchMetadata")


class ContributorSearchResolution(ContributorSearchModel):
    outcome: ContributorSearchOutcome
    selected_candidate: SearchCandidate | None = Field(
        default=None,
        alias="selectedCandidate",
    )
    shortlist: list[SearchCandidate]
    related_candidates: list[SearchCandidate] = Field(
        default_factory=list,
        alias="relatedCandidates",
    )
    rejected_candidates: list[SearchCandidate] = Field(
        default_factory=list,
        alias="rejectedCandidates",
    )
    confidence: ContributorSearchConfidence
    reason: str
    degraded: ContributorSearchDegradedOutcome | None = None
    search_metadata: ContributorSearchMetadata = Field(..., alias="searchMetadata")


class ContributorSearchValidationExpectation(ContributorSearchModel):
    outcome: ContributorSearchOutcome
    selected_candidate_id: str | None = Field(default=None, alias="selectedCandidateId")
    degraded_reason_code: ContributorSearchDegradedReasonCode | None = Field(
        default=None,
        alias="degradedReasonCode",
    )


class ContributorSearchValidationResolution(ContributorSearchModel):
    outcome: ContributorSearchOutcome
    selected_candidate_id: str | None = Field(default=None, alias="selectedCandidateId")
    shortlist_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="shortlistCandidateIds",
    )
    related_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="relatedCandidateIds",
    )
    rejected_candidate_ids: list[str] = Field(
        default_factory=list,
        alias="rejectedCandidateIds",
    )
    confidence: ContributorSearchConfidence
    reason: str
    degraded_reason_code: ContributorSearchDegradedReasonCode | None = Field(
        default=None,
        alias="degradedReasonCode",
    )


class ContributorSearchValidationArtifact(ContributorSearchModel):
    version: str
    generated_at: str = Field(..., alias="generatedAt")
    case_id: str = Field(..., alias="caseId")
    case_kind: ContributorSearchValidationCaseKind = Field(..., alias="caseKind")
    raw_request: str = Field(..., alias="rawRequest")
    intents: list[SearchIntent]
    candidates: list[SearchCandidate]
    resolution: ContributorSearchValidationResolution
    search_metadata: ContributorSearchMetadata = Field(..., alias="searchMetadata")
    expectation: ContributorSearchValidationExpectation | None = None


@dataclass(slots=True)
class ResolveContributorSearchParams:
    raw_request: str
    intents: list[SearchIntent]
    candidates: list[SearchCandidate]
    judge: ContributorSearchJudge | None = None
    helper_config: ContributorSearchConfig | None = None
    contributor_config: ContributorSearchConfig | None = None
    overrides: ContributorSearchConfig | None = None
    instructions: str | None = None
    is_candidate_valid: CandidateValidator | None = None
    trace_label: str | None = None


class ContributorSearchBudgetExceededError(Exception):
    def __init__(self, message: str = "Contributor search budget exceeded") -> None:
        super().__init__(message)
