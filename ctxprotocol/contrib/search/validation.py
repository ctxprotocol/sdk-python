from __future__ import annotations

from datetime import datetime, timezone

from ctxprotocol.contrib.search.types import (
    CONTRIBUTOR_SEARCH_VALIDATION_VERSION,
    ContributorSearchResolution,
    ContributorSearchValidationArtifact,
    ContributorSearchValidationCaseKind,
    ContributorSearchValidationExpectation,
    ContributorSearchValidationResolution,
    SearchCandidate,
    SearchIntent,
)


def build_contributor_search_validation_artifact(
    *,
    case_id: str,
    case_kind: ContributorSearchValidationCaseKind,
    raw_request: str,
    intents: list[SearchIntent],
    candidates: list[SearchCandidate],
    resolution: ContributorSearchResolution,
    expectation: ContributorSearchValidationExpectation | None = None,
    generated_at: str | None = None,
) -> ContributorSearchValidationArtifact:
    return ContributorSearchValidationArtifact(
        version=CONTRIBUTOR_SEARCH_VALIDATION_VERSION,
        generatedAt=generated_at or datetime.now(timezone.utc).isoformat(),
        caseId=case_id,
        caseKind=case_kind,
        rawRequest=raw_request,
        intents=intents,
        candidates=candidates,
        resolution=ContributorSearchValidationResolution(
            outcome=resolution.outcome,
            selectedCandidateId=(
                resolution.selected_candidate.candidate_id
                if resolution.selected_candidate is not None
                else None
            ),
            shortlistCandidateIds=[
                candidate.candidate_id for candidate in resolution.shortlist
            ],
            relatedCandidateIds=[
                candidate.candidate_id for candidate in resolution.related_candidates
            ],
            rejectedCandidateIds=[
                candidate.candidate_id for candidate in resolution.rejected_candidates
            ],
            confidence=resolution.confidence,
            reason=resolution.reason,
            degradedReasonCode=(
                resolution.degraded.reason_code
                if resolution.degraded is not None
                else None
            ),
        ),
        searchMetadata=resolution.search_metadata,
        expectation=expectation,
    )
