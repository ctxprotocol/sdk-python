from __future__ import annotations

import pytest

from ctxprotocol.client.types import QueryDeveloperTrace
from ctxprotocol.contrib.search import (
    ContributorSearchBudgetExceededError,
    ContributorSearchConfig,
    ResolveContributorSearchParams,
    attach_contributor_search_metadata,
    extract_contributor_search_metadata,
    extract_contributor_searches_from_developer_trace,
    resolve_contributor_search,
)
from ctxprotocol.contrib.search.types import (
    ContributorSearchJudgeContext,
    ContributorSearchJudgeInput,
    ContributorSearchJudgeResult,
)
from tests.contrib_search_validation_cases import (
    TARIFFS_PROMPT,
    VALIDATION_CASES,
    ValidationCase,
    build_validation_artifact_for_case,
    read_validation_artifact,
    resolve_validation_case,
)


def _get_validation_case(case_id: str) -> ValidationCase:
    for validation_case in VALIDATION_CASES:
        if validation_case.case_id == case_id:
            return validation_case
    raise AssertionError(f"Missing validation case: {case_id}")


class SpyJudge:
    def __init__(self, result: ContributorSearchJudgeResult) -> None:
        self.result = result
        self.calls: list[
            tuple[ContributorSearchJudgeInput, ContributorSearchJudgeContext]
        ] = []

    async def evaluate(
        self,
        input: ContributorSearchJudgeInput,
        context: ContributorSearchJudgeContext,
    ) -> ContributorSearchJudgeResult:
        self.calls.append((input, context))
        return self.result


class BudgetExceededJudge:
    async def evaluate(
        self,
        input: ContributorSearchJudgeInput,
        context: ContributorSearchJudgeContext,
    ) -> ContributorSearchJudgeResult:
        del input, context
        raise ContributorSearchBudgetExceededError(
            "Budget exhausted while ranking the single surviving candidate."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "validation_case",
    VALIDATION_CASES,
    ids=[validation_case.case_id for validation_case in VALIDATION_CASES],
)
async def test_validation_cases_match_saved_artifacts(
    validation_case: ValidationCase,
) -> None:
    artifact, resolution = await build_validation_artifact_for_case(validation_case)

    assert resolution.outcome == validation_case.expectation.outcome
    assert (
        resolution.selected_candidate.candidate_id
        if resolution.selected_candidate is not None
        else None
    ) == validation_case.expectation.selected_candidate_id
    assert (
        resolution.degraded.reason_code if resolution.degraded is not None else None
    ) == validation_case.expectation.degraded_reason_code
    assert artifact.model_dump(by_alias=True) == read_validation_artifact(
        validation_case.artifact_path
    )


@pytest.mark.asyncio
async def test_resolve_contributor_search_passes_merged_config_to_judge_context() -> None:
    validation_case = _get_validation_case("generic-overlap-best-match")
    if validation_case.judge_result is None:
        raise AssertionError("generic overlap validation case requires a judge result")

    judge = SpyJudge(validation_case.judge_result)
    await resolve_contributor_search(
        ResolveContributorSearchParams(
            raw_request=validation_case.raw_request,
            intents=validation_case.intents,
            candidates=validation_case.candidates,
            judge=judge,
            helper_config=validation_case.helper_config,
            contributor_config=validation_case.contributor_config,
            overrides=validation_case.overrides,
            trace_label=validation_case.trace_label,
        )
    )

    assert len(judge.calls) == 1
    input, context = judge.calls[0]
    assert input.policy.provider == "openrouter"
    assert input.policy.model == "glm-turbo-model"
    assert input.policy.timeout_ms == 900
    assert input.policy.budget_usd == "0.002500"
    assert input.policy.max_shortlist_size == 3
    assert context.provider == "openrouter"
    assert context.model == "glm-turbo-model"
    assert context.timeout_ms == 900
    assert context.budget_usd == "0.002500"
    assert context.trace_label == "generic-overlap-parity"


@pytest.mark.asyncio
async def test_resolve_contributor_search_allows_low_confidence_selected_on_budget_exhaustion() -> None:
    tariffs_case = _get_validation_case("kalshi-supreme-court-tariffs")

    resolution = await resolve_contributor_search(
        ResolveContributorSearchParams(
            raw_request=TARIFFS_PROMPT,
            intents=tariffs_case.intents,
            candidates=[tariffs_case.candidates[0]],
            judge=BudgetExceededJudge(),
            contributor_config=ContributorSearchConfig(
                degraded_outcome_policy="allow_low_confidence_selected"
            ),
        )
    )

    assert resolution.outcome == "selected"
    assert resolution.confidence == "low"
    assert resolution.selected_candidate is not None
    assert resolution.selected_candidate.candidate_id == "kalshi-kxdjtvostariffs"
    assert resolution.degraded is not None
    assert resolution.degraded.reason_code == "judge_budget_exceeded"
    assert resolution.search_metadata.judge.applied is True


@pytest.mark.asyncio
async def test_extract_contributor_search_metadata_from_wrapped_results_and_trace() -> None:
    validation_case = _get_validation_case("generic-overlap-best-match")
    resolution = await resolve_validation_case(validation_case)
    wrapped_result = attach_contributor_search_metadata({"ok": True}, resolution)
    trace = QueryDeveloperTrace.model_validate(
        {
            "timeline": [
                {
                    "tool": {"id": "tool-1", "name": "Tariff Markets"},
                    "timestampMs": 123,
                    "metadata": {"result": wrapped_result},
                }
            ]
        }
    )

    extracted = extract_contributor_search_metadata(wrapped_result)
    trace_records = extract_contributor_searches_from_developer_trace(trace)

    assert extracted is not None
    assert extracted.selected_candidate_id == "generic-scotus-tariffs"
    assert len(trace_records) == 1
    assert trace_records[0].tool_id == "tool-1"
    assert trace_records[0].tool_name == "Tariff Markets"
    assert trace_records[0].timestamp_ms == 123
    assert trace_records[0].search_metadata == resolution.search_metadata
