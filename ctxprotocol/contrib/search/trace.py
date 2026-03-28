from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ctxprotocol.contrib.search.types import (
    CONTRIBUTOR_SEARCH_METADATA_VERSION,
    ContributorSearchMetadata,
    ContributorSearchTraceRecord,
)

if TYPE_CHECKING:
    from ctxprotocol.client.types import QueryDeveloperTrace


def _parse_contributor_search_metadata(value: Any) -> ContributorSearchMetadata | None:
    if isinstance(value, ContributorSearchMetadata):
        return value
    if value is None:
        return None

    try:
        metadata = ContributorSearchMetadata.model_validate(value)
    except ValidationError:
        return None

    if metadata.version != CONTRIBUTOR_SEARCH_METADATA_VERSION:
        return None
    return metadata


def _extract_metadata_from_unknown(value: Any) -> ContributorSearchMetadata | None:
    metadata = _parse_contributor_search_metadata(value)
    if metadata is not None:
        return metadata

    if isinstance(value, dict):
        return _parse_contributor_search_metadata(value.get("searchMetadata"))

    return None


def extract_contributor_search_metadata(result: Any) -> ContributorSearchMetadata | None:
    return _extract_metadata_from_unknown(result)


def extract_contributor_searches_from_developer_trace(
    trace: QueryDeveloperTrace | None,
) -> list[ContributorSearchTraceRecord]:
    diagnostics = trace.diagnostics if trace is not None else None
    if diagnostics is not None and diagnostics.contributor_searches:
        return diagnostics.contributor_searches

    extracted: list[ContributorSearchTraceRecord] = []
    for step in trace.timeline or [] if trace is not None else []:
        metadata = step.metadata
        if not isinstance(metadata, dict):
            continue

        direct_metadata = _extract_metadata_from_unknown(metadata.get("contributorSearch"))
        result_metadata = _extract_metadata_from_unknown(metadata.get("result"))
        search_metadata = direct_metadata or result_metadata
        if search_metadata is None:
            continue

        extracted.append(
            ContributorSearchTraceRecord(
                toolId=step.tool.id if step.tool is not None else None,
                toolName=step.tool.name if step.tool is not None else None,
                timestampMs=step.timestamp_ms,
                searchMetadata=search_metadata,
            )
        )

    return extracted
