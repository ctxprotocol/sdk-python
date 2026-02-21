"""
Contract tests for the Discovery resource with surface-aware filters.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock, patch

from ctxprotocol.client.client import ContextClient


def _parsed_query(endpoint: str) -> dict[str, list[str]]:
    parsed = urlparse(endpoint)
    return parse_qs(parsed.query)


async def test_search_legacy_signature() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"tools": [], "mode": "query", "query": "gas prices", "count": 0}
        await client.discovery.search("gas prices", limit=8)

    endpoint = mock_fetch.call_args[0][0]
    query = _parsed_query(endpoint)
    assert query.get("q") == ["gas prices"]
    assert query.get("limit") == ["8"]
    await client.close()


async def test_search_forwards_surface_filters() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"tools": [], "mode": "execute", "query": "prices", "count": 0}
        await client.discovery.search(
            "prices",
            limit=4,
            mode="execute",
            surface="execute",
            query_eligible=False,
            require_execute_pricing=True,
            exclude_latency_classes=["streaming", "slow"],
            exclude_slow=True,
        )

    endpoint = mock_fetch.call_args[0][0]
    query = _parsed_query(endpoint)
    assert query.get("q") == ["prices"]
    assert query.get("limit") == ["4"]
    assert query.get("mode") == ["execute"]
    assert query.get("surface") == ["execute"]
    assert query.get("queryEligible") == ["false"]
    assert query.get("requireExecutePricing") == ["true"]
    assert query.get("excludeLatency") == ["streaming,slow"]
    assert query.get("excludeSlow") == ["true"]
    await client.close()


async def test_get_featured_accepts_execute_filters() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with patch.object(client, "fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"tools": [], "mode": "execute", "query": "", "count": 0}
        await client.discovery.get_featured(
            3,
            mode="execute",
            require_execute_pricing=True,
        )

    endpoint = mock_fetch.call_args[0][0]
    query = _parsed_query(endpoint)
    assert query.get("limit") == ["3"]
    assert query.get("mode") == ["execute"]
    assert query.get("requireExecutePricing") == ["true"]
    await client.close()
