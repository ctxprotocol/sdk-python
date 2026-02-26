"""
Tests for low-level HTTP client retry and timeout behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ctxprotocol.client.client import ContextClient
from ctxprotocol.client.types import ContextError


def _response(
    status_code: int,
    body: dict[str, object],
    method: str = "POST",
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request(method, "https://www.ctxprotocol.com/test"),
    )


async def test_fetch_retries_on_5xx_then_succeeds() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = [
            _response(502, {"error": "Bad gateway"}),
            _response(200, {"ok": True}),
        ]

        result = await client.fetch("/api/test", method="POST", json_body={"x": 1})

    assert result == {"ok": True}
    assert mock_post.call_count == 2
    mock_sleep.assert_awaited_once()
    await client.close()


async def test_fetch_retries_on_transport_error_then_succeeds() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = [
            httpx.TransportError("socket reset"),
            _response(200, {"ok": True}),
        ]

        result = await client.fetch("/api/test", method="POST", json_body={"x": 1})

    assert result == {"ok": True}
    assert mock_post.call_count == 2
    mock_sleep.assert_awaited_once()
    await client.close()


async def test_fetch_timeout_raises_context_error_after_retries() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(ContextError) as exc_info:
            await client.fetch("/api/test", method="POST", json_body={"x": 1})

    assert exc_info.value.status_code == 408
    assert mock_post.call_count == 4
    assert mock_sleep.await_count == 3
    await client.close()
