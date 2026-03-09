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


async def test_fetch_retries_safe_get_requests_on_5xx_then_succeeds() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "get", new_callable=AsyncMock) as mock_get,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_get.side_effect = [
            _response(502, {"error": "Bad gateway"}, method="GET"),
            _response(200, {"ok": True}, method="GET"),
        ]

        result = await client.fetch("/api/test", method="GET")

    assert result == {"ok": True}
    assert mock_get.call_count == 2
    mock_sleep.assert_awaited_once()
    await client.close()


async def test_fetch_retries_post_when_idempotency_key_present() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = [
            httpx.TransportError("socket reset"),
            _response(200, {"ok": True}),
        ]

        result = await client.fetch(
            "/api/test",
            method="POST",
            json_body={"x": 1},
            extra_headers={"Idempotency-Key": "01f7db54-43ca-4c30-a8da-0d4d71d2a573"},
        )

    assert result == {"ok": True}
    assert mock_post.call_count == 2
    mock_sleep.assert_awaited_once()
    await client.close()


async def test_fetch_does_not_retry_non_idempotent_post_requests() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = httpx.TransportError("socket reset")

        with pytest.raises(ContextError) as exc_info:
            await client.fetch("/api/test", method="POST", json_body={"x": 1})

    assert "socket reset" in str(exc_info.value)
    assert mock_post.call_count == 1
    mock_sleep.assert_not_awaited()
    await client.close()


async def test_fetch_timeout_raises_context_error_after_idempotent_retries() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")

    with (
        patch.object(client._client, "post", new_callable=AsyncMock) as mock_post,
        patch("ctxprotocol.client.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(ContextError) as exc_info:
            await client.fetch(
                "/api/test",
                method="POST",
                json_body={"x": 1},
                extra_headers={"Idempotency-Key": "db56398b-d383-411c-a0c4-4ff3184dd725"},
            )

    assert exc_info.value.status_code == 408
    assert mock_post.call_count == 4
    assert mock_sleep.await_count == 3
    await client.close()


async def test_fetch_does_not_retry_after_successful_response_when_json_parse_fails() -> None:
    client = ContextClient(api_key="ctx_test_key_1234567890abcdef12345678")
    response = _response(200, {"ok": True})
    response.json = lambda: (_ for _ in ()).throw(ValueError("bad json"))  # type: ignore[method-assign]

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response

        with pytest.raises(ContextError) as exc_info:
            await client.fetch("/api/test", method="POST", json_body={"x": 1})

    assert "Failed to parse JSON response" in str(exc_info.value)
    assert mock_post.call_count == 1
    await client.close()
