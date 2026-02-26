"""
The official Python client for the Context Protocol.

Use this client to discover and execute AI tools programmatically.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from ctxprotocol.client.types import ContextError

DEFAULT_BASE_URL = "https://www.ctxprotocol.com"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_STREAM_TIMEOUT_SECONDS = 600.0


class ContextClient:
    """The official Python client for the Context Protocol.

    Use this client to discover and execute AI tools programmatically.

    Example:
        >>> from ctxprotocol import ContextClient
        >>>
        >>> async with ContextClient(api_key="sk_live_...") as client:
        ...     # Pay-per-response: Ask a question, get a curated answer
        ...     answer = await client.query.run("What are the top whale movements on Base?")
        ...     print(answer.response)
        ...
        ...     # Pay-per-request: Execute a specific tool for raw data
        ...     tools = await client.discovery.search("gas prices")
        ...     result = await client.tools.execute(
        ...         tool_id=tools[0].id,
        ...         tool_name=tools[0].mcp_tools[0].name,
        ...         args={"chainId": 1}
        ...     )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stream_timeout_seconds: float = DEFAULT_STREAM_TIMEOUT_SECONDS,
    ) -> None:
        """Creates a new Context Protocol client.

        Args:
            api_key: Your Context Protocol API key (format: sk_live_...)
            base_url: Optional base URL override (defaults to https://www.ctxprotocol.com)
            request_timeout_seconds: Timeout for non-streaming requests (default 300.0s)
            stream_timeout_seconds: Timeout for establishing streaming requests (default 600.0s)

        Raises:
            ContextError: If API key is not provided or timeout values are invalid
        """
        if not api_key:
            raise ContextError("API key is required")

        if (
            not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ContextError("request_timeout_seconds must be a positive number")

        if (
            not math.isfinite(stream_timeout_seconds)
            or stream_timeout_seconds <= 0
        ):
            raise ContextError("stream_timeout_seconds must be a positive number")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._stream_timeout_seconds = stream_timeout_seconds
        self._http_client: httpx.AsyncClient | None = None
        self._stream_http_client: httpx.AsyncClient | None = None

        # Import here to avoid circular imports
        from ctxprotocol.client.resources.discovery import Discovery
        from ctxprotocol.client.resources.query import Query
        from ctxprotocol.client.resources.tools import Tools

        # Initialize resources
        self.discovery = Discovery(self)
        self.tools = Tools(self)
        self.query = Query(self)

    def _build_http_client(self, timeout_seconds: float) -> httpx.AsyncClient:
        """Create a configured HTTP client with auth headers and timeout."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def _client(self) -> httpx.AsyncClient:
        """Get or create the non-streaming HTTP client."""
        if self._http_client is None:
            self._http_client = self._build_http_client(self._request_timeout_seconds)
        return self._http_client

    @property
    def _stream_client(self) -> httpx.AsyncClient:
        """Get or create the streaming HTTP client."""
        if self._stream_http_client is None:
            self._stream_http_client = self._build_http_client(
                self._stream_timeout_seconds
            )
        return self._stream_http_client

    async def __aenter__(self) -> ContextClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and close HTTP client."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._stream_http_client is not None:
            await self._stream_http_client.aclose()
            self._stream_http_client = None

    async def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Internal method for making authenticated HTTP requests.

        All requests include the Authorization header with the API key.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            json_body: Optional JSON body for POST requests
            extra_headers: Optional headers merged into the request

        Returns:
            Parsed JSON response

        Raises:
            ContextError: If the request fails
        """
        max_retries = 3
        timeout_seconds = self._request_timeout_seconds
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                if method == "GET":
                    response = await self._client.get(endpoint, headers=extra_headers)
                elif method == "POST":
                    response = await self._client.post(
                        endpoint,
                        json=json_body,
                        headers=extra_headers,
                    )
                else:
                    raise ContextError(f"Unsupported HTTP method: {method}")

                if not response.is_success:
                    # Retry transient 5xx errors
                    if response.status_code >= 500 and attempt < max_retries:
                        delay = min(2**attempt, 10)
                        await asyncio.sleep(delay)
                        continue

                    error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                    error_code: str | None = None
                    help_url: str | None = None

                    try:
                        error_body = response.json()
                        if "error" in error_body:
                            error_message = error_body["error"]
                            error_code = error_body.get("code")
                            help_url = error_body.get("helpUrl")
                    except Exception:
                        # Use default error message if JSON parsing fails
                        pass

                    raise ContextError(
                        message=error_message,
                        code=error_code,
                        status_code=response.status_code,
                        help_url=help_url,
                    )

                return response.json()
            except ContextError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < max_retries:
                    delay = min(2**attempt, 10)
                    await asyncio.sleep(delay)
                    continue

                if isinstance(exc, httpx.TimeoutException):
                    raise ContextError(
                        message=f"Request timed out after {int(timeout_seconds)}s",
                        status_code=408,
                    ) from exc

                raise ContextError(f"HTTP request failed: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ContextError(f"HTTP request failed: {exc}") from exc

        raise ContextError(f"Request failed after retries: {last_error}")

    async def fetch_stream(
        self,
        endpoint: str,
        method: str = "POST",
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Internal method for making authenticated streaming HTTP requests.

        Returns the raw httpx.Response with streaming enabled. The caller
        is responsible for iterating over the response body.

        Args:
            endpoint: API endpoint path
            method: HTTP method (POST, etc.)
            json_body: Optional JSON body
            extra_headers: Optional headers merged into the request

        Returns:
            Raw httpx.Response with stream open

        Raises:
            ContextError: If the request fails
        """
        max_retries = 3
        timeout_seconds = self._stream_timeout_seconds
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._stream_client.send(
                    self._stream_client.build_request(
                        method,
                        endpoint,
                        json=json_body,
                        headers=extra_headers,
                    ),
                    stream=True,
                )

                if not response.is_success:
                    # Read body before retrying/raising
                    await response.aread()

                    if response.status_code >= 500 and attempt < max_retries:
                        delay = min(2**attempt, 10)
                        await asyncio.sleep(delay)
                        continue

                    error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                    error_code: str | None = None
                    help_url: str | None = None

                    try:
                        error_body = response.json()
                        if "error" in error_body:
                            error_message = error_body["error"]
                            error_code = error_body.get("code")
                            help_url = error_body.get("helpUrl")
                    except Exception:
                        pass

                    raise ContextError(
                        message=error_message,
                        code=error_code,
                        status_code=response.status_code,
                        help_url=help_url,
                    )

                return response
            except ContextError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < max_retries:
                    delay = min(2**attempt, 10)
                    await asyncio.sleep(delay)
                    continue

                if isinstance(exc, httpx.TimeoutException):
                    raise ContextError(
                        message=f"Request timed out after {int(timeout_seconds)}s",
                        status_code=408,
                    ) from exc

                raise ContextError(f"HTTP streaming request failed: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ContextError(f"HTTP streaming request failed: {exc}") from exc

        raise ContextError(f"Streaming request failed after retries: {last_error}")