"""
The official Python client for the Context Protocol.

Use this client to discover and execute AI tools programmatically.
"""

from __future__ import annotations

from typing import Any

import httpx

from ctxprotocol.client.types import ContextError


class ContextClient:
    """The official Python client for the Context Protocol.

    Use this client to discover and execute AI tools programmatically.

    Example:
        >>> from ctxprotocol import ContextClient
        >>>
        >>> async with ContextClient(api_key="sk_live_...") as client:
        ...     # Discover tools
        ...     tools = await client.discovery.search("gas prices")
        ...
        ...     # Execute a tool method
        ...     result = await client.tools.execute(
        ...         tool_id=tools[0].id,
        ...         tool_name=tools[0].mcp_tools[0].name,
        ...         args={"chainId": 1}
        ...     )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ctxprotocol.com",
    ) -> None:
        """Creates a new Context Protocol client.

        Args:
            api_key: Your Context Protocol API key (format: sk_live_...)
            base_url: Optional base URL override (defaults to https://ctxprotocol.com)

        Raises:
            ContextError: If API key is not provided
        """
        if not api_key:
            raise ContextError("API key is required")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_client: httpx.AsyncClient | None = None

        # Import here to avoid circular imports
        from ctxprotocol.client.resources.discovery import Discovery
        from ctxprotocol.client.resources.tools import Tools

        # Initialize resources
        self.discovery = Discovery(self)
        self.tools = Tools(self)

    @property
    def _client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                timeout=httpx.Timeout(30.0),
            )
        return self._http_client

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

    async def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Internal method for making authenticated HTTP requests.

        All requests include the Authorization header with the API key.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            json_body: Optional JSON body for POST requests

        Returns:
            Parsed JSON response

        Raises:
            ContextError: If the request fails
        """
        try:
            if method == "GET":
                response = await self._client.get(endpoint)
            elif method == "POST":
                response = await self._client.post(endpoint, json=json_body)
            else:
                raise ContextError(f"Unsupported HTTP method: {method}")

            if not response.is_success:
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

        except httpx.HTTPError as e:
            raise ContextError(f"HTTP request failed: {e}") from e

