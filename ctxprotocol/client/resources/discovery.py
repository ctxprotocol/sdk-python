"""
Discovery resource for searching and finding tools on the Context Protocol marketplace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ctxprotocol.client.types import SearchResponse, Tool

if TYPE_CHECKING:
    from ctxprotocol.client.client import ContextClient


class Discovery:
    """Discovery resource for searching and finding tools on the Context Protocol marketplace."""

    def __init__(self, client: ContextClient) -> None:
        """Initialize the Discovery resource.

        Args:
            client: The parent ContextClient instance
        """
        self._client = client

    async def search(self, query: str, limit: int | None = None) -> list[Tool]:
        """Search for tools matching a query string.

        Args:
            query: The search query (e.g., "gas prices", "nft metadata")
            limit: Maximum number of results (1-50, default 10)

        Returns:
            Array of matching tools

        Example:
            >>> tools = await client.discovery.search("gas prices")
            >>> print(tools[0].name)  # "Gas Price Oracle"
            >>> print(tools[0].mcp_tools)  # Available methods
        """
        params: dict[str, str] = {}

        if query:
            params["q"] = query

        if limit is not None:
            params["limit"] = str(limit)

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        endpoint = f"/api/v1/tools/search{'?' + query_string if query_string else ''}"

        response = await self._client.fetch(endpoint)
        search_response = SearchResponse.model_validate(response)

        return search_response.tools

    async def get_featured(self, limit: int | None = None) -> list[Tool]:
        """Get featured/popular tools (empty query search).

        Args:
            limit: Maximum number of results (1-50, default 10)

        Returns:
            Array of featured tools

        Example:
            >>> featured = await client.discovery.get_featured(5)
        """
        return await self.search("", limit)

