"""
Discovery resource for searching and finding tools on the Context Protocol marketplace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

    async def search(
        self,
        query: str,
        limit: int | None = None,
        *,
        mode: Literal["query", "execute"] | None = None,
        surface: Literal["answer", "execute", "both"] | None = None,
        query_eligible: bool | None = None,
        require_execute_pricing: bool | None = None,
        exclude_latency_classes: list[
            Literal["instant", "fast", "slow", "streaming"]
        ] | None = None,
        exclude_slow: bool | None = None,
    ) -> list[Tool]:
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

        if mode is not None:
            params["mode"] = mode

        if surface is not None:
            params["surface"] = surface

        if query_eligible is not None:
            params["queryEligible"] = "true" if query_eligible else "false"

        if require_execute_pricing is not None:
            params["requireExecutePricing"] = (
                "true" if require_execute_pricing else "false"
            )

        if exclude_latency_classes:
            params["excludeLatency"] = ",".join(exclude_latency_classes)

        if exclude_slow is not None:
            params["excludeSlow"] = "true" if exclude_slow else "false"

        from urllib.parse import urlencode
        query_string = urlencode(params) if params else ""
        endpoint = f"/api/v1/tools/search{'?' + query_string if query_string else ''}"

        response = await self._client.fetch(endpoint)
        search_response = SearchResponse.model_validate(response)

        return search_response.tools

    async def get_featured(
        self,
        limit: int | None = None,
        *,
        mode: Literal["query", "execute"] | None = None,
        surface: Literal["answer", "execute", "both"] | None = None,
        query_eligible: bool | None = None,
        require_execute_pricing: bool | None = None,
        exclude_latency_classes: list[
            Literal["instant", "fast", "slow", "streaming"]
        ] | None = None,
        exclude_slow: bool | None = None,
    ) -> list[Tool]:
        """Get featured/popular tools (empty query search).

        Args:
            limit: Maximum number of results (1-50, default 10)

        Returns:
            Array of featured tools

        Example:
            >>> featured = await client.discovery.get_featured(5)
        """
        return await self.search(
            "",
            limit,
            mode=mode,
            surface=surface,
            query_eligible=query_eligible,
            require_execute_pricing=require_execute_pricing,
            exclude_latency_classes=exclude_latency_classes,
            exclude_slow=exclude_slow,
        )

