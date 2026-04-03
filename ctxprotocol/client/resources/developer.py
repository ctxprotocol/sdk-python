"""
Developer resource for managing tool listings on the Context Protocol marketplace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from ctxprotocol.client.types import ContextError

if TYPE_CHECKING:
    from ctxprotocol.client.client import ContextClient


class Developer:
    """Developer resource for managing tool listings.

    Scoped to contributor/developer concerns (listing management), separate
    from the consumer-facing ``tools.execute()`` and ``query.run()``.
    """

    def __init__(self, client: ContextClient) -> None:
        self._client = client

    async def update_tool(
        self,
        tool_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Update a tool listing's metadata.

        Requires an API key belonging to the tool's owner.

        Args:
            tool_id: The UUID of the tool to update.
            name: New display name for the tool.
            description: New marketplace description.
            category: New category (e.g. "crypto", "finance", "data").

        Returns:
            Dict with updated tool fields (id, name, description, category, updatedAt).

        Raises:
            ContextError: If authentication fails or the caller does not own the tool.

        Example:
            >>> updated = await client.developer.update_tool(
            ...     "tool-uuid",
            ...     description="Updated description",
            ...     category="crypto",
            ... )
            >>> print(updated["updatedAt"])
        """
        if not tool_id:
            raise ContextError("tool_id is required")

        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if category is not None:
            payload["category"] = category

        if not payload:
            raise ContextError(
                "At least one field required: name, description, or category"
            )

        encoded_tool_id = quote(tool_id, safe="")
        return await self._client.fetch(
            f"/api/v1/tools/{encoded_tool_id}",
            method="PATCH",
            json_body=payload,
        )
