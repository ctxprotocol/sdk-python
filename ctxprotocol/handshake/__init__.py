"""
Handshake Module - Types and helpers for MCP tools that need user interaction.

Use these types when your tool needs to request user actions:
- Signatures (EIP-712 typed data for Hyperliquid, Polymarket, dYdX)
- Transactions (direct on-chain actions for Uniswap, NFT mints)
- OAuth (authentication flows for Discord, Twitter)

Example:
    >>> from ctxprotocol.handshake import create_signature_request, wrap_handshake_response
    >>>
    >>> # In your MCP tool handler:
    >>> def handle_place_order(args):
    ...     return wrap_handshake_response(
    ...         create_signature_request(
    ...             domain={"name": "Hyperliquid", "version": "1", "chainId": 42161},
    ...             types={"Order": [{"name": "asset", "type": "uint32"}, ...]},
    ...             primary_type="Order",
    ...             message={"asset": 4, "isBuy": True, ...},
    ...             meta={"description": "Place order", "protocol": "Hyperliquid"}
    ...         )
    ...     )

For more information, see: https://docs.ctxprotocol.com/guides/handshake-architecture
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict, Union

# =============================================================================
# Shared Meta Types
# =============================================================================


class HandshakeMeta(TypedDict, total=False):
    """UI metadata for handshake approval cards."""

    description: str
    """Human-readable description of the action."""

    protocol: str
    """Protocol name (e.g., 'Hyperliquid', 'Polymarket')."""

    action: str
    """Action verb (e.g., 'Place Order', 'Place Bid')."""

    token_symbol: str
    """Token symbol if relevant."""

    token_amount: str
    """Human-readable token amount."""

    warning_level: Literal["info", "caution", "danger"]
    """UI warning level."""


# =============================================================================
# Web3: Signature Requests (EIP-712)
# =============================================================================


class EIP712Domain(TypedDict, total=False):
    """EIP-712 domain separator."""

    name: str
    """Domain name (e.g., 'Hyperliquid', 'ClobAuthDomain')."""

    version: str
    """Domain version."""

    chainId: int
    """Chain ID (informational - signing is chain-agnostic)."""

    verifyingContract: str
    """Optional verifying contract address (0x...)."""


class EIP712TypeField(TypedDict):
    """A single field in an EIP-712 type definition."""

    name: str
    type: str


class SignatureRequest(TypedDict, total=False):
    """
    Signature Request for EIP-712 typed data signing.

    Use this for platforms with proxy wallets (Hyperliquid, Polymarket, dYdX).

    Benefits:
    - No gas required (user signs a message, not a transaction)
    - No network switching needed (signing is chain-agnostic)
    - Works with Privy embedded wallets on any chain
    """

    _action: Literal["signature_request"]
    """Action type identifier (required)."""

    domain: EIP712Domain
    """EIP-712 domain separator (required)."""

    types: dict[str, list[EIP712TypeField]]
    """EIP-712 type definitions (required)."""

    primaryType: str
    """The primary type being signed (required)."""

    message: dict[str, Any]
    """The message data to sign (required)."""

    meta: HandshakeMeta
    """UI metadata for the approval card."""

    callbackToolName: str
    """Optional: Tool name to call with the signature result."""


# =============================================================================
# Web3: Transaction Proposals
# =============================================================================


class TransactionProposalMeta(HandshakeMeta, total=False):
    """Extended metadata for transaction proposals."""

    estimated_gas: str
    """Estimated gas cost (informational - Context may sponsor)."""

    explorer_url: str
    """Link to contract on block explorer."""


class TransactionProposal(TypedDict, total=False):
    """
    Transaction Proposal for direct on-chain actions.

    Use this for protocols without proxy wallets (Uniswap, NFT mints, etc.).

    Note: May require network switching and gas fees.
    """

    _action: Literal["transaction_proposal"]
    """Action type identifier (required)."""

    chainId: int
    """EVM chain ID (e.g., 137 for Polygon, 8453 for Base) (required)."""

    to: str
    """Target contract address (0x...) (required)."""

    data: str
    """Encoded calldata (0x...) (required)."""

    value: str
    """Wei to send (as string, default '0')."""

    meta: TransactionProposalMeta
    """UI metadata for the approval card."""


# =============================================================================
# Web2: OAuth Requests
# =============================================================================


class AuthRequiredMeta(TypedDict, total=False):
    """Metadata for OAuth requests."""

    display_name: str
    """Human-friendly service name."""

    scopes: list[str]
    """Permissions being requested."""

    description: str
    """Description of what access is needed."""

    icon_url: str
    """Tool's icon URL."""

    expires_in: str
    """How long authorization lasts."""


class AuthRequired(TypedDict, total=False):
    """
    Auth Required for OAuth flows.

    Use this when your tool needs the user to authenticate with an external service.
    """

    _action: Literal["auth_required"]
    """Action type identifier (required)."""

    provider: str
    """Service identifier (e.g., 'discord', 'slack') (required)."""

    authUrl: str
    """Your OAuth initiation endpoint (MUST be HTTPS) (required)."""

    meta: AuthRequiredMeta
    """UI metadata for the auth card."""


# =============================================================================
# Union Type
# =============================================================================

HandshakeAction = Union[SignatureRequest, TransactionProposal, AuthRequired]

# =============================================================================
# Type Guards
# =============================================================================


def is_handshake_action(value: Any) -> bool:
    """Check if a value is a handshake action."""
    if not isinstance(value, dict):
        return False
    action = value.get("_action")
    return action in ("signature_request", "transaction_proposal", "auth_required")


def is_signature_request(value: Any) -> bool:
    """Check if a value is a signature request."""
    return is_handshake_action(value) and value.get("_action") == "signature_request"


def is_transaction_proposal(value: Any) -> bool:
    """Check if a value is a transaction proposal."""
    return is_handshake_action(value) and value.get("_action") == "transaction_proposal"


def is_auth_required(value: Any) -> bool:
    """Check if a value is an auth required action."""
    return is_handshake_action(value) and value.get("_action") == "auth_required"


# =============================================================================
# Helper Functions
# =============================================================================


def create_signature_request(
    *,
    domain: EIP712Domain,
    types: dict[str, list[EIP712TypeField]],
    primary_type: str,
    message: dict[str, Any],
    meta: HandshakeMeta | None = None,
    callback_tool_name: str | None = None,
) -> SignatureRequest:
    """
    Create a signature request response.

    Use this for platforms with proxy wallets (Hyperliquid, Polymarket, dYdX).
    Benefits: No gas required, no network switching needed.

    Args:
        domain: EIP-712 domain separator
        types: EIP-712 type definitions
        primary_type: The primary type being signed
        message: The message data to sign
        meta: Optional UI metadata for the approval card
        callback_tool_name: Optional tool name to call with signature result

    Returns:
        A SignatureRequest dict ready to be wrapped in a handshake response
    """
    result: SignatureRequest = {
        "_action": "signature_request",
        "domain": domain,
        "types": types,
        "primaryType": primary_type,
        "message": message,
    }
    if meta:
        result["meta"] = meta
    if callback_tool_name:
        result["callbackToolName"] = callback_tool_name
    return result


def create_transaction_proposal(
    *,
    chain_id: int,
    to: str,
    data: str,
    value: str = "0",
    meta: TransactionProposalMeta | None = None,
) -> TransactionProposal:
    """
    Create a transaction proposal response.

    Use this for protocols without proxy wallets (Uniswap, NFT mints, etc.).
    Note: May require network switching and gas.

    Args:
        chain_id: EVM chain ID (e.g., 137 for Polygon, 8453 for Base)
        to: Target contract address (0x...)
        data: Encoded calldata (0x...)
        value: Wei to send (as string, default '0')
        meta: Optional UI metadata for the approval card

    Returns:
        A TransactionProposal dict ready to be wrapped in a handshake response
    """
    result: TransactionProposal = {
        "_action": "transaction_proposal",
        "chainId": chain_id,
        "to": to,
        "data": data,
        "value": value,
    }
    if meta:
        result["meta"] = meta
    return result


def create_auth_required(
    *,
    provider: str,
    auth_url: str,
    meta: AuthRequiredMeta | None = None,
) -> AuthRequired:
    """
    Create an auth required response.

    Use this when your tool needs the user to authenticate via OAuth.

    Args:
        provider: Service identifier (e.g., 'discord', 'slack')
        auth_url: Your OAuth initiation endpoint (MUST be HTTPS)
        meta: Optional UI metadata for the auth card

    Returns:
        An AuthRequired dict ready to be wrapped in a handshake response
    """
    result: AuthRequired = {
        "_action": "auth_required",
        "provider": provider,
        "authUrl": auth_url,
    }
    if meta:
        result["meta"] = meta
    return result


def wrap_handshake_response(action: HandshakeAction) -> dict[str, Any]:
    """
    Wrap a handshake action in the proper MCP response format.

    MCP tools should return handshake actions in `_meta.handshakeAction` to prevent
    the MCP SDK from stripping unknown fields.

    Example:
        >>> return wrap_handshake_response(create_signature_request(
        ...     domain={"name": "Hyperliquid", "version": "1", "chainId": 42161},
        ...     types={"Order": [...]},
        ...     primary_type="Order",
        ...     message=order_data,
        ...     meta={"description": "Place order", "protocol": "Hyperliquid"}
        ... ))

    Args:
        action: The handshake action (SignatureRequest, TransactionProposal, or AuthRequired)

    Returns:
        A dict with the proper MCP response format including structuredContent._meta.handshakeAction
    """
    action_type = action.get("_action", "unknown").replace("_", " ")
    description = action.get("meta", {}).get("description", f"{action_type} required")

    return {
        "content": [
            {
                "type": "text",
                "text": f"Handshake required: {action_type}. Please approve in the Context app.",
            }
        ],
        "structuredContent": {
            "_meta": {
                "handshakeAction": action,
            },
            "status": "handshake_required",
            "message": description,
        },
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "HandshakeMeta",
    "EIP712Domain",
    "EIP712TypeField",
    "SignatureRequest",
    "TransactionProposalMeta",
    "TransactionProposal",
    "AuthRequiredMeta",
    "AuthRequired",
    "HandshakeAction",
    # Type guards
    "is_handshake_action",
    "is_signature_request",
    "is_transaction_proposal",
    "is_auth_required",
    # Helper functions
    "create_signature_request",
    "create_transaction_proposal",
    "create_auth_required",
    "wrap_handshake_response",
]
