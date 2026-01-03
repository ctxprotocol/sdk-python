"""
Wallet context types for portfolio tracking.

These types represent wallet and token holdings that can be
injected into MCP tools for personalized analysis.
"""

from pydantic import BaseModel, Field


class WalletContext(BaseModel):
    """Base wallet context - address and chain info.

    Attributes:
        address: Wallet address (checksummed)
        chain_id: Chain ID (137 for Polygon, 1 for Ethereum, etc.)
        native_balance: Native token balance in wei (string for precision)
    """

    address: str = Field(..., description="Wallet address (checksummed)")
    chain_id: int = Field(
        ...,
        alias="chainId",
        description="Chain ID (137 for Polygon, 1 for Ethereum, etc.)",
    )
    native_balance: str | None = Field(
        default=None,
        alias="nativeBalance",
        description="Native token balance in wei (string for precision)",
    )

    model_config = {"populate_by_name": True}


class ERC20TokenBalance(BaseModel):
    """ERC20 token holdings.

    Attributes:
        address: Token contract address
        symbol: Token symbol (e.g., "USDC")
        decimals: Token decimals
        balance: Balance in smallest unit (string for precision)
    """

    address: str = Field(..., description="Token contract address")
    symbol: str = Field(..., description="Token symbol (e.g., 'USDC')")
    decimals: int = Field(..., description="Token decimals")
    balance: str = Field(..., description="Balance in smallest unit (string for precision)")


class ERC20Context(BaseModel):
    """Collection of ERC20 token balances.

    Attributes:
        tokens: Array of token balances
    """

    tokens: list[ERC20TokenBalance] = Field(..., description="Array of token balances")

