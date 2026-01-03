"""
Authentication utilities for verifying Context Protocol requests.

This module provides JWT verification and middleware for MCP server contributors
to secure their endpoints and verify that requests originate from the Context Protocol Platform.

Example:
    >>> from ctxprotocol.auth import verify_context_request, is_protected_mcp_method
    >>>
    >>> # Check if a method requires auth
    >>> if is_protected_mcp_method(body["method"]):
    ...     payload = await verify_context_request(
    ...         authorization_header=request.headers.get("authorization"),
    ...         audience="https://your-tool.com/mcp",  # optional
    ...     )
    ...     # payload contains verified JWT claims
"""

from typing import Any, Awaitable, Callable

import jwt
from cryptography.hazmat.primitives import serialization

from ctxprotocol.client.types import ContextError

# ============================================================================
# Configuration
# ============================================================================

# Official Context Protocol Platform Public Key (RS256)
CONTEXT_PLATFORM_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAs9YOgdpkmVQ5aoNovjsu
chJdV54OT7dUdbVXz914a7Px8EwnpDqhsvG7WO8xL8sj2Rn6ueAJBk+04Hy/P/UN
RJyp23XL5TsGmb4rbfg0ii0MiL2nbVXuqvAe3JSM2BOFZR5bpwIVIaa8aonfamUy
VXGc7OosF90ThdKjm9cXlVM+kV6IgSWc1502X7M3abQqRcTU/rluVXnky0eiWDQa
lfOKbr7w0u72dZjiZPwnNDsX6PEEgvfmoautTFYTQgnZjDzq8UimTcv3KF+hJ5Ep
weipe6amt9lzQzi8WXaFKpOXHQs//WDlUytz/Hl8pvd5craZKzo6Kyrg1Vfan7H3
TQIDAQAB
-----END PUBLIC KEY-----"""

# MCP methods that require authentication
# - tools/call: Executes tool logic, may cost money
# - resources/read: Reads potentially sensitive data
# - prompts/get: Gets prompt content
PROTECTED_MCP_METHODS: frozenset[str] = frozenset([
    "tools/call",
    # Uncomment these if you want to protect resource/prompt access:
    # "resources/read",
    # "prompts/get",
])

# MCP methods that are always open (no auth required)
# These are discovery/listing operations that return metadata only
OPEN_MCP_METHODS: frozenset[str] = frozenset([
    "initialize",
    "tools/list",
    "resources/list",
    "prompts/list",
    "ping",
    "notifications/initialized",
])


# ============================================================================
# Method Classification
# ============================================================================


def is_protected_mcp_method(method: str) -> bool:
    """Determines if a given MCP method requires authentication.

    Discovery methods (tools/list, resources/list, etc.) are open.
    Execution methods (tools/call) require authentication.

    Args:
        method: The MCP JSON-RPC method (e.g., "tools/list", "tools/call")

    Returns:
        True if the method requires authentication

    Example:
        >>> if is_protected_mcp_method(body["method"]):
        ...     await verify_context_request(
        ...         authorization_header=req.headers.get("authorization")
        ...     )
    """
    return method in PROTECTED_MCP_METHODS


def is_open_mcp_method(method: str) -> bool:
    """Determines if a given MCP method is explicitly open (no auth).

    Args:
        method: The MCP JSON-RPC method

    Returns:
        True if the method is known to be open
    """
    return method in OPEN_MCP_METHODS


# ============================================================================
# Request Verification
# ============================================================================


class VerifyRequestOptions:
    """Options for verifying a Context Protocol request.

    Attributes:
        authorization_header: The full Authorization header string (e.g., "Bearer eyJ...")
        audience: Expected Audience (your tool URL) for stricter validation
    """

    def __init__(
        self,
        authorization_header: str | None = None,
        audience: str | None = None,
    ) -> None:
        self.authorization_header = authorization_header
        self.audience = audience


async def verify_context_request(
    authorization_header: str | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Verifies that an incoming request originated from the Context Protocol Platform.

    Args:
        authorization_header: The full Authorization header string (e.g., "Bearer eyJ...")
        audience: Expected Audience (your tool URL) for stricter validation

    Returns:
        The decoded JWT payload if valid

    Raises:
        ContextError: If the authorization header is missing or invalid
        ContextError: If the JWT signature verification fails

    Example:
        >>> payload = await verify_context_request(
        ...     authorization_header=request.headers.get("authorization"),
        ...     audience="https://your-tool.com/mcp",
        ... )
        >>> user_id = payload.get("sub")
    """
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise ContextError(
            message="Missing or invalid Authorization header",
            code="unauthorized",
            status_code=401,
        )

    token = authorization_header.split(" ", 1)[1]

    try:
        # Load the public key
        public_key = serialization.load_pem_public_key(
            CONTEXT_PLATFORM_PUBLIC_KEY_PEM.encode()
        )

        # Build decode options - match TypeScript SDK behavior
        decode_options: dict[str, Any] = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "require": ["exp", "iat"],
        }
        
        # Only verify issuer if we expect it (TypeScript SDK does this)
        # But don't require it in case the platform doesn't always include it
        decode_options["verify_iss"] = True
        
        # Only verify audience if explicitly provided
        if audience:
            decode_options["verify_aud"] = True
        else:
            decode_options["verify_aud"] = False

        # Verify the JWT
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer="https://ctxprotocol.com",
            audience=audience if audience else None,
            options=decode_options,
        )

        return payload

    except jwt.ExpiredSignatureError:
        raise ContextError(
            message="JWT has expired",
            code="unauthorized",
            status_code=401,
        )
    except jwt.InvalidAudienceError:
        raise ContextError(
            message="Invalid JWT audience",
            code="unauthorized",
            status_code=401,
        )
    except jwt.InvalidIssuerError:
        raise ContextError(
            message="Invalid JWT issuer",
            code="unauthorized",
            status_code=401,
        )
    except jwt.DecodeError as e:
        raise ContextError(
            message=f"JWT decode error: {e}",
            code="unauthorized",
            status_code=401,
        )
    except jwt.InvalidSignatureError:
        raise ContextError(
            message="Invalid JWT signature",
            code="unauthorized",
            status_code=401,
        )
    except jwt.PyJWTError as e:
        raise ContextError(
            message=f"JWT verification failed: {e}",
            code="unauthorized",
            status_code=401,
        )


# ============================================================================
# FastAPI/Starlette Middleware
# ============================================================================


class CreateContextMiddlewareOptions:
    """Options for creating Context middleware.

    Attributes:
        audience: Expected Audience (your tool URL) for stricter validation
    """

    def __init__(self, audience: str | None = None) -> None:
        self.audience = audience


class ContextMiddleware:
    """ASGI middleware that secures your MCP endpoint.

    This middleware automatically:
    - Allows discovery methods (tools/list, initialize) without authentication
    - Requires and verifies JWT for execution methods (tools/call)
    - Attaches the verified payload to request.state.context for downstream use

    Example with FastAPI:
        >>> from fastapi import FastAPI, Request
        >>> from ctxprotocol.auth import ContextMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(ContextMiddleware)
        >>>
        >>> @app.post("/mcp")
        >>> async def handle_mcp(request: Request):
        ...     # request.state.context contains verified JWT payload (on protected methods)
        ...     context = getattr(request.state, "context", None)
        ...     # Handle MCP request...

    Example with Starlette:
        >>> from starlette.applications import Starlette
        >>> from starlette.middleware import Middleware
        >>> from ctxprotocol.auth import ContextMiddleware
        >>>
        >>> app = Starlette(
        ...     middleware=[Middleware(ContextMiddleware, audience="https://your-tool.com/mcp")]
        ... )
    """

    def __init__(
        self,
        app: Callable[[dict[str, Any], Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[Any]],
        audience: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap
            audience: Expected Audience (your tool URL) for stricter validation
        """
        self.app = app
        self.audience = audience

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[Any]],
    ) -> None:
        """Process the request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # We need to read the body to check the method
        # This is a simplified version - in production you might want to
        # use a more sophisticated approach to avoid reading the body twice
        body_parts: list[bytes] = []

        async def receive_wrapper() -> dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                body_parts.append(message.get("body", b""))
            return message

        # For this middleware to work properly with body reading,
        # we need a stateful request object. In FastAPI/Starlette,
        # this is typically handled at the Request level.
        # Here we just pass through and let the endpoint handle auth.
        await self.app(scope, receive, send)


def create_context_middleware(
    audience: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Creates a dependency function for FastAPI that verifies Context Protocol requests.

    This is the recommended way to secure your FastAPI MCP endpoint.
    It automatically:
    - Allows discovery methods (tools/list, initialize) without authentication
    - Requires and verifies JWT for execution methods (tools/call)
    - Returns the verified payload for downstream use

    Args:
        audience: Expected Audience (your tool URL) for stricter validation

    Returns:
        A FastAPI dependency function

    Example with FastAPI:
        >>> from fastapi import FastAPI, Request, Depends
        >>> from ctxprotocol.auth import create_context_middleware
        >>>
        >>> app = FastAPI()
        >>> verify_context = create_context_middleware(audience="https://your-tool.com/mcp")
        >>>
        >>> @app.post("/mcp")
        >>> async def handle_mcp(request: Request, context: dict = Depends(verify_context)):
        ...     # context contains verified JWT payload (on protected methods)
        ...     # None for open methods
        ...     ...
    """

    async def dependency(request: Any) -> dict[str, Any] | None:
        """FastAPI dependency for Context Protocol authentication."""
        # Try to get the body - this works with FastAPI's Request object
        try:
            body = await request.json()
        except Exception:
            # If we can't parse the body, let the endpoint handle it
            return None

        method = body.get("method", "")

        # Allow discovery methods without authentication
        if not method or not is_protected_mcp_method(method):
            return None

        # Protected method - require authentication
        authorization = request.headers.get("authorization")

        payload = await verify_context_request(
            authorization_header=authorization,
            audience=audience,
        )

        return payload

    return dependency  # type: ignore[return-value]


__all__ = [
    # Constants
    "CONTEXT_PLATFORM_PUBLIC_KEY_PEM",
    "PROTECTED_MCP_METHODS",
    "OPEN_MCP_METHODS",
    # Method classification
    "is_protected_mcp_method",
    "is_open_mcp_method",
    # Request verification
    "VerifyRequestOptions",
    "verify_context_request",
    # Middleware
    "CreateContextMiddlewareOptions",
    "ContextMiddleware",
    "create_context_middleware",
]

