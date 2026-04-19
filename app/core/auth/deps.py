import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=True)

_JWKS_URL = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
_jwks_client = PyJWKClient(_JWKS_URL, cache_keys=True, lifespan=3600)


@dataclass
class AuthUser:
    user_id: str
    email: Optional[str]
    role: str  # "admin" | "client" | "authenticated"
    raw_claims: dict


def _decode_supabase_jwt(token: str) -> dict:
    """Verify token. Supports ES256/RS256 (signing keys via JWKS) and HS256 (legacy secret)."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        logger.warning(f"Bad JWT header: {e}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    alg = header.get("alg", "HS256")

    try:
        if alg == "HS256":
            return jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp", "sub"]},
            )
        # Asymmetric (ES256/RS256) — fetch signing key from JWKS
        signing_key = _jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            signing_key,
            algorithms=[alg],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT: {e}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> AuthUser:
    claims = _decode_supabase_jwt(creds.credentials)
    role = (claims.get("app_metadata") or {}).get("role") or claims.get("role") or "authenticated"
    return AuthUser(
        user_id=claims["sub"],
        email=claims.get("email"),
        role=role,
        raw_claims=claims,
    )


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user
