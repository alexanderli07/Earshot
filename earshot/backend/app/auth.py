"""Password hashing, session tokens, and the current-user dependency.

Security posture:
- Passwords are hashed with bcrypt; the plaintext is never stored or logged.
- Session tokens are random 256-bit strings handed to the client in an
  httpOnly cookie; only their SHA-256 hash is stored, so a database leak
  cannot be replayed as a live session.
- Auth only guards the per-user (/me/*) and account (/auth/*) surface. The
  live alert path (/ws, /debug/event, device rules) is intentionally left
  open: a login screen must never stand between a user and a smoke alarm.
"""

import hashlib
import secrets

import bcrypt
from fastapi import Cookie, Depends, HTTPException, Request

from . import config

# bcrypt hashes at most the first 72 bytes; enforce a sane bound so long
# inputs fail fast rather than being silently truncated.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LEN = 8


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"),
                              password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def validate_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LEN:
        raise ValueError(
            f"password must be at least {MIN_PASSWORD_LEN} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes")


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_store(request: Request):
    """The UserStore, or a 503 if Mongo isn't configured (auth disabled)."""
    store = getattr(request.app.state, "users", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="user accounts are not configured (set EARSHOT_MONGO_URI)")
    return store


async def current_user(request: Request,
                       earshot_session: str | None = Cookie(default=None)):
    """Resolve the logged-in user from the session cookie, or 401."""
    store = get_store(request)
    if not earshot_session:
        raise HTTPException(status_code=401, detail="not authenticated")
    session = await store.get_active_session(hash_token(earshot_session))
    if session is None:
        raise HTTPException(status_code=401, detail="session expired")
    user = await store.get_user(session["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="account not found")
    return user


def public_user(user: dict) -> dict:
    """A user document with only client-safe fields (never the hash)."""
    return {
        "id": user["_id"],
        "username": user["username"],
        "display_name": user.get("display_name", user["username"]),
        "created_at": user.get("created_at"),
    }


SESSION_COOKIE = config.SESSION_COOKIE
