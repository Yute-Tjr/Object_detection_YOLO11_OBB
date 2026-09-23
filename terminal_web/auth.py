from __future__ import annotations

import hashlib
import secrets
import unicodedata

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


PASSWORD_MIN_LENGTH = 6
SESSION_COOKIE_NAME = "terminal_session"

_PASSWORD_HASHER = PasswordHasher()
DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash(
    "terminal-inspection-dummy-password-never-used"
)


def normalize_username(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("username must be text")
    normalized = value.strip()
    if not 1 <= len(normalized) <= 64:
        raise ValueError("username must contain 1-64 characters")
    if any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in normalized
    ):
        raise ValueError("username cannot contain whitespace or control characters")
    return normalized


def hash_password(password: str) -> str:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"password must contain at least {PASSWORD_MIN_LENGTH} characters"
        )
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError):
        return False


def session_token_digest(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_session_token() -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(48)
    return raw_token, session_token_digest(raw_token)
