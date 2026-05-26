"""Unit tests for config validators and security helpers."""

import pytest
from pydantic import ValidationError

from src.core.config import DatabaseSettings, Settings
from src.core.security import hash_password, verify_password


async def test_verify_password_returns_false_for_overlong_input() -> None:
    hashed = await hash_password("normal_password")
    # bcrypt 4+ raises ValueError for passwords > 72 bytes; verify_password must
    # catch it and return False rather than propagating the exception.
    long_password = "a" * 73
    assert await verify_password(long_password, hashed) is False


def test_rejects_default_secret_key_in_any_environment() -> None:
    with pytest.raises(
        ValidationError, match="SECRET_KEY must be set to a strong random value"
    ):
        Settings(
            ENVIRONMENT="development",
            SECRET_KEY="dev-secret-key-change-before-production",
        )


def test_production_rejects_default_db_password() -> None:
    with pytest.raises(ValidationError, match="POSTGRES_PASSWORD must be changed"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="a-very-long-and-secret-production-key-abc123",
        )


def test_async_url_derived_from_standard_postgresql_scheme() -> None:
    s = DatabaseSettings(DATABASE_URL="postgresql://user:pass@localhost/db")
    assert s.ASYNC_DATABASE_URL.startswith("postgresql+asyncpg://")
    assert "+asyncpg" in s.ASYNC_DATABASE_URL
