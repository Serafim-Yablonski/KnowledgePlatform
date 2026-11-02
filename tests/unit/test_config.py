"""Unit tests for config validators and security helpers."""

import pytest
from pydantic import ValidationError

from src.core.config import (
    CelerySettings,
    DatabaseSettings,
    PipelineSettings,
    Settings,
    get_settings,
    settings,
)
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


def test_settings_override_reaches_get_settings(
    settings_override: pytest.MonkeyPatch,
) -> None:
    # Code reads settings both ways: the module-level object and get_settings().
    settings_override.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    assert get_settings().MAX_UPLOAD_SIZE_MB == 1


@pytest.mark.parametrize(
    "limits",
    [
        # The child's kill must come before the soft limit, or the soft limit (which
        # pypdf can swallow, S6-10) is what bounds the parse again.
        {"CELERY_EXTRACT_PARSE_TIMEOUT": 120, "CELERY_EXTRACT_SOFT_TIME_LIMIT": 120},
        {"CELERY_EXTRACT_SOFT_TIME_LIMIT": 180, "CELERY_EXTRACT_TIME_LIMIT": 180},
    ],
)
def test_extract_limits_must_be_ordered(limits: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="CELERY_EXTRACT_PARSE_TIMEOUT <"):
        CelerySettings(**limits)


def test_the_default_extract_limits_are_ordered() -> None:
    s = CelerySettings()
    assert (
        s.CELERY_EXTRACT_PARSE_TIMEOUT
        < s.CELERY_EXTRACT_SOFT_TIME_LIMIT
        < s.CELERY_EXTRACT_TIME_LIMIT
    )


@pytest.mark.parametrize(
    "limits",
    [
        # The parse child's kill must come before the attempt's timeout: the child's
        # exit says whose fault a failure is, the attempt's timeout can't.
        {"PIPELINE_EXTRACT_PARSE_TIMEOUT": 120, "PIPELINE_EXTRACT_TIMEOUT": 120},
        # A live attempt must never lose its row: the lease outlasts the timeout.
        {"PIPELINE_EXTRACT_TIMEOUT": 180, "PIPELINE_EXTRACT_LEASE": 180},
    ],
)
def test_extract_pipeline_limits_must_be_ordered(limits: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="PIPELINE_EXTRACT_PARSE_TIMEOUT <"):
        PipelineSettings(**limits)


def test_the_embed_lease_must_outlast_its_timeout() -> None:
    with pytest.raises(ValidationError, match="PIPELINE_EMBED_TIMEOUT <"):
        PipelineSettings(PIPELINE_EMBED_TIMEOUT=360, PIPELINE_EMBED_LEASE=360)


def test_the_default_pipeline_limits_are_ordered() -> None:
    s = PipelineSettings()
    assert (
        s.PIPELINE_EXTRACT_PARSE_TIMEOUT
        < s.PIPELINE_EXTRACT_TIMEOUT
        < s.PIPELINE_EXTRACT_LEASE
    )
    assert s.PIPELINE_EMBED_TIMEOUT < s.PIPELINE_EMBED_LEASE
