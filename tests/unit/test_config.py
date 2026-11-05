"""Unit tests for config validators and security helpers."""

import re

import pytest
from pydantic import SecretStr, ValidationError

from src.core.config import (
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
    assert s.ASYNC_DATABASE_URL.get_secret_value() == (
        "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert s.SYNC_DATABASE_URL.get_secret_value() == (
        "postgresql+psycopg://user:pass@localhost/db"
    )


def test_settings_override_reaches_get_settings(
    settings_override: pytest.MonkeyPatch,
) -> None:
    # Code reads settings both ways: the module-level object and get_settings().
    settings_override.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    assert get_settings().MAX_UPLOAD_SIZE_MB == 1


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


# Every pipeline setting and the first value past its bound. 0 breaks the ones that
# must be positive: no attempt ever runs (a budget of 0 fails every upload as
# interrupted; 0 slots never claim; a 0 s parse fails every PDF as a timeout), or an
# idle worker claims in a busy loop. 0 is a choice for a backoff (retry at once) and
# a grace (cancel at once); only a negative one is refused.
_PIPELINE_BOUNDS = {
    "PIPELINE_MAX_ATTEMPTS": 0,
    "PIPELINE_POLL_INTERVAL": 0,
    "PIPELINE_SHUTDOWN_GRACE": -1,
    "PIPELINE_EXTRACT_CONCURRENCY": 0,
    "PIPELINE_EMBED_CONCURRENCY": 0,
    "PIPELINE_EXTRACT_PARSE_TIMEOUT": 0,
    "PIPELINE_EXTRACT_TIMEOUT": 0,
    "PIPELINE_EXTRACT_LEASE": 0,
    "PIPELINE_EMBED_TIMEOUT": 0,
    "PIPELINE_EMBED_LEASE": 0,
    "PIPELINE_EXTRACT_BACKOFF_BASE": -1,
    "PIPELINE_EMBED_BACKOFF_BASE": -1,
}


def test_every_pipeline_setting_has_a_bound() -> None:
    assert set(_PIPELINE_BOUNDS) == set(PipelineSettings.model_fields)


@pytest.mark.parametrize(("name", "value"), _PIPELINE_BOUNDS.items())
def test_a_pipeline_setting_past_its_bound_is_refused(name: str, value: float) -> None:
    with pytest.raises(ValidationError) as refused:
        PipelineSettings(**{name: value})

    # The field's own bound, not the ordering checks that run after it.
    assert [e["loc"] for e in refused.value.errors()] == [(name,)]


def test_a_zero_backoff_and_grace_are_allowed() -> None:
    PipelineSettings(
        PIPELINE_EXTRACT_BACKOFF_BASE=0,
        PIPELINE_EMBED_BACKOFF_BASE=0,
        PIPELINE_SHUTDOWN_GRACE=0,
    )


def test_the_default_pipeline_limits_are_ordered() -> None:
    s = PipelineSettings()
    assert (
        s.PIPELINE_EXTRACT_PARSE_TIMEOUT
        < s.PIPELINE_EXTRACT_TIMEOUT
        < s.PIPELINE_EXTRACT_LEASE
    )
    assert s.PIPELINE_EMBED_TIMEOUT < s.PIPELINE_EMBED_LEASE


# Every setting that grants access, with a value no default contains.
_SECRETS = {
    "SECRET_KEY": "fake-jwt-secret-0123456789abcdef",
    "GOOGLE_API_KEY": "fake-google-key",
    "ANTHROPIC_API_KEY": "fake-anthropic-key",
    "EMBEDDING_API_KEY": "fake-embedding-key",
    "LOGFIRE_TOKEN": "fake-logfire-token",
    "POSTGRES_PASSWORD": "fake-postgres-password",
    "DATABASE_URL": "postgresql://u:fake-db-password@db:5432/kp",
    "REDIS_URL": "redis://:fake-redis-password@redis:6379/0",
}
# A setting named like these is a secret: a plain str would print with the rest.
_SECRET_NAME = re.compile(
    r"_API_KEY|_TOKEN|_PASSWORD|SECRET_KEY|DATABASE_URL|REDIS_URL"
)


def test_settings_never_print_their_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    # repr, str and a dump are what a traceback, a log line or a test failure
    # shows of the object (P7a: a Settings in a frame printed every secret).
    for name, value in _SECRETS.items():
        monkeypatch.setenv(name, value)
    s = Settings(_env_file=None)  # type: ignore[call-arg]

    shown = [repr(s), str(s), str(s.model_dump()), s.model_dump_json()]

    leaked = {
        name
        for name, value in _SECRETS.items()
        for out in shown
        if value.split("@")[0] in out
    }
    assert leaked == set()
    assert s.SECRET_KEY.get_secret_value() == _SECRETS["SECRET_KEY"]
    assert "fake-db-password" in s.ASYNC_DATABASE_URL.get_secret_value()


def test_every_setting_named_like_a_secret_is_one() -> None:
    plain = {
        name
        for name, field in Settings.model_fields.items()
        if _SECRET_NAME.search(name)
        and field.annotation is not SecretStr
        and field.annotation not in (int, float, bool)
    }
    assert plain == set()
    assert {n for n in Settings.model_fields if _SECRET_NAME.search(n)} >= set(_SECRETS)
