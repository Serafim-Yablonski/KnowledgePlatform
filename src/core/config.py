from functools import lru_cache
from typing import Self

from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


# Every setting that grants access is a SecretStr: printing Settings (a repr in a
# traceback, a dump, a test failure) shows "**********"; code reads the value with
# get_secret_value(). Database and Redis URLs carry their passwords.


class DatabaseSettings(BaseSettings):
    model_config = _ENV_CONFIG

    DATABASE_URL: SecretStr = SecretStr(
        "postgresql+asyncpg://user:password@localhost:5432/knowledge_platform"
    )
    # Derived from DATABASE_URL if not set — replaces postgresql:// with postgresql+asyncpg://
    ASYNC_DATABASE_URL: SecretStr = SecretStr("")
    # Derived from DATABASE_URL if not set — psycopg3, for LangGraph's checkpointer
    SYNC_DATABASE_URL: SecretStr = SecretStr("")
    POSTGRES_DB: str = "knowledge_platform"
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: SecretStr = SecretStr("password")

    @model_validator(mode="after")
    def _derive_async_url(self) -> Self:
        if not self.ASYNC_DATABASE_URL.get_secret_value():
            url = self.DATABASE_URL.get_secret_value()
            # Handle both postgresql:// (standard) and postgres:// (cloud providers)
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix) and "+asyncpg" not in url:
                    url = "postgresql+asyncpg://" + url[len(prefix) :]
                    break
            self.ASYNC_DATABASE_URL = SecretStr(url)
        return self

    @model_validator(mode="after")
    def _derive_sync_url(self) -> Self:
        if not self.SYNC_DATABASE_URL.get_secret_value():
            url = self.DATABASE_URL.get_secret_value()
            # Strip any existing driver suffix to get a bare postgresql:// URL,
            # then re-prefix with the psycopg3 (sync) driver.
            for prefix in (
                "postgresql+asyncpg://",
                "postgresql+psycopg2://",
                "postgresql+psycopg://",
                "postgres://",
                "postgresql://",
            ):
                if url.startswith(prefix):
                    url = "postgresql+psycopg://" + url[len(prefix) :]
                    break
            self.SYNC_DATABASE_URL = SecretStr(url)
        return self


class RedisSettings(BaseSettings):
    model_config = _ENV_CONFIG

    REDIS_URL: SecretStr = SecretStr("redis://localhost:6379/0")


class AuthSettings(BaseSettings):
    model_config = _ENV_CONFIG

    SECRET_KEY: SecretStr = SecretStr("dev-secret-key-change-before-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Resolution of api_keys.last_used_at: at most one DB write per key per window
    # (seconds, ADR-019).
    API_KEY_LAST_USED_RESOLUTION: int = 300


class AISettings(BaseSettings):
    model_config = _ENV_CONFIG

    LLM_MODEL: str = "google-gla:gemini-3.5-flash-lite"
    LLM_STRONG_MODEL: str = "google-gla:gemini-3.8-flash"
    GOOGLE_API_KEY: SecretStr = SecretStr("")
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_DIMENSIONS: int = 768
    # The provider's API root, version included; empty: Gemini's. A test seam: the
    # acceptance suite points the workers at a fake provider.
    EMBEDDING_BASE_URL: str = ""


class ObservabilitySettings(BaseSettings):
    model_config = _ENV_CONFIG

    LOGFIRE_TOKEN: SecretStr = SecretStr("")
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SERVICE_NAME: str = "knowledge-platform"


class AppSettings(BaseSettings):
    model_config = _ENV_CONFIG

    UPLOAD_DIR: str = "/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 25
    # Set True when running behind a reverse proxy. Controls ProxyHeadersMiddleware.
    TRUSTED_PROXY_HEADERS: bool = False
    # Comma-separated trusted proxy IPs or CIDR ranges; "*" trusts all proxies.
    # Only used when TRUSTED_PROXY_HEADERS=True. In Docker Compose set this to the
    # nginx/load-balancer container IP or bridge subnet.
    TRUSTED_PROXY_IPS: str = "127.0.0.1"


class PipelineSettings(BaseSettings):
    """The document pipeline's workers, which claim work from the documents table
    (ADR-027). Times in seconds.

    Every setting is bounded. A 0 where a positive one is needed would leave the
    pipeline running and doing nothing useful (a budget of 0 fails every upload as
    interrupted, 0 slots never claim, a 0 s parse fails every PDF), or polling in a
    busy loop. A backoff or grace of 0 is a choice; only a negative one is refused.
    """

    model_config = _ENV_CONFIG

    # Attempts per phase; the claim that finds them used up fails it as interrupted.
    PIPELINE_MAX_ATTEMPTS: PositiveInt = 5
    # An idle worker's claim query, about this often (jittered).
    PIPELINE_POLL_INTERVAL: PositiveFloat = 1.0
    # On SIGTERM, running attempts get this long to finish; the rest are cancelled
    # and released. Must end inside compose's stop_grace_period.
    PIPELINE_SHUTDOWN_GRACE: NonNegativeFloat = 20

    # Attempts at once, per worker process. Low for local dev; production: 4 and 10.
    PIPELINE_EXTRACT_CONCURRENCY: PositiveInt = 2
    PIPELINE_EMBED_CONCURRENCY: PositiveInt = 2

    # Extraction: a text-heavy PDF at the 25 MB upload cap parsed in 28–38 s
    # (pipeline step 8c-0). The parse child is killed at its own timeout, inside the
    # attempt's; the lease outlasts the attempt (validated), and the defaults leave
    # 60 s for its outcome write.
    PIPELINE_EXTRACT_PARSE_TIMEOUT: PositiveFloat = 90
    PIPELINE_EXTRACT_TIMEOUT: PositiveFloat = 120
    PIPELINE_EXTRACT_LEASE: PositiveFloat = 180

    # Embedding: ~500 chunks × ~100 ms/call ≈ 50 s typical.
    PIPELINE_EMBED_TIMEOUT: PositiveFloat = 300
    PIPELINE_EMBED_LEASE: PositiveFloat = 360

    # A transient failure's retry waits up to base · 2^(attempt-1), capped (full
    # jitter, queue.backoff). Settings so the acceptance suite needn't wait them out.
    PIPELINE_EXTRACT_BACKOFF_BASE: NonNegativeFloat = 30
    PIPELINE_EMBED_BACKOFF_BASE: NonNegativeFloat = 60

    @model_validator(mode="after")
    def _order_extract_limits(self) -> Self:
        if not (
            self.PIPELINE_EXTRACT_PARSE_TIMEOUT
            < self.PIPELINE_EXTRACT_TIMEOUT
            < self.PIPELINE_EXTRACT_LEASE
        ):
            raise ValueError(
                "PIPELINE_EXTRACT_PARSE_TIMEOUT < PIPELINE_EXTRACT_TIMEOUT < "
                "PIPELINE_EXTRACT_LEASE must hold"
            )
        return self

    @model_validator(mode="after")
    def _order_embed_limits(self) -> Self:
        if not self.PIPELINE_EMBED_TIMEOUT < self.PIPELINE_EMBED_LEASE:
            raise ValueError("PIPELINE_EMBED_TIMEOUT < PIPELINE_EMBED_LEASE must hold")
        return self


class RateLimitSettings(BaseSettings):
    model_config = _ENV_CONFIG

    # Auth endpoints (IP-scoped, unauthenticated)
    RATE_LIMIT_LOGIN_REQUESTS: int = 10
    RATE_LIMIT_LOGIN_WINDOW: int = 60
    RATE_LIMIT_REGISTER_REQUESTS: int = 5
    RATE_LIMIT_REGISTER_WINDOW: int = 60
    RATE_LIMIT_REFRESH_REQUESTS: int = 20
    RATE_LIMIT_REFRESH_WINDOW: int = 60
    # Authenticated endpoints (user-scoped)
    RATE_LIMIT_SEARCH_REQUESTS: int = 20
    RATE_LIMIT_SEARCH_WINDOW: int = 60
    RATE_LIMIT_AI_ASK_REQUESTS: int = 20
    RATE_LIMIT_AI_ASK_WINDOW: int = 60
    RATE_LIMIT_RESEARCH_START_REQUESTS: int = 5
    RATE_LIMIT_RESEARCH_START_WINDOW: int = 60
    RATE_LIMIT_RESEARCH_STREAM_REQUESTS: int = 30
    RATE_LIMIT_RESEARCH_STREAM_WINDOW: int = 60
    RATE_LIMIT_RESEARCH_REVIEW_REQUESTS: int = 10
    RATE_LIMIT_RESEARCH_REVIEW_WINDOW: int = 60
    RATE_LIMIT_RESEARCH_STATUS_REQUESTS: int = 60
    RATE_LIMIT_RESEARCH_STATUS_WINDOW: int = 60
    # Workspace-scoped (keyed on user_id:workspace_id)
    RATE_LIMIT_DOCUMENT_UPLOAD_REQUESTS: int = 10
    RATE_LIMIT_DOCUMENT_UPLOAD_WINDOW: int = 60
    # Workspace management
    RATE_LIMIT_WORKSPACE_CREATE_REQUESTS: int = 10
    RATE_LIMIT_WORKSPACE_CREATE_WINDOW: int = 60
    # API key management
    RATE_LIMIT_API_KEY_CREATE_REQUESTS: int = 10
    RATE_LIMIT_API_KEY_CREATE_WINDOW: int = 60
    # When True (default), allow requests through if Redis is unavailable.
    # Set False to reject with 503 when the rate-limit backend is down.
    RATE_LIMIT_FAIL_OPEN: bool = True


_DEV_SECRET = "dev-secret-key-change-before-production"


class Settings(
    DatabaseSettings,
    RedisSettings,
    AuthSettings,
    AISettings,
    ObservabilitySettings,
    AppSettings,
    PipelineSettings,
    RateLimitSettings,
):
    model_config = _ENV_CONFIG

    @model_validator(mode="after")
    def _reject_dev_secrets_in_production(self) -> Self:
        if self.SECRET_KEY.get_secret_value() == _DEV_SECRET:
            raise ValueError(
                "SECRET_KEY must be set to a strong random value. "
                "Generate one with: "
                "python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        if (
            self.ENVIRONMENT in ("production", "staging")
            and self.POSTGRES_PASSWORD.get_secret_value() == "password"
        ):
            raise ValueError(
                "POSTGRES_PASSWORD must be changed before deploying to production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for direct import; use Depends(get_settings) in FastAPI.
settings: Settings = get_settings()
