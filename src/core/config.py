from functools import lru_cache
from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


class DatabaseSettings(BaseSettings):
    model_config = _ENV_CONFIG

    DATABASE_URL: str = (
        "postgresql+asyncpg://user:password@localhost:5432/knowledge_platform"
    )
    # Derived from DATABASE_URL if not set — replaces postgresql:// with postgresql+asyncpg://
    ASYNC_DATABASE_URL: str = ""
    # Derived from DATABASE_URL if not set — psycopg3 sync driver for Celery workers
    SYNC_DATABASE_URL: str = ""
    POSTGRES_DB: str = "knowledge_platform"
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"

    @model_validator(mode="after")
    def _derive_async_url(self) -> Self:
        if not self.ASYNC_DATABASE_URL:
            url = self.DATABASE_URL
            # Handle both postgresql:// (standard) and postgres:// (cloud providers)
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix) and "+asyncpg" not in url:
                    url = "postgresql+asyncpg://" + url[len(prefix) :]
                    break
            self.ASYNC_DATABASE_URL = url
        return self

    @model_validator(mode="after")
    def _derive_sync_url(self) -> Self:
        if not self.SYNC_DATABASE_URL:
            url = self.DATABASE_URL
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
            self.SYNC_DATABASE_URL = url
        return self


class RedisSettings(BaseSettings):
    model_config = _ENV_CONFIG

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"


class AuthSettings(BaseSettings):
    model_config = _ENV_CONFIG

    SECRET_KEY: str = "dev-secret-key-change-before-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7


class AISettings(BaseSettings):
    model_config = _ENV_CONFIG

    LLM_MODEL: str = "google-gla:gemini-2.0-flash"
    LLM_STRONG_MODEL: str = "anthropic:claude-sonnet-4-5"
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_MODEL: str = "text-embedding-005"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSIONS: int = 768


class ObservabilitySettings(BaseSettings):
    model_config = _ENV_CONFIG

    LOGFIRE_TOKEN: str = ""
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SERVICE_NAME: str = "knowledge-platform"


class AppSettings(BaseSettings):
    model_config = _ENV_CONFIG

    UPLOAD_DIR: str = "/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    # Set True when running behind a reverse proxy. Controls ProxyHeadersMiddleware.
    TRUSTED_PROXY_HEADERS: bool = False
    # Comma-separated trusted proxy IPs or CIDR ranges; "*" trusts all proxies.
    # Only used when TRUSTED_PROXY_HEADERS=True. In Docker Compose set this to the
    # nginx/load-balancer container IP or bridge subnet.
    TRUSTED_PROXY_IPS: str = "127.0.0.1"


class MCPSettings(BaseSettings):
    model_config = _ENV_CONFIG

    # Bearer token auth reuses the JWT stack. API key is a simpler alternative
    # for local development — set both variables to enable it.
    MCP_API_KEY: str | None = None
    MCP_API_KEY_USER_EMAIL: str | None = None

    @model_validator(mode="after")
    def _mcp_key_min_length(self) -> Self:
        if self.MCP_API_KEY is not None and len(self.MCP_API_KEY) < 32:
            raise ValueError("MCP_API_KEY must be at least 32 characters")
        return self


class CacheSettings(BaseSettings):
    model_config = _ENV_CONFIG

    CACHE_TTL_WORKSPACE: int = 300  # seconds; override per environment via env var
    CACHE_TTL_MEMBERSHIP: int = 300
    # Safety-net TTL — revocation propagates via explicit cache.delete(), not expiry.
    CACHE_TTL_API_KEY: int = 300


class CelerySettings(BaseSettings):
    model_config = _ENV_CONFIG

    # Restart workers after N tasks — prevents memory creep from asyncio.run() loops.
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = 200

    # extract_text: CPU-bound PDF parsing; 50 MB upload ceiling → ~2 min worst case.
    CELERY_EXTRACT_SOFT_TIME_LIMIT: int = 120  # SoftTimeLimitExceeded at this point
    CELERY_EXTRACT_TIME_LIMIT: int = 180  # SIGKILL if soft limit is ignored

    # embed_chunks: I/O-bound embedding API; ~500 chunks × ~100 ms/call ≈ 50 s typical.
    CELERY_EMBED_SOFT_TIME_LIMIT: int = 300
    CELERY_EMBED_TIME_LIMIT: int = 360

    # Worker concurrency per queue type. Default 2 is intentionally low for local dev
    # (git clone on a laptop). Set higher in production: extract→4, embed→10.
    CELERY_EXTRACT_CONCURRENCY: int = 2
    CELERY_EMBED_CONCURRENCY: int = 2


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
    MCPSettings,
    CacheSettings,
    CelerySettings,
    RateLimitSettings,
):
    model_config = _ENV_CONFIG

    @model_validator(mode="after")
    def _reject_dev_secrets_in_production(self) -> Self:
        if self.SECRET_KEY == _DEV_SECRET:
            raise ValueError(
                "SECRET_KEY must be set to a strong random value. "
                "Generate one with: "
                "python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        if (
            self.ENVIRONMENT in ("production", "staging")
            and self.POSTGRES_PASSWORD == "password"
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
