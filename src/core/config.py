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


class AppSettings(BaseSettings):
    model_config = _ENV_CONFIG

    UPLOAD_DIR: str = "/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50


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


_DEV_SECRET = "dev-secret-key-change-before-production"


class Settings(
    DatabaseSettings,
    RedisSettings,
    AuthSettings,
    AISettings,
    ObservabilitySettings,
    AppSettings,
    MCPSettings,
):
    model_config = _ENV_CONFIG

    @model_validator(mode="after")
    def _reject_dev_secrets_in_production(self) -> Self:
        if self.ENVIRONMENT in ("production", "staging"):
            if self.SECRET_KEY == _DEV_SECRET:
                raise ValueError(
                    "SECRET_KEY must be changed before deploying to production"
                )
            if self.POSTGRES_PASSWORD == "password":
                raise ValueError(
                    "POSTGRES_PASSWORD must be changed before deploying to production"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for direct import; use Depends(get_settings) in FastAPI.
settings: Settings = get_settings()
