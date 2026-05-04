from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://user:password@localhost:5432/knowledge_platform"
    )

    # ─── Redis / Celery ──────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ─── Security / JWT ──────────────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-before-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ─── LLM — PydanticAI model strings (provider:model-name) ───────────────
    # Any provider supported by PydanticAI works here; change without code edits.
    # gemini-2.0-flash is the cheap default for local dev and CI.
    LLM_MODEL: str = "google-gla:gemini-2.0-flash"
    # Stronger model for tasks that need higher reasoning (summarisation, eval).
    LLM_STRONG_MODEL: str = "anthropic:claude-sonnet-4-5"

    # ─── Embeddings ──────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_MODEL: str = "text-embedding-005"
    EMBEDDING_API_KEY: str = ""
    # MUST match the pgvector column width: Vector(settings.EMBEDDING_DIMENSIONS).
    # Switching providers requires re-embedding all documents — the re-indexing
    # pipeline handles this via the chunk version column.
    EMBEDDING_DIMENSIONS: int = 768

    # ─── Observability ───────────────────────────────────────────────────────
    LOGFIRE_TOKEN: str = ""
    ENVIRONMENT: str = "development"

    # ─── File Uploads ────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "/data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for direct import; use Depends(get_settings) in FastAPI.
settings: Settings = get_settings()
