"""
config.py — Single source of truth for all application settings.

All environment variables are read here and nowhere else.
Import `settings` from this module throughout the application:
    from app.config import settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    Pydantic validates types and raises at startup — not at runtime.
    """

    upload_dir: str = "./uploads"
    max_upload_size_bytes: int = 50 * 1024 * 1024
    allowed_extensions: list[str] = [".pdf"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "Research Contradiction Analyzer"
    app_version: str = "0.1.0"
    app_port: int = Field(default=8000, ge=1024, le=65535)
    app_secret_key: str = Field(default="change-me-in-production", min_length=16)
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "mistral", "ollama"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=8192)
    llm_request_timeout: int = Field(default=60, ge=10, le=300)

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_api_key: str = ""
    qdrant_collection_claims: str = "claims"
    qdrant_collection_evidence: str = "evidence"
    qdrant_timeout: int = Field(default=30, ge=5, le=120)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "rca_db"
    postgres_user: str = "rca_user"
    postgres_password: str = "change-me"
    database_url: str = ""  # overrides individual fields if set
    db_pool_size: int = Field(default=10, ge=1, le=50)
    db_max_overflow: int = Field(default=20, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=5, le=120)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = Field(default=3600, ge=60)
    redis_embedding_ttl_seconds: int = Field(default=86400, ge=3600)
    redis_nli_ttl_seconds: int = Field(default=604800, ge=3600)  # 7 days

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_timeout: int = Field(default=300, ge=30, le=3600)

    # ── ML Models ─────────────────────────────────────────────────────────────
    embedding_model: str = "jinaai/jina-embeddings-v2-base-en"
    embedding_dimension: int = Field(default=768, ge=64, le=4096)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_contradiction_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = Field(default=5, ge=1, le=20)
    hf_cache_dir: str = "./models"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_dense_top_k: int = Field(default=20, ge=5, le=100)
    retrieval_bm25_top_k: int = Field(default=20, ge=5, le=100)
    retrieval_rrf_k: int = Field(default=60, ge=10, le=200)
    max_candidate_pairs: int = Field(default=50, ge=5, le=200)

    # ── External APIs ─────────────────────────────────────────────────────────
    semantic_scholar_api_key: str = ""
    arxiv_max_results: int = Field(default=20, ge=1, le=100)
    arxiv_timeout: int = Field(default=30, ge=5, le=120)

    # ── Observability ─────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Allow CORS_ORIGINS as comma-separated string in .env."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        """Build database URL from parts if not explicitly provided."""
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Fail fast if production is misconfigured."""
        if self.app_env == "production":
            if self.app_secret_key == "change-me-in-production":
                raise ValueError("APP_SECRET_KEY must be changed in production")
            if not self.openai_api_key and self.llm_provider == "openai":
                raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        return self

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def langfuse_active(self) -> bool:
        return self.langfuse_enabled and bool(self.langfuse_public_key)

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached settings singleton.
    Use this in FastAPI dependencies:
        Depends(get_settings)
    Use this everywhere else:
        from app.config import settings
    """
    return Settings()


# Module-level singleton — import this directly in non-DI contexts
settings: Settings = get_settings()
