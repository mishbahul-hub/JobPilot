from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://jobpilot:jobpilot@localhost:5432/jobpilot"

    # LLM + embeddings
    google_api_key: str = ""  # one key for chat and embeddings
    llm_model: str = "gemini-3.5-flash-lite"  # fast, reliable on the free tier
    # Tried in order when the main model errors (Gemini often answers 503 under load).
    llm_fallback_models: list[str] = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]
    llm_requests_per_minute: int = 15  # free-tier flash-lite allows more; lower if you see 429s
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embed_texts_per_minute: int = 90  # Gemini free tier allows 100

    # Job source (https://developer.adzuna.com)
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    adzuna_country: str = "in"

    # API
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    log_level: str = "INFO"
    # Per client IP, per window (protects the free-tier quotas on a public demo).
    chat_limit_per_10min: int = 30
    resume_limit_per_hour: int = 5
    search_limit_per_min: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
