"""Configuration settings for AI Safety Evaluation Framework."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_prefix="SAFETY_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API Keys (all optional)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    together_api_key: str | None = None

    # Other settings
    default_model: str = "anthropic/claude-3-sonnet-20240229"
    db_path: str = "safety_eval.duckdb"
    cache_dir: str | None = None


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
