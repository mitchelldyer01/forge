"""Settings via pydantic-settings + .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FORGE configuration loaded from environment / .env file."""

    db_path: str = "data/forge.db"
    llama_url: str = "http://127.0.0.1:8080"
    llama_timeout: float = 60.0
    log_level: str = "INFO"

    model_config = {"env_prefix": "FORGE_"}


def get_settings() -> Settings:
    """Return a Settings instance (reads from env / .env)."""
    return Settings()
