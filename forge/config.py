"""Settings via pydantic-settings + .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FORGE configuration, loaded from environment / .env file."""

    db_path: str = "data/forge.db"
    llama_url: str = "http://127.0.0.1:8080"
    llama_timeout: float = 120.0
    default_swarm_size: int = 30
    simulation_rounds: int = 3
    relevance_threshold: float = 0.6
    log_level: str = "INFO"

    model_config = {"env_prefix": "FORGE_"}
