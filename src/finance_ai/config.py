from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="dev", validation_alias="APP_ENV")
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct", validation_alias="OLLAMA_MODEL"
    )
    ollama_base_url: str | None = Field(
        default=None, validation_alias="OLLAMA_BASE_URL"
    )
    embedding_mode: str = Field(
        default="sentence", validation_alias="FINANCE_AI_EMBEDDING_MODE"
    )
    vectorstore_root: str = Field(
        default="data/vectorstore", validation_alias="FINANCE_AI_VECTORSTORE_ROOT"
    )
    low_grounding_threshold: float = Field(
        default=0.45, validation_alias="FINANCE_AI_LOW_GROUNDING_THRESHOLD"
    )
    show_debug_panels: bool = Field(
        default=False, validation_alias="FINANCE_AI_SHOW_DEBUG_PANELS"
    )
    agent_planner_mode: str = Field(
        default="hybrid", validation_alias="FINANCE_AI_AGENT_PLANNER_MODE"
    )
    enable_rerank: bool = Field(
        default=True, validation_alias="FINANCE_AI_ENABLE_RERANK"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="FINANCE_AI_RERANK_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
