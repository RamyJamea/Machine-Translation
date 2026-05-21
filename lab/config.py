import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env")
    )

    HUGGINGFACE_TOKEN: str
    WANDB_API_KEY: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
