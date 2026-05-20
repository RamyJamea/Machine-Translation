import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    HF_TOKEN: str

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
