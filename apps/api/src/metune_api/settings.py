from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://metune:metune@localhost:5432/metune"
    session_secret: str = "dev-insecure-change-me"
    cookie_secure: bool = True
    session_max_age_s: int = 12 * 3600

    admin_user: str = "admin"
    admin_password_hash: str = ""          # argon2, from `make hash-password`

    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = 2048
    gemini_api_key: str = ""
    metune_config: str = "config.yaml"

    worker_concurrency: int = 4
    file_timeout_s: int = 600
    max_attempts: int = 3
    git_sha: str = "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()
