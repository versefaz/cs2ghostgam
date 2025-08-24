from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "cs2_analytics"
    postgres_user: str = "postgres"
    postgres_password: str = "cs2bet"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Feature settings
    feature_ttl: int = 3600  # seconds
    refresh_interval: int = 300  # seconds

    # API
    api_port: int = 8003

    class Config:
        env_file = ".env"

settings = Settings()
