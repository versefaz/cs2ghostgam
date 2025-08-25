"""Configuration settings for HLTV Scraper"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "hltv-scraper"
    environment: str = "production"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://user:pass@postgres:5432/cs2_analytics"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Kafka
    kafka_servers: str = "kafka:9092"

    # Scraping
    browser_pool_size: int = 5
    headless_mode: bool = True
    max_retries: int = 3
    request_timeout: int = 30

    # Proxy
    proxy_enabled: bool = True
    proxy_rotation_interval: int = 300

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Monitoring
    sentry_dsn: Optional[str] = None
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
