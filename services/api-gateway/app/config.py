import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Application configuration"""

    # HLTV Configuration
    HLTV_SOCKET_URL: str = os.getenv("HLTV_SOCKET_URL", "wss://scorebot-secure.hltv.org")
    HLTV_API_URL: str = os.getenv("HLTV_API_URL", "https://hltv-api.vercel.app/api")
    HLTV_RECONNECT_ATTEMPTS: int = int(os.getenv("HLTV_RECONNECT_ATTEMPTS", "10"))
    HLTV_INITIAL_BACKOFF: float = float(os.getenv("HLTV_INITIAL_BACKOFF", "1.0"))
    HLTV_MAX_BACKOFF: float = float(os.getenv("HLTV_MAX_BACKOFF", "60.0"))

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # PostgreSQL/TimescaleDB Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/cs2_betting",
    )

    # Feature Builder Service (default aligned to 8003)
    FEATURE_BUILDER_URL: str = os.getenv("FEATURE_BUILDER_URL", "http://localhost:8003")

    # Monitoring
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

    # Message Processing
    MESSAGE_QUEUE_SIZE: int = int(os.getenv("MESSAGE_QUEUE_SIZE", "10000"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    BATCH_TIMEOUT: float = float(os.getenv("BATCH_TIMEOUT", "5.0"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

config = Config()
