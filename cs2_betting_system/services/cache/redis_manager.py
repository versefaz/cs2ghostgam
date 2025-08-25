import redis
from cs2_betting_system.config import settings

class RedisManager:
    def __init__(self, db: int = 0):
        self.client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=db, decode_responses=True)

    def get(self, key: str):
        return self.client.get(key)

    def setex(self, key: str, ttl: int, value: str):
        return self.client.setex(key, ttl, value)
