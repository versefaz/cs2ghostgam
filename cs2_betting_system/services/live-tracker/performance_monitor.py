import json
from datetime import datetime
import redis
from cs2_betting_system.config import settings

class PerformanceMonitor:
    def __init__(self):
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

    def log_metrics(self, success: int, failed: int):
        total = success + failed
        rate = (success / total * 100) if total else 0
        metric = {
            'timestamp': datetime.now().isoformat(),
            'successful_scrapes': success,
            'failed_scrapes': failed,
            'scrape_success_rate': rate,
        }
        self.redis.lpush('performance_metrics', json.dumps(metric))
        self.redis.ltrim('performance_metrics', 0, 999)
