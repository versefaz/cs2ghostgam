import json
import time
import redis

from cs2_betting_system.config import settings

r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

if __name__ == '__main__':
    while True:
        metrics = r.lrange('performance_metrics', 0, 10)
        print('Latest metrics:')
        for m in metrics:
            print(m)
        time.sleep(30)
