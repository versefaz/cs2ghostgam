import json
import redis
from cs2_betting_system.config import settings

class SignalGenerator:
    def __init__(self):
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

    def emit_signal(self, prediction_record: dict):
        ev = prediction_record.get('prediction', {}).get('ev')
        conf = prediction_record.get('prediction', {}).get('confidence')
        if ev and conf and ev > settings.MIN_VALUE - 1 and conf > settings.MIN_CONFIDENCE:
            self.redis.publish('betting_signals', json.dumps(prediction_record))
