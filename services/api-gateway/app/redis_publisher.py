import redis
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class RedisPublisher:
     def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                  password: Optional[str] = None, max_retries: int = 3, retry_delay: int = 1):
         """Initialize Redis publisher with connection pooling and retry logic"""
         self.pool = redis.ConnectionPool(
             host=host,
             port=port,
             db=db,
             password=password,
             decode_responses=True,
             max_connections=50,
             socket_keepalive=True,
             socket_keepalive_options={
                 1: 1,  # TCP_KEEPIDLE
                 2: 1,  # TCP_KEEPINTVL
                 3: 3,  # TCP_KEEPCNT
             }
         )
         self.redis_client = redis.Redis(connection_pool=self.pool)
         self.max_retries = max_retries
         self.retry_delay = retry_delay

         # Test connection
         self._test_connection()

     def _test_connection(self):
         """Test Redis connection on init"""
         try:
             self.redis_client.ping()
             logger.info("✅ Redis connection successful")
         except redis.ConnectionError as e:
             logger.error(f"❌ Redis connection failed: {e}")
             raise

     def publish(self, channel: str, data: Dict[Any, Any], priority: str = 'normal') -> bool:
         """
         Publish data to Redis channel with retry logic

         Args:
             channel: Redis channel name
             data: Data to publish
             priority: 'high', 'normal', 'low'
         """
         # Add metadata
         message = {
             'data': data,
             'timestamp': datetime.utcnow().isoformat(),
             'priority': priority
         }

         retries = 0
         payload = json.dumps(message)
         while retries < self.max_retries:
             try:
                 # Publish to main channel
                 subscribers = self.redis_client.publish(channel, payload)

                 # Also push to queue for persistence
                 queue_key = f"queue:{channel}"
                 if priority == 'high':
                     self.redis_client.lpush(queue_key, payload)
                 else:
                     self.redis_client.rpush(queue_key, payload)

                 # Log metrics
                 self._log_metrics(channel, len(payload), subscribers)

                 logger.info(f"✅ Published to {channel} ({subscribers} subscribers)")
                 return True

             except redis.RedisError as e:
                 retries += 1
                 logger.warning(f"⚠️ Publish attempt {retries} failed: {e}")
                 if retries < self.max_retries:
                     time.sleep(self.retry_delay * retries)
                 else:
                     logger.error(f"❌ Failed to publish after {self.max_retries} attempts")
                     self._store_failed_message(channel, message)
                     return False

     def publish_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, bool]:
         """Publish multiple messages efficiently using pipeline"""
         results: Dict[str, bool] = {}
         pipeline = self.redis_client.pipeline()

         try:
             for msg in messages:
                 channel = msg.get('channel')
                 data = msg.get('data')
                 priority = msg.get('priority', 'normal')
                 message = {
                     'data': data,
                     'timestamp': datetime.utcnow().isoformat(),
                     'priority': priority,
                 }
                 payload = json.dumps(message)
                 pipeline.publish(channel, payload)
                 queue_key = f"queue:{channel}"
                 if priority == 'high':
                     pipeline.lpush(queue_key, payload)
                 else:
                     pipeline.rpush(queue_key, payload)

             pipeline.execute()
             for msg in messages:
                 results[msg['channel']] = True
             logger.info(f"✅ Batch published {len(messages)} messages")
         except redis.RedisError as e:
             logger.error(f"❌ Batch publish failed: {e}")
             for msg in messages:
                 results[msg['channel']] = False

         return results

     def _log_metrics(self, channel: str, size: int, subscribers: int):
         """Log publishing metrics"""
         metrics = {
             'channel': channel,
             'message_size': size,
             'subscribers': subscribers,
             'timestamp': datetime.utcnow().isoformat()
         }
         try:
             self.redis_client.lpush('metrics:publish', json.dumps(metrics))
             # Keep only last 1000 metrics
             self.redis_client.ltrim('metrics:publish', 0, 999)
         except redis.RedisError as e:
             logger.warning(f"Metrics logging failed: {e}")

     def _store_failed_message(self, channel: str, message: dict):
         """Store failed messages for retry"""
         failed_msg = {
             'channel': channel,
             'message': message,
             'failed_at': datetime.utcnow().isoformat()
         }
         try:
             self.redis_client.rpush('failed_messages', json.dumps(failed_msg))
         except redis.RedisError as e:
             logger.error(f"Failed to store failed message: {e}")

     def get_queue_size(self, channel: str) -> int:
         """Get number of messages in queue"""
         try:
             return int(self.redis_client.llen(f"queue:{channel}"))
         except redis.RedisError as e:
             logger.error(f"Failed to get queue size: {e}")
             return 0

     def health_check(self) -> Dict[str, Any]:
         """Health check for monitoring"""
         try:
             ping = self.redis_client.ping()
             info = self.redis_client.info()
             return {
                 'status': 'healthy' if ping else 'unhealthy',
                 'connected_clients': info.get('connected_clients', 0),
                 'used_memory': info.get('used_memory_human', 'unknown'),
                 'uptime': info.get('uptime_in_seconds', 0)
             }
         except Exception as e:
             return {
                 'status': 'unhealthy',
                 'error': str(e)
             }
