import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import aioredis

logger = logging.getLogger(__name__)


class RedisSignalPublisher:
    """ส่งสัญญาณผ่าน Redis Pub/Sub และเก็บคิวถาวร"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        channels: Optional[Dict[str, str]] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password

        # Default channels
        self.channels = channels or {
            "signals": "betting:signals",
            "high_priority": "betting:signals:high",
            "critical": "betting:signals:critical",
            "executed": "betting:signals:executed",
            "monitoring": "betting:monitoring",
        }

        # Queue names
        self.queues = {
            "pending": "queue:signals:pending",
            "processing": "queue:signals:processing",
            "completed": "queue:signals:completed",
            "failed": "queue:signals:failed",
            "history": "queue:signals:history",
        }

        self.redis_client: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> bool:
        """เชื่อมต่อ Redis"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                encoding="utf-8",
            )
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """ปิดการเชื่อมต่อ"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            self._connected = False
            logger.info("Disconnected from Redis")

    async def publish_signal(self, signal: "BettingSignal", persist: bool = True) -> bool:
        """ส่งสัญญาณไปยัง channel และเก็บในคิว"""
        if not self._connected:
            await self.connect()

        try:
            # Prepare signal data
            signal_data = signal.to_dict()
            signal_data["published_at"] = datetime.utcnow().isoformat()
            signal_json = json.dumps(signal_data)

            # Determine channel based on priority
            channel = self._get_channel_for_signal(signal)

            # Publish to channel
            subscribers = await self.redis_client.publish(channel, signal_json)
            logger.info(
                f"Published signal {signal.signal_id} to {channel} ({subscribers} subscribers)"
            )

            # Persist to queue if requested
            if persist:
                await self._persist_signal(signal_data)

            # Store in hash for quick lookup
            await self.redis_client.hset(
                f"signal:{signal.signal_id}",
                mapping={
                    "data": signal_json,
                    "status": signal.status.value,
                    "created_at": signal.created_at.isoformat(),
                    "match_id": signal.match_id,
                },
            )

            # Set expiry
            if signal.expires_at:
                ttl = int((signal.expires_at - datetime.utcnow()).total_seconds())
                if ttl > 0:
                    await self.redis_client.expire(f"signal:{signal.signal_id}", ttl)

            # Update indices
            await self._update_indices(signal)

            return True

        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            return False

    async def _persist_signal(self, signal_data: Dict[str, Any]):
        """เก็บสัญญาณในคิวถาวร"""
        try:
            # Add to pending queue
            await self.redis_client.lpush(self.queues["pending"], json.dumps(signal_data))

            # Add to sorted set for time-based queries
            score = datetime.fromisoformat(signal_data["created_at"]).timestamp()

            await self.redis_client.zadd("signals:by_time", score, signal_data["signal_id"])

            # Add to history list (capped)
            await self.redis_client.lpush(self.queues["history"], json.dumps(signal_data))

            # Keep only last 10000 signals in history
            await self.redis_client.ltrim(self.queues["history"], 0, 9999)

            logger.debug(f"Persisted signal {signal_data['signal_id']} to queues")

        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

    async def _update_indices(self, signal: "BettingSignal"):
        """อัพเดท indices สำหรับการค้นหา"""
        try:
            # Index by match_id
            await self.redis_client.sadd(
                f"match:{signal.match_id}:signals", signal.signal_id
            )

            # Index by status
            await self.redis_client.sadd(
                f"signals:status:{signal.status.value}", signal.signal_id
            )

            # Index by priority
            await self.redis_client.sadd(
                f"signals:priority:{signal.priority.value}", signal.signal_id
            )

            # Index by source
            if signal.source:
                await self.redis_client.sadd(
                    f"signals:source:{signal.source}", signal.signal_id
                )

            # Index by strategy
            if signal.strategy:
                await self.redis_client.sadd(
                    f"signals:strategy:{signal.strategy}", signal.signal_id
                )

        except Exception as e:
            logger.error(f"Failed to update indices: {e}")

    def _get_channel_for_signal(self, signal: "BettingSignal") -> str:
        """เลือก channel ตาม priority"""
        from models.signal import SignalPriority

        if signal.priority == SignalPriority.CRITICAL:
            return self.channels["critical"]
        elif signal.priority == SignalPriority.HIGH:
            return self.channels["high_priority"]
        else:
            return self.channels["signals"]

    async def get_pending_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """ดึงสัญญาณที่รอดำเนินการ"""
        if not self._connected:
            await self.connect()

        try:
            signals: List[Dict[str, Any]] = []
            items = await self.redis_client.lrange(self.queues["pending"], 0, limit - 1)

            for item in items:
                try:
                    signal_data = json.loads(item)
                    signals.append(signal_data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in queue: {item}")

            return signals

        except Exception as e:
            logger.error(f"Failed to get pending signals: {e}")
            return []

    async def move_to_processing(self, signal_id: str) -> bool:
        """ย้ายสัญญาณไป processing queue"""
        try:
            # Get signal data
            signal_data = await self.redis_client.hget(f"signal:{signal_id}", "data")

            if not signal_data:
                return False

            # Move from pending to processing
            await self.redis_client.lrem(self.queues["pending"], 1, signal_data)

            await self.redis_client.lpush(self.queues["processing"], signal_data)

            # Update status
            await self.redis_client.hset(
                f"signal:{signal_id}", "status", "processing"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to move signal to processing: {e}")
            return False

    async def mark_completed(self, signal_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """ทำเครื่องหมายสัญญาณว่าเสร็จสมบูรณ์"""
        try:
            # Get signal data
            signal_data = await self.redis_client.hget(f"signal:{signal_id}", "data")

            if not signal_data:
                return False

            # Parse and update
            data = json.loads(signal_data)
            data["completed_at"] = datetime.utcnow().isoformat()
            data["status"] = "completed"

            if result:
                data["result"] = result

            updated_json = json.dumps(data)

            # Move to completed queue
            await self.redis_client.lrem(self.queues["processing"], 1, signal_data)

            await self.redis_client.lpush(self.queues["completed"], updated_json)

            # Update hash
            await self.redis_client.hset(
                f"signal:{signal_id}",
                mapping={
                    "data": updated_json,
                    "status": "completed",
                    "completed_at": data["completed_at"],
                },
            )

            # Publish to executed channel
            await self.redis_client.publish(self.channels["executed"], updated_json)

            return True

        except Exception as e:
            logger.error(f"Failed to mark signal as completed: {e}")
            return False

    async def get_signal_stats(self) -> Dict[str, Any]:
        """ดึงสถิติของสัญญาณ"""
        if not self._connected:
            await self.connect()

        try:
            stats: Dict[str, Any] = {
                "pending": await self.redis_client.llen(self.queues["pending"]),
                "processing": await self.redis_client.llen(self.queues["processing"]),
                "completed": await self.redis_client.llen(self.queues["completed"]),
                "failed": await self.redis_client.llen(self.queues["failed"]),
                "total_history": await self.redis_client.llen(self.queues["history"]),
                "active_signals": await self.redis_client.zcard("signals:by_time"),
            }

            # Get counts by priority
            for priority in ["1", "2", "3", "4"]:
                count = await self.redis_client.scard(f"signals:priority:{priority}")
                stats[f"priority_{priority}"] = count

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
