import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict

import aiohttp
import socketio

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - fallback stubs
    class _Metric:
        def labels(self, **kwargs):
            return self
        def inc(self, *_a, **_k):
            return self
        def set(self, *_a, **_k):
            return self
        def time(self):
            class _T:
                def __enter__(self, *a, **k):
                    return self
                def __exit__(self, *a, **k):
                    return False
            return _T()
    Counter = Gauge = Histogram = _Metric

from .config import config
from .message_parser import MessageParser
from .data_validator import DataValidator
from .redis_publisher import RedisPublisher
from .db_writer import DatabaseWriter
from .health_monitor import HealthMonitor

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

messages_received = Counter('hltv_messages_received_total', 'Total messages received', ['event_type'])
messages_processed = Counter('hltv_messages_processed_total', 'Total messages processed', ['event_type'])
messages_failed = Counter('hltv_messages_failed_total', 'Total messages failed', ['event_type', 'reason'])
connection_status = Gauge('hltv_connection_status', 'Connection status (1=connected, 0=disconnected)')
processing_time = Histogram('hltv_processing_time_seconds', 'Message processing time', ['event_type'])
reconnect_attempts = Counter('hltv_reconnect_attempts_total', 'Total reconnection attempts')

class HLTVSocketClient:
    """HLTV Socket.io client with auto-reconnection and fallback"""

    def __init__(self):
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=config.HLTV_RECONNECT_ATTEMPTS,
            reconnection_delay=config.HLTV_INITIAL_BACKOFF,
            reconnection_delay_max=config.HLTV_MAX_BACKOFF,
            logger=False,
            engineio_logger=False,
        )

        self.parser = MessageParser()
        self.validator = DataValidator()
        self.redis_publisher = RedisPublisher()
        self.db_writer = DatabaseWriter()
        self.health_monitor = HealthMonitor(self)

        self.connected = False
        self.fallback_mode = False
        self.last_message_time = datetime.now()
        self.reconnect_count = 0
        self.message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=config.MESSAGE_QUEUE_SIZE)
        self.processing_tasks: list[asyncio.Task] = []

        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.event
        async def connect():
            logger.info("Connected to HLTV Socket.io server")
            self.connected = True
            self.fallback_mode = False
            self.reconnect_count = 0
            connection_status.set(1)
            await self.redis_publisher.publish_system_event({
                "type": "connection_established",
                "timestamp": datetime.now().isoformat(),
            })

        @self.sio.event
        async def connect_error(data):
            logger.error(f"Connection error: {data}")
            connection_status.set(0)
            messages_failed.labels(event_type='connection', reason='error').inc()

        @self.sio.event
        async def disconnect():
            logger.warning("Disconnected from HLTV Socket.io server")
            self.connected = False
            connection_status.set(0)
            await self.redis_publisher.publish_system_event({
                "type": "connection_lost",
                "timestamp": datetime.now().isoformat(),
            })

        # HLTV events
        @self.sio.on('scoreUpdate')
        async def on_score_update(data):
            await self._handle_message('scoreUpdate', data)

        @self.sio.on('matchData')
        async def on_match_data(data):
            await self._handle_message('matchData', data)

        @self.sio.on('roundEnd')
        async def on_round_end(data):
            await self._handle_message('roundEnd', data)

        @self.sio.on('playerKill')
        async def on_player_kill(data):
            await self._handle_message('playerKill', data)

        @self.sio.on('bombPlanted')
        async def on_bomb_planted(data):
            await self._handle_message('bombPlanted', data)

        @self.sio.on('bombDefused')
        async def on_bomb_defused(data):
            await self._handle_message('bombDefused', data)

        @self.sio.on('roundStart')
        async def on_round_start(data):
            await self._handle_message('roundStart', data)

        @self.sio.on('mapEnd')
        async def on_map_end(data):
            await self._handle_message('mapEnd', data)

        @self.sio.on('matchEnd')
        async def on_match_end(data):
            await self._handle_message('matchEnd', data)

    async def _handle_message(self, event_type: str, data: Any):
        try:
            messages_received.labels(event_type=event_type).inc()
            self.last_message_time = datetime.now()
            await self.message_queue.put({
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now(),
            })
        except asyncio.QueueFull:
            logger.error(f"Message queue full, dropping message: {event_type}")
            messages_failed.labels(event_type=event_type, reason='queue_full').inc()

    async def connect(self) -> bool:
        try:
            logger.info(f"Connecting to HLTV Socket.io: {config.HLTV_SOCKET_URL}")
            await self.sio.connect(config.HLTV_SOCKET_URL, transports=['websocket', 'polling'])
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            reconnect_attempts.inc()
            if not self.fallback_mode:
                logger.info("Attempting fallback to HTTP long-polling")
                return await self._connect_with_fallback()
            return False

    async def _connect_with_fallback(self) -> bool:
        try:
            self.fallback_mode = True
            await self.sio.connect(config.HLTV_SOCKET_URL, transports=['polling'])
            logger.info("Connected via HTTP long-polling fallback")
            return True
        except Exception as e:
            logger.error(f"Fallback connection failed: {e}")
            return False

    async def _process_messages(self):
        batch: list[Dict[str, Any]] = []
        last_batch_time = time.time()
        while True:
            try:
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=config.BATCH_TIMEOUT)
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass

                now = time.time()
                if (len(batch) >= config.BATCH_SIZE) or (batch and now - last_batch_time >= config.BATCH_TIMEOUT):
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = now
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, batch: list[Dict[str, Any]]):
        for message in batch:
            event_type = message['event_type']
            data = message['data']
            ts = message['timestamp']
            with processing_time.labels(event_type=event_type).time():
                try:
                    parsed = self.parser.parse(event_type, data)
                    if not self.validator.validate(event_type, parsed):
                        messages_failed.labels(event_type=event_type, reason='validation').inc()
                        continue
                    parsed.setdefault('_metadata', {})
                    parsed['_metadata'].update({
                        'event_type': event_type,
                        'received_at': ts.isoformat(),
                        'processed_at': datetime.now().isoformat(),
                        'fallback_mode': self.fallback_mode,
                    })
                    await self.redis_publisher.publish_event(event_type, parsed)
                    await self.db_writer.write_event(event_type, parsed)
                    if event_type in ['roundEnd', 'mapEnd', 'matchEnd']:
                        await self._trigger_feature_calculation(parsed)
                    messages_processed.labels(event_type=event_type).inc()
                except Exception as e:
                    logger.error(f"Error processing {event_type}: {e}")
                    messages_failed.labels(event_type=event_type, reason='processing').inc()

    async def _trigger_feature_calculation(self, data: Dict[str, Any]):
        try:
            match_id = data.get('match_id')
            if not match_id:
                return
            async with aiohttp.ClientSession() as session:
                url = f"{config.FEATURE_BUILDER_URL}/features/refresh"
                async with session.post(url, json={"match_id": match_id}) as resp:
                    if resp.status == 200:
                        logger.info(f"Triggered feature calculation for match {match_id}")
                    else:
                        logger.warning(f"Feature calculation trigger failed: {resp.status}")
        except Exception as e:
            logger.error(f"Error triggering feature calculation: {e}")

    async def reconnect_with_backoff(self) -> bool:
        backoff = config.HLTV_INITIAL_BACKOFF
        while self.reconnect_count < config.HLTV_RECONNECT_ATTEMPTS:
            self.reconnect_count += 1
            reconnect_attempts.inc()
            logger.info(f"Reconnection attempt {self.reconnect_count}/{config.HLTV_RECONNECT_ATTEMPTS}")
            if await self.connect():
                return True
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, config.HLTV_MAX_BACKOFF)
        logger.error("Max reconnection attempts reached")
        return False

    async def start(self) -> bool:
        if not await self.connect():
            logger.error("Initial connection failed")
            return False
        for _ in range(3):
            self.processing_tasks.append(asyncio.create_task(self._process_messages()))
        asyncio.create_task(self.health_monitor.start())
        asyncio.create_task(self._monitor_connection())
        logger.info("HLTV Socket.io client started successfully")
        return True

    async def _monitor_connection(self):
        while True:
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)
            if datetime.now() - self.last_message_time > timedelta(minutes=5) and self.connected:
                logger.warning("No messages received for 5 minutes, checking connection")
                if not self.sio.connected:
                    await self.reconnect_with_backoff()

    async def stop(self):
        logger.info("Stopping HLTV Socket.io client")
        for t in self.processing_tasks:
            t.cancel()
        await self.sio.disconnect()
        await self.redis_publisher.close()
        await self.db_writer.close()
        logger.info("HLTV Socket.io client stopped")
