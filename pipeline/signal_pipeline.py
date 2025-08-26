import asyncio
from typing import Dict, Any, List
import logging
from datetime import datetime

from core.signal_generator import SignalGenerator, SignalConfig
from publishers.redis_publisher import RedisSignalPublisher
from models.signal import BettingSignal, SignalStatus

logger = logging.getLogger(__name__)


class SignalPipeline:
    """ระบบประมวลผลสัญญาณแบบครบวงจร"""

    def __init__(self, generator: SignalGenerator, publisher: RedisSignalPublisher, config: Dict[str, Any] = None):
        self.generator = generator
        self.publisher = publisher
        self.config = config or {}

        # Pipeline settings
        self.batch_size = int(self.config.get("batch_size", 10))
        self.process_interval = float(self.config.get("process_interval", 1))  # seconds
        self.retry_attempts = int(self.config.get("retry_attempts", 3))

        # State management
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_processed = datetime.utcnow()

        # Queues
        self.input_queue: asyncio.Queue[BettingSignal] = asyncio.Queue()
        self.output_queue: asyncio.Queue[BettingSignal] = asyncio.Queue()

    async def start(self):
        """เริ่มต้น pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return

        self.is_running = True
        logger.info("Starting signal pipeline...")

        # Connect to Redis
        await self.publisher.connect()

        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_input_queue()),
            asyncio.create_task(self._process_output_queue()),
            asyncio.create_task(self._monitor_pipeline()),
            asyncio.create_task(self._cleanup_expired_signals()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.stop()

    async def stop(self):
        """หยุด pipeline"""
        logger.info("Stopping signal pipeline...")
        self.is_running = False
        await self.publisher.disconnect()

    async def process_match_data(self, match_id: str, data: Dict[str, Any]) -> List[BettingSignal]:
        """ประมวลผลข้อมูลแมทช์และสร้างสัญญาณ"""
        signals: List[BettingSignal] = []

        try:
            # Generate signals from different sources
            tasks = []

            # Value betting signals
            if data.get("our_probability") and data.get("market_odds"):
                tasks.append(
                    self.generator.generate_value_signal(
                        match_id=match_id,
                        our_prob=float(data["our_probability"]),
                        market_odds=float(data["market_odds"]),
                        side=data.get("side"),
                        source="value_analysis",
                    )
                )

            # ML signals
            if data.get("ml_predictions"):
                tasks.append(
                    self.generator.generate_ml_signal(
                        match_id=match_id,
                        predictions=data["ml_predictions"],
                        source="ml_model",
                    )
                )

            # Pattern signals
            if data.get("pattern_data"):
                tasks.append(
                    self.generator.generate_pattern_signal(
                        match_id=match_id,
                        pattern_data=data["pattern_data"],
                        source="pattern_detection",
                    )
                )

            # Arbitrage signals
            if data.get("arbitrage_odds"):
                tasks.append(
                    self.generator.generate_arbitrage_signal(
                        match_id=match_id, odds_dict=data["arbitrage_odds"], source="arbitrage_scanner"
                    )
                )

            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid signals
            for result in results:
                if isinstance(result, BettingSignal):
                    signals.append(result)
                elif isinstance(result, list):  # Arbitrage returns list
                    signals.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Signal generation error: {result}")

            # Validate and filter signals
            validated_signals: List[BettingSignal] = []
            for signal in signals:
                if signal and await self.generator.validate_signal(signal):
                    validated_signals.append(signal)
                    await self.generator.add_active_signal(signal)

            return validated_signals

        except Exception as e:
            logger.error(f"Error processing match data: {e}")
            return []

    async def submit_signal(self, signal: BettingSignal):
        """ส่งสัญญาณเข้า pipeline"""
        await self.input_queue.put(signal)

    async def submit_batch(self, signals: List[BettingSignal]):
        """ส่งสัญญาณเป็นชุด"""
        for signal in signals:
            await self.input_queue.put(signal)

    async def _process_input_queue(self):
        """ประมวลผล input queue"""
        while self.is_running:
            try:
                # Get batch of signals
                batch: List[BettingSignal] = []
                deadline = asyncio.get_event_loop().time() + self.process_interval

                while len(batch) < self.batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        signal = await asyncio.wait_for(self.input_queue.get(), timeout=timeout)
                        batch.append(signal)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                if batch:
                    await self._process_signal_batch(batch)

                await asyncio.sleep(0.1)  # Small delay

            except Exception as e:
                logger.error(f"Error in input queue processing: {e}")
                await asyncio.sleep(1)

    async def _process_signal_batch(self, signals: List[BettingSignal]):
        """ประมวลผลสัญญาณเป็นชุด"""
        for signal in signals:
            try:
                # Enrich signal with additional data
                await self._enrich_signal(signal)

                # Apply filters
                if await self._apply_filters(signal):
                    # Add to output queue
                    await self.output_queue.put(signal)
                    self.processed_count += 1
                else:
                    logger.debug(f"Signal {signal.signal_id} filtered out")

            except Exception as e:
                logger.error(f"Error processing signal {signal.signal_id}: {e}")
                self.error_count += 1

    async def _enrich_signal(self, signal: BettingSignal):
        """เพิ่มข้อมูลเสริมให้สัญญาณ"""
        try:
            # Add market context
            signal.metadata.setdefault("market_conditions", await self._get_market_conditions())

            # Add risk assessment
            signal.metadata.setdefault("risk_score", await self._calculate_risk_score(signal))

            # Add execution instructions
            if signal.odds:
                min_odds = signal.odds * 0.95
            else:
                min_odds = None
            signal.metadata.setdefault(
                "execution",
                {
                    "min_odds": min_odds,  # Accept 5% slippage
                    "max_stake": signal.stake * 1.1,  # Allow 10% increase
                    "timeout_seconds": 30,
                    "retry_count": 2,
                },
            )

        except Exception as e:
            logger.error(f"Error enriching signal: {e}")

    async def _apply_filters(self, signal: BettingSignal) -> bool:
        """กรองสัญญาณตามเงื่อนไข"""
        try:
            # Check if expired
            if signal.expires_at and signal.expires_at < datetime.utcnow():
                signal.status = SignalStatus.EXPIRED
                return False

            # Check minimum requirements
            if signal.ev < 0.03:  # Less than 3% EV
                return False

            if signal.confidence < 0.60:  # Less than 60% confidence
                return False

            # Check risk limits
            risk_score = signal.metadata.get("risk_score", 0)
            if risk_score > 0.8:  # High risk
                return False

            return True

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False

    async def _process_output_queue(self):
        """ประมวลผล output queue และส่งไป Redis"""
        while self.is_running:
            try:
                # Get batch from output queue
                batch: List[BettingSignal] = []
                deadline = asyncio.get_event_loop().time() + self.process_interval

                while len(batch) < self.batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        signal = await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
                        batch.append(signal)
                    except asyncio.TimeoutError:
                        break

                # Publish batch
                if batch:
                    for signal in batch:
                        success = await self.publisher.publish_signal(signal, persist=True)
                        if success:
                            signal.status = SignalStatus.PUBLISHED
                            logger.info(
                                f"Published signal {signal.signal_id} for match {signal.match_id}"
                            )
                        else:
                            signal.status = SignalStatus.ERROR

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in output queue processing: {e}")
                await asyncio.sleep(1)

    async def _monitor_pipeline(self):
        """ติดตามสถานะ pipeline"""
        while self.is_running:
            try:
                # Log stats every minute
                await asyncio.sleep(60)

                stats = {
                    "processed": self.processed_count,
                    "errors": self.error_count,
                    "input_queue": self.input_queue.qsize(),
                    "output_queue": self.output_queue.qsize(),
                    "last_processed": self.last_processed.isoformat(),
                }

                # Get Redis stats
                redis_stats = await self.publisher.get_signal_stats()
                stats.update(redis_stats)

                logger.info(f"Pipeline stats: {stats}")

                # Reset counters
                self.processed_count = 0
                self.error_count = 0

            except Exception as e:
                logger.error(f"Error in pipeline monitoring: {e}")

    async def _cleanup_expired_signals(self):
        """ล้างสัญญาณที่หมดอายุ"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Clean expired signals from generator
                async with self.generator._lock:
                    now = datetime.utcnow()
                    expired = [
                        sid
                        for sid, s in list(self.generator.active_signals.items())
                        if s.expires_at and s.expires_at < now
                    ]
                    for sid in expired:
                        del self.generator.active_signals[sid]

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Mock: ดึง market conditions (placeholder)"""
        # TODO: Integrate with actual odds manager or market feed
        return {"liquidity": "medium", "volatility": "moderate"}

    async def _calculate_risk_score(self, signal: BettingSignal) -> float:
        """Mock: คำนวณ risk score (placeholder)"""
        base = 1 - min(1.0, max(0.0, signal.confidence))
        ev_factor = 0.5 if signal.ev >= 0.10 else 0.8
        return max(0.0, min(1.0, base * ev_factor))
