from typing import Dict, Any, AsyncIterator
import asyncio
import logging

from risk_management.risk_system import RiskManagementSystem, RiskSignal

logger = logging.getLogger(__name__)


class SignalRiskIntegration:
    """Integrate signal generation/consumption with risk management"""

    def __init__(
        self,
        signal_consumer: Any,  # expected to expose async def consume() -> AsyncIterator[Dict]
        risk_system: RiskManagementSystem,
        auto_execute: bool = False,
    ):
        self.signal_consumer = signal_consumer
        self.risk_system = risk_system
        self.auto_execute = auto_execute
        self.execution_queue: asyncio.Queue[RiskSignal] = asyncio.Queue()
        self._stop = asyncio.Event()

    async def process_signal_stream(self):
        """Process signals from consumer through risk system and optionally queue for execution"""
        async for signal in self.signal_consumer.consume():
            if self._stop.is_set():
                break
            try:
                risk_signal = await self.risk_system.process_signal(signal)
                logger.info(
                    "Signal %s: Stake=$%.2f Risk=%.2f Execute=%s",
                    signal.get("id", "<unknown>"),
                    risk_signal.kelly_stake,
                    risk_signal.risk_score,
                    risk_signal.can_execute,
                )

                if risk_signal.can_execute and self.auto_execute:
                    await self.execution_queue.put(risk_signal)

                await self.store_decision(risk_signal)

            except Exception as e:
                logger.exception("Error processing signal: %s", e)

    async def execution_loop(self):
        """Execute queued approved signals sequentially"""
        while not self._stop.is_set():
            try:
                risk_signal = await self.execution_queue.get()
                await self.risk_system.execute_signal(risk_signal)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Execution error")

    async def store_decision(self, risk_signal: RiskSignal):
        """Placeholder for persistence/analytics of decisions"""
        # Integrate with DB, analytics, or Redis streams as needed.
        return

    def stop(self):
        self._stop.set()
