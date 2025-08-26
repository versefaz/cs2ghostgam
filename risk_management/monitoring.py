from typing import List, Dict
import asyncio
from datetime import datetime, timedelta

from .risk_system import RiskManagementSystem
from .cooldown_manager import CooldownLevel


class RiskMonitor:
    """Real-time risk monitoring and alerts"""

    def __init__(self, risk_system: RiskManagementSystem):
        self.risk_system = risk_system
        self.alerts: List[Dict] = []
        self.metrics: Dict[str, float] = {}

    async def monitor_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                await self.check_exposure_limits()
                await self.check_loss_streaks()
                await self.check_bankroll_health()
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(5)

    async def check_exposure_limits(self):
        """Check if approaching exposure limits"""
        exposure = self.risk_system.position_manager.exposure
        max_exposure = max(1.0, float(self.risk_system.limits.max_exposure))
        exposure_pct = exposure.total_exposure / max_exposure

        if exposure_pct > 0.9:
            self.alerts.append(
                {
                    "level": "critical",
                    "message": f"Near max exposure: {exposure_pct:.1%}",
                    "timestamp": datetime.now(),
                }
            )
        elif exposure_pct > 0.75:
            self.alerts.append(
                {
                    "level": "warning",
                    "message": f"High exposure: {exposure_pct:.1%}",
                    "timestamp": datetime.now(),
                }
            )

    async def check_loss_streaks(self):
        """Monitor for dangerous loss streaks"""
        cooldowns = self.risk_system.cooldown_manager

        for key, streak in cooldowns.loss_streaks.items():
            if streak >= 5:
                self.alerts.append(
                    {
                        "level": "critical",
                        "message": f"Loss streak on {key}: {streak} consecutive losses",
                        "timestamp": datetime.now(),
                    }
                )
                # Auto-increase cooldowns
                try:
                    level_str, entity_id = key.split(":", 1)
                    level_enum = CooldownLevel(level_str)
                except Exception:
                    level_enum = CooldownLevel.GLOBAL
                    entity_id = "global"
                from .cooldown_manager import CooldownConfig

                cooldowns.add_cooldown(level_enum, entity_id, custom_duration=timedelta(days=1))

    async def check_bankroll_health(self):
        """Monitor bankroll status"""
        current_bankroll = float(self.risk_system.bankroll)
        initial_bankroll = float(self.risk_system.initial_bankroll)
        if initial_bankroll <= 0:
            return
        drawdown = (initial_bankroll - current_bankroll) / initial_bankroll

        if drawdown > 0.3:
            self.alerts.append(
                {
                    "level": "critical",
                    "message": f"Severe drawdown: {drawdown:.1%}",
                    "timestamp": datetime.now(),
                    "action": "Consider stopping all betting",
                }
            )
            # Auto-reduce Kelly fraction
            self.risk_system.kelly_fraction *= 0.5
        elif drawdown > 0.15:
            self.alerts.append(
                {
                    "level": "warning",
                    "message": f"Significant drawdown: {drawdown:.1%}",
                    "timestamp": datetime.now(),
                }
            )
