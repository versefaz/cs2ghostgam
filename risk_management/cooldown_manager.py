from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import asyncio
from collections import defaultdict
from dataclasses import dataclass


class CooldownLevel(Enum):
    MATCH = "match"
    TEAM = "team"
    TOURNAMENT = "tournament"
    LEAGUE = "league"
    MARKET = "market"
    GLOBAL = "global"


@dataclass
class CooldownConfig:
    """Configuration for cooldown periods"""
    match_cooldown: timedelta = timedelta(hours=2)
    team_cooldown: timedelta = timedelta(hours=6)
    tournament_cooldown: timedelta = timedelta(hours=12)
    league_cooldown: timedelta = timedelta(days=1)
    market_cooldown: timedelta = timedelta(hours=3)
    global_cooldown: timedelta = timedelta(minutes=30)

    # Loss-based cooldowns
    loss_multiplier: float = 2.0  # Multiply cooldown on loss
    consecutive_loss_multiplier: float = 1.5
    max_cooldown_multiplier: float = 5.0


class CooldownManager:
    """Comprehensive cooldown management system"""

    def __init__(self, config: CooldownConfig):
        self.config = config
        self.cooldowns: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.loss_streaks: Dict[str, int] = defaultdict(int)
        self.blocked_entities: set[str] = set()
        self._lock = asyncio.Lock()

    def is_available(
        self, level: CooldownLevel, entity_id: str, check_all: bool = True
    ) -> Tuple[bool, Optional[timedelta]]:
        """
        Check if entity is available for betting
        Returns (is_available, time_remaining | None)
        """
        # Permanently blocked
        if entity_id in self.blocked_entities:
            return False, None

        cooldown_key = f"{level.value}:{entity_id}"
        if cooldown_key in self.cooldowns[level.value]:
            cooldown_end = self.cooldowns[level.value][cooldown_key]
            now = datetime.now()
            if now < cooldown_end:
                return False, (cooldown_end - now)

        if check_all:
            avail, remain = self._check_hierarchy_cooldowns(level, entity_id)
            if not avail:
                return avail, remain

        return True, None

    def add_cooldown(
        self,
        level: CooldownLevel,
        entity_id: str,
        result: str = "neutral",
        custom_duration: Optional[timedelta] = None,
    ):
        """Add cooldown with optional loss multipliers"""
        base_duration = custom_duration or self._get_base_duration(level)

        loss_key = f"{level.value}:{entity_id}"
        if result == "loss":
            self.loss_streaks[loss_key] += 1
            multiplier = min(
                self.config.loss_multiplier
                * (self.config.consecutive_loss_multiplier ** (self.loss_streaks[loss_key] - 1)),
                self.config.max_cooldown_multiplier,
            )
            base_duration *= multiplier
        elif result == "win":
            self.loss_streaks[loss_key] = 0

        cooldown_key = f"{level.value}:{entity_id}"
        self.cooldowns[level.value][cooldown_key] = datetime.now() + base_duration

        self._propagate_cooldowns(level, entity_id, base_duration)

    def _get_base_duration(self, level: CooldownLevel) -> timedelta:
        durations = {
            CooldownLevel.MATCH: self.config.match_cooldown,
            CooldownLevel.TEAM: self.config.team_cooldown,
            CooldownLevel.TOURNAMENT: self.config.tournament_cooldown,
            CooldownLevel.LEAGUE: self.config.league_cooldown,
            CooldownLevel.MARKET: self.config.market_cooldown,
            CooldownLevel.GLOBAL: self.config.global_cooldown,
        }
        return durations.get(level, timedelta(hours=1))

    def _check_hierarchy_cooldowns(
        self, level: CooldownLevel, entity_id: str
    ) -> Tuple[bool, Optional[timedelta]]:
        """Simplified hierarchy check: consult global cooldown only"""
        # In a full system, resolve team/tournament/league parents here.
        # Check global cooldown as a catch-all
        global_key = f"{CooldownLevel.GLOBAL.value}:global"
        if global_key in self.cooldowns[CooldownLevel.GLOBAL.value]:
            end = self.cooldowns[CooldownLevel.GLOBAL.value][global_key]
            now = datetime.now()
            if now < end:
                return False, (end - now)
        return True, None

    def _propagate_cooldowns(
        self, level: CooldownLevel, entity_id: str, duration: timedelta
    ):
        """Placeholder for propagation logic across related entities"""
        # Example: team cooldown might propagate to all matches with that team.
        return
