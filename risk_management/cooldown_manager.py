from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class CooldownLevel(Enum):
    MATCH = "match"
    TEAM = "team"
    MARKET = "market"  # e.g., match_winner, map1_winner
    LEAGUE = "league"  # e.g., ESL Pro League, BLAST Premier
    TOURNAMENT = "tournament"
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
    base_cooldown_minutes: int = 30
    max_loss_multiplier: float = 10.0


class CooldownManager:
    """Comprehensive cooldown management system"""

    def __init__(self):
        self.cooldowns: Dict[str, Dict[CooldownLevel, datetime]] = {}
        self.config = CooldownConfig()
        self.market_cooldowns: Dict[str, datetime] = {}  # market_type -> cooldown_end
        self.league_cooldowns: Dict[str, datetime] = {}  # league_id -> cooldown_end
        self.tournament_cooldowns: Dict[str, datetime] = {}  # tournament_id -> cooldown_end
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

        if level in [CooldownLevel.MATCH, CooldownLevel.TEAM, CooldownLevel.GLOBAL]:
            # Traditional cooldowns
            if entity_id not in self.cooldowns:
                self.cooldowns[entity_id] = {}
            if level in self.cooldowns[entity_id]:
                cooldown_end = self.cooldowns[entity_id][level]
                now = datetime.now()
                if now < cooldown_end:
                    return False, (cooldown_end - now)
        elif level == CooldownLevel.MARKET:
            # Market-specific cooldowns (e.g., match_winner, map1_winner)
            if entity_id in self.market_cooldowns:
                cooldown_end = self.market_cooldowns[entity_id]
                now = datetime.now()
                if now < cooldown_end:
                    return False, (cooldown_end - now)
        elif level == CooldownLevel.LEAGUE:
            # League-specific cooldowns (e.g., ESL Pro League)
            if entity_id in self.league_cooldowns:
                cooldown_end = self.league_cooldowns[entity_id]
                now = datetime.now()
                if now < cooldown_end:
                    return False, (cooldown_end - now)
        elif level == CooldownLevel.TOURNAMENT:
            # Tournament-specific cooldowns
            if entity_id in self.tournament_cooldowns:
                cooldown_end = self.tournament_cooldowns[entity_id]
                now = datetime.now()
                if now < cooldown_end:
                    return False, (cooldown_end - now)

        if check_all:
            avail, remain = self._check_hierarchy_cooldowns(level, entity_id)
            if not avail:
                return avail, remain

        return True, None

    def set_cooldown(self, identifier: str, level: CooldownLevel, duration_minutes: int = None, context: Dict = None):
        """Set cooldown for specific identifier and level with context support"""
        if duration_minutes is None:
            duration_minutes = self.config.base_cooldown_minutes
        
        cooldown_end = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        if level in [CooldownLevel.MATCH, CooldownLevel.TEAM, CooldownLevel.GLOBAL]:
            # Traditional cooldowns
            if identifier not in self.cooldowns:
                self.cooldowns[identifier] = {}
            self.cooldowns[identifier][level] = cooldown_end
        elif level == CooldownLevel.MARKET:
            # Market-specific cooldowns (e.g., match_winner, map1_winner)
            market_type = context.get('market_type', identifier) if context else identifier
            self.market_cooldowns[market_type] = cooldown_end
        elif level == CooldownLevel.LEAGUE:
            # League-specific cooldowns (e.g., ESL Pro League)
            league_id = context.get('league_id', identifier) if context else identifier
            self.league_cooldowns[league_id] = cooldown_end
        elif level == CooldownLevel.TOURNAMENT:
            # Tournament-specific cooldowns
            tournament_id = context.get('tournament_id', identifier) if context else identifier
            self.tournament_cooldowns[tournament_id] = cooldown_end
        
        logger.info(f"Cooldown set: {identifier} ({level.value}) until {cooldown_end}")
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
