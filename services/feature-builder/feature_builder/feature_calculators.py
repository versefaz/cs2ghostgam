from __future__ import annotations
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TeamFeatureCalculator:
    def __init__(self, db_manager):
        self.db = db_manager  # expects DatabaseManager with asyncpg pool

    async def calculate_team_features(self, team_id: int, map_name: Optional[str] = None) -> Dict[str, Any]:
        defaults = {
            "elo_rating": 1600.0,
            "form_last_10": 0.5,
            "map_win_rate": 0.5,
            "map_played_count": 0,
            "lan_win_rate": 0.5,
        }
        if not self.db or not self.db.pool:
            logger.warning("DB pool not initialized; returning defaults for team features")
            return defaults

        try:
            async with self.db.pool.acquire() as conn:
                # Total matches played
                total_played = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM matches
                    WHERE team1_id = $1 OR team2_id = $1
                    """,
                    team_id,
                )

                # Map-specific matches played
                if map_name:
                    map_played = await conn.fetchval(
                        """
                        SELECT COUNT(*)
                        FROM matches
                        WHERE (team1_id = $1 OR team2_id = $1)
                          AND map_name = $2
                        """,
                        team_id,
                        map_name,
                    )
                else:
                    map_played = total_played or 0

                # Recent form proxy: fraction of last 10 appearances relative to recent 20 period
                recent_appearances = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT match_date
                        FROM matches
                        WHERE team1_id = $1 OR team2_id = $1
                        ORDER BY match_date DESC
                        LIMIT 10
                    ) t
                    """,
                    team_id,
                ) or 0
                form_last_10 = (recent_appearances / 10.0) if recent_appearances else 0.0

                # Since we don't have results in schema here, keep win rates neutral with slight adjustment by activity
                activity_factor = min(1.0, (total_played or 0) / 50.0)
                map_win_rate = 0.45 + 0.1 * activity_factor
                lan_win_rate = 0.45 + 0.1 * activity_factor

                return {
                    "elo_rating": 1600.0 + 50.0 * activity_factor,
                    "form_last_10": round(form_last_10, 3),
                    "map_win_rate": round(map_win_rate, 3),
                    "map_played_count": int(map_played or 0),
                    "lan_win_rate": round(lan_win_rate, 3),
                }
        except Exception as e:
            logger.exception(f"calculate_team_features failed: {e}")
            return defaults

    async def calculate_h2h_features(self, team1_id: int, team2_id: int) -> Dict[str, Any]:
        defaults = {"h2h_win_rate": 0.5, "h2h_round_diff": 0.0}
        if not self.db or not self.db.pool:
            logger.warning("DB pool not initialized; returning defaults for h2h features")
            return defaults
        try:
            async with self.db.pool.acquire() as conn:
                meetings = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM matches
                    WHERE (team1_id = $1 AND team2_id = $2)
                       OR (team1_id = $2 AND team2_id = $1)
                    """,
                    team1_id,
                    team2_id,
                ) or 0
                # Without scores in schema, approximate win rate by symmetry with a slight bias by frequency
                bias = min(0.05, meetings / 200.0)
                h2h_win_rate = 0.5 + bias
                return {"h2h_win_rate": round(h2h_win_rate, 3), "h2h_round_diff": 0.0}
        except Exception as e:
            logger.exception(f"calculate_h2h_features failed: {e}")
            return defaults


class PlayerFeatureCalculator:
    def __init__(self, db_manager):
        self.db = db_manager

    async def calculate_player_features(self, player_id: int) -> Dict[str, Any]:
        # Placeholder: no player tables in current schema snapshot; return safe defaults
        return {"rating_2_0": 1.0, "adr_avg": 75.0, "kast_avg": 0.7}


class MatchContextCalculator:
    def __init__(self, db_manager):
        self.db = db_manager

    async def calculate_match_context(self, match_id: str) -> Dict[str, Any]:
        defaults = {"is_lan": False, "tournament_tier": 2, "map_name": None}
        if not self.db or not self.db.pool:
            logger.warning("DB pool not initialized; returning defaults for match context")
            return defaults
        try:
            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT match_id, match_date, map_name
                    FROM matches
                    WHERE match_id = $1
                    """,
                    match_id,
                )
                if not row:
                    return defaults
                # Without LAN/tier fields, infer dummy tier based on recency
                is_recent = 1 if row["match_date"] else 0
                tier = 1 if is_recent else 2
                return {"is_lan": False, "tournament_tier": tier, "map_name": row["map_name"]}
        except Exception as e:
            logger.exception(f"calculate_match_context failed: {e}")
            return defaults
