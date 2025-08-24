from __future__ import annotations
from typing import Dict, Any

class TeamFeatureCalculator:
    def __init__(self, db):
        self.db = db

    async def calculate_team_features(self, team_id: int, map_name: str | None = None) -> Dict[str, Any]:
        # TODO: real aggregation from DB
        return {
            "elo_rating": 1600.0,
            "form_last_10": 0.55,
            "map_win_rate": 0.52,
            "map_played_count": 20,
            "lan_win_rate": 0.5,
        }

    async def calculate_h2h_features(self, team1_id: int, team2_id: int) -> Dict[str, Any]:
        return {"h2h_win_rate": 0.5, "h2h_round_diff": 0.0}

class PlayerFeatureCalculator:
    def __init__(self, db):
        self.db = db

    async def calculate_player_features(self, player_id: int) -> Dict[str, Any]:
        return {"rating_2_0": 1.05, "adr_avg": 78.0, "kast_avg": 0.72}

class MatchContextCalculator:
    def __init__(self, db):
        self.db = db

    async def calculate_match_context(self, match_id: str) -> Dict[str, Any]:
        return {"is_lan": False, "tournament_tier": 2}
