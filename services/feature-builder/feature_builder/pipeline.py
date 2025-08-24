from typing import Dict, Any

def build_vector(team_stats: Dict[str, float], player_stats: Dict[str, float], odds: Dict[str, float]) -> dict:
    vec = {
        "elo_diff": team_stats.get("elo_a", 0.0) - team_stats.get("elo_b", 0.0),
        "form_a": team_stats.get("form_a10", 0.0),
        "odds_a": odds.get("moneyline_a", 0.0),
    }
    return vec
