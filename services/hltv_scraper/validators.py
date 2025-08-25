from typing import Dict

class DataValidator:
    def validate_match(self, data: Dict) -> bool:
        required = ["match_id", "team1", "team2"]
        return all(data.get(k) for k in required)

    def validate_team(self, data: Dict) -> bool:
        return bool(data.get("team_id") and data.get("name"))

    def validate_live_score(self, data: Dict) -> bool:
        required = ["match_id", "team1", "team2", "score1", "score2"]
        return all(data.get(k) is not None for k in required)
