from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate HLTV event data"""

    def __init__(self):
        self.required_fields = {
            'scoreUpdate': ['match_id', 'team1', 'team2'],
            'matchData': ['match_id', 'team1', 'team2'],
            'roundEnd': ['match_id', 'round_number', 'winner'],
            'playerKill': ['match_id', 'killer', 'victim'],
            'bombPlanted': ['match_id', 'round_number', 'planter'],
            'bombDefused': ['match_id', 'round_number', 'defuser'],
            'roundStart': ['match_id', 'round_number'],
            'mapEnd': ['match_id', 'map_number', 'winner'],
            'matchEnd': ['match_id', 'winner'],
        }

    def validate(self, event_type: str, data: Dict[str, Any]) -> bool:
        if not data:
            logger.error(f"Empty data for event {event_type}")
            return False

        # Check required fields
        required = self.required_fields.get(event_type, [])
        for field in required:
            if field not in data or data[field] is None:
                logger.error(f"Missing required field '{field}' in {event_type}")
                return False

        # Specific validators
        validator_method = getattr(self, f"_validate_{event_type}", None)
        if validator_method:
            return validator_method(data)
        return True

    def _validate_scoreUpdate(self, data: Dict[str, Any]) -> bool:
        team1_score = data.get('team1', {}).get('score', 0)
        team2_score = data.get('team2', {}).get('score', 0)
        if team1_score < 0 or team2_score < 0:
            logger.error("Invalid score values")
            return False
        round_number = data.get('round_number', 0)
        if round_number < 0 or round_number > 60:
            logger.error(f"Invalid round number: {round_number}")
            return False
        return True

    def _validate_roundEnd(self, data: Dict[str, Any]) -> bool:
        valid_win_types = ['elimination', 'bomb_defused', 'bomb_exploded', 'time']
        win_type = data.get('win_type')
        if win_type and win_type not in valid_win_types:
            logger.error(f"Invalid win type: {win_type}")
            return False
        return True

    def _validate_playerKill(self, data: Dict[str, Any]) -> bool:
        weapon = data.get('weapon')
        if weapon and not self._is_valid_weapon(weapon):
            logger.error(f"Invalid weapon: {weapon}")
            return False
        return True

    def _validate_bombPlanted(self, data: Dict[str, Any]) -> bool:
        site = data.get('site')
        if site and site not in ['A', 'B']:
            logger.error(f"Invalid bomb site: {site}")
            return False
        return True

    def _is_valid_weapon(self, weapon: str) -> bool:
        """Check if weapon name is valid"""
        valid_weapons = [
            # Pistols
            'usp_silencer', 'glock', 'p250', 'cz75a', 'tec9',
            'fiveseven', 'p2000', 'deagle', 'elite', 'revolver',
            # SMGs
            'mac10', 'mp9', 'mp7', 'ump45', 'p90', 'bizon', 'mp5sd',
            # Rifles
            'famas', 'galil', 'ak47', 'm4a1', 'm4a1_silencer',
            'sg556', 'aug', 'ssg08', 'awp', 'scar20', 'g3sg1',
            # Heavy
            'nova', 'xm1014', 'mag7', 'sawedoff', 'negev', 'm249',
            # Grenades
            'hegrenade', 'flashbang', 'smokegrenade', 'molotov',
            'incgrenade', 'decoy',
            # Other
            'knife', 'taser', 'c4', 'healthshot',
        ]
        normalized_weapon = weapon.lower().strip()
        weapon_aliases = {
            'smoke': 'smokegrenade',
            'flash': 'flashbang',
            'he': 'hegrenade',
            'nade': 'hegrenade',
            'fire': 'molotov',
            'inc': 'incgrenade',
            'ak': 'ak47',
            'm4': 'm4a1',
            'm4s': 'm4a1_silencer',
            'awp': 'awp',
            'scout': 'ssg08',
            'deag': 'deagle',
            'zeus': 'taser',
            'bomb': 'c4',
        }
        if normalized_weapon in weapon_aliases:
            normalized_weapon = weapon_aliases[normalized_weapon]
        return normalized_weapon in valid_weapons
