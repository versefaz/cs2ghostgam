import json
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MessageParser:
    """Parse HLTV Socket.io messages"""

    def parse(self, event_type: str, data: Any) -> Dict[str, Any]:
        """Parse message based on event type"""
        # Ensure data is dict
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {data}")
                return {}

        parser_method = getattr(self, f"_parse_{event_type}", None)
        if parser_method:
            return parser_method(data)
        else:
            logger.warning(f"No parser for event type: {event_type}")
            return data

    def _parse_scoreUpdate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'team1': {
                'id': data.get('team1', {}).get('id'),
                'name': data.get('team1', {}).get('name'),
                'score': data.get('team1', {}).get('score', 0),
                'side': data.get('team1', {}).get('side', 'CT'),
            },
            'team2': {
                'id': data.get('team2', {}).get('id'),
                'name': data.get('team2', {}).get('name'),
                'score': data.get('team2', {}).get('score', 0),
                'side': data.get('team2', {}).get('side', 'T'),
            },
            'round_number': data.get('roundNumber', 0),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_matchData(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'event_name': data.get('eventName'),
            'format': data.get('format', 'bo1'),
            'maps': data.get('maps', []),
            'team1': {
                'id': data.get('team1', {}).get('id'),
                'name': data.get('team1', {}).get('name'),
                'rank': data.get('team1', {}).get('rank'),
                'players': self._parse_players(data.get('team1', {}).get('players', [])),
            },
            'team2': {
                'id': data.get('team2', {}).get('id'),
                'name': data.get('team2', {}).get('name'),
                'rank': data.get('team2', {}).get('rank'),
                'players': self._parse_players(data.get('team2', {}).get('players', [])),
            },
            'start_time': data.get('startTime'),
            'status': data.get('status', 'live'),
        }

    def _parse_roundEnd(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'round_number': data.get('roundNumber'),
            'winner': data.get('winner'),
            'win_type': data.get('winType'),
            'team1_economy': data.get('team1Economy', {}),
            'team2_economy': data.get('team2Economy', {}),
            'round_duration': data.get('duration'),
            'players_alive': {
                'team1': data.get('team1PlayersAlive', 0),
                'team2': data.get('team2PlayersAlive', 0),
            },
            'damage_stats': data.get('damageStats', []),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_playerKill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'round_number': data.get('roundNumber'),
            'killer': {
                'id': data.get('killer', {}).get('id'),
                'name': data.get('killer', {}).get('name'),
                'team': data.get('killer', {}).get('team'),
            },
            'victim': {
                'id': data.get('victim', {}).get('id'),
                'name': data.get('victim', {}).get('name'),
                'team': data.get('victim', {}).get('team'),
            },
            'weapon': data.get('weapon'),
            'headshot': data.get('headshot', False),
            'wallbang': data.get('wallbang', False),
            'flash_assist': data.get('flashAssist', False),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_bombPlanted(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'round_number': data.get('roundNumber'),
            'planter': {
                'id': data.get('planter', {}).get('id'),
                'name': data.get('planter', {}).get('name'),
            },
            'site': data.get('site'),
            'time_remaining': data.get('timeRemaining'),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_bombDefused(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'round_number': data.get('roundNumber'),
            'defuser': {
                'id': data.get('defuser', {}).get('id'),
                'name': data.get('defuser', {}).get('name'),
            },
            'time_remaining': data.get('timeRemaining'),
            'had_kit': data.get('hadKit', False),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_roundStart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber', 1),
            'round_number': data.get('roundNumber'),
            'team1_money': data.get('team1Money', 0),
            'team2_money': data.get('team2Money', 0),
            'team1_equipment_value': data.get('team1EquipmentValue', 0),
            'team2_equipment_value': data.get('team2EquipmentValue', 0),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_mapEnd(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'map_number': data.get('mapNumber'),
            'map_name': data.get('mapName'),
            'final_score': {
                'team1': data.get('team1Score', 0),
                'team2': data.get('team2Score', 0),
            },
            'winner': data.get('winner'),
            'overtime': data.get('overtime', False),
            'duration': data.get('duration'),
            'player_stats': self._parse_player_stats(data.get('playerStats', [])),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_matchEnd(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'match_id': data.get('matchId'),
            'winner': data.get('winner'),
            'final_score': {
                'team1': data.get('team1MapScore', 0),
                'team2': data.get('team2MapScore', 0),
            },
            'maps_played': data.get('mapsPlayed', []),
            'total_duration': data.get('totalDuration'),
            'mvp': data.get('mvp'),
            'timestamp': datetime.now().isoformat(),
        }

    def _parse_players(self, players: list) -> list:
        return [
            {
                'id': p.get('id'),
                'name': p.get('name'),
                'rating': p.get('rating', 1.0),
                'kills': p.get('kills', 0),
                'deaths': p.get('deaths', 0),
                'adr': p.get('adr', 0),
                'kast': p.get('kast', 0),
            }
            for p in players
        ]

    def _parse_player_stats(self, stats: list) -> list:
        return [
            {
                'player_id': s.get('playerId'),
                'player_name': s.get('playerName'),
                'team': s.get('team'),
                'kills': s.get('kills', 0),
                'deaths': s.get('deaths', 0),
                'assists': s.get('assists', 0),
                'adr': s.get('adr', 0),
                'kast': s.get('kast', 0),
                'rating2': s.get('rating2', 1.0),
                'first_kills': s.get('firstKills', 0),
                'clutches': s.get('clutches', 0),
                'multi_kills': s.get('multiKills', {}),
            }
            for s in stats
        ]
