import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    return {
        'match_id': 'test_match_001',
        'team1': 'Natus Vincere',
        'team2': 'FaZe Clan',
        'status': 'upcoming',
        'map': 'mirage',
        'tournament_tier': 1,
        'is_lan': True,
        'prize_pool': 1000000,
        'start_time': (datetime.now() + timedelta(hours=2)).isoformat(),
        'tournament': 'IEM Katowice 2024'
    }


@pytest.fixture
def sample_team_stats():
    """Sample team statistics data"""
    return {
        'team_name': 'Natus Vincere',
        'team_id': 4608,
        'last_updated': datetime.now().isoformat(),
        'overview': {
            'world_ranking': 1,
            'country': 'Ukraine',
            'avg_player_age': 23.2,
            'weeks_in_top30': 156,
            'trophies': 15,
            'current_lineup': ['s1mple', 'electronic', 'Perfecto', 'b1t', 'sdy']
        },
        'recent_matches': [
            {
                'date': '2024-01-15',
                'opponent': 'G2 Esports',
                'score': '16-12',
                'our_score': 16,
                'opp_score': 12,
                'result': 'W',
                'map': 'mirage',
                'event': 'IEM Katowice'
            },
            {
                'date': '2024-01-14',
                'opponent': 'Astralis',
                'score': '16-8',
                'our_score': 16,
                'opp_score': 8,
                'result': 'W',
                'map': 'inferno',
                'event': 'IEM Katowice'
            }
        ],
        'map_stats': {
            'mirage': {
                'matches_played': 25,
                'wins': 18,
                'draws': 0,
                'losses': 7,
                'win_rate': 0.72,
                'rounds_won': 380,
                'rounds_lost': 295,
                'round_winrate': 0.563
            },
            'inferno': {
                'matches_played': 22,
                'wins': 16,
                'draws': 0,
                'losses': 6,
                'win_rate': 0.727,
                'rounds_won': 342,
                'rounds_lost': 268,
                'round_winrate': 0.561
            }
        },
        'player_stats': [
            {
                'name': 's1mple',
                'maps_played': 47,
                'kd_diff': 245,
                'kd_ratio': 1.31,
                'rating_2': 1.28,
                'adr': 82.5,
                'kast': 0.74
            },
            {
                'name': 'electronic',
                'maps_played': 47,
                'kd_diff': 156,
                'kd_ratio': 1.18,
                'rating_2': 1.15,
                'adr': 78.2,
                'kast': 0.71
            }
        ],
        'processed': {
            'form_last_10': 0.8,
            'form_last_5': 0.9,
            'momentum': 0.85,
            'avg_round_diff': 4.2,
            'avg_team_rating': 1.12,
            'star_player_rating': 1.28,
            'consistency': 0.78,
            'avg_map_strength': 0.72,
            'overall_strength': 0.89
        }
    }


@pytest.fixture
def sample_h2h_stats():
    """Sample head-to-head statistics"""
    return {
        'total_matches': 8,
        'team1_wins': 5,
        'team2_wins': 3,
        'team1_winrate': 0.625,
        'recent_matches': [
            {
                'date': '2023-12-10',
                'score': '2-1',
                'map': 'bo3',
                'winner': 'Natus Vincere'
            },
            {
                'date': '2023-11-15',
                'score': '16-14',
                'map': 'mirage',
                'winner': 'FaZe Clan'
            }
        ],
        'map_breakdown': {
            'mirage': {
                'matches': 3,
                'team1_wins': 2,
                'team2_wins': 1
            },
            'inferno': {
                'matches': 2,
                'team1_wins': 1,
                'team2_wins': 1
            }
        }
    }


@pytest.fixture
def sample_odds_data():
    """Sample odds data for testing"""
    return {
        'sources': {
            'bet365': {
                'team1_odds': 1.85,
                'team2_odds': 1.95,
                'timestamp': datetime.now().isoformat()
            },
            'pinnacle': {
                'team1_odds': 1.88,
                'team2_odds': 1.92,
                'timestamp': datetime.now().isoformat()
            },
            '1xbet': {
                'team1_odds': 1.82,
                'team2_odds': 1.98,
                'timestamp': datetime.now().isoformat()
            }
        },
        'average': {
            'team1': 1.85,
            'team2': 1.95
        },
        'best': {
            'team1': 1.88,
            'team2': 1.98
        },
        'implied_prob_team1': 0.541,
        'implied_prob_team2': 0.513,
        'num_sources': 3,
        'confidence': 0.85,
        'movement': {
            'direction': 'stable',
            'magnitude': 0.02,
            'velocity': 0.001
        }
    }


@pytest.fixture
def enhanced_match_with_stats(sample_match_data, sample_team_stats, sample_h2h_stats, sample_odds_data):
    """Complete match data with all statistics"""
    match = sample_match_data.copy()
    
    # Add team stats
    match['team1_stats'] = sample_team_stats
    match['team2_stats'] = {
        **sample_team_stats,
        'team_name': 'FaZe Clan',
        'team_id': 6667,
        'overview': {
            **sample_team_stats['overview'],
            'world_ranking': 3,
            'country': 'Europe',
            'current_lineup': ['karrigan', 'rain', 'Twistzz', 'ropz', 'broky']
        },
        'processed': {
            **sample_team_stats['processed'],
            'form_last_10': 0.7,
            'overall_strength': 0.82
        }
    }
    
    # Add H2H and odds
    match['h2h_stats'] = sample_h2h_stats
    match['odds'] = sample_odds_data
    match['features_ready'] = True
    match['stats_timestamp'] = datetime.now().isoformat()
    
    return match


@pytest.fixture
def sample_prediction_result():
    """Sample prediction result"""
    return {
        'match_id': 'test_match_001',
        'team1': 'Natus Vincere',
        'team2': 'FaZe Clan',
        'prediction': {
            'team1_probability': 0.6234,
            'team2_probability': 0.3766,
            'predicted_winner': 'Natus Vincere',
            'confidence': 0.78
        },
        'model_predictions': {
            'xgboost': {
                'team1_prob': 0.625,
                'team2_prob': 0.375
            },
            'lightgbm': {
                'team1_prob': 0.621,
                'team2_prob': 0.379
            }
        },
        'expected_value': {
            'team1_ev': 0.0834,
            'team2_ev': -0.0456,
            'team1_ev_percent': 8.34,
            'team2_ev_percent': -4.56,
            'best_bet': 'team1',
            'edge': 0.0834
        },
        'features': {
            'strength_diff': 0.07,
            'form_diff': 0.1,
            'rating_diff': 0.05,
            'h2h_advantage': 0.125
        },
        'recommendation': {
            'action': 'BET',
            'team': 'team1',
            'suggested_stake': 2.1,
            'kelly_criterion': 0.021,
            'reasoning': ['Expected value: 8.3%', 'Model confidence: 78%', 'Win probability: 62.3%']
        }
    }


@pytest.fixture
def sample_signal_data():
    """Sample betting signal data"""
    return {
        'id': 'signal_test_001',
        'match_id': 'test_match_001',
        'team': 'Natus Vincere',
        'market_type': 'match_winner',
        'confidence': 78,
        'odds': 1.85,
        'expected_value': 0.0834,
        'stake_recommendation': 2.1,
        'kelly_fraction': 0.021,
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'model_confidence': 0.78,
            'data_quality': 0.85,
            'h2h_matches': 8,
            'team1_form': 0.8,
            'team2_form': 0.7
        }
    }


@pytest.fixture
def mock_redis_data():
    """Mock Redis data structure"""
    return {
        'prediction:test_match_001': json.dumps({
            'match_id': 'test_match_001',
            'prediction': {
                'team1_probability': 0.6234,
                'team2_probability': 0.3766,
                'predicted_winner': 'Natus Vincere'
            },
            'timestamp': datetime.now().isoformat()
        }),
        'predictions:recent': [
            json.dumps({
                'match_id': 'test_match_001',
                'teams': ['Natus Vincere', 'FaZe Clan'],
                'prediction': {'confidence': 0.78}
            }),
            json.dumps({
                'match_id': 'test_match_002',
                'teams': ['G2 Esports', 'Astralis'],
                'prediction': {'confidence': 0.72}
            })
        ],
        'signal:active:signal_test_001': json.dumps({
            'id': 'signal_test_001',
            'match_id': 'test_match_001',
            'team': 'Natus Vincere',
            'confidence': 78,
            'timestamp': datetime.now().isoformat()
        }),
        'queue:high_priority': [
            json.dumps({
                'type': 'prediction_request',
                'match_id': 'test_match_001',
                'timestamp': datetime.now().isoformat()
            })
        ]
    }


@pytest.fixture
def sample_backtest_events():
    """Sample backtesting events"""
    base_time = datetime.now() - timedelta(days=30)
    
    return [
        {
            'timestamp': base_time + timedelta(hours=i),
            'event_type': 'odds_update',
            'match_id': f'match_{i//4}',
            'data': {
                'odds_team1': 1.8 + (i % 4) * 0.05,
                'odds_team2': 2.0 - (i % 4) * 0.05,
                'team1': 'Team A',
                'team2': 'Team B'
            }
        } for i in range(0, 120, 4)
    ] + [
        {
            'timestamp': base_time + timedelta(hours=i+2),
            'event_type': 'match_end',
            'match_id': f'match_{i//4}',
            'data': {
                'winner': 'Team A' if i % 8 < 5 else 'Team B',
                'score': '16-12' if i % 8 < 5 else '12-16'
            }
        } for i in range(0, 120, 4)
    ]


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing"""
    return {
        'total_bets': 50,
        'winning_bets': 32,
        'losing_bets': 18,
        'hit_rate': 64.0,
        'roi': 12.5,
        'total_staked': 5000.0,
        'net_profit': 625.0,
        'max_drawdown': 8.2,
        'sharpe_ratio': 1.45,
        'avg_bet_size': 100.0,
        'longest_winning_streak': 7,
        'longest_losing_streak': 4
    }


class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}
        self.lists = {}
    
    def get(self, key: str):
        return self.data.get(key)
    
    def set(self, key: str, value: str):
        self.data[key] = value
        return True
    
    def setex(self, key: str, ttl: int, value: str):
        self.data[key] = value
        return True
    
    def lpush(self, key: str, *values):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key] = list(values) + self.lists[key]
        return len(self.lists[key])
    
    def lrange(self, key: str, start: int, end: int):
        if key not in self.lists:
            return []
        return self.lists[key][start:end+1 if end != -1 else None]
    
    def llen(self, key: str):
        return len(self.lists.get(key, []))
    
    def keys(self, pattern: str):
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return [k for k in self.data.keys() if k == pattern]
    
    def ping(self):
        return True
    
    def info(self, section=None):
        return {
            'redis_version': '6.2.0',
            'used_memory': 1024000,
            'connected_clients': 5
        }


@pytest.fixture
def mock_redis_client(mock_redis_data):
    """Mock Redis client with sample data"""
    client = MockRedisClient(mock_redis_data)
    client.lists = {
        'predictions:recent': mock_redis_data['predictions:recent'],
        'queue:high_priority': mock_redis_data['queue:high_priority']
    }
    return client
