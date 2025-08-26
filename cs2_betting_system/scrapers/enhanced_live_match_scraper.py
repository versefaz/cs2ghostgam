# enhanced_live_match_scraper.py

import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .hltv_stats_scraper import HLTVStatsScraper
from .live_match_scraper import LiveMatchScraper
from .odds_scraper_robust import MultiSourceOddsFetcher

logger = logging.getLogger(__name__)

class EnhancedLiveMatchScraper:
    """
    Enhanced Live Match Scraper ที่ผูกกับ HLTV Stats
    """
    
    def __init__(self):
        self.hltv_scraper = HLTVStatsScraper()
        self.prediction_model = EnhancedPredictionModel()
        self.multi_odds_fetcher: Optional[MultiSourceOddsFetcher] = None
        self.matches_cache: Dict[str, Dict] = {}
        self.stats_cache: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """Initialize all components"""
        await self.hltv_scraper.initialize()
        await self.prediction_model.initialize()
        self.multi_odds_fetcher = MultiSourceOddsFetcher()
        await self.multi_odds_fetcher.initialize()
        logger.info("Enhanced Live Match Scraper initialized")

    async def get_live_matches(self) -> List[Dict]:
        """Fetch live/upcoming matches via LiveMatchScraper and normalize fields to team1/team2."""
        def _scrape():
            scraper = LiveMatchScraper()
            try:
                return scraper.scrape_all_sources()
            finally:
                scraper.cleanup()

        loop = asyncio.get_running_loop()
        raw_matches: List[Dict] = await loop.run_in_executor(self.executor, _scrape)
        normalized: List[Dict] = []
        for m in raw_matches:
            team1 = m.get('team1') or m.get('team_a') or m.get('home')
            team2 = m.get('team2') or m.get('team_b') or m.get('away')
            normalized.append({
                'match_id': m.get('match_id') or f"{team1}_{team2}_{int(datetime.utcnow().timestamp())}",
                'team1': team1,
                'team2': team2,
                'status': m.get('status', 'upcoming'),
                'map': (m.get('map') or 'TBA'),
                'tournament_tier': m.get('tournament_tier', 2),
                'is_lan': m.get('is_lan', False),
                'prize_pool': m.get('prize_pool', 0),
                # optional robust identifiers placeholder if calling code provides
                'odds_identifiers': m.get('odds_identifiers'),
            })
        return normalized
        
    async def get_live_matches_with_stats(self) -> List[Dict]:
        """
        ดึง live matches พร้อมสถิติแบบครบถ้วน
        """
        # Get base live matches (from your existing scraper)
        live_matches = await self.get_live_matches()
        
        # Enhance each match with HLTV stats in parallel
        enhanced_matches = []
        tasks = []
        
        for match in live_matches:
            task = self._enhance_match_with_stats(match)
            tasks.append(task)
        
        # Process all matches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for match, result in zip(live_matches, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to enhance match {match.get('match_id')}: {result}")
                enhanced_matches.append(match)  # Use original if enhancement fails
            else:
                enhanced_matches.append(result)
        
        return enhanced_matches
    
    async def _enhance_match_with_stats(self, match: Dict) -> Dict:
        """
        เพิ่มสถิติจาก HLTV เข้าไปใน match data
        """
        team1 = match.get('team1')
        team2 = match.get('team2')
        
        # Parallel fetch both teams' stats
        tasks = [
            self.hltv_scraper.get_team_stats(team1),
            self.hltv_scraper.get_team_stats(team2),
            self.hltv_scraper.get_h2h_stats(team1, team2)
        ]
        
        team1_stats, team2_stats, h2h_stats = await asyncio.gather(*tasks)
        
        # Add stats to match
        enhanced_match = match.copy()
        enhanced_match['team1_stats'] = team1_stats
        enhanced_match['team2_stats'] = team2_stats
        enhanced_match['h2h_stats'] = h2h_stats
        
        # Add processed metrics for quick access
        enhanced_match['team1_metrics'] = team1_stats.get('processed', {})
        enhanced_match['team2_metrics'] = team2_stats.get('processed', {})
        
        # Add current odds if available
        enhanced_match['odds'] = await self._fetch_live_odds(match)
        
        # Calculate feature-ready data
        enhanced_match['features_ready'] = True
        enhanced_match['stats_timestamp'] = datetime.now().isoformat()
        
        return enhanced_match
    
    async def _fetch_live_odds(self, match: Dict) -> Dict:
        """
        ดึง odds แบบ real-time จากหลายแหล่ง
        """
        # 1) Try robust multi-source odds if identifiers are provided
        if self.multi_odds_fetcher and match.get('odds_identifiers'):
            try:
                records = await self.multi_odds_fetcher.fetch_odds_with_fallback(match['odds_identifiers'])
                aggregated = self.multi_odds_fetcher.aggregate_odds(records)
                if aggregated:
                    key = f"{match.get('team1')}_vs_{match.get('team2')}"
                    data = aggregated.get(key) or next(iter(aggregated.values()))
                    odds_data: Dict[str, Any] = {
                        'sources': {r.source: {'team1_odds': r.odds_1, 'team2_odds': r.odds_2, 'timestamp': r.timestamp.isoformat()} for r in records},
                        'average': {'team1': float(data['odds_1']), 'team2': float(data['odds_2'])},
                        'best': {'team1': float(data['best_odds_1']), 'team2': float(data['best_odds_2'])},
                        'implied_prob_team1': 1.0 / float(data['odds_1']) if data['odds_1'] else None,
                        'implied_prob_team2': 1.0 / float(data['odds_2']) if data['odds_2'] else None,
                        'num_sources': int(data['num_sources']),
                        'confidence': float(data['confidence']),
                    }
                    odds_data['movement'] = await self._detect_odds_movement(match, odds_data)
                    return odds_data
            except Exception as e:
                logger.warning(f"Robust odds fetch failed, falling back: {e}")

        # 2) Fallback to legacy per-source odds via LiveMatchScraper
        odds_sources = ['bet365', '1xbet', 'pinnacle', 'ggbet', 'thunderpick']
        odds_data: Dict[str, Any] = {
            'sources': {},
            'average': {},
            'best': {}
        }
        tasks = [self._get_odds_from_source(match, s) for s in odds_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        team1_odds: List[float] = []
        team2_odds: List[float] = []
        for source, result in zip(odds_sources, results):
            if not isinstance(result, Exception) and result:
                odds_data['sources'][source] = result
                if 'team1_odds' in result:
                    team1_odds.append(result['team1_odds'])
                if 'team2_odds' in result:
                    team2_odds.append(result['team2_odds'])
        if team1_odds:
            odds_data['average']['team1'] = np.mean(team1_odds)
            odds_data['best']['team1'] = max(team1_odds)
            odds_data['implied_prob_team1'] = 1 / odds_data['average']['team1']
        if team2_odds:
            odds_data['average']['team2'] = np.mean(team2_odds)
            odds_data['best']['team2'] = max(team2_odds)
            odds_data['implied_prob_team2'] = 1 / odds_data['average']['team2']
        odds_data['movement'] = await self._detect_odds_movement(match, odds_data)
        return odds_data
    
    async def _get_odds_from_source(self, match: Dict, source: str) -> Dict:
        """
        ดึง odds จาก source แต่ละแหล่ง
        """
        try:
            team1 = match.get('team1')
            team2 = match.get('team2')

            # Run blocking Selenium + Redis calls in a thread
            def _fetch():
                scraper = LiveMatchScraper()
                try:
                    odds = scraper.get_cached_odds(team1, team2)
                finally:
                    scraper.cleanup()
                return odds

            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(self.executor, _fetch)

            if not raw:
                return None

            o1 = raw.get('odds_team1')
            o2 = raw.get('odds_team2')
            if not (o1 and o2):
                return None

            return {
                'team1_odds': float(o1),
                'team2_odds': float(o2),
                'source': raw.get('odds_source') or source,
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to get odds from {source}: {e}")
            return None
    
    async def _detect_odds_movement(self, match: Dict, current_odds: Dict) -> Dict:
        """
        ตรวจจับการเคลื่อนไหวของ odds
        """
        match_id = match.get('match_id')
        
        # Get historical odds from cache
        historical = self.matches_cache.get(match_id, {}).get('odds_history', [])
        
        movement = {
            'direction': 'stable',
            'magnitude': 0,
            'velocity': 0
        }
        
        if historical and len(historical) > 0:
            last_odds = historical[-1]
            
            # Calculate movement
            if 'average' in current_odds and 'average' in last_odds:
                team1_change = current_odds['average'].get('team1', 0) - last_odds['average'].get('team1', 0)
                
                if abs(team1_change) > 0.05:  # Significant movement threshold
                    movement['direction'] = 'up' if team1_change > 0 else 'down'
                    movement['magnitude'] = abs(team1_change)
                    
                    # Calculate velocity (change per hour)
                    time_diff = (datetime.now() - datetime.fromisoformat(last_odds['timestamp'])).seconds / 3600
                    if time_diff > 0:
                        movement['velocity'] = movement['magnitude'] / time_diff
        
        # Store current odds in history
        if match_id not in self.matches_cache:
            self.matches_cache[match_id] = {'odds_history': []}
        
        current_odds['timestamp'] = datetime.now().isoformat()
        self.matches_cache[match_id]['odds_history'].append(current_odds)
        
        # Keep only last 20 records
        if len(self.matches_cache[match_id]['odds_history']) > 20:
            self.matches_cache[match_id]['odds_history'] = self.matches_cache[match_id]['odds_history'][-20:]
        
        return movement


class EnhancedPredictionModel:
    """
    Enhanced Prediction Model ที่ใช้ HLTV stats จริง
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_importance = {}
        
    async def initialize(self):
        """Load all trained models"""
        await self.load_models()
        await self.load_scaler()
        logger.info("Enhanced Prediction Model initialized")
    
    async def extract_features(self, match: Dict) -> pd.DataFrame:
        """
        Extract features จากข้อมูลแมตช์ที่มี HLTV stats
        ใช้ข้อมูลจริงทั้งหมด
        """
        features = {}
        
        # Basic match info
        features['match_id'] = match.get('match_id', 'unknown')
        features['is_live'] = 1 if match.get('status') == 'live' else 0
        features['map'] = self._encode_map(match.get('map', 'TBA'))
        
        # Team stats from HLTV
        team1_stats = match.get('team1_stats', {})
        team2_stats = match.get('team2_stats', {})
        team1_metrics = team1_stats.get('processed', {})
        team2_metrics = team2_stats.get('processed', {})
        
        # 1. World Ranking Features
        features['team1_rank'] = team1_stats.get('overview', {}).get('world_ranking', 50)
        features['team2_rank'] = team2_stats.get('overview', {}).get('world_ranking', 50)
        features['rank_diff'] = features['team1_rank'] - features['team2_rank']
        features['rank_ratio'] = features['team1_rank'] / max(features['team2_rank'], 1)
        
        # 2. Form Features (Recent Performance)
        features['team1_form_10'] = team1_metrics.get('form_last_10', 0.5)
        features['team2_form_10'] = team2_metrics.get('form_last_10', 0.5)
        features['team1_form_5'] = team1_metrics.get('form_last_5', 0.5)
        features['team2_form_5'] = team2_metrics.get('form_last_5', 0.5)
        features['form_diff_10'] = features['team1_form_10'] - features['team2_form_10']
        features['form_diff_5'] = features['team1_form_5'] - features['team2_form_5']
        
        # 3. Momentum Features
        features['team1_momentum'] = team1_metrics.get('momentum', 0.5)
        features['team2_momentum'] = team2_metrics.get('momentum', 0.5)
        features['momentum_diff'] = features['team1_momentum'] - features['team2_momentum']
        
        # 4. Round Difference Average
        features['team1_round_diff_avg'] = team1_metrics.get('avg_round_diff', 0)
        features['team2_round_diff_avg'] = team2_metrics.get('avg_round_diff', 0)
        features['round_diff_comparison'] = features['team1_round_diff_avg'] - features['team2_round_diff_avg']
        
        # 5. Player Rating Features
        features['team1_avg_rating'] = team1_metrics.get('avg_team_rating', 1.0)
        features['team2_avg_rating'] = team2_metrics.get('avg_team_rating', 1.0)
        features['team1_star_rating'] = team1_metrics.get('star_player_rating', 1.0)
        features['team2_star_rating'] = team2_metrics.get('star_player_rating', 1.0)
        features['rating_diff'] = features['team1_avg_rating'] - features['team2_avg_rating']
        features['star_diff'] = features['team1_star_rating'] - features['team2_star_rating']
        features['team1_consistency'] = team1_metrics.get('consistency', 0.5)
        features['team2_consistency'] = team2_metrics.get('consistency', 0.5)
        
        # 6. Map-specific Features
        map_name = match.get('map', 'TBA').lower()
        if map_name != 'tba':
            team1_map = team1_stats.get('map_stats', {}).get(map_name, {})
            team2_map = team2_stats.get('map_stats', {}).get(map_name, {})
            
            features['team1_map_winrate'] = team1_map.get('win_rate', 0.5)
            features['team2_map_winrate'] = team2_map.get('win_rate', 0.5)
            features['team1_map_matches'] = team1_map.get('matches_played', 0)
            features['team2_map_matches'] = team2_map.get('matches_played', 0)
            features['map_winrate_diff'] = features['team1_map_winrate'] - features['team2_map_winrate']
            features['map_experience_diff'] = features['team1_map_matches'] - features['team2_map_matches']
            
            # Map round statistics
            features['team1_map_round_wr'] = team1_map.get('round_winrate', 0.5)
            features['team2_map_round_wr'] = team2_map.get('round_winrate', 0.5)
        else:
            # Use overall map pool strength if map not decided
            features['team1_map_strength'] = team1_metrics.get('avg_map_strength', 0.5)
            features['team2_map_strength'] = team2_metrics.get('avg_map_strength', 0.5)
            features['map_strength_diff'] = features['team1_map_strength'] - features['team2_map_strength']
        
        # 7. Head-to-Head Features
        h2h = match.get('h2h_stats', {})
        features['h2h_total_matches'] = h2h.get('total_matches', 0)
        features['h2h_team1_wins'] = h2h.get('team1_wins', 0)
        features['h2h_team2_wins'] = h2h.get('team2_wins', 0)
        features['h2h_team1_winrate'] = h2h.get('team1_winrate', 0.5)
        
        # H2H on specific map
        if map_name != 'tba' and 'map_breakdown' in h2h:
            map_h2h = h2h['map_breakdown'].get(map_name, {})
            features['h2h_map_matches'] = map_h2h.get('matches', 0)
            features['h2h_map_team1_wins'] = map_h2h.get('team1_wins', 0)
        
        # 8. Odds Features
        odds_data = match.get('odds', {})
        if odds_data:
            features['odds_team1_avg'] = odds_data.get('average', {}).get('team1', 2.0)
            features['odds_team2_avg'] = odds_data.get('average', {}).get('team2', 2.0)
            features['implied_prob_team1'] = odds_data.get('implied_prob_team1', 0.5)
            features['implied_prob_team2'] = odds_data.get('implied_prob_team2', 0.5)
            
            # Odds movement
            movement = odds_data.get('movement', {})
            features['odds_movement_direction'] = 1 if movement.get('direction') == 'up' else (-1 if movement.get('direction') == 'down' else 0)
            features['odds_movement_magnitude'] = movement.get('magnitude', 0)
            features['odds_movement_velocity'] = movement.get('velocity', 0)
            
            # Value detection
            features['odds_sources_count'] = len(odds_data.get('sources', {}))
        
        # 9. Tournament Context Features
        features['tournament_tier'] = match.get('tournament_tier', 2)
        features['is_lan'] = 1 if match.get('is_lan', False) else 0
        features['prize_pool_log'] = np.log10(match.get('prize_pool', 50000) + 1)
        
        # 10. Team Experience Features
        team1_overview = team1_stats.get('overview', {})
        team2_overview = team2_stats.get('overview', {})
        
        features['team1_weeks_top30'] = team1_overview.get('weeks_in_top30', 0)
        features['team2_weeks_top30'] = team2_overview.get('weeks_in_top30', 0)
        features['team1_trophies'] = team1_overview.get('trophies', 0)
        features['team2_trophies'] = team2_overview.get('trophies', 0)
        features['experience_diff'] = features['team1_weeks_top30'] - features['team2_weeks_top30']
        
        # 11. Overall Strength Score
        features['team1_strength'] = team1_metrics.get('overall_strength', 0.5)
        features['team2_strength'] = team2_metrics.get('overall_strength', 0.5)
        features['strength_diff'] = features['team1_strength'] - features['team2_strength']
        
        # 12. Composite Features (Interactions)
        features['rank_form_interaction'] = features['rank_diff'] * features['form_diff_10']
        features['rating_momentum_interaction'] = features['rating_diff'] * features['momentum_diff']
        features['map_form_interaction'] = features.get('map_winrate_diff', 0) * features['form_diff_5']
        
        # 13. Confidence Features
        features['data_completeness'] = self._calculate_data_completeness(features)
        features['prediction_confidence'] = self._calculate_confidence_score(features)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure all required features are present
        df = self._ensure_feature_columns(df)
        
        return df
    
    def _encode_map(self, map_name: str) -> int:
        """Encode map name to numeric"""
        map_encoding = {
            'ancient': 1, 'anubis': 2, 'inferno': 3, 'mirage': 4,
            'nuke': 5, 'overpass': 6, 'vertigo': 7, 'tba': 0
        }
        return map_encoding.get(map_name.lower(), 0)
    
    def _calculate_data_completeness(self, features: Dict) -> float:
        """Calculate how complete the feature data is"""
        important_features = [
            'team1_rank', 'team2_rank', 'team1_form_10', 'team2_form_10',
            'team1_avg_rating', 'team2_avg_rating', 'h2h_total_matches'
        ]
        
        available = sum(1 for f in important_features if features.get(f, 0) != 0)
        return available / len(important_features)
    
    def _calculate_confidence_score(self, features: Dict) -> float:
        """Calculate confidence in prediction based on data quality"""
        confidence = 0.5  # Base confidence
        
        # More H2H matches = higher confidence
        h2h_matches = features.get('h2h_total_matches', 0)
        confidence += min(h2h_matches * 0.02, 0.2)
        
        # More map-specific data = higher confidence
        if features.get('team1_map_matches', 0) > 10 and features.get('team2_map_matches', 0) > 10:
            confidence += 0.15
        
        # Consistent form = higher confidence
        if abs(features.get('form_diff_10', 0)) > 0.3:
            confidence += 0.1
        
        # Multiple odds sources = higher confidence
        if features.get('odds_sources_count', 0) >= 3:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist"""
        required_columns = [
            'team1_rank', 'team2_rank', 'rank_diff', 'team1_form_10', 'team2_form_10',
            'form_diff_10', 'team1_momentum', 'team2_momentum', 'momentum_diff',
            'team1_avg_rating', 'team2_avg_rating', 'rating_diff',
            'h2h_total_matches', 'h2h_team1_winrate', 'odds_team1_avg', 'odds_team2_avg',
            'strength_diff', 'data_completeness', 'prediction_confidence'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    async def predict_match(self, match: Dict) -> Dict:
        """
        ทำนายผลการแข่งขันด้วยข้อมูลจริงจาก HLTV
        """
        try:
            # Extract features
            features = await self.extract_features(match)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(features_scaled)[0]
                    predictions[name] = {
                        'team1_prob': pred_proba[1],
                        'team2_prob': pred_proba[0]
                    }
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
            
            # Ensemble prediction
            if predictions:
                avg_team1 = np.mean([p['team1_prob'] for p in predictions.values()])
                avg_team2 = np.mean([p['team2_prob'] for p in predictions.values()])
                
                # Normalize
                total = avg_team1 + avg_team2
                team1_prob = avg_team1 / total
                team2_prob = avg_team2 / total
            else:
                team1_prob = 0.5
                team2_prob = 0.5
            
            # Calculate expected value
            odds = match.get('odds', {})
            ev_analysis = self._calculate_expected_value(
                team1_prob, team2_prob,
                odds.get('average', {}).get('team1', 2.0),
                odds.get('average', {}).get('team2', 2.0)
            )
            
            result = {
                'match_id': match.get('match_id'),
                'team1': match.get('team1'),
                'team2': match.get('team2'),
                'prediction': {
                    'team1_probability': round(team1_prob, 4),
                    'team2_probability': round(team2_prob, 4),
                    'predicted_winner': match.get('team1') if team1_prob > team2_prob else match.get('team2'),
                    'confidence': features.iloc[0]['prediction_confidence']
                },
                'model_predictions': predictions,
                'expected_value': ev_analysis,
                'features': {
                    'strength_diff': features.iloc[0]['strength_diff'],
                    'form_diff': features.iloc[0]['form_diff_10'],
                    'rating_diff': features.iloc[0]['rating_diff'],
                    'h2h_advantage': features.iloc[0]['h2h_team1_winrate'] - 0.5
                },
                'recommendation': self._generate_betting_recommendation(
                    team1_prob, team2_prob, ev_analysis,
                    features.iloc[0]['prediction_confidence']
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for match {match.get('match_id')}: {e}")
            return {
                'error': str(e),
                'match_id': match.get('match_id')
            }
    
    def _calculate_expected_value(self, team1_prob: float, team2_prob: float,
                                  team1_odds: float, team2_odds: float) -> Dict:
        """Calculate expected value for betting"""
        ev1 = (team1_prob * (team1_odds - 1)) - (1 - team1_prob)
        ev2 = (team2_prob * (team2_odds - 1)) - (1 - team2_prob)
        
        return {
            'team1_ev': round(ev1, 4),
            'team2_ev': round(ev2, 4),
            'team1_ev_percent': round(ev1 * 100, 2),
            'team2_ev_percent': round(ev2 * 100, 2),
            'best_bet': 'team1' if ev1 > ev2 and ev1 > 0 else ('team2' if ev2 > 0 else 'no_bet'),
            'edge': max(ev1, ev2) if max(ev1, ev2) > 0 else 0
        }
    
    def _generate_betting_recommendation(self, team1_prob: float, team2_prob: float,
                                        ev_analysis: Dict, confidence: float) -> Dict:
        """Generate betting recommendations"""
        recommendation = {
            'action': 'SKIP',
            'team': None,
            'suggested_stake': 0,
            'kelly_criterion': 0,
            'reasoning': []
        }
        
        # Check for value
        if ev_analysis['edge'] > 0.05 and confidence > 0.7:
            best_bet = ev_analysis['best_bet']
            
            if best_bet != 'no_bet':
                recommendation['action'] = 'BET'
                recommendation['team'] = best_bet
                
                # Kelly Criterion calculation
                if best_bet == 'team1':
                    p = team1_prob
                    b = ev_analysis.get('team1_odds', 2.0) - 1
                else:
                    p = team2_prob
                    b = ev_analysis.get('team2_odds', 2.0) - 1
                
                kelly = (p * b - (1 - p)) / b if b > 0 else 0
                kelly_adjusted = kelly * 0.25  # Conservative Kelly (1/4 Kelly)
                
                recommendation['kelly_criterion'] = round(max(0, min(kelly_adjusted, 0.1)), 4)
                recommendation['suggested_stake'] = round(recommendation['kelly_criterion'] * 100, 1)
                
                recommendation['reasoning'].append(f"Expected value: {ev_analysis['edge']*100:.1f}%")
                recommendation['reasoning'].append(f"Model confidence: {confidence*100:.0f}%")
                recommendation['reasoning'].append(f"Win probability: {p*100:.1f}%")
        
        elif confidence < 0.5:
            recommendation['reasoning'].append("Insufficient data for reliable prediction")
        elif ev_analysis['edge'] < 0:
            recommendation['reasoning'].append("No positive expected value found")
        else:
            recommendation['reasoning'].append(f"Edge too small: {ev_analysis['edge']*100:.1f}%")
        
        return recommendation
    
    async def load_models(self):
        """Load all trained models"""
        model_names = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'neural_network']
        
        for name in model_names:
            try:
                import pickle
                with open(f'models/{name}_model.pkl', 'rb') as f:
                    self.models[name] = pickle.load(f)
                    logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.warning(f"Could not load {name} model: {e}")
    
    async def load_scaler(self):
        """Load feature scaler"""
        try:
            import pickle
            with open('models/feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                logger.info("Loaded feature scaler")
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")


# Example usage
async def main():
    """
    ตัวอย่างการใช้งานระบบ
    """
    scraper = EnhancedLiveMatchScraper()
    await scraper.initialize()
    
    # Get enhanced matches with stats
    matches = await scraper.get_live_matches_with_stats()
    
    for match in matches:
        # Make prediction
        prediction = await scraper.prediction_model.predict_match(match)
        
        print(f"\n{'='*60}")
        print(f"{match['team1']} vs {match['team2']}")
        print(f"Map: {match.get('map', 'TBA')}")
        print(f"Prediction: {prediction['prediction']['predicted_winner']} "
              f"({prediction['prediction']['team1_probability']*100:.1f}% vs "
              f"{prediction['prediction']['team2_probability']*100:.1f}%)")
        print(f"Confidence: {prediction['prediction']['confidence']*100:.0f}%")
        
        if prediction['recommendation']['action'] == 'BET':
            print(f"💰 BETTING OPPORTUNITY: {prediction['recommendation']['team']}")
            print(f"   Suggested stake: {prediction['recommendation']['suggested_stake']}%")
            print(f"   Reasoning: {', '.join(prediction['recommendation']['reasoning'])}")
        
        print(f"Expected Value: Team1={prediction['expected_value']['team1_ev_percent']}%, "
              f"Team2={prediction['expected_value']['team2_ev_percent']}%")
    
    await scraper.hltv_scraper.close()

if __name__ == "__main__":
    asyncio.run(main())