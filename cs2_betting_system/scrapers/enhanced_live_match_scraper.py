import asyncio
import logging
from typing import Any, Dict, List, Optional

# Local imports guarded to avoid hard failures if optional deps missing
try:
    from .hltv_stats_scraper import HLTVStatsScraper  # type: ignore
except Exception:  # pragma: no cover
    HLTVStatsScraper = None  # type: ignore

logger = logging.getLogger(__name__)


class EnhancedPredictionModel:
    """
    Placeholder prediction model that consumes enriched match features.
    Replace with your actual model manager integration.
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic placeholder
        strength_a = features.get('team_a_strength', 0.5)
        strength_b = features.get('team_b_strength', 0.5)
        prob_a = min(max(0.01, 0.5 + (strength_a - strength_b) * 0.4), 0.99)
        return {
            'team_a_win_prob': prob_a,
            'team_b_win_prob': 1 - prob_a,
            'recommend_team': 'A' if prob_a >= self.threshold else 'B',
            'confidence': abs(prob_a - 0.5) * 2,
        }


class EnhancedLiveMatchScraper:
    """
    Scraper that combines live match data with HLTV team stats for richer analytics.
    - fetches live matches from your existing pipeline/source (stub)
    - enriches with HLTV stats via HLTVStatsScraper
    - extracts features for prediction
    """

    def __init__(self, prediction_model: Optional[EnhancedPredictionModel] = None):
        self.prediction_model = prediction_model or EnhancedPredictionModel()
        self.hltv = HLTVStatsScraper() if HLTVStatsScraper else None

    async def initialize(self):
        if self.hltv and hasattr(self.hltv, 'initialize'):
            await self.hltv.initialize()

    async def close(self):
        if self.hltv and hasattr(self.hltv, 'close'):
            await self.hltv.close()

    async def fetch_live_matches(self) -> List[Dict[str, Any]]:
        """
        Replace this stub with your live source.
        Expected match dict keys: team_a, team_b, best_of, event, maps (optional)
        """
        # Stub returning a synthetic live match
        return [{
            'team_a': 'Team A',
            'team_b': 'Team B',
            'best_of': 3,
            'event': 'Showmatch',
            'maps': ['mirage', 'nuke', 'ancient'],
        }]

    async def enrich_match_with_stats(self, match: Dict[str, Any]) -> Dict[str, Any]:
        if not self.hltv:
            logger.warning('HLTVStatsScraper not available; returning match unchanged')
            return {**match, 'enrichment': {}}

        team_a, team_b = match.get('team_a'), match.get('team_b')
        try:
            a_stats_task = self.hltv.get_team_stats(team_a)
            b_stats_task = self.hltv.get_team_stats(team_b)
            h2h_task = self.hltv.get_h2h_stats(team_a, team_b)
            a_stats, b_stats, h2h = await asyncio.gather(a_stats_task, b_stats_task, h2h_task, return_exceptions=True)

            if isinstance(a_stats, Exception):
                logger.error(f"Failed to get stats for {team_a}: {a_stats}")
                a_stats = {}
            if isinstance(b_stats, Exception):
                logger.error(f"Failed to get stats for {team_b}: {b_stats}")
                b_stats = {}
            if isinstance(h2h, Exception):
                logger.error(f"Failed to get h2h for {team_a} vs {team_b}: {h2h}")
                h2h = {}

            features = self._extract_features(match, a_stats, b_stats, h2h)
            prediction = await self.prediction_model.predict(features)

            return {
                **match,
                'enrichment': {
                    'team_a_stats': a_stats,
                    'team_b_stats': b_stats,
                    'h2h': h2h,
                    'features': features,
                    'prediction': prediction,
                }
            }
        except Exception as e:
            logger.exception(f"enrich_match_with_stats error: {e}")
            return {**match, 'enrichment': {}}

    def _extract_features(self, match: Dict[str, Any], a_stats: Dict[str, Any], b_stats: Dict[str, Any], h2h: Dict[str, Any]) -> Dict[str, Any]:
        def safe(d: Optional[Dict[str, Any]], *keys, default=0.0):
            cur = d or {}
            for k in keys:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return default
            return cur if isinstance(cur, (int, float)) else default

        a_strength = safe(a_stats, 'processed', 'overall_strength', default=0.5)
        b_strength = safe(b_stats, 'processed', 'overall_strength', default=0.5)
        a_form = safe(a_stats, 'processed', 'form_last_5', default=0.5)
        b_form = safe(b_stats, 'processed', 'form_last_5', default=0.5)
        a_rating = safe(a_stats, 'processed', 'avg_team_rating', default=1.0)
        b_rating = safe(b_stats, 'processed', 'avg_team_rating', default=1.0)
        a_map_strength = safe(a_stats, 'processed', 'avg_map_strength', default=0.5)
        b_map_strength = safe(b_stats, 'processed', 'avg_map_strength', default=0.5)

        total_h2h = int((h2h or {}).get('total_matches', 0) or 0)
        team_a_h2h_wr = float((h2h or {}).get('team1_winrate', 0.0) or 0.0)

        features = {
            'best_of': int(match.get('best_of', 1) or 1),
            'team_a_strength': a_strength,
            'team_b_strength': b_strength,
            'team_a_form': a_form,
            'team_b_form': b_form,
            'team_a_rating': a_rating,
            'team_b_rating': b_rating,
            'team_a_map_strength': a_map_strength,
            'team_b_map_strength': b_map_strength,
            'team_a_h2h_wr': team_a_h2h_wr,
            'h2h_samples': total_h2h,
        }

        # Optional: map veto preferences signal
        maps = [str(m).lower() for m in match.get('maps', [])]
        if maps:
            a_map_pool = (a_stats or {}).get('processed', {}).get('map_pool', {})
            b_map_pool = (b_stats or {}).get('processed', {}).get('map_pool', {})
            features['map_bias'] = sum((a_map_pool.get(m, {}).get('strength', 0.0) - b_map_pool.get(m, {}).get('strength', 0.0)) for m in maps) / max(1, len(maps))
        else:
            features['map_bias'] = 0.0
        return features

    async def run_once(self) -> List[Dict[str, Any]]:
        matches = await self.fetch_live_matches()
        enriched: List[Dict[str, Any]] = []
        for m in matches:
            enriched.append(await self.enrich_match_with_stats(m))
        return enriched
