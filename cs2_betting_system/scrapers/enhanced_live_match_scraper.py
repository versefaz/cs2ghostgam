import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Optional heavy deps guarded
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

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
        self.models: Dict[str, object] = {}
        self.scaler: Optional[object] = None
        self.feature_importance: Dict[str, float] = {}

    async def initialize(self) -> None:
        await self.load_models()
        await self.load_scaler()

    async def load_models(self) -> None:
        # Stub: load trained models from disk or registry
        self.models['default'] = None

    async def load_scaler(self) -> None:
        # Stub: load feature scaler if used
        self.scaler = None

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic placeholder
        strength_a = features.get('team_a_strength', 0.5)
        strength_b = features.get('team_b_strength', 0.5)
        base = 0.5 + (strength_a - strength_b) * 0.4
        # Optional odds adjustment if provided
        implied_a = features.get('implied_prob_team1')
        if isinstance(implied_a, (int, float)) and 0 < implied_a < 1:
            base = (base + implied_a) / 2
        prob_a = min(max(0.01, base), 0.99)
        return {
            'team_a_win_prob': prob_a,
            'team_b_win_prob': 1 - prob_a,
            'recommend_team': 'A' if prob_a >= self.threshold else 'B',
            'confidence': abs(prob_a - 0.5) * 2,
        }

    async def extract_features(self, match: Dict[str, Any]) -> "pd.DataFrame":  # type: ignore
        if pd is None:
            # Minimal fallback: return dict wrapped as a single-row-like object
            return {k: [v] for k in (match.get('enrichment', {}).get('features', {}) or {}).items()}  # type: ignore

        f = match.get('enrichment', {}).get('features', {}) or {}
        return pd.DataFrame([f])


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
        self.matches_cache: Dict[str, Dict] = {}
        self.stats_cache: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def initialize(self):
        if self.hltv and hasattr(self.hltv, 'initialize'):
            await self.hltv.initialize()
        if hasattr(self.prediction_model, 'initialize'):
            await self.prediction_model.initialize()

    async def close(self):
        if self.hltv and hasattr(self.hltv, 'close'):
            await self.hltv.close()
        self.executor.shutdown(wait=True)

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

    async def get_live_matches_with_stats(self) -> List[Dict[str, Any]]:
        """Fetch live matches and enhance each with HLTV stats, odds, features, predictions."""
        live_matches = await self.fetch_live_matches()
        tasks = [self.enrich_match_with_stats(m) for m in live_matches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        enhanced: List[Dict[str, Any]] = []
        for m, r in zip(live_matches, results):
            if isinstance(r, Exception):
                logger.error(f"Failed to enhance match for {m.get('team_a')} vs {m.get('team_b')}: {r}")
                enhanced.append(m)
            else:
                enhanced.append(r)
        return enhanced

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
            # augment with odds
            odds = await self._fetch_live_odds(match)
            if odds:
                features['implied_prob_team1'] = odds.get('implied_prob_team1')
                features['implied_prob_team2'] = odds.get('implied_prob_team2')
            prediction = await self.prediction_model.predict(features)

            return {
                **match,
                'enrichment': {
                    'team_a_stats': a_stats,
                    'team_b_stats': b_stats,
                    'h2h': h2h,
                    'features': features,
                    'odds': odds,
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

    async def _fetch_live_odds(self, match: Dict[str, Any]) -> Dict[str, Any]:
        sources = ['bet365', '1xbet', 'pinnacle', 'ggbet', 'thunderpick']
        tasks = [self._get_odds_from_source(match, s) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        team1_odds: List[float] = []
        team2_odds: List[float] = []
        data = {'sources': {}, 'average': {}, 'best': {}}
        for src, res in zip(sources, results):
            if not isinstance(res, Exception) and res:
                data['sources'][src] = res
                if isinstance(res.get('team1_odds'), (int, float)):
                    team1_odds.append(float(res['team1_odds']))
                if isinstance(res.get('team2_odds'), (int, float)):
                    team2_odds.append(float(res['team2_odds']))
        if team1_odds:
            avg1 = float(np.mean(team1_odds)) if np is not None else sum(team1_odds) / len(team1_odds)
            data['average']['team1'] = avg1
            data['best']['team1'] = max(team1_odds)
            data['implied_prob_team1'] = 1.0 / avg1 if avg1 > 0 else None
        if team2_odds:
            avg2 = float(np.mean(team2_odds)) if np is not None else sum(team2_odds) / len(team2_odds)
            data['average']['team2'] = avg2
            data['best']['team2'] = max(team2_odds)
            data['implied_prob_team2'] = 1.0 / avg2 if avg2 > 0 else None
        data['movement'] = await self._detect_odds_movement(match, data)
        return data

    async def _get_odds_from_source(self, match: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        try:
            # Placeholder: integrate with your robust odds scraper here
            return {
                'team1_odds': 1.90,
                'team2_odds': 1.95,
                'source': source,
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to get odds from {source}: {e}")
            return None

    async def _detect_odds_movement(self, match: Dict[str, Any], current_odds: Dict[str, Any]) -> Dict[str, Any]:
        match_key = f"{match.get('team_a')}|{match.get('team_b')}"
        history = self.matches_cache.get(match_key, {}).get('odds_history', [])
        movement = {'direction': 'stable', 'magnitude': 0.0, 'velocity': 0.0}
        if history:
            last = history[-1]
            last_avg1 = (last.get('average') or {}).get('team1')
            cur_avg1 = (current_odds.get('average') or {}).get('team1')
            if isinstance(last_avg1, (int, float)) and isinstance(cur_avg1, (int, float)):
                delta = cur_avg1 - last_avg1
                if abs(delta) > 0.05:
                    movement['direction'] = 'up' if delta > 0 else 'down'
                    movement['magnitude'] = abs(delta)
                    try:
                        t1 = datetime.fromisoformat(last.get('timestamp'))
                        t2 = datetime.utcnow()
                        hours = max((t2 - t1).total_seconds() / 3600, 1e-6)
                        movement['velocity'] = movement['magnitude'] / hours
                    except Exception:
                        movement['velocity'] = 0.0
        # append
        if match_key not in self.matches_cache:
            self.matches_cache[match_key] = {'odds_history': []}
        current_odds['timestamp'] = datetime.utcnow().isoformat()
        self.matches_cache[match_key]['odds_history'].append(current_odds)
        self.matches_cache[match_key]['odds_history'] = self.matches_cache[match_key]['odds_history'][-20:]
        return movement

    def _encode_map(self, map_name: str) -> int:
        maps = ['ancient', 'anubis', 'inferno', 'mirage', 'nuke', 'overpass', 'vertigo', 'tba']
        m = (map_name or 'tba').lower()
        return maps.index(m) if m in maps else len(maps) - 1
