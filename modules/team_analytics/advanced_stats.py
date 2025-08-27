import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Optional heavy deps guarded
try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

import numpy as np
import pandas as pd


class AdvancedTeamAnalytics:
    """Advanced team analytics with multi-source, async scraping and safe fallbacks.

    Notes:
    - External sites can change DOM; parsers are best-effort and fail-safe to empty results.
    - Methods return consistent shapes with zeros/defaults when data is unavailable.
    """

    def __init__(self):
        self.data_sources = {
            'hltv': 'https://www.hltv.org',
            'esea': 'https://play.esea.net',
            'faceit': 'https://www.faceit.com',
            'blast': 'https://blast.tv',
            'esl': 'https://pro.eslgaming.com',
        }

    async def fetch_team_form(self, team_name: str, days: int = 30) -> Dict:
        """Fetch recent team form across sources and compute summary stats.

        Returns dict with: matches, win_rate, round_difference, map_stats, momentum_score
        """
        results = {
            'matches': [],
            'win_rate': 0.0,
            'round_difference': 0,
            'map_stats': {},
            'momentum_score': 0.0,
        }

        # Fan out to multiple sources (guard when deps missing)
        tasks: List[asyncio.Future] = []
        tasks.append(self._fetch_hltv_form(team_name, days))
        tasks.append(self._fetch_faceit_form(team_name, days))
        tasks.append(self._fetch_esea_form(team_name, days))

        data_sources = await asyncio.gather(*tasks, return_exceptions=True)

        all_matches: List[Dict] = []
        for source_data in data_sources:
            if isinstance(source_data, Exception) or not isinstance(source_data, dict):
                continue
            all_matches.extend(source_data.get('matches', []))

        if not all_matches:
            return results

        df = pd.DataFrame(all_matches)
        # Basic normalization
        if 'result' not in df.columns:
            df['result'] = 'loss'
        if 'rounds_won' not in df.columns:
            df['rounds_won'] = 0
        if 'rounds_lost' not in df.columns:
            df['rounds_lost'] = 0
        if 'map' not in df.columns:
            df['map'] = 'unknown'

        results['matches'] = all_matches[-10:]  # last 10
        results['win_rate'] = float((df['result'] == 'win').mean())
        results['round_difference'] = int(df['rounds_won'].sum() - df['rounds_lost'].sum())

        for map_name in df['map'].dropna().unique():
            map_df = df[df['map'] == map_name]
            results['map_stats'][str(map_name)] = {
                'played': int(len(map_df)),
                'win_rate': float((map_df['result'] == 'win').mean()) if len(map_df) else 0.0,
                'avg_rounds': float(map_df['rounds_won'].mean()) if len(map_df) else 0.0,
            }

        # Momentum: exponentially weighted recent performance (most recent last row)
        n = len(df)
        weights = np.exp(-0.1 * np.arange(n))[::-1]  # higher weight for recent
        win_binary = (df['result'] == 'win').astype(float).to_numpy()
        try:
            results['momentum_score'] = float(np.average(win_binary, weights=weights))
        except ZeroDivisionError:
            results['momentum_score'] = float(win_binary.mean() if len(win_binary) else 0.0)

        return results

    async def _fetch_hltv_form(self, team_name: str, days: int) -> Dict:
        """Fetch team recent matches from HLTV (best-effort).

        HLTV uses numeric team IDs in canonical URLs. We attempt a name route for
        best-effort scraping; on failure, return empty.
        """
        if aiohttp is None or BeautifulSoup is None:
            return {'matches': []}

        # Note: real HLTV team URLs are like /team/<id>/<name>. Using name only as fallback.
        url = f"{self.data_sources['hltv']}/search?query={team_name}"
        try:
            timeout = aiohttp.ClientTimeout(total=10)
        except AttributeError:
            return {'matches': []}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                    if response.status != 200:
                        return {'matches': []}
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # This is a placeholder parse; DOM can differ.
                    # We look for result containers generically.
                    match_containers = soup.find_all('div', class_='result-con') if soup else []
                    matches: List[Dict] = []
                    for mc in match_containers[:20]:
                        try:
                            date_text = self._safe_text(mc.find('div', class_='date'))
                            opponent = self._safe_text(mc.find('div', class_='team'))
                            score_text = self._safe_text(mc.find('span', class_='score'))
                            map_name = self._safe_text(mc.find('div', class_='map')) or 'unknown'

                            won_class = mc.get('class', [])
                            result = 'win' if any('won' in c for c in won_class) else 'loss'

                            rw, rl = self._parse_score(score_text)
                            matches.append({
                                'date': self._parse_date(date_text),
                                'opponent': opponent,
                                'result': result,
                                'score': score_text,
                                'map': map_name,
                                'rounds_won': rw,
                                'rounds_lost': rl,
                            })
                        except Exception:
                            continue
                    return {'matches': matches}
        except Exception:
            return {'matches': []}

    async def _fetch_faceit_form(self, team_name: str, days: int) -> Dict:
        # Placeholder: FACEIT pages are dynamic; return empty to keep system robust.
        return {'matches': []}

    async def _fetch_esea_form(self, team_name: str, days: int) -> Dict:
        # Placeholder: ESEA scrape stub.
        return {'matches': []}

    async def calculate_head_to_head(self, team1: str, team2: str) -> Dict:
        """Analyze head-to-head history (best-effort HLTV scrape)."""
        h2h_stats = {
            'total_matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'map_records': {},
            'avg_round_diff': 0,
            'recent_form': [],
            'psychological_edge': None,
        }

        if aiohttp is None or BeautifulSoup is None:
            return h2h_stats

        url = f"{self.data_sources['hltv']}/results?team={team1}&team={team2}"
        try:
            timeout = aiohttp.ClientTimeout(total=10)
        except AttributeError:
            return h2h_stats
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                    if response.status != 200:
                        return h2h_stats
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    matches = soup.find_all('div', class_='result-con') if soup else []

                    round_diffs: List[int] = []
                    for match in matches:
                        info = self._parse_match_details(match)
                        if not info:
                            continue
                        h2h_stats['total_matches'] += 1
                        winner = info.get('winner')
                        if winner == team1:
                            h2h_stats['team1_wins'] += 1
                        elif winner == team2:
                            h2h_stats['team2_wins'] += 1

                        map_name = info.get('map')
                        if map_name:
                            rec = h2h_stats['map_records'].setdefault(map_name, {'team1': 0, 'team2': 0})
                            if winner == team1:
                                rec['team1'] += 1
                            elif winner == team2:
                                rec['team2'] += 1

                        rd = info.get('round_diff')
                        if isinstance(rd, int):
                            round_diffs.append(rd)

                    # Psychological edge
                    if h2h_stats['total_matches'] > 0:
                        wr = h2h_stats['team1_wins'] / h2h_stats['total_matches']
                        if wr > 0.65:
                            h2h_stats['psychological_edge'] = team1
                        elif wr < 0.35:
                            h2h_stats['psychological_edge'] = team2
                        else:
                            h2h_stats['psychological_edge'] = 'balanced'

                    h2h_stats['avg_round_diff'] = int(np.mean(round_diffs)) if round_diffs else 0
                    return h2h_stats
        except Exception:
            return h2h_stats

    async def get_map_pool_analysis(self, team_name: str) -> Dict:
        """Analyze team's map pool using source stats (stubbed best-effort)."""
        map_pool = {
            'permaban': [],
            'strong_maps': [],
            'weak_maps': [],
            'float_maps': [],
            'map_statistics': {},
        }

        maps = ['dust2', 'mirage', 'inferno', 'nuke', 'overpass', 'vertigo', 'ancient']
        for m in maps:
            stats = await self._fetch_map_statistics(team_name, m)
            map_pool['map_statistics'][m] = stats
            wr = stats.get('win_rate', 0.0)
            pr = stats.get('play_rate', 0.0)
            if pr < 0.05:
                map_pool['permaban'].append(m)
            elif wr > 0.65:
                map_pool['strong_maps'].append(m)
            elif wr < 0.35:
                map_pool['weak_maps'].append(m)
            else:
                map_pool['float_maps'].append(m)
        return map_pool

    # -------------------- helpers --------------------
    def _safe_text(self, node: Optional[object]) -> str:
        try:
            return (node.text or '').strip()
        except Exception:
            return ''

    def _parse_date(self, txt: str) -> str:
        # Attempt to parse common formats; return ISO string
        for fmt in ("%d %b %Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(txt.strip(), fmt).isoformat()
            except Exception:
                continue
        # Fallback: recent days heuristic or now
        return datetime.utcnow().isoformat()

    def _parse_score(self, txt: str) -> (int, int):
        try:
            t = txt.replace('\xa0', ' ').strip()
            parts = t.replace(':', '-').split('-')
            if len(parts) >= 2:
                rw = int(''.join(filter(str.isdigit, parts[0])))
                rl = int(''.join(filter(str.isdigit, parts[1])))
                return rw, rl
        except Exception:
            pass
        return 0, 0

    def _parse_match_details(self, match_node) -> Dict:
        try:
            date_text = self._safe_text(getattr(match_node, 'find', lambda *a, **k: None)('div', class_='date'))
            score_text = self._safe_text(getattr(match_node, 'find', lambda *a, **k: None)('span', class_='score'))
            rw, rl = self._parse_score(score_text)
            return {
                'date': self._parse_date(date_text),
                'winner': None,  # unknown without richer parse
                'map': self._safe_text(getattr(match_node, 'find', lambda *a, **k: None)('div', class_='map')) or 'unknown',
                'round_diff': rw - rl,
            }
        except Exception:
            return {}

    async def _fetch_map_statistics(self, team_name: str, map_name: str) -> Dict:
        # Stub: in absence of reliable API, return neutral priors
        return {
            'play_rate': 0.1,  # 10% play rate
            'win_rate': 0.5,   # 50% win rate
            'avg_rounds_won': 8.0,
            'sample': 10,
        }
