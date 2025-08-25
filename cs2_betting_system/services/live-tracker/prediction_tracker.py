import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from cs2_betting_system.scrapers.live_match_scraper import LiveMatchScraper
from cs2_betting_system.models.ensemble_predictor import EnsemblePredictor
from cs2_betting_system.config import settings
from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_MADE = Counter('cs2_predictions_total', 'Total predictions generated')
SCRAPE_ERRORS = Counter('cs2_scrape_errors_total', 'Total scrape errors')
SIGNALS_SENT = Counter('cs2_signals_sent_total', 'Total betting signals sent')
LOOP_LATENCY = Histogram('cs2_tracker_loop_seconds', 'Time to process one tracker loop')
ACTIVE_SIGNALS = Gauge('cs2_active_signals', 'Currently active signals in queue')

class EnhancedPredictionTracker:
    def __init__(self):
        self.scraper = LiveMatchScraper()
        self.model = EnsemblePredictor()
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
        self.signals = SignalGenerator()
        self._cooldown: Dict[str, float] = {}

    async def process_matches_once(self):
        with LOOP_LATENCY.time():
            matches = await asyncio.to_thread(self.scraper.scrape_all_sources)
            for m in matches:
                pred = self.model.predict(m)
                record = {
                    'match_id': m.get('match_id'),
                    'teams': [m.get('team1'), m.get('team2')],
                    'status': m.get('status'),
                    'timestamp': datetime.now().isoformat(),
                    'odds': {
                        'team1': m.get('odds_team1'),
                        'team2': m.get('odds_team2'),
                        'source': m.get('odds_source')
                    },
                    'prediction': pred,
                }
                key = f"prediction:{record['match_id']}"
                self.redis.setex(key, 3600, json.dumps(record, default=str))
                self.redis.lpush('predictions:recent', json.dumps(record, default=str))
                self.redis.ltrim('predictions:recent', 0, 499)
                PREDICTIONS_MADE.inc()

                # Auto-signal publishing with basic cooldown (30m per match)
                try:
                    ev = record['prediction'].get('ev') or record['prediction'].get('expected_value') or 0
                    conf = record['prediction'].get('confidence') or 0
                    match_id = record['match_id'] or 'unknown'
                    cooldown_key = match_id
                    should_send = ev and conf and ev > (settings.MIN_VALUE - 1) and conf > settings.MIN_CONFIDENCE
                    last = float(self._cooldown.get(cooldown_key, 0))
                    now_ts = datetime.now().timestamp()
                    if should_send and (now_ts - last > 30 * 60):
                        ACTIVE_SIGNALS.inc()
                        self.signals.emit_signal(record)
                        SIGNALS_SENT.inc()
                        ACTIVE_SIGNALS.dec()
                        self._cooldown[cooldown_key] = now_ts
                except Exception as e:
                    logger.warning(f"Signal emit failed: {e}")

    async def run(self, interval: int = 60):
        # start metrics server
        try:
            start_http_server(settings.METRICS_PORT_PREDICTION)
            logger.info(f"Metrics server started on :{settings.METRICS_PORT_PREDICTION}")
        except Exception as e:
            logger.warning(f"Metrics server failed to start: {e}")

        while True:
            try:
                await self.process_matches_once()
            except Exception as e:
                logger.error(f"Tracker loop error: {e}")
                SCRAPE_ERRORS.inc()
            await asyncio.sleep(interval)

async def main():
    logging.basicConfig(level=logging.INFO)
    tracker = EnhancedPredictionTracker()
    await tracker.run(settings.SCRAPE_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main())
