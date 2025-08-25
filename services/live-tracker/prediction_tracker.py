import asyncio
import json
import logging
import os
import pickle
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp
import numpy as np
import pandas as pd
import psycopg2
import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'cs2_predictions')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'your_password')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
HLTV_API_URL = os.getenv('HLTV_API_URL', 'https://api.example.com/live-matches')
HLTV_API_KEY = os.getenv('HLTV_API_KEY', 'your_api_key')

PROM_PORT = int(os.getenv('PROMETHEUS_PORT', '8080'))


# Prometheus metrics
REQUESTS_TOTAL = Counter('tracker_requests_total', 'Total fetch cycles')
PREDICTIONS_TOTAL = Counter('tracker_predictions_total', 'Total predictions created')
PREDICTIONS_CORRECT = Counter('tracker_predictions_correct_total', 'Total correct predictions')
ACTIVE_PREDICTIONS = Gauge('tracker_active_predictions', 'Active predictions in memory')
CYCLE_DURATION = Histogram('tracker_cycle_duration_seconds', 'Duration of tracker loop cycle seconds')


@dataclass
class PredictionRecord:
    prediction_id: str
    match_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    team1: str
    team2: str
    predicted_winner: str
    win_probability: float
    confidence_score: float
    odds_team1: float
    odds_team2: float
    expected_value: float
    features_used: Dict
    actual_winner: Optional[str] = None
    is_correct: Optional[bool] = None
    profit_loss: Optional[float] = None


class LivePredictionTracker:
    def __init__(self):
        # Database connections
        self.pg_conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT,
        )
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

        # Load models
        self.models = self._load_models()
        self.active_predictions: Dict[str, PredictionRecord] = {}

    def _load_models(self):
        """Load trained ML models from models/ directory."""
        models = {}
        model_files = {
            'xgboost_v1': 'models/xgb_model.pkl',
            'lightgbm_v1': 'models/lgb_model.pkl',
            'ensemble_v1': 'models/ensemble_model.pkl',
        }
        for name, path in model_files.items():
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        return models

    async def fetch_live_matches(self):
        """Fetch ongoing matches. Replace API endpoint with real one when available."""
        async with aiohttp.ClientSession() as session:
            headers = {"API-Key": HLTV_API_KEY}
            try:
                async with session.get(HLTV_API_URL, headers=headers, timeout=20) as response:
                    if response.status == 200:
                        return await response.json()
                    logger.error(f"Live matches API HTTP {response.status}")
            except Exception as e:
                logger.error(f"Error fetching live matches: {e}")
        return []

    def extract_features(self, match_data: Dict) -> pd.DataFrame:
        features = {
            'team1_rating': match_data.get('team1_rating', 1500),
            'team2_rating': match_data.get('team2_rating', 1500),
            'team1_recent_winrate': match_data.get('team1_winrate', 0.5),
            'team2_recent_winrate': match_data.get('team2_winrate', 0.5),
            'head_to_head_ratio': match_data.get('h2h_ratio', 0.5),
            'team1_avg_rounds': match_data.get('team1_avg_rounds', 16),
            'team2_avg_rounds': match_data.get('team2_avg_rounds', 16),
            'map_team1_winrate': match_data.get('map_team1_wr', 0.5),
            'map_team2_winrate': match_data.get('map_team2_wr', 0.5),
            'team1_form': match_data.get('team1_form', 0),
            'team2_form': match_data.get('team2_form', 0),
            'tournament_tier': match_data.get('tournament_tier', 2),
            'bo_format': match_data.get('best_of', 1),
        }
        return pd.DataFrame([features])

    async def make_prediction(self, match_data: Dict) -> Optional[PredictionRecord]:
        features_df = self.extract_features(match_data)
        model = self.models.get('ensemble_v1')
        if not model:
            logger.error("No model available")
            return None

        # Predict
        prediction_proba = model.predict_proba(features_df)[0]
        predicted_class = int(model.predict(features_df)[0])

        # Confidence and EV
        confidence = float(np.max(prediction_proba))
        odds_team1 = float(match_data.get('odds_team1', 2.0))
        odds_team2 = float(match_data.get('odds_team2', 2.0))

        if predicted_class == 0:  # Team 1 wins
            predicted_winner = match_data['team1']
            win_prob = float(prediction_proba[0])
            expected_value = (win_prob * odds_team1) - 1.0
        else:  # Team 2 wins (2-way)
            predicted_winner = match_data['team2']
            win_prob = float(prediction_proba[-1])
            expected_value = (win_prob * odds_team2) - 1.0

        prediction = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            match_id=str(match_data['match_id']),
            timestamp=datetime.now(timezone.utc),
            model_name='ensemble_v1',
            model_version='1.0.0',
            team1=str(match_data['team1']),
            team2=str(match_data['team2']),
            predicted_winner=str(predicted_winner),
            win_probability=float(win_prob),
            confidence_score=float(confidence),
            odds_team1=odds_team1,
            odds_team2=odds_team2,
            expected_value=float(expected_value),
            features_used=features_df.to_dict('records')[0],
        )
        return prediction

    async def store_prediction(self, prediction: PredictionRecord):
        try:
            # Store in PostgreSQL
            with self.pg_conn.cursor() as cursor:
                insert_query = """
                INSERT INTO predictions (
                    prediction_id, match_id, timestamp, model_name, model_version,
                    team1, team2, predicted_winner, win_probability, confidence_score,
                    odds_team1, odds_team2, expected_value, features_used
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(insert_query, (
                    prediction.prediction_id,
                    prediction.match_id,
                    prediction.timestamp,
                    prediction.model_name,
                    prediction.model_version,
                    prediction.team1,
                    prediction.team2,
                    prediction.predicted_winner,
                    prediction.win_probability,
                    prediction.confidence_score,
                    prediction.odds_team1,
                    prediction.odds_team2,
                    prediction.expected_value,
                    json.dumps(prediction.features_used),
                ))
                self.pg_conn.commit()

            # Store in Redis
            redis_key = f"prediction:{prediction.match_id}"
            self.redis_client.setex(redis_key, 86400, json.dumps(asdict(prediction), default=str))

            # Active set
            self.active_predictions[prediction.match_id] = prediction
            PREDICTIONS_TOTAL.inc()
            ACTIVE_PREDICTIONS.set(len(self.active_predictions))
            logger.info(f"Stored prediction: {prediction.prediction_id}")
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            self.pg_conn.rollback()

    async def update_with_result(self, match_id: str, actual_winner: str):
        try:
            prediction = self.active_predictions.get(match_id)
            if not prediction:
                redis_data = self.redis_client.get(f"prediction:{match_id}")
                if redis_data:
                    prediction = PredictionRecord(**json.loads(redis_data))

            if prediction:
                prediction.actual_winner = actual_winner
                prediction.is_correct = (prediction.predicted_winner == actual_winner)
                bet_amount = 100.0
                if prediction.is_correct:
                    if prediction.predicted_winner == prediction.team1:
                        prediction.profit_loss = bet_amount * (prediction.odds_team1 - 1.0)
                    else:
                        prediction.profit_loss = bet_amount * (prediction.odds_team2 - 1.0)
                    PREDICTIONS_CORRECT.inc()
                else:
                    prediction.profit_loss = -bet_amount

                with self.pg_conn.cursor() as cursor:
                    update_query = """
                    UPDATE predictions
                    SET actual_winner = %s, is_correct = %s, profit_loss = %s
                    WHERE prediction_id = %s
                    """
                    cursor.execute(update_query, (
                        prediction.actual_winner,
                        prediction.is_correct,
                        prediction.profit_loss,
                        prediction.prediction_id,
                    ))
                    self.pg_conn.commit()
                logger.info(f"Updated result for {match_id}: {prediction.is_correct}")
        except Exception as e:
            logger.error(f"Error updating result: {e}")

    async def send_high_confidence_alert(self, prediction: PredictionRecord):
        if not DISCORD_WEBHOOK_URL:
            return
        alert_data = {
            'match': f"{prediction.team1} vs {prediction.team2}",
            'predicted_winner': prediction.predicted_winner,
            'confidence': f"{prediction.confidence_score:.2%}",
            'expected_value': f"{prediction.expected_value:.3f}",
            'timestamp': prediction.timestamp.isoformat(),
        }
        async with aiohttp.ClientSession() as session:
            await session.post(DISCORD_WEBHOOK_URL, json={
                'content': f"🎯 **High Confidence Prediction**\n```{json.dumps(alert_data, indent=2)}```"
            })

    async def run_live_tracker(self):
        logger.info("Starting live prediction tracker...")
        start_http_server(PROM_PORT)
        while True:
            with CYCLE_DURATION.time():
                try:
                    REQUESTS_TOTAL.inc()
                    live_matches = await self.fetch_live_matches()
                    for match in live_matches:
                        if match.get('match_id') not in self.active_predictions:
                            prediction = await self.make_prediction(match)
                            if prediction:
                                await self.store_prediction(prediction)
                                if prediction.confidence_score > 0.75:
                                    await self.send_high_confidence_alert(prediction)
                    ACTIVE_PREDICTIONS.set(len(self.active_predictions))
                    await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"Error in live tracker: {e}")
                    await asyncio.sleep(60)


if __name__ == "__main__":
    tracker = LivePredictionTracker()
    asyncio.run(tracker.run_live_tracker())
