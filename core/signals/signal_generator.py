from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class SignalStrength(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class BettingSignal:
    match_id: str
    timestamp: datetime
    team1: str
    team2: str
    recommended_bet: str
    confidence: float
    signal_strength: SignalStrength
    expected_value: float
    kelly_fraction: float
    reasons: List[str]
    risk_level: str
    metadata: Dict


class SignalGenerator:
    """Professional Signal Generator with modular analyses and validations.

    Dependencies (model_manager, odds_analyzer, team_analyzer) are injected and
    must provide the following minimal interfaces:
      - model_manager.predict_match(match) -> List[{'probability': float}]
      - model_manager.get_true_probability(match) -> {'team1': float, 'team2': float}
      - odds_analyzer.get_best_odds(match) -> {'team1': float, 'team2': float}
      - team_analyzer.get_team_form(team_name) -> {'momentum_score': float, 'win_rate': float, 'round_difference': int}
    """

    def __init__(self, model_manager, odds_analyzer, team_analyzer):
        self.model_manager = model_manager
        self.odds_analyzer = odds_analyzer
        self.team_analyzer = team_analyzer

        self.thresholds = {
            'min_edge': 0.05,
            'min_confidence': 0.65,
            'max_kelly': 0.25,
            'min_expected_value': 1.05,
        }
        self.weights = {
            'ml_prediction': 0.35,
            'value_betting': 0.25,
            'team_form': 0.20,
            'h2h_history': 0.10,
            'market_movement': 0.10,
        }

    async def generate_signals(self, matches: List[Dict]) -> List[BettingSignal]:
        signals: List[BettingSignal] = []
        for m in matches:
            sig = await self.analyze_match(m)
            if sig and self._validate_signal(sig):
                signals.append(sig)
        return self._rank_signals(signals)

    async def analyze_match(self, match: Dict) -> Optional[BettingSignal]:
        ml_signal, value_signal, form_signal, h2h_signal, market_signal = await asyncio.gather(
            self._ml_analysis(match),
            self._value_analysis(match),
            self._form_analysis(match),
            self._h2h_analysis(match),
            self._market_analysis(match),
        )

        combined_signal = self._combine_signals({
            'ml': ml_signal,
            'value': value_signal,
            'form': form_signal,
            'h2h': h2h_signal,
            'market': market_signal,
        })
        if not combined_signal:
            return None

        confidence = self._calculate_confidence(combined_signal)
        expected_value = self._calculate_expected_value(combined_signal, match)
        kelly_fraction = self._calculate_kelly_criterion(combined_signal, match)
        signal_strength = self._determine_signal_strength(confidence, expected_value)
        recommended_bet = self._determine_recommendation(combined_signal, confidence, kelly_fraction)
        reasons = self._generate_reasoning(combined_signal)
        risk_level = self._assess_risk(combined_signal, match)

        return BettingSignal(
            match_id=str(match.get('match_id', f"{match.get('team1','')}vs{match.get('team2','')}")),
            timestamp=datetime.utcnow(),
            team1=str(match.get('team1', 'T1')),
            team2=str(match.get('team2', 'T2')),
            recommended_bet=recommended_bet,
            confidence=float(confidence),
            signal_strength=signal_strength,
            expected_value=float(expected_value),
            kelly_fraction=float(kelly_fraction),
            reasons=reasons,
            risk_level=risk_level,
            metadata={
                'ml_confidence': float(ml_signal.get('confidence', 0.0)),
                'value_edge': float(value_signal.get('edge', 0.0)),
                'form_score': float(form_signal.get('score', 0.0)),
                'h2h_advantage': float(h2h_signal.get('advantage', 0.0)),
                'market_sentiment': float(market_signal.get('sentiment', 0.0)),
            },
        )

    # ---------------- analyses ----------------
    async def _ml_analysis(self, match: Dict) -> Dict:
        try:
            predictions = await self.model_manager.predict_match(match)
        except Exception:
            return {'predicted_winner': None, 'probability': 0.5, 'confidence': 0.0, 'model_agreement': 0.0}
        probs = [float(p.get('probability', 0.5)) for p in (predictions or [])] or [0.5]
        ensemble_prob = float(np.mean(probs))
        dispersion = float(np.std(probs))
        if ensemble_prob > 0.5:
            predicted_winner = match.get('team1')
            win_probability = ensemble_prob
        else:
            predicted_winner = match.get('team2')
            win_probability = 1 - ensemble_prob
        return {
            'predicted_winner': predicted_winner,
            'probability': float(win_probability),
            'confidence': float(max(0.0, 1.0 - dispersion)),
            'model_agreement': float(1.0 - dispersion),
        }

    async def _value_analysis(self, match: Dict) -> Dict:
        try:
            best_odds = self.odds_analyzer.get_best_odds(match)
            true_prob = await self.model_manager.get_true_probability(match)
        except Exception:
            return {'value_bet': None, 'edge': 0.0}
        t1_val = float(best_odds.get('team1', 0) * true_prob.get('team1', 0) - 1)
        t2_val = float(best_odds.get('team2', 0) * true_prob.get('team2', 0) - 1)
        if t1_val > t2_val and t1_val > self.thresholds['min_edge']:
            return {
                'value_bet': match.get('team1'),
                'edge': t1_val,
                'implied_prob': float(1 / max(1e-9, best_odds.get('team1', 1e9))),
                'true_prob': float(true_prob.get('team1', 0.0)),
            }
        if t2_val > self.thresholds['min_edge']:
            return {
                'value_bet': match.get('team2'),
                'edge': t2_val,
                'implied_prob': float(1 / max(1e-9, best_odds.get('team2', 1e9))),
                'true_prob': float(true_prob.get('team2', 0.0)),
            }
        return {'value_bet': None, 'edge': 0.0}

    async def _form_analysis(self, match: Dict) -> Dict:
        try:
            t1 = await self.team_analyzer.get_team_form(str(match.get('team1', '')))
            t2 = await self.team_analyzer.get_team_form(str(match.get('team2', '')))
        except Exception:
            return {'form_favorite': None, 'score': 0.0}
        s1 = float(t1.get('momentum_score', 0.0))
        s2 = float(t2.get('momentum_score', 0.0))
        diff = s1 - s2
        if abs(diff) > 0.2:
            if diff > 0:
                return {'form_favorite': match.get('team1'), 'score': float(diff), 'team1_momentum': s1, 'team2_momentum': s2}
            return {'form_favorite': match.get('team2'), 'score': float(-diff), 'team1_momentum': s1, 'team2_momentum': s2}
        return {'form_favorite': None, 'score': 0.0}

    async def _h2h_analysis(self, match: Dict) -> Dict:
        # Placeholder: return neutral unless provided via external analyzer
        return {'advantage': 0.0}

    async def _market_analysis(self, match: Dict) -> Dict:
        # Placeholder: market sentiment not implemented
        return {'sentiment': 0.0}

    # ---------------- combine and scoring ----------------
    def _combine_signals(self, signals: Dict) -> Dict:
        combined = {'team1_score': 0.0, 'team2_score': 0.0, 'signals': []}
        ml = signals.get('ml', {})
        if ml.get('predicted_winner'):
            weight = self.weights['ml_prediction'] * float(ml.get('confidence', 0.0))
            if ml['predicted_winner'] == signals.get('team1'):
                combined['team1_score'] += weight
            else:
                combined['team2_score'] += weight
            combined['signals'].append('ml_prediction')
        val = signals.get('value', {})
        if val.get('value_bet'):
            weight = self.weights['value_betting'] * (1.0 + float(val.get('edge', 0.0)))
            if val['value_bet'] == signals.get('team1'):
                combined['team1_score'] += weight
            else:
                combined['team2_score'] += weight
            combined['signals'].append('value_betting')
        form = signals.get('form', {})
        if form.get('form_favorite'):
            weight = self.weights['team_form'] * float(form.get('score', 0.0))
            if form['form_favorite'] == signals.get('team1'):
                combined['team1_score'] += weight
            else:
                combined['team2_score'] += weight
            combined['signals'].append('team_form')
        return combined

    def _calculate_confidence(self, signal: Dict) -> float:
        total = signal['team1_score'] + signal['team2_score']
        return float(total / (total + 1e-6))  # bounded (0,1)

    def _calculate_expected_value(self, signal: Dict, match: Dict) -> float:
        # naive EV proxy: higher of team scores translated via odds if present
        t1, t2 = signal['team1_score'], signal['team2_score']
        if t1 + t2 == 0:
            return 1.0
        prob = t1 / (t1 + t2) if t1 >= t2 else t2 / (t1 + t2)
        odds = float(match.get('odds1' if t1 >= t2 else 'odds2', 2.0))
        return float(prob * odds)

    def _calculate_kelly_criterion(self, signal: Dict, match: Dict) -> float:
        t1, t2 = signal['team1_score'], signal['team2_score']
        if t1 + t2 == 0:
            return 0.0
        prob = t1 / (t1 + t2) if t1 >= t2 else t2 / (t1 + t2)
        odds = float(match.get('odds1' if t1 >= t2 else 'odds2', 2.0))
        b = max(1e-6, odds - 1.0)
        q = 1.0 - prob
        k = (prob * b - q) / b
        k = max(0.0, k) * 0.25
        return float(min(k, self.thresholds['max_kelly']))

    def _determine_signal_strength(self, confidence: float, ev: float) -> SignalStrength:
        if confidence > 0.8 and ev > 1.15:
            return SignalStrength.STRONG_BUY
        if confidence > 0.7 and ev > 1.08:
            return SignalStrength.BUY
        if confidence < 0.5 or ev < 0.95:
            return SignalStrength.STRONG_SELL
        if confidence < 0.6 or ev < 1.0:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL

    def _validate_signal(self, signal: BettingSignal) -> bool:
        return (
            signal.confidence >= self.thresholds['min_confidence']
            and signal.expected_value >= self.thresholds['min_expected_value']
            and signal.kelly_fraction > 0
            and signal.signal_strength not in {SignalStrength.NEUTRAL, SignalStrength.SELL, SignalStrength.STRONG_SELL}
        )

    def _determine_recommendation(self, combined: Dict, confidence: float, kelly: float) -> str:
        return 'TEAM1' if combined['team1_score'] >= combined['team2_score'] else 'TEAM2'

    def _generate_reasoning(self, combined: Dict) -> List[str]:
        reasons = []
        if 'ml_prediction' in combined['signals']:
            reasons.append('ML models consensus supports the pick')
        if 'value_betting' in combined['signals']:
            reasons.append('Positive value edge versus market odds')
        if 'team_form' in combined['signals']:
            reasons.append('Recent form and momentum advantage')
        if not reasons:
            reasons.append('Composite signal meets thresholds')
        return reasons

    def _assess_risk(self, combined: Dict, match: Dict) -> str:
        spread = abs(combined['team1_score'] - combined['team2_score'])
        if spread > 0.5:
            return 'LOW'
        if spread > 0.25:
            return 'MEDIUM'
        return 'HIGH'
