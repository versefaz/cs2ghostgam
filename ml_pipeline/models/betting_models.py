import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class BettingPrediction:
    """Betting prediction result"""
    match_id: str
    team1_win_prob: float
    team2_win_prob: float
    draw_prob: float
    confidence: float
    expected_value: Dict[str, float]
    kelly_fraction: Dict[str, float]


class ProbabilityCalibrator:
    """Simple probability calibration via Isotonic Regression per class.

    For multi-class (t1, draw, t2), we calibrate one-vs-rest probabilities and
    renormalize to 1.0.
    """

    def __init__(self):
        self.iso_t1: Optional[IsotonicRegression] = None
        self.iso_draw: Optional[IsotonicRegression] = None
        self.iso_t2: Optional[IsotonicRegression] = None

    def fit(self, raw_proba: np.ndarray, y: np.ndarray):
        """Fit calibrators.
        raw_proba: shape (n_samples, 3)
        y: integer labels in {0,1,2} for (t1, draw, t2)
        """
        if raw_proba.ndim != 2 or raw_proba.shape[1] != 3:
            raise ValueError("raw_proba must be shape (n_samples, 3)")
        self.iso_t1 = IsotonicRegression(out_of_bounds="clip").fit(raw_proba[:, 0], (y == 0).astype(float))
        self.iso_draw = IsotonicRegression(out_of_bounds="clip").fit(raw_proba[:, 1], (y == 1).astype(float))
        self.iso_t2 = IsotonicRegression(out_of_bounds="clip").fit(raw_proba[:, 2], (y == 2).astype(float))
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        if any(x is None for x in (self.iso_t1, self.iso_draw, self.iso_t2)):
            raise RuntimeError("Calibrator not fitted")
        p1 = self.iso_t1.predict(raw_proba[:, 0])
        pd = self.iso_draw.predict(raw_proba[:, 1])
        p2 = self.iso_t2.predict(raw_proba[:, 2])
        P = np.vstack([p1, pd, p2]).T
        # Renormalize
        s = P.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return P / s


def implied_prob_from_decimal_odds(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0


def expected_value(prob: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV for a decimal odds bet with probability prob and stake."""
    return prob * (decimal_odds - 1) * stake - (1 - prob) * stake


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Kelly fraction for decimal odds.
    f* = (bp - q) / b, with b = odds - 1, q = 1 - p
    Clip to [0,1].
    """
    b = max(decimal_odds - 1.0, 0.0)
    if b == 0:
        return 0.0
    q = 1 - prob
    f_star = (b * prob - q) / b
    return float(np.clip(f_star, 0.0, 1.0))


class BettingModel:
    """Betting helper that converts model probabilities to suggested bets."""

    def __init__(self, calibrator: Optional[ProbabilityCalibrator] = None, min_edge: float = 0.01, kelly_cap: float = 0.25):
        self.calibrator = calibrator
        self.min_edge = min_edge
        self.kelly_cap = kelly_cap

    def predict_probs(self, raw_proba: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.predict(raw_proba)
        return raw_proba

    def evaluate_market(self, probs: np.ndarray, odds_t1: float, odds_draw: Optional[float], odds_t2: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute EV and Kelly per outcome. odds_draw can be None for 2-way markets.
        Returns (evs, kelly_fracs)
        """
        evs: Dict[str, float] = {}
        kellys: Dict[str, float] = {}

        evs['team1'] = expected_value(probs[0], odds_t1)
        kellys['team1'] = min(kelly_fraction(probs[0], odds_t1), self.kelly_cap)

        if odds_draw is not None and len(probs) == 3:
            evs['draw'] = expected_value(probs[1], odds_draw)
            kellys['draw'] = min(kelly_fraction(probs[1], odds_draw), self.kelly_cap)
            evs['team2'] = expected_value(probs[2], odds_t2)
            kellys['team2'] = min(kelly_fraction(probs[2], odds_t2), self.kelly_cap)
        else:
            # Two-way: normalize to t1/t2
            p1 = probs[0]
            p2 = probs[-1]
            s = max(p1 + p2, 1e-9)
            p1 /= s
            p2 /= s
            evs['team2'] = expected_value(p2, odds_t2)
            kellys['team2'] = min(kelly_fraction(p2, odds_t2), self.kelly_cap)

        return evs, kellys

    def suggest_bets(self, match_id: str, raw_proba: np.ndarray, odds_t1: float, odds_t2: float, odds_draw: Optional[float] = None) -> BettingPrediction:
        probs = self.predict_probs(raw_proba.reshape(-1, 3))[0]
        evs, kellys = self.evaluate_market(probs, odds_t1, odds_draw, odds_t2)

        # Confidence via entropy (lower entropy => higher confidence)
        entropy = stats.entropy(probs)
        confidence = float(1.0 - entropy / np.log(len(probs)))

        # Filter minimal edge bets (optional for external consumers)
        evs = {k: v for k, v in evs.items() if v > self.min_edge}
        kellys = {k: v for k, v in kellys.items() if k in evs}

        return BettingPrediction(
            match_id=match_id,
            team1_win_prob=float(probs[0]),
            draw_prob=float(probs[1]) if len(probs) == 3 else 0.0,
            team2_win_prob=float(probs[2] if len(probs) == 3 else probs[-1]),
            confidence=confidence,
            expected_value=evs,
            kelly_fraction=kellys,
        )
