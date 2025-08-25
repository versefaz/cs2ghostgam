class KellyCriterion:
    def __init__(self, max_kelly_fraction: float = 0.25):
        self.max_kelly_fraction = max_kelly_fraction

    def calculate(self, win_prob: float, odds: float, bankroll: float) -> float:
        """
        f* = (p*b - q) / b
        p = win probability
        q = 1 - p
        b = odds - 1
        Returns fraction of bankroll to bet (capped and non-negative)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        b = odds - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - win_prob
        kelly = (win_prob * b - q) / b
        kelly = min(kelly, self.max_kelly_fraction)
        return max(0.0, kelly)
