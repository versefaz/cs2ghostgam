def calculate_kelly_fraction(probability: float, odds: float, multiplier: float = 1.0) -> float:
    """Calculate Kelly fraction for given probability and decimal odds.

    Kelly = (bp - q) / b where b = odds-1, p = probability, q = 1-p
    Returns 0 for invalid inputs or negative results.
    """
    try:
        if odds <= 1 or probability <= 0 or probability >= 1:
            return 0.0
        b = odds - 1.0
        p = probability
        q = 1.0 - p
        kelly = (b * p - q) / b
        kelly = max(0.0, kelly)
        kelly *= max(0.0, multiplier)
        return max(0.0, min(1.0, kelly))
    except Exception:
        return 0.0
