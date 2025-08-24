class KellyCalculator:
    """คำนวณขนาดเดิมพันที่เหมาะสมตาม Kelly Criterion พร้อม Safety margin"""
    def calculate_stake(self, win_prob: float, odds: float, bankroll: float, max_exposure: float = 0.25) -> float:
        b = odds - 1
        p = win_prob
        q = 1 - p
        if b <= 0:
            return 0.0
        f = (b * p - q) / b
        f = max(0.0, min(f, max_exposure))
        return bankroll * (f * 0.25)
