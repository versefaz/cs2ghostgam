from cs2_betting_system.models.ensemble_predictor import EnsemblePredictor

def test_ensemble():
    e = EnsemblePredictor()
    res = e.predict({'team1':'A','team2':'B','odds_team1':2.1,'odds_team2':1.8})
    assert res.get('win_prob') is not None
