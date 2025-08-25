from cs2_betting_system.models.prediction_model import PredictionModel

def test_predict_basic():
    m = PredictionModel()
    out = m.predict({'team1':'A','team2':'B','odds_team1':2.0,'odds_team2':2.0})
    assert 'win_prob' in out
