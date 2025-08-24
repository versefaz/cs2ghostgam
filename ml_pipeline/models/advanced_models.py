def get_xgboost(**kwargs):
    import xgboost as xgb
    return xgb.XGBClassifier(**kwargs)


def get_lightgbm(**kwargs):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**kwargs)
