from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_random_forest(**kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(**kwargs)


def get_logistic_regression(**kwargs) -> LogisticRegression:
    return LogisticRegression(**kwargs)
