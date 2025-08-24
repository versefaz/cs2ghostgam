from sklearn.ensemble import VotingClassifier
from typing import List, Tuple


def build_voting_ensemble(estimators: List[Tuple[str, object]], voting: str = 'soft') -> VotingClassifier:
    return VotingClassifier(estimators=estimators, voting=voting)
