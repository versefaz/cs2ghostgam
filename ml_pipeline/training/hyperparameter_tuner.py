from typing import Dict
import optuna


def run_optuna(objective, n_trials: int = 50, direction: str = 'maximize', study_name: str = 'tuning') -> Dict:
    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    return {'best_params': study.best_params, 'best_value': study.best_value}
