import pickle
from pathlib import Path


def verify_models() -> bool:
    required = ['xgb_model.pkl', 'lgb_model.pkl', 'ensemble_model.pkl']
    models_dir = Path('services/live-tracker/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    missing = [m for m in required if not (models_dir / m).exists()]
    if missing:
        print(f"Missing models: {missing}")
        return False
    ok = True
    for m in required:
        try:
            with open(models_dir / m, 'rb') as f:
                _ = pickle.load(f)
            print(f"OK: {m}")
        except Exception as e:
            print(f"Error loading {m}: {e}")
            ok = False
    return ok


if __name__ == '__main__':
    print('Verifying models...')
    ready = verify_models()
    print('Ready!' if ready else 'Model preparation failed')
