from typing import Dict
import logging

logger = logging.getLogger(__name__)


def log_drift_metrics(metrics: Dict):
    logger.info(f"Drift metrics: {metrics}")
