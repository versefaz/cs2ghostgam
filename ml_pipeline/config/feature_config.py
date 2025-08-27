"""Feature configuration definitions"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    include_groups: List[str]
    exclude: List[str]
    feature_sets: Dict[str, Any]
    
    def __init__(self):
        self.include_groups = ['basic', 'map', 'momentum', 'economy']
        self.exclude = []
        self.feature_sets = FEATURE_SETS


FEATURE_SETS = {
    'default': {
        'include_groups': ['basic', 'map', 'momentum', 'economy'],
        'exclude': [],
    }
}
