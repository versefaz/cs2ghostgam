from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

class FeatureRequest(BaseModel):
    match_id: str
    team1_id: int
    team2_id: int
    map_name: str
    force_refresh: bool = False

class FeatureResponse(BaseModel):
    match_id: str
    features: Dict[str, Any]
    timestamp: datetime
