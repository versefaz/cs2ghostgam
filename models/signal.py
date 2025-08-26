from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
import json


class SignalSide(Enum):
    """ฝั่งของการเดิมพัน"""
    TEAM1_WIN = "team1_win"
    TEAM2_WIN = "team2_win"
    OVER = "over"
    UNDER = "under"
    TEAM1_HANDICAP = "team1_handicap"
    TEAM2_HANDICAP = "team2_handicap"
    DRAW = "draw"


class SignalStatus(Enum):
    """สถานะของสัญญาณ"""
    PENDING = "pending"
    PUBLISHED = "published"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    ERROR = "error"


class SignalPriority(Enum):
    """ระดับความสำคัญ"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BettingSignal:
    """โครงสร้างข้อมูลสัญญาณการเดิมพัน"""
    # Required fields
    match_id: str
    side: SignalSide
    stake: float  # จำนวนเงินเดิมพันแนะนำ (% ของ bankroll)
    ev: float  # Expected Value
    confidence: float  # ความมั่นใจ (0-1)

    # Auto-generated fields
    signal_id: Optional[str] = None
    created_at: Optional[datetime] = None

    # Optional fields
    odds: Optional[float] = None
    probability: Optional[float] = None
    kelly_fraction: Optional[float] = None

    # Metadata
    source: Optional[str] = None  # แหล่งที่มาของสัญญาณ
    strategy: Optional[str] = None  # กลยุทธ์ที่ใช้
    priority: SignalPriority = SignalPriority.MEDIUM
    status: SignalStatus = SignalStatus.PENDING

    # Additional data
    metadata: Optional[Dict[str, Any]] = None
    reasons: Optional[List[str]] = None  # เหตุผลของสัญญาณ

    # Timing
    expires_at: Optional[datetime] = None
    execution_deadline: Optional[datetime] = None

    def __post_init__(self):
        """Initialize auto fields"""
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.reasons is None:
            self.reasons = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert enums to strings
        data['side'] = self.side.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        # Convert datetime to ISO format
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        if self.execution_deadline:
            data['execution_deadline'] = self.execution_deadline.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BettingSignal':
        """Create from dictionary"""
        # Convert string enums back
        if isinstance(data.get('side'), str):
            data['side'] = SignalSide(data['side'])
        if isinstance(data.get('status'), str):
            data['status'] = SignalStatus(data['status'])
        if isinstance(data.get('priority'), (str, int)):
            # Allow int or str
            data['priority'] = SignalPriority(int(data['priority']))
        # Convert ISO strings to datetime
        for field in ['created_at', 'expires_at', 'execution_deadline']:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)
