from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from godmode.core.enums import DirectionBias


class MarkerType(str, Enum):
    CONSOLIDATION = "consolidation"
    SUPPORT_BOUNCE = "support_bounce"
    BREAKDOWN = "breakdown"
    DOWNTREND_BREAK = "downtrend_break"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_BOTTOM = "triple_bottom"
    DOUBLE_TOP = "double_top"
    TRIPLE_TOP = "triple_top"


@dataclass(frozen=True, slots=True)
class Marker:
    marker_id: str
    session_id: str
    ticker: str
    ts_ms: int
    marker_type: MarkerType
    direction_bias: Optional[DirectionBias] = None
    notes: Optional[str] = None


