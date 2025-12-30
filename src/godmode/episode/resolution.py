from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.enums import DirectionBias, ResolutionTrigger


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    trigger: ResolutionTrigger
    resolution_time: int


def evaluate_resolution(
    *,
    ts_ms: int,
    last_price: float,
    level_price: float,
    atr_value: float,
    direction_bias: DirectionBias,
    success_threshold_atr: float,
    failure_threshold_atr: float,
    zone_exit_time: Optional[int],
    timeout_seconds: int,
) -> Optional[ResolutionDecision]:
    """
    Evaluate whether an episode resolves at this timestamp.

    Deterministic triggers (Addendum B):
    - Success: +success_threshold_atr ATR from level (thesis direction)
    - Failure: -failure_threshold_atr ATR against thesis before success
    - Timeout: timeout_seconds after zone_exit_time
    """
    if atr_value <= 0:
        raise ValueError("atr_value must be > 0")

    if zone_exit_time is not None:
        if ts_ms >= zone_exit_time + (timeout_seconds * 1000):
            return ResolutionDecision(trigger=ResolutionTrigger.TIMEOUT, resolution_time=int(ts_ms))

    move_atr = (float(last_price) - float(level_price)) / float(atr_value)

    if direction_bias == DirectionBias.LONG:
        if move_atr >= float(success_threshold_atr):
            return ResolutionDecision(trigger=ResolutionTrigger.THRESHOLD_HIT, resolution_time=int(ts_ms))
        if move_atr <= -float(failure_threshold_atr):
            return ResolutionDecision(trigger=ResolutionTrigger.INVALIDATION, resolution_time=int(ts_ms))
        return None

    # SHORT thesis: reverse sign
    if move_atr <= -float(success_threshold_atr):
        return ResolutionDecision(trigger=ResolutionTrigger.THRESHOLD_HIT, resolution_time=int(ts_ms))
    if move_atr >= float(failure_threshold_atr):
        return ResolutionDecision(trigger=ResolutionTrigger.INVALIDATION, resolution_time=int(ts_ms))
    return None


