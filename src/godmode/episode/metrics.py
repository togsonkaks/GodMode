from __future__ import annotations

from typing import Optional

from godmode.core.enums import DirectionBias, Outcome, ResolutionTrigger
from godmode.core.models import Episode, Snapshot


def _move_atr(*, last_price: float, level_price: float, atr: float, direction_bias: DirectionBias) -> float:
    """
    Signed move in ATR units in the thesis direction.
    - LONG: positive when price rises from level
    - SHORT: positive when price falls from level
    """
    if atr <= 0:
        return 0.0
    if direction_bias == DirectionBias.LONG:
        return (float(last_price) - float(level_price)) / float(atr)
    return (float(level_price) - float(last_price)) / float(atr)


def apply_outcome_and_metrics(*, ep: Episode, snapshots: list[Snapshot]) -> None:
    """
    Deterministically populate:
    - outcome (from resolution_trigger)
    - mfe/mae (in ATR units, thesis-direction favorable/adverse)
    - time_to_mfe_ms (from zone_entry_time to first max-favorable)
    - time_to_failure_ms (from zone_entry_time to first time adverse >= failure_threshold_atr)

    This intentionally does NOT set resolution_type (interpretive without further spec).
    """
    # Outcome from trigger (minimal deterministic mapping)
    if ep.resolution_trigger == ResolutionTrigger.THRESHOLD_HIT:
        ep.outcome = Outcome.WIN
    elif ep.resolution_trigger == ResolutionTrigger.INVALIDATION:
        ep.outcome = Outcome.LOSS
    elif ep.resolution_trigger == ResolutionTrigger.TIMEOUT:
        ep.outcome = Outcome.SCRATCH

    atr = float(ep.atr_value)
    if atr <= 0:
        return

    # Determine computation window: stress+resolution segment.
    start = int(ep.zone_entry_time)
    end: Optional[int] = int(ep.end_time) if ep.end_time is not None else (int(ep.resolution_time) if ep.resolution_time is not None else None)

    # Filter snapshots in the window.
    window: list[Snapshot] = []
    for s in snapshots:
        t = int(s.timestamp)
        if t < start:
            continue
        if end is not None and t > end:
            continue
        window.append(s)

    if not window:
        return

    # Compute max favorable / max adverse excursion in ATR units.
    max_fav = float("-inf")
    max_adv = float("-inf")  # adverse is positive magnitude against thesis
    ts_max_fav: Optional[int] = None

    # Failure time (first time adverse reaches failure threshold)
    fail_thr = float(ep.failure_threshold_atr)
    ts_first_fail: Optional[int] = None if fail_thr > 0 else None

    for s in window:
        t = int(s.timestamp)
        mv = _move_atr(last_price=float(s.last_price), level_price=float(ep.level_price), atr=atr, direction_bias=ep.direction_bias)
        fav = mv
        adv = -mv

        if fav > max_fav:
            max_fav = fav
            ts_max_fav = t

        if adv > max_adv:
            max_adv = adv

        if ts_first_fail is None and fail_thr > 0 and adv >= fail_thr:
            ts_first_fail = t

    # Clamp excursions to >= 0 for interpretability
    ep.mfe = float(max(0.0, max_fav))
    ep.mae = float(max(0.0, max_adv))

    if ts_max_fav is not None:
        ep.time_to_mfe_ms = int(ts_max_fav - start)

    if ts_first_fail is not None:
        ep.time_to_failure_ms = int(ts_first_fail - start)


