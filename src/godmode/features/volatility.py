from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from godmode.core.models import Snapshot


def _pop_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return sqrt(v)


def _log_return(p_now: float, p_prev: float) -> Optional[float]:
    if p_now <= 0.0 or p_prev <= 0.0:
        return None
    import math

    return float(math.log(p_now / p_prev))


def _snapshot_at_or_before(snaps: list[Snapshot], ts_ms: int) -> Optional[Snapshot]:
    # snaps assumed in increasing timestamp order
    best: Optional[Snapshot] = None
    for s in snaps:
        if int(s.timestamp) <= ts_ms:
            best = s
        else:
            break
    return best


@dataclass(frozen=True, slots=True)
class VolatilityFeatures:
    realized_volatility_60s: float
    approach_return_60s: float
    approach_volatility_60s: float


class VolatilityFeatureEngine:
    """
    Deterministic volatility/approach features per SPEC.md §5F.

    - realized_volatility_60s: pop std dev of 10s log returns over last 60s (inclusive)
    - approach_return_60s: ln(p_t / p_{t-60s})
    - approach_volatility_60s: realized_volatility_60s
    """

    def __init__(self, *, rolling_window_seconds: int = 60) -> None:
        self._win_ms = int(rolling_window_seconds * 1000)

    def compute(self, *, current: Snapshot, history: list[Snapshot]) -> VolatilityFeatures:
        end_ms = int(current.timestamp)
        start_ms = end_ms - self._win_ms

        window = [s for s in history if start_ms <= int(s.timestamp) <= end_ms] + [current]
        window = sorted(window, key=lambda s: int(s.timestamp))

        # 10s log returns
        rets: list[float] = []
        prev_valid_price: Optional[float] = None
        for s in window:
            p = float(s.last_price)
            if p <= 0.0:
                continue
            if prev_valid_price is None:
                prev_valid_price = p
                continue
            r = _log_return(p, prev_valid_price)
            prev_valid_price = p
            if r is not None:
                rets.append(r)

        rv = _pop_std(rets)

        # approach_return_60s
        target = end_ms - self._win_ms
        snaps_all = sorted(history + [current], key=lambda s: int(s.timestamp))
        s_prev = _snapshot_at_or_before(snaps_all, target)
        if s_prev is None:
            ar = 0.0
        else:
            p_now = float(current.last_price)
            p_prev = float(s_prev.last_price)
            r = _log_return(p_now, p_prev)
            ar = float(r) if r is not None else 0.0

        return VolatilityFeatures(
            realized_volatility_60s=float(rv),
            approach_return_60s=float(ar),
            approach_volatility_60s=float(rv),
        )

    def apply_to_snapshot(self, snapshot: Snapshot, feats: VolatilityFeatures) -> None:
        snapshot.realized_volatility_60s = feats.realized_volatility_60s
        snapshot.approach_return_60s = feats.approach_return_60s
        snapshot.approach_volatility_60s = feats.approach_volatility_60s


