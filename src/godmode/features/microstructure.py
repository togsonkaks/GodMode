from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from godmode.core.models import Snapshot


def _pop_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return sqrt(v)


@dataclass(frozen=True, slots=True)
class MicrostructureFeatures:
    spread_volatility_60s: float
    spread_zscore: float


class MicrostructureFeatureEngine:
    """
    Deterministic microstructure features (SPEC §5B deterministic definitions).

    Window: last 60s of snapshots (inclusive), cadence aligned.
    Include only samples with valid NBBO (bid>0 and ask>0).
    """

    def __init__(self, *, rolling_window_seconds: int = 60, epsilon: float = 1e-9) -> None:
        self._win_ms = int(rolling_window_seconds * 1000)
        self._eps = float(epsilon)

    def compute(self, *, current: Snapshot, history: list[Snapshot]) -> MicrostructureFeatures:
        end_ms = int(current.timestamp)
        start_ms = end_ms - self._win_ms

        window = [s for s in history if start_ms <= int(s.timestamp) <= end_ms] + [current]

        spreads = [float(s.spread_pct) for s in window if float(s.bid) > 0.0 and float(s.ask) > 0.0]
        vol = _pop_std(spreads)

        if len(spreads) < 2 or vol == 0.0:
            z = 0.0
        else:
            m = sum(spreads) / len(spreads)
            z = (float(current.spread_pct) - m) / (vol + self._eps)

        return MicrostructureFeatures(spread_volatility_60s=float(vol), spread_zscore=float(z))

    def apply_to_snapshot(self, snapshot: Snapshot, feats: MicrostructureFeatures) -> None:
        snapshot.spread_volatility_60s = feats.spread_volatility_60s
        snapshot.spread_zscore = feats.spread_zscore


