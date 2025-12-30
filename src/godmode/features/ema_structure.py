from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.models import Episode, Snapshot


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _snapshot_at_or_before(snaps: list[Snapshot], ts_ms: int) -> Optional[Snapshot]:
    best: Optional[Snapshot] = None
    for s in snaps:
        if int(s.timestamp) <= ts_ms:
            best = s
        else:
            break
    return best


@dataclass(frozen=True, slots=True)
class EMAStructureFeatures:
    slope_ema9_60s: float
    slope_ema20_60s: float
    slope_ema30_60s: float
    ema_spread_9_20: float
    ema_spread_20_30: float
    compression_index: float
    ema_confluence_score: float
    stack_state: str
    price_vs_emas: int
    stretch_200_atr: float


class EMAStructureFeatureEngine:
    """
    Deterministic EMA structure features per SPEC.md §5D.
    """

    def __init__(self) -> None:
        pass

    def compute(
        self,
        *,
        episode: Episode,
        current: Snapshot,
        history: list[Snapshot],
        atr_14_1m: float,
    ) -> EMAStructureFeatures:
        if atr_14_1m <= 0:
            raise ValueError("atr_14_1m must be > 0")

        end_ms = int(current.timestamp)
        target = end_ms - 60_000

        snaps_all = sorted(history + [current], key=lambda s: int(s.timestamp))
        prev = _snapshot_at_or_before(snaps_all, target)

        def slope(cur: float, prev_val: Optional[float]) -> float:
            if prev_val is None:
                return 0.0
            return (cur - prev_val) / 60.0

        prev_ema9 = float(prev.ema9) if prev is not None else None
        prev_ema20 = float(prev.ema20) if prev is not None else None
        prev_ema30 = float(prev.ema30) if prev is not None else None

        ema9 = float(current.ema9)
        ema20 = float(current.ema20)
        ema30 = float(current.ema30)
        ema200 = float(current.ema200)

        slope9 = slope(ema9, prev_ema9)
        slope20 = slope(ema20, prev_ema20)
        slope30 = slope(ema30, prev_ema30)

        spread_9_20 = ema9 - ema20
        spread_20_30 = ema20 - ema30

        compression = (abs(spread_9_20) + abs(spread_20_30)) / float(atr_14_1m)
        confluence_score = clamp01(1.0 - (compression / float(episode.ema_confluence_ref)))

        if ema9 > ema20 > ema30:
            stack = "bull"
        elif ema9 < ema20 < ema30:
            stack = "bear"
        else:
            stack = "mixed"

        price = float(current.last_price)
        bit0 = 1 if price > ema9 and ema9 != 0.0 else 0
        bit1 = 1 if price > ema20 and ema20 != 0.0 else 0
        bit2 = 1 if price > ema30 and ema30 != 0.0 else 0
        bit3 = 1 if price > ema200 and ema200 != 0.0 else 0
        mask = bit0 + 2 * bit1 + 4 * bit2 + 8 * bit3

        stretch_200 = (price - ema200) / float(atr_14_1m) if ema200 != 0.0 and price != 0.0 else 0.0

        return EMAStructureFeatures(
            slope_ema9_60s=float(slope9),
            slope_ema20_60s=float(slope20),
            slope_ema30_60s=float(slope30),
            ema_spread_9_20=float(spread_9_20),
            ema_spread_20_30=float(spread_20_30),
            compression_index=float(compression),
            ema_confluence_score=float(confluence_score),
            stack_state=stack,
            price_vs_emas=int(mask),
            stretch_200_atr=float(stretch_200),
        )

    def apply_to_snapshot(self, snapshot: Snapshot, feats: EMAStructureFeatures) -> None:
        snapshot.slope_ema9_60s = feats.slope_ema9_60s
        snapshot.slope_ema20_60s = feats.slope_ema20_60s
        snapshot.slope_ema30_60s = feats.slope_ema30_60s
        snapshot.ema_spread_9_20 = feats.ema_spread_9_20
        snapshot.ema_spread_20_30 = feats.ema_spread_20_30
        snapshot.compression_index = feats.compression_index
        snapshot.ema_confluence_score = feats.ema_confluence_score
        snapshot.stack_state = feats.stack_state
        snapshot.price_vs_emas = feats.price_vs_emas
        snapshot.stretch_200_atr = feats.stretch_200_atr


