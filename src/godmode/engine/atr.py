from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from godmode.core.enums import ATRSeedSource, ATRStatus
from godmode.core.models import Bar


def true_range(*, high: float, low: float, prev_close: Optional[float]) -> float:
    """
    TR_t = max(high_t - low_t, abs(high_t - close_{t-1}), abs(low_t - close_{t-1}))
    """
    hl = high - low
    if prev_close is None:
        return float(hl)
    hc = abs(high - prev_close)
    lc = abs(low - prev_close)
    return float(max(hl, hc, lc))


@dataclass(frozen=True, slots=True)
class ATRSnapshot:
    value: float
    status: ATRStatus
    seed_source: Optional[ATRSeedSource]
    blend_alpha: float
    is_warm: bool
    bars_today: int


class ATRCalculator:
    """
    ATR_14_1m computed on internal 1-minute bars using Wilder's ATR (RMA), per SPEC.md.

    Cold start policy (implementation decisions):
    - If bars_today >= n: ATR is live.
    - Else: ATR = alpha * ATR_seed + (1-alpha) * ATR_partial.
      If no partial exists yet, ATR = ATR_seed and status is 'seeded'.

    Notes:
    - This class is deterministic given deterministic bar ordering.
    - Bars must be non-decreasing by event-time; equal timestamps are allowed.
    """

    def __init__(self, *, period_bars: int = 14, atr_blend_alpha: float = 0.7) -> None:
        if period_bars <= 1:
            raise ValueError("period_bars must be > 1")
        if not (0.0 <= atr_blend_alpha <= 1.0):
            raise ValueError("atr_blend_alpha must be in [0, 1]")

        self._n = int(period_bars)
        self._alpha = float(atr_blend_alpha)

        self._seed: Optional[float] = None
        self._seed_source: Optional[ATRSeedSource] = None

        self._bars_today = 0
        self._prev_close: Optional[float] = None

        # Partial/live ATR computed from today's bars only.
        self._atr_partial: Optional[float] = None
        self._atr_live: Optional[float] = None

        self._last_bar_ts_ms: Optional[int] = None

    @property
    def period_bars(self) -> int:
        return self._n

    @property
    def atr_blend_alpha(self) -> float:
        return self._alpha

    def reset_session(
        self,
        *,
        atr_seed: float,
        seed_source: ATRSeedSource,
        atr_blend_alpha: Optional[float] = None,
    ) -> None:
        """Start a new session with a deterministic seed."""
        if atr_seed <= 0:
            raise ValueError("atr_seed must be > 0")
        self._seed = float(atr_seed)
        self._seed_source = seed_source
        if atr_blend_alpha is not None:
            if not (0.0 <= atr_blend_alpha <= 1.0):
                raise ValueError("atr_blend_alpha must be in [0, 1]")
            self._alpha = float(atr_blend_alpha)

        self._bars_today = 0
        self._prev_close = None
        self._atr_partial = None
        self._atr_live = None
        self._last_bar_ts_ms = None

    def update(self, bar: Bar) -> ATRSnapshot:
        if self._seed is None or self._seed_source is None:
            raise RuntimeError("ATRCalculator requires reset_session(atr_seed=..., seed_source=...) before update().")

        ts_ms = int(bar.ts_ms)
        if self._last_bar_ts_ms is not None and ts_ms < self._last_bar_ts_ms:
            raise ValueError(f"out-of-order bar ts_ms={ts_ms} < last_bar_ts_ms={self._last_bar_ts_ms}")
        self._last_bar_ts_ms = ts_ms

        tr = true_range(high=bar.high, low=bar.low, prev_close=self._prev_close)
        self._prev_close = bar.close
        self._bars_today += 1

        # ATR partial:
        # We need a deterministic partial value before we are warm.
        # We use a running SMA of TR until we have n bars; then Wilder smoothing applies as live.
        if self._bars_today == 1:
            self._atr_partial = tr
        else:
            assert self._atr_partial is not None
            k = self._bars_today
            if k <= self._n:
                # Running mean of TR_1..TR_k
                self._atr_partial = ((self._atr_partial * (k - 1)) + tr) / k
            else:
                # After warmup, partial follows Wilder same as live.
                self._atr_partial = ((self._atr_partial * (self._n - 1)) + tr) / self._n

        # Live ATR uses Wilder once warm; until then it is not considered live.
        is_warm = self._bars_today >= self._n
        if is_warm:
            if self._atr_live is None:
                # Initialization: on the first warm point, use the SMA(TR_1..TR_n) already held in atr_partial.
                self._atr_live = float(self._atr_partial)
            else:
                self._atr_live = ((self._atr_live * (self._n - 1)) + tr) / self._n

        if is_warm and self._atr_live is not None:
            return ATRSnapshot(
                value=float(self._atr_live),
                status=ATRStatus.LIVE,
                seed_source=self._seed_source,
                blend_alpha=self._alpha,
                is_warm=True,
                bars_today=self._bars_today,
            )

        # Not warm yet: blend seed with partial if available.
        if self._atr_partial is None:
            return ATRSnapshot(
                value=float(self._seed),
                status=ATRStatus.SEEDED,
                seed_source=self._seed_source,
                blend_alpha=self._alpha,
                is_warm=False,
                bars_today=self._bars_today,
            )

        blended = (self._alpha * float(self._seed)) + ((1.0 - self._alpha) * float(self._atr_partial))
        return ATRSnapshot(
            value=float(blended),
            status=ATRStatus.BLENDING,
            seed_source=self._seed_source,
            blend_alpha=self._alpha,
            is_warm=False,
            bars_today=self._bars_today,
        )


def daily_atr_fallback_seed(*, daily_atr: float) -> float:
    """Bulletproofing fallback: ATR_seed = daily_ATR / sqrt(390)."""
    if daily_atr <= 0:
        raise ValueError("daily_atr must be > 0")
    return float(daily_atr) / sqrt(390.0)


