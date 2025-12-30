from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.models import Bar


def ema_alpha(period: int) -> float:
    if period <= 0:
        raise ValueError("period must be > 0")
    return 2.0 / (period + 1.0)


@dataclass(frozen=True, slots=True)
class EMAUpdate:
    period: int
    prev: Optional[float]
    value: float


class EMACalculator:
    """
    Deterministic EMA calculator updated on internal 1-minute bars (uses bar.close).

    Initialization:
    - First observation initializes EMA to the first close (deterministic).
    """

    def __init__(self, *, period: int) -> None:
        self._period = int(period)
        self._alpha = ema_alpha(self._period)
        self._value: Optional[float] = None
        self._last_bar_ts_ms: Optional[int] = None

    @property
    def period(self) -> int:
        return self._period

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, bar: Bar) -> EMAUpdate:
        ts_ms = int(bar.ts_ms)
        if self._last_bar_ts_ms is not None and ts_ms < self._last_bar_ts_ms:
            raise ValueError(f"out-of-order bar ts_ms={ts_ms} < last_bar_ts_ms={self._last_bar_ts_ms}")
        self._last_bar_ts_ms = ts_ms

        prev = self._value
        x = float(bar.close)
        if prev is None:
            self._value = x
        else:
            self._value = (self._alpha * x) + ((1.0 - self._alpha) * prev)

        return EMAUpdate(period=self._period, prev=prev, value=float(self._value))


