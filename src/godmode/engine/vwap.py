from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from godmode.core.models import Trade


def _parse_hhmmss(s: str) -> time:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError("time must be HH:MM:SS")
    h, m, sec = (int(parts[0]), int(parts[1]), int(parts[2]))
    return time(hour=h, minute=m, second=sec)


def _dt_from_ts_ms(ts_ms: int, tz: ZoneInfo) -> datetime:
    # Deterministic conversion for session boundary logic.
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=tz)


def _rth_open_ts_ms(*, d: date, tz: ZoneInfo, rth_open: time) -> int:
    dt = datetime(d.year, d.month, d.day, rth_open.hour, rth_open.minute, rth_open.second, tzinfo=tz)
    return int(dt.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class VWAPSnapshot:
    vwap: Optional[float]
    session_date: date
    is_active: bool  # True once VWAP accumulation has started for the session


class VWAPSessionCalculator:
    """
    VWAP using trades (price * size / total size).

    Determinism:
    - VWAP is computed in event-time using trade.ts_ms.
    - VWAP resets daily at RTH open by default (SPEC Addendum A3).
    - include_premarket_in_vwap controls whether trades before RTH open count.
    """

    def __init__(
        self,
        *,
        exchange_timezone: str = "America/New_York",
        rth_open_time: str = "09:30:00",
        include_premarket_in_vwap: bool = False,
    ) -> None:
        self._tz = ZoneInfo(exchange_timezone)
        self._rth_open = _parse_hhmmss(rth_open_time)
        self._include_premarket = include_premarket_in_vwap

        self._session_date: Optional[date] = None
        self._session_rth_open_ts_ms: Optional[int] = None

        self._pv = 0.0
        self._v = 0.0
        self._is_active = False

        self._last_ts_ms: Optional[int] = None

    def reset(self) -> None:
        self._session_date = None
        self._session_rth_open_ts_ms = None
        self._pv = 0.0
        self._v = 0.0
        self._is_active = False
        self._last_ts_ms = None

    def _ensure_session(self, ts_ms: int) -> None:
        dt = _dt_from_ts_ms(ts_ms, self._tz)
        d = dt.date()

        if self._session_date != d:
            self._session_date = d
            self._session_rth_open_ts_ms = _rth_open_ts_ms(d=d, tz=self._tz, rth_open=self._rth_open)
            self._pv = 0.0
            self._v = 0.0
            self._is_active = False

    def update(self, trade: Trade) -> VWAPSnapshot:
        ts_ms = int(trade.ts_ms)
        if self._last_ts_ms is not None and ts_ms < self._last_ts_ms:
            raise ValueError(f"out-of-order trade ts_ms={ts_ms} < last_ts_ms={self._last_ts_ms}")
        self._last_ts_ms = ts_ms

        self._ensure_session(ts_ms)
        assert self._session_date is not None
        assert self._session_rth_open_ts_ms is not None

        if (not self._include_premarket) and ts_ms < self._session_rth_open_ts_ms:
            # VWAP not active yet for this session.
            return VWAPSnapshot(vwap=None, session_date=self._session_date, is_active=False)

        # Start (or continue) VWAP accumulation.
        self._is_active = True

        size = float(trade.size)
        if size > 0:
            self._pv += float(trade.price) * size
            self._v += size

        vwap = (self._pv / self._v) if self._v > 0 else None
        return VWAPSnapshot(vwap=vwap, session_date=self._session_date, is_active=self._is_active)

    @property
    def current_vwap(self) -> Optional[float]:
        return (self._pv / self._v) if (self._is_active and self._v > 0) else None


