from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.models import Bar, Trade

@dataclass(slots=True)
class _BarState:
    ts_ms: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int


class BarBuilder:
    """
    Internal time-bucket OHLCV builder from trades (Addendum A1).

    Contract:
    - Input trades MUST be non-decreasing by event-time (ts_ms). Equal timestamps allowed.
    - Bars are keyed by bucket_seconds bucket: floor(ts_ms / (bucket_seconds*1000)) * (bucket_seconds*1000).
    - A bar closes when we observe a trade in a later bucket.
    """

    def __init__(self, *, symbol: str, bucket_seconds: int = 60, allow_out_of_order: bool = False) -> None:
        self._symbol = symbol
        self._allow_out_of_order = allow_out_of_order
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be > 0")
        self._bucket_ms = int(bucket_seconds) * 1000

        self._state: Optional[_BarState] = None
        self._last_ts_ms: Optional[int] = None

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def current_minute_ts_ms(self) -> Optional[int]:
        return self._state.ts_ms if self._state else None

    def update(self, trade: Trade) -> Optional[Bar]:
        """
        Consume one trade. Returns a completed Bar if this trade advances the minute;
        otherwise returns None.
        """
        if trade.symbol != self._symbol:
            raise ValueError(f"BarBuilder(symbol={self._symbol}) got trade for {trade.symbol}")

        ts_ms = int(trade.ts_ms)
        if (not self._allow_out_of_order) and (self._last_ts_ms is not None) and ts_ms < self._last_ts_ms:
            raise ValueError(f"out-of-order trade ts_ms={ts_ms} < last_ts_ms={self._last_ts_ms}")
        self._last_ts_ms = ts_ms

        minute_ts = (ts_ms // self._bucket_ms) * self._bucket_ms

        if self._state is None:
            self._state = _BarState(
                ts_ms=minute_ts,
                symbol=self._symbol,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=float(trade.size),
                trade_count=1,
            )
            return None

        if minute_ts == self._state.ts_ms:
            # Same minute: update OHLCV
            self._state.high = max(self._state.high, trade.price)
            self._state.low = min(self._state.low, trade.price)
            self._state.close = trade.price
            self._state.volume += float(trade.size)
            self._state.trade_count += 1
            return None

        if minute_ts < self._state.ts_ms and self._allow_out_of_order:
            # If out-of-order is allowed, we do not backfill past minutes; we simply reject
            # because otherwise we'd be redefining finalized OHLCV behavior.
            raise ValueError(
                f"out-of-order minute: trade minute {minute_ts} < current minute {self._state.ts_ms}"
            )

        # Minute advanced: finalize previous bar and start new bar.
        finished = Bar(
            ts_ms=self._state.ts_ms,
            symbol=self._state.symbol,
            open=self._state.open,
            high=self._state.high,
            low=self._state.low,
            close=self._state.close,
            volume=self._state.volume,
            trade_count=self._state.trade_count,
        )

        self._state = _BarState(
            ts_ms=minute_ts,
            symbol=self._symbol,
            open=trade.price,
            high=trade.price,
            low=trade.price,
            close=trade.price,
            volume=float(trade.size),
            trade_count=1,
        )
        return finished

    def flush(self) -> Optional[Bar]:
        """Finalize and return the current in-progress bar, if any."""
        if self._state is None:
            return None

        finished = Bar(
            ts_ms=self._state.ts_ms,
            symbol=self._state.symbol,
            open=self._state.open,
            high=self._state.high,
            low=self._state.low,
            close=self._state.close,
            volume=self._state.volume,
            trade_count=self._state.trade_count,
        )
        self._state = None
        return finished


