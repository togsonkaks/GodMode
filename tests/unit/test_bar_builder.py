from __future__ import annotations

import pytest

from godmode.core.models import Trade
from godmode.engine.bar_builder import BarBuilder


def _t(ts_ms: int, price: float, size: float = 1.0, symbol: str = "AAPL") -> Trade:
    return Trade(ts_ms=ts_ms, symbol=symbol, price=price, size=size)


def test_builds_single_minute_ohlcv() -> None:
    b = BarBuilder(symbol="AAPL")
    assert b.update(_t(0, 10.0, 1)) is None
    assert b.update(_t(1_000, 11.0, 2)) is None
    assert b.update(_t(2_000, 9.5, 3)) is None
    assert b.update(_t(59_999, 10.5, 4)) is None

    bar = b.flush()
    assert bar is not None
    assert bar.ts_ms == 0
    assert bar.open == 10.0
    assert bar.high == 11.0
    assert bar.low == 9.5
    assert bar.close == 10.5
    assert bar.volume == 1 + 2 + 3 + 4
    assert bar.trade_count == 4


def test_returns_finished_bar_when_minute_advances() -> None:
    b = BarBuilder(symbol="AAPL")
    b.update(_t(10_000, 100.0, 1))
    b.update(_t(20_000, 101.0, 1))

    # Next trade is in minute starting at 60_000 -> should finalize previous minute (0).
    finished = b.update(_t(60_000, 200.0, 5))
    assert finished is not None
    assert finished.ts_ms == 0
    assert finished.open == 100.0
    assert finished.high == 101.0
    assert finished.low == 100.0
    assert finished.close == 101.0
    assert finished.volume == 2.0
    assert finished.trade_count == 2

    # Current bar now is minute 60_000 and flush should finalize it.
    bar2 = b.flush()
    assert bar2 is not None
    assert bar2.ts_ms == 60_000
    assert bar2.open == 200.0
    assert bar2.high == 200.0
    assert bar2.low == 200.0
    assert bar2.close == 200.0
    assert bar2.volume == 5.0
    assert bar2.trade_count == 1


def test_rejects_wrong_symbol() -> None:
    b = BarBuilder(symbol="AAPL")
    with pytest.raises(ValueError):
        b.update(_t(0, 10.0, 1, symbol="TSLA"))


def test_rejects_out_of_order_ts_ms_by_default() -> None:
    b = BarBuilder(symbol="AAPL")
    b.update(_t(2_000, 10.0))
    with pytest.raises(ValueError):
        b.update(_t(1_000, 9.0))


def test_equal_ts_ms_is_allowed_and_updates_close() -> None:
    b = BarBuilder(symbol="AAPL")
    b.update(_t(1_000, 10.0, 1))
    b.update(_t(1_000, 11.0, 1))  # equal ts_ms, same minute
    bar = b.flush()
    assert bar is not None
    assert bar.open == 10.0
    assert bar.close == 11.0
    assert bar.high == 11.0
    assert bar.low == 10.0
    assert bar.trade_count == 2


