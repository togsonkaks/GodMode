from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from godmode.core.models import Bar, Trade
from godmode.engine.ema import EMACalculator, ema_alpha
from godmode.engine.vwap import VWAPSessionCalculator


def _utc_ms(y: int, mo: int, d: int, hh: int, mm: int, ss: int) -> int:
    return int(datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp() * 1000)


def test_vwap_excludes_premarket_when_disabled() -> None:
    # Use UTC tz and a simple "RTH open" at 00:01:00 for deterministic testing.
    v = VWAPSessionCalculator(exchange_timezone="UTC", rth_open_time="00:01:00", include_premarket_in_vwap=False)

    # 00:00:30 is before open -> not active
    t1 = Trade(ts_ms=_utc_ms(2025, 1, 1, 0, 0, 30), symbol="AAPL", price=100.0, size=10)
    s1 = v.update(t1)
    assert s1.is_active is False
    assert s1.vwap is None

    # 00:01:10 is after open -> active; VWAP uses only post-open trades
    t2 = Trade(ts_ms=_utc_ms(2025, 1, 1, 0, 1, 10), symbol="AAPL", price=110.0, size=10)
    s2 = v.update(t2)
    assert s2.is_active is True
    assert s2.vwap == 110.0


def test_vwap_includes_premarket_when_enabled() -> None:
    v = VWAPSessionCalculator(exchange_timezone="UTC", rth_open_time="00:01:00", include_premarket_in_vwap=True)
    t1 = Trade(ts_ms=_utc_ms(2025, 1, 1, 0, 0, 30), symbol="AAPL", price=100.0, size=10)
    t2 = Trade(ts_ms=_utc_ms(2025, 1, 1, 0, 1, 10), symbol="AAPL", price=110.0, size=10)

    assert v.update(t1).vwap == 100.0
    s2 = v.update(t2)
    assert s2.is_active is True
    assert s2.vwap == 105.0


def test_vwap_resets_by_session_day() -> None:
    v = VWAPSessionCalculator(exchange_timezone="UTC", rth_open_time="00:00:00", include_premarket_in_vwap=True)

    t1 = Trade(ts_ms=_utc_ms(2025, 1, 1, 12, 0, 0), symbol="AAPL", price=100.0, size=10)
    assert v.update(t1).vwap == 100.0

    # Next day should reset
    t2 = Trade(ts_ms=_utc_ms(2025, 1, 2, 12, 0, 0), symbol="AAPL", price=200.0, size=10)
    assert v.update(t2).vwap == 200.0


def test_ema_update_formula() -> None:
    ema = EMACalculator(period=3)
    a = ema_alpha(3)

    b1 = Bar(ts_ms=0, symbol="AAPL", open=0, high=0, low=0, close=10.0, volume=0, trade_count=0)
    u1 = ema.update(b1)
    assert u1.prev is None
    assert u1.value == 10.0

    b2 = Bar(ts_ms=60_000, symbol="AAPL", open=0, high=0, low=0, close=13.0, volume=0, trade_count=0)
    u2 = ema.update(b2)
    expected = a * 13.0 + (1 - a) * 10.0
    assert u2.prev == 10.0
    assert math.isclose(u2.value, expected, rel_tol=1e-12)


def test_ema_rejects_out_of_order_bars() -> None:
    ema = EMACalculator(period=9)
    ema.update(Bar(ts_ms=60_000, symbol="AAPL", open=0, high=0, low=0, close=10.0, volume=0, trade_count=0))
    with pytest.raises(ValueError):
        ema.update(Bar(ts_ms=0, symbol="AAPL", open=0, high=0, low=0, close=10.0, volume=0, trade_count=0))


