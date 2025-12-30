from __future__ import annotations

from datetime import datetime, timezone

import pytest

from godmode.core.config import AppConfig
from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias
from godmode.core.models import Episode, Level, Quote, Trade
from godmode.engine.ring_buffer import TimeRingBuffer
from godmode.engine.vwap import VWAPSessionCalculator
from godmode.episode.snapshot_builder import SnapshotBuilder
from godmode.zone.zone_gate import ZoneGate


def _utc_ms(y: int, mo: int, d: int, hh: int, mm: int, ss: int) -> int:
    return int(datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc).timestamp() * 1000)


def test_latest_quote_in_window_used_for_snapshot_quote() -> None:
    cfg = AppConfig()
    sb = SnapshotBuilder(config=cfg, interval_ms=10_000)

    trades = TimeRingBuffer[Trade](max_age_ms=300_000)
    quotes = TimeRingBuffer[Quote](max_age_ms=300_000)

    lvl = Level(level_id="L1", level_price=100.0, level_type="support", level_source="manual", level_width_atr=0.25)
    zg = ZoneGate(ticker="AAPL")
    atr = 4.0

    # Snapshot window [0..9999]
    quotes.append(Quote(ts_ms=1000, symbol="AAPL", bid=99.0, bid_size=1, ask=101.0, ask_size=1))
    quotes.append(Quote(ts_ms=9000, symbol="AAPL", bid=98.0, bid_size=1, ask=102.0, ask_size=1))  # latest in window

    zsnap, _ = zg.update(ts_ms=9_999, last_price=99.5, atr_value=atr, levels=[lvl])
    zone = zsnap[0]

    ep = Episode(
        episode_id="E1",
        session_id="S1",
        ticker="AAPL",
        level_id="L1",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side="below",
        start_time=-300_000,
        zone_entry_time=0,
        atr_value=atr,
        atr_status=ATRStatus.BLENDING,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
        direction_bias=DirectionBias.LONG,
    )

    vwap = VWAPSessionCalculator(exchange_timezone="UTC", rth_open_time="00:00:00", include_premarket_in_vwap=True)
    snap = sb.build(ts_ms=9_999, episode=ep, zone=zone, atr_value=atr, trades=trades, quotes=quotes, vwap=vwap)
    assert snap.bid == 98.0
    assert snap.ask == 102.0
    assert snap.quote_age_ms == 9999 - 9000


def test_trade_classification_uses_quote_at_or_before_trade_time() -> None:
    cfg = AppConfig()
    sb = SnapshotBuilder(config=cfg, interval_ms=10_000)

    trades = TimeRingBuffer[Trade](max_age_ms=300_000)
    quotes = TimeRingBuffer[Quote](max_age_ms=300_000)

    lvl = Level(level_id="L1", level_price=100.0, level_type="support", level_source="manual", level_width_atr=0.25)
    zg = ZoneGate(ticker="AAPL")
    atr = 4.0

    # Quotes: one earlier with tight spread, one later with wider spread
    quotes.append(Quote(ts_ms=1000, symbol="AAPL", bid=99.0, bid_size=1, ask=101.0, ask_size=1))
    quotes.append(Quote(ts_ms=8000, symbol="AAPL", bid=90.0, bid_size=1, ask=110.0, ask_size=1))

    # Trades in window [0..9999]
    # At t=1500, using quote(1000): ask=101 -> trade=101 classified BUY
    trades.append(Trade(ts_ms=1500, symbol="AAPL", price=101.0, size=10))
    # At t=9000, using quote(8000): bid=90 -> trade=90 classified SELL
    trades.append(Trade(ts_ms=9000, symbol="AAPL", price=90.0, size=5))

    zsnap, _ = zg.update(ts_ms=9_999, last_price=100.0, atr_value=atr, levels=[lvl])
    zone = zsnap[0]

    ep = Episode(
        episode_id="E1",
        session_id="S1",
        ticker="AAPL",
        level_id="L1",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side="below",
        start_time=-300_000,
        zone_entry_time=0,
        atr_value=atr,
        atr_status=ATRStatus.BLENDING,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
        direction_bias=DirectionBias.LONG,
    )

    vwap = VWAPSessionCalculator(exchange_timezone="UTC", rth_open_time="00:00:00", include_premarket_in_vwap=True)
    snap = sb.build(ts_ms=9_999, episode=ep, zone=zone, atr_value=atr, trades=trades, quotes=quotes, vwap=vwap)

    assert snap.buy_volume == 10.0
    assert snap.sell_volume == 5.0
    assert snap.unknown_volume == 0.0
    assert snap.delta == 5.0
    assert snap.trade_count == 2


