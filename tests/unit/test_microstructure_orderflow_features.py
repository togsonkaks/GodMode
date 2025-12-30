from __future__ import annotations

import math

from godmode.core.enums import EpisodePhase
from godmode.core.models import Snapshot, Trade
from godmode.features.microstructure import MicrostructureFeatureEngine
from godmode.features.orderflow import OrderFlowFeatureEngine


def _snap(ts: int) -> Snapshot:
    return Snapshot(
        episode_id="E1",
        sequence_id=0,
        timestamp=ts,
        phase=EpisodePhase.STRESS,
    )


def test_spread_volatility_skips_missing_quotes() -> None:
    eng = MicrostructureFeatureEngine(rolling_window_seconds=60)
    h = []

    # Missing quotes -> bid/ask 0 => excluded from samples
    s1 = _snap(0)
    s1.bid = 0.0
    s1.ask = 0.0
    s1.spread_pct = 0.0
    h.append(s1)

    s2 = _snap(10_000)
    s2.bid = 100.0
    s2.ask = 102.0
    s2.spread_pct = (2.0 / 101.0)
    h.append(s2)

    cur = _snap(20_000)
    cur.bid = 100.0
    cur.ask = 101.0
    cur.spread_pct = (1.0 / 100.5)

    feats = eng.compute(current=cur, history=h)
    assert feats.spread_volatility_60s > 0.0  # computed from s2 + cur


def test_relative_aggression_zscore_is_zero_when_std_zero() -> None:
    eng = OrderFlowFeatureEngine(rolling_window_seconds=60)
    h = []
    for ts in [0, 10_000, 20_000]:
        s = _snap(ts)
        s.relative_aggression = 0.5
        h.append(s)
    cur = _snap(30_000)
    cur.relative_aggression = 0.5

    feats = eng.compute(episode_id="E1", current=cur, history=h, window_trades=[])
    assert feats.relative_aggression_zscore_60s == 0.0


def test_trade_size_stats_and_top_decile_share() -> None:
    eng = OrderFlowFeatureEngine()
    cur = _snap(0)
    cur.delta = 0.0
    cur.last_price = 100.0
    cur.relative_aggression = 0.0

    trades = [
        Trade(ts_ms=0, symbol="AAPL", price=100.0, size=1.0),
        Trade(ts_ms=1, symbol="AAPL", price=100.0, size=1.0),
        Trade(ts_ms=2, symbol="AAPL", price=100.0, size=8.0),
        Trade(ts_ms=3, symbol="AAPL", price=100.0, size=10.0),
    ]

    feats = eng.compute(episode_id="E1", current=cur, history=[], window_trades=trades)
    assert math.isclose(feats.avg_trade_size, (1 + 1 + 8 + 10) / 4, rel_tol=1e-12)
    assert feats.trade_size_std > 0.0
    # top decile => ceil(10% of 4) = 1 -> top is size 10, share = 10 / 20
    assert math.isclose(feats.top_decile_volume_share, 10.0 / 20.0, rel_tol=1e-12)


def test_absorption_index_uses_delta_and_price_return() -> None:
    eng = OrderFlowFeatureEngine()

    # First snapshot has no prev price -> price_return_10s = 0 => absorption uses epsilon
    s1 = _snap(0)
    s1.delta = -10.0
    s1.last_price = 100.0
    s1.relative_aggression = 0.0
    f1 = eng.compute(episode_id="E1", current=s1, history=[], window_trades=[])
    assert f1.absorption_index_10s > 0.0

    # Next snapshot: price moved by 2, delta -10 => absorption = 10 / (2 + eps)
    s2 = _snap(10_000)
    s2.delta = -10.0
    s2.last_price = 102.0
    s2.relative_aggression = 0.0
    f2 = eng.compute(episode_id="E1", current=s2, history=[s1], window_trades=[])
    assert math.isclose(f2.absorption_index_10s, 10.0 / 2.0, rel_tol=1e-6)


