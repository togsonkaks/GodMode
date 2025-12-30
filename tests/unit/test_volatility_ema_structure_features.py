from __future__ import annotations

import math

from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, EpisodePhase
from godmode.core.models import Episode, Snapshot
from godmode.features.ema_structure import EMAStructureFeatureEngine
from godmode.features.volatility import VolatilityFeatureEngine


def _ep() -> Episode:
    return Episode(
        episode_id="E1",
        session_id="S1",
        ticker="AAPL",
        level_id="L1",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side="below",
        start_time=0,
        zone_entry_time=0,
        atr_value=4.0,
        atr_status=ATRStatus.LIVE,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=True,
        direction_bias=DirectionBias.LONG,
        ema_confluence_ref=0.25,
    )


def _snap(ts: int, price: float) -> Snapshot:
    s = Snapshot(episode_id="E1", sequence_id=0, timestamp=ts, phase=EpisodePhase.STRESS)
    s.last_price = price
    return s


def test_realized_volatility_log_returns() -> None:
    eng = VolatilityFeatureEngine(rolling_window_seconds=60)
    h = [_snap(0, 100.0), _snap(10_000, 101.0), _snap(20_000, 102.0)]
    cur = _snap(30_000, 103.0)

    feats = eng.compute(current=cur, history=h)
    assert feats.realized_volatility_60s >= 0.0
    # non-zero volatility for changing returns
    assert feats.realized_volatility_60s > 0.0


def test_approach_return_60s_uses_price_60s_ago_or_before() -> None:
    eng = VolatilityFeatureEngine(rolling_window_seconds=60)
    # current at 70s, target at 10s -> should use snapshot at 10s
    h = [_snap(0, 100.0), _snap(10_000, 110.0)]
    cur = _snap(70_000, 121.0)
    feats = eng.compute(current=cur, history=h)
    assert math.isclose(feats.approach_return_60s, math.log(121.0 / 110.0), rel_tol=1e-12)


def test_ema_structure_slope_compression_confluence_and_bitmask() -> None:
    ep = _ep()
    eng = EMAStructureFeatureEngine()

    prev = _snap(0, 100.0)
    prev.ema9 = 90.0
    prev.ema20 = 80.0
    prev.ema30 = 70.0
    prev.ema200 = 50.0

    cur = _snap(60_000, 101.0)
    cur.ema9 = 96.0
    cur.ema20 = 88.0
    cur.ema30 = 79.0
    cur.ema200 = 60.0

    feats = eng.compute(episode=ep, current=cur, history=[prev], atr_14_1m=4.0)
    # per-second slopes over 60s
    assert math.isclose(feats.slope_ema9_60s, (96.0 - 90.0) / 60.0, rel_tol=1e-12)
    assert feats.stack_state == "bull"

    # bitmask: price=101 > all emas => 1+2+4+8 = 15
    assert feats.price_vs_emas == 15


