from __future__ import annotations

import math

from godmode.core.config import AppConfig
from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, EpisodePhase
from godmode.core.models import Episode, Snapshot
from godmode.features.smart_money import SmartMoneyFeatureEngine


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
    )


def _snap(ts: int, phase: EpisodePhase, *, price: float, delta: float, vol: float, tc: int, atr: float = 4.0) -> Snapshot:
    s = Snapshot(episode_id="E1", sequence_id=0, timestamp=ts, phase=phase)
    s.last_price = price
    s.delta = delta
    s.total_volume = vol
    s.trade_count = tc
    s.buy_volume = max(0.0, delta)  # simple proxy for test
    s.sell_volume = max(0.0, -delta)
    s.atr_value = atr
    return s


def test_cvd_windows_and_slopes() -> None:
    cfg = AppConfig()
    eng = SmartMoneyFeatureEngine(config=cfg, snapshot_interval_seconds=10)
    ep = _ep()

    h = [
        _snap(0, EpisodePhase.BASELINE, price=100, delta=1, vol=10, tc=1),
        _snap(10_000, EpisodePhase.BASELINE, price=99, delta=2, vol=10, tc=1),
    ]
    cur = _snap(20_000, EpisodePhase.STRESS, price=98, delta=3, vol=10, tc=1)

    feats = eng.compute(episode=ep, current=cur, history=h)
    assert feats.cvd_10s == 3
    assert feats.cvd_30s == 1 + 2 + 3
    # Not enough for 60s -> sum of available
    assert feats.cvd_60s == 1 + 2 + 3


def test_buy_on_red_and_sell_on_green_are_capped() -> None:
    cfg = AppConfig()
    cfg.smart_money.return_floor_atr = 0.02
    eng = SmartMoneyFeatureEngine(config=cfg, snapshot_interval_seconds=10)
    ep = _ep()

    # ret_10 is tiny -> denominator should use floor_atr*atr = 0.08
    h = [_snap(0, EpisodePhase.BASELINE, price=100.00, delta=-10, vol=100, tc=10)]
    cur = _snap(10_000, EpisodePhase.STRESS, price=99.99, delta=10, vol=100, tc=10)
    cur.buy_volume = 50
    cur.sell_volume = 0

    feats = eng.compute(episode=ep, current=cur, history=h)
    assert feats.return_10s < 0  # price down
    assert feats.buy_on_red > 0
    # Expected approx 50 / 0.08
    assert math.isclose(feats.buy_on_red, 50 / 0.08, rel_tol=1e-3)


def test_divergence_flag_and_score() -> None:
    cfg = AppConfig()
    eng = SmartMoneyFeatureEngine(config=cfg, snapshot_interval_seconds=10)
    ep = _ep()

    # Baseline: stable vol so baseline_vol_mean > 0
    h = [
        _snap(0, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
        _snap(10_000, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
        _snap(20_000, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
        _snap(30_000, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
        _snap(40_000, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
        _snap(50_000, EpisodePhase.BASELINE, price=100, delta=-5, vol=100, tc=10),
    ]
    # Stress: price up but delta still negative -> divergence
    cur = _snap(60_000, EpisodePhase.STRESS, price=101, delta=-5, vol=100, tc=10)

    feats = eng.compute(episode=ep, current=cur, history=h)
    assert feats.return_norm > 0
    assert feats.delta_norm < 0
    assert feats.divergence_flag == 1
    assert feats.div_score > 0


