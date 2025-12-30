from __future__ import annotations

from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, EpisodePhase
from godmode.core.models import Episode, Snapshot
from godmode.features.level_spine import LevelSpineFeatureEngine


def _ep(*, entry_side: str) -> Episode:
    return Episode(
        episode_id="E1",
        session_id="S1",
        ticker="AAPL",
        level_id="L1",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side=entry_side,
        start_time=0,
        zone_entry_time=0,
        atr_value=4.0,
        atr_status=ATRStatus.LIVE,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=True,
        direction_bias=DirectionBias.LONG,
    )


def _snap(ts: int, signed_atr: float) -> Snapshot:
    return Snapshot(
        episode_id="E1",
        sequence_id=0,
        timestamp=ts,
        phase=EpisodePhase.STRESS,
        signed_distance_to_level_atr=signed_atr,
        signed_distance_to_level=signed_atr * 4.0,
        abs_distance_to_level_atr=abs(signed_atr),
    )


def test_cross_count_and_oscillation_amplitude() -> None:
    eng = LevelSpineFeatureEngine(snapshot_interval_ms=10_000, rolling_window_seconds=60)
    ep = _ep(entry_side="below")

    # History over 60s: alternating signs => crossings
    h = [
        _snap(0, -0.1),
        _snap(10_000, 0.2),
        _snap(20_000, -0.2),
        _snap(30_000, 0.1),
    ]
    cur = _snap(40_000, -0.05)

    feats = eng.compute(episode=ep, current=cur, history=h, in_zone=True, touch_count=0)
    # Signs: - + - + - => 4 sign changes
    assert feats.cross_count_60s == 4
    assert feats.oscillation_amplitude_atr_60s == max([s.signed_distance_to_level_atr for s in h] + [cur.signed_distance_to_level_atr]) - min(
        [s.signed_distance_to_level_atr for s in h] + [cur.signed_distance_to_level_atr]
    )


def test_max_penetration_depends_on_entry_side() -> None:
    eng = LevelSpineFeatureEngine(snapshot_interval_ms=10_000, rolling_window_seconds=60)

    ep_above = _ep(entry_side="above")
    # penetration for entry_side=above uses max(0, -signed_atr)
    feats1 = eng.compute(episode=ep_above, current=_snap(0, 0.4), history=[], in_zone=False, touch_count=0)
    assert feats1.max_penetration_atr == 0.0
    feats2 = eng.compute(episode=ep_above, current=_snap(10_000, -0.3), history=[], in_zone=False, touch_count=0)
    assert feats2.max_penetration_atr == 0.3


def test_total_time_in_zone_and_avg_time_per_touch() -> None:
    eng = LevelSpineFeatureEngine(snapshot_interval_ms=10_000, rolling_window_seconds=60)
    ep = _ep(entry_side="below")
    cur = _snap(0, 0.1)  # in-zone since width=0.25

    feats = eng.compute(episode=ep, current=cur, history=[], in_zone=True, touch_count=0)
    assert feats.total_time_in_zone_episode == 10.0
    assert feats.avg_time_per_touch == 0.0

    # Next snapshot also in-zone; touch_count=1 => avg_time_per_touch should be total_time/1
    cur2 = _snap(10_000, 0.05)
    feats2 = eng.compute(episode=ep, current=cur2, history=[cur], in_zone=True, touch_count=1)
    assert feats2.total_time_in_zone_episode == 20.0
    assert feats2.avg_time_per_touch == 20.0


