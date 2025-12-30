from __future__ import annotations

import pytest

from godmode.core.enums import DirectionBias, EpisodePhase, Outcome, ResolutionTrigger
from godmode.core.models import Episode, Snapshot
from godmode.episode.metrics import apply_outcome_and_metrics


def _snap(ep_id: str, seq: int, ts: int, price: float) -> Snapshot:
    return Snapshot(
        episode_id=ep_id,
        sequence_id=seq,
        timestamp=ts,
        phase=EpisodePhase.STRESS,
        atr_value=1.0,
        last_price=price,
    )


def test_apply_outcome_and_metrics_long() -> None:
    ep = Episode(
        episode_id="ep1",
        session_id="s",
        ticker="T",
        level_id="L",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side="below",
        start_time=0,
        zone_entry_time=1_000,
        zone_exit_time=2_000,
        resolution_time=4_000,
        end_time=4_000,
        resolution_trigger=ResolutionTrigger.THRESHOLD_HIT,
        direction_bias=DirectionBias.LONG,
        success_threshold_atr=0.50,
        failure_threshold_atr=0.35,
        timeout_seconds=300,
        atr_value=2.0,  # 1 ATR = $2
    )

    snaps = [
        _snap("ep1", 0, 900, 100.0),   # before zone_entry_time (ignored)
        _snap("ep1", 1, 1_000, 100.0), # mv=0
        _snap("ep1", 2, 2_000, 99.0),  # mv=-0.5 ATR => adverse=0.5 (failure threshold hit here)
        _snap("ep1", 3, 3_000, 101.0), # mv=+0.5 ATR
        _snap("ep1", 4, 4_000, 102.0), # mv=+1.0 ATR (MFE)
        _snap("ep1", 5, 5_000, 103.0), # after end_time (ignored)
    ]

    apply_outcome_and_metrics(ep=ep, snapshots=snaps)

    assert ep.outcome == Outcome.WIN
    assert ep.mfe == 1.0
    assert ep.mae == 0.5
    assert ep.time_to_mfe_ms == 3_000  # 4000 - 1000
    assert ep.time_to_failure_ms == 1_000  # 2000 - 1000


def test_apply_outcome_and_metrics_short() -> None:
    ep = Episode(
        episode_id="ep2",
        session_id="s",
        ticker="T",
        level_id="L",
        level_price=50.0,
        level_type="resistance",
        level_source="manual",
        level_width_atr=0.25,
        level_entry_side="above",
        start_time=0,
        zone_entry_time=1_000,
        zone_exit_time=2_000,
        resolution_time=4_000,
        end_time=4_000,
        resolution_trigger=ResolutionTrigger.INVALIDATION,
        direction_bias=DirectionBias.SHORT,
        success_threshold_atr=0.50,
        failure_threshold_atr=0.35,
        timeout_seconds=300,
        atr_value=1.0,
    )

    snaps = [
        _snap("ep2", 1, 1_000, 50.0),  # mv=0
        _snap("ep2", 2, 2_000, 49.5),  # mv=+0.5 ATR favorable for short
        _snap("ep2", 3, 3_000, 50.4),  # mv=-0.4 ATR => adverse=0.4 hits failure threshold
        _snap("ep2", 4, 4_000, 50.2),
    ]

    apply_outcome_and_metrics(ep=ep, snapshots=snaps)

    assert ep.outcome == Outcome.LOSS
    assert ep.mfe == 0.5
    assert ep.mae == pytest.approx(0.4)
    assert ep.time_to_mfe_ms == 1_000
    assert ep.time_to_failure_ms == 2_000


