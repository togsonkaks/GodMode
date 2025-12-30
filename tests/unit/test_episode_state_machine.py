from __future__ import annotations

import pytest

from godmode.core.config import AppConfig
from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, ResolutionTrigger
from godmode.core.models import Level
from godmode.episode.state_machine import EpisodeStateMachine
from godmode.zone.zone_gate import ZoneGate


def test_episode_lifecycle_success_after_exit() -> None:
    cfg = AppConfig()
    lvl = Level(
        level_id="L1",
        level_price=100.0,
        level_type="support",
        level_source="manual",
        level_width_atr=0.25,
    )

    zg = ZoneGate(ticker="AAPL")
    sm = EpisodeStateMachine(ticker="AAPL", session_id="S1", config=cfg, baseline_window_seconds=300)
    atr = 4.0

    # Enter zone at t=10s
    _, events = zg.update(ts_ms=0, last_price=98.0, atr_value=atr, levels=[lvl])
    assert events == []

    _, events = zg.update(ts_ms=10_000, last_price=99.5, atr_value=atr, levels=[lvl])
    assert len(events) == 1 and events[0].type.value == "enter"

    ep = sm.on_zone_event(
        event=events[0],
        direction_bias=DirectionBias.LONG,
        atr_value=atr,
        atr_status=ATRStatus.BLENDING,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
    )
    assert ep is not None
    assert ep.zone_entry_time == 10_000
    assert ep.start_time == 10_000 - 300_000
    assert ep.level_type == "support"
    assert ep.level_source == "manual"

    # Exit at t=20s
    _, events = zg.update(ts_ms=20_000, last_price=98.5, atr_value=atr, levels=[lvl])
    assert len(events) == 1 and events[0].type.value == "exit"
    sm.on_zone_event(
        event=events[0],
        direction_bias=DirectionBias.LONG,
        atr_value=atr,
        atr_status=ATRStatus.BLENDING,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
    )

    active = sm.get_active("L1")
    assert active is not None
    assert active.zone_exit_time == 20_000

    # Resolve success: price >= level + 0.50 ATR => 100 + 2.0 = 102.0
    resolved = sm.update_resolution(level_id="L1", ts_ms=30_000, last_price=102.1, atr_value=atr)
    assert resolved is not None
    assert resolved.resolution_trigger == ResolutionTrigger.THRESHOLD_HIT
    assert resolved.resolution_time == 30_000
    assert sm.get_active("L1") is None


def test_episode_timeout_after_exit() -> None:
    cfg = AppConfig()
    cfg.resolution.timeout_seconds = 5  # shorten for test determinism
    lvl = Level(level_id="L1", level_price=100.0, level_type="support", level_source="manual", level_width_atr=0.25)
    zg = ZoneGate(ticker="AAPL")
    sm = EpisodeStateMachine(ticker="AAPL", session_id="S1", config=cfg, baseline_window_seconds=300)
    atr = 4.0

    _, events = zg.update(ts_ms=0, last_price=99.5, atr_value=atr, levels=[lvl])
    ep = sm.on_zone_event(
        event=events[0],
        direction_bias=DirectionBias.LONG,
        atr_value=atr,
        atr_status=ATRStatus.SEEDED,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
    )
    assert ep is not None

    _, events = zg.update(ts_ms=10_000, last_price=98.0, atr_value=atr, levels=[lvl])
    sm.on_zone_event(
        event=events[0],
        direction_bias=DirectionBias.LONG,
        atr_value=atr,
        atr_status=ATRStatus.SEEDED,
        atr_seed_source=ATRSeedSource.PRIOR_SESSION,
        atr_is_warm=False,
    )

    # At t=16s (>= exit 10s + timeout 5s) => timeout
    resolved = sm.update_resolution(level_id="L1", ts_ms=16_000, last_price=100.0, atr_value=atr)
    assert resolved is not None
    assert resolved.resolution_trigger == ResolutionTrigger.TIMEOUT


