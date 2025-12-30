from __future__ import annotations

from pathlib import Path

import pytest

from godmode.core.models import Level
from godmode.zone.level_manager import LevelManager
from godmode.zone.zone_gate import ZoneEventType, ZoneGate


def test_level_manager_loads_yaml(tmp_path: Path) -> None:
    p = tmp_path / "AAPL.yaml"
    p.write_text(
        "\n".join(
            [
                "ticker: AAPL",
                "levels:",
                "  - level_id: L1",
                "    level_price: 100.0",
                "    level_type: support",
                "    level_source: manual",
                "    level_width_atr: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    lm = LevelManager()
    ls = lm.load_yaml(p)
    assert ls.ticker == "AAPL"
    assert len(ls.levels) == 1
    assert ls.levels[0].level_id == "L1"


def test_zone_gate_entry_exit_and_touch_count() -> None:
    lvl = Level(level_id="L1", level_price=100.0, level_type="support", level_source="manual", level_width_atr=0.25)
    zg = ZoneGate(ticker="AAPL")
    atr = 4.0  # 0.25 ATR => 1.0 point zone

    # Snapshot 0: outside (price far)
    snaps, ev = zg.update(ts_ms=0, last_price=98.0, atr_value=atr, levels=[lvl])
    assert snaps[0].in_zone is False
    assert ev == []

    # Snapshot 10s: enter (within 1.0 point of level)
    snaps, ev = zg.update(ts_ms=10_000, last_price=99.5, atr_value=atr, levels=[lvl])
    assert snaps[0].in_zone is True
    assert len(ev) == 1 and ev[0].type == ZoneEventType.ENTER
    assert ev[0].level_entry_side == "below"

    # Snapshot 20s: exit (outside) => exit event but touch not counted yet
    snaps, ev = zg.update(ts_ms=20_000, last_price=98.5, atr_value=atr, levels=[lvl])
    assert snaps[0].in_zone is False
    assert len(ev) == 1 and ev[0].type == ZoneEventType.EXIT
    assert snaps[0].touch_count == 0

    # Snapshot 30s: still outside => confirms exit for >=1 full snapshot, touch increments
    snaps, ev = zg.update(ts_ms=30_000, last_price=98.5, atr_value=atr, levels=[lvl])
    assert snaps[0].in_zone is False
    assert snaps[0].touch_count == 1


def test_zone_entry_side_above() -> None:
    lvl = Level(level_id="L1", level_price=100.0, level_type="resistance", level_source="manual", level_width_atr=0.25)
    zg = ZoneGate(ticker="AAPL")
    atr = 4.0

    # Enter from above (signed >= 0)
    _, ev = zg.update(ts_ms=0, last_price=100.5, atr_value=atr, levels=[lvl])
    assert len(ev) == 1
    assert ev[0].type == ZoneEventType.ENTER
    assert ev[0].level_entry_side == "above"


