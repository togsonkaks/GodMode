from __future__ import annotations

from godmode.orchestrator.polygon_export import parse_dt_to_ts_ms


def test_parse_dt_to_ts_ms_with_tz_assumption() -> None:
    # No offset -> assume UTC here.
    ts = parse_dt_to_ts_ms("2025-01-01 00:00:00", tz="UTC")
    assert ts == 1735689600000


def test_parse_dt_to_ts_ms_with_offset() -> None:
    # Explicit offset should be respected, not replaced by tz param.
    ts = parse_dt_to_ts_ms("2025-01-01T00:00:00-05:00", tz="UTC")
    assert ts == 1735707600000


