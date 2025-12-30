from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from godmode.core.config import AppConfig
from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, EpisodePhase
from godmode.core.models import Episode, Snapshot
from godmode.storage.partitioning import PartitionKey, date_str_from_ts_ms, partition_dir_for
from godmode.storage.writer import ParquetStorageWriter


def test_partition_dir_format(tmp_path: Path) -> None:
    key = PartitionKey(kind="snapshots", date="2025-01-01", ticker="AAPL", session_id="S1")
    p = partition_dir_for(root_dir=tmp_path, key=key)
    assert str(p).endswith(r"snapshots\date=2025-01-01\ticker=AAPL\session=S1")


def test_date_str_from_ts_ms_timezone() -> None:
    # 2025-01-01 00:00:00 UTC should be 2024-12-31 in America/New_York (EST)
    ds = date_str_from_ts_ms(ts_ms=1735689600000, exchange_timezone="America/New_York")
    assert ds == "2024-12-31"


def test_atomic_parquet_write_and_schema_aliasing(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.storage.root_dir = str(tmp_path)
    cfg.session.exchange_timezone = "UTC"

    w = ParquetStorageWriter(config=cfg)

    snap = Snapshot(
        episode_id="E1",
        sequence_id=1,
        timestamp=1_000,
        phase=EpisodePhase.STRESS,
    )
    snap.pct_at_ask = 0.6
    snap.pct_at_bid = 0.4
    snap.bid = 100.0
    snap.ask = 101.0
    snap.mid_price = 100.5
    snap.spread_pct = 0.01

    out_paths = w.write_snapshots([snap], session_id="S1", ticker="AAPL")
    assert len(out_paths) == 1
    out_path = out_paths[0]
    assert out_path.exists()
    # Ensure tmp file is not left behind
    assert not out_path.with_suffix(out_path.suffix + ".tmp").exists()

    table = pq.read_table(out_path)
    cols = set(table.column_names)
    assert "%_at_ask" in cols
    assert "%_at_bid" in cols
    assert "pct_at_ask" not in cols
    assert "pct_at_bid" not in cols


