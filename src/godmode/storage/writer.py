from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from godmode.core.config import AppConfig
from godmode.core.enums import EpisodePhase
from godmode.core.markers import Marker
from godmode.core.models import Episode, Snapshot
from godmode.storage.partitioning import PartitionKey, date_str_from_ts_ms, partition_dir_for


Kind = Literal["snapshots", "episodes", "labels", "tf_indicators", "markers", "session_stream"]


def _to_primitive(x: Any) -> Any:
    # Enums -> value; dataclasses -> dict; leave primitives.
    if hasattr(x, "value") and not isinstance(x, (str, bytes, bytearray)):
        try:
            return x.value
        except Exception:
            pass
    if is_dataclass(x):
        return {k: _to_primitive(v) for k, v in asdict(x).items()}
    return x


def _snapshot_record(s: Snapshot) -> dict[str, Any]:
    d = _to_primitive(s)
    assert isinstance(d, dict)

    # Schema aliasing: official storage names are `%_at_ask` / `%_at_bid`
    d["%_at_ask"] = d.pop("pct_at_ask", 0.0)
    d["%_at_bid"] = d.pop("pct_at_bid", 0.0)

    # Ensure enums are stored as strings
    if isinstance(s.phase, EpisodePhase):
        d["phase"] = s.phase.value

    return d


def _episode_record(e: Episode) -> dict[str, Any]:
    d = _to_primitive(e)
    assert isinstance(d, dict)
    return d


def _atomic_write_parquet(*, table: pa.Table, out_path: Path, compression: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    pq.write_table(table, tmp_path, compression=compression)
    os.replace(tmp_path, out_path)  # atomic on same filesystem


class ParquetStorageWriter:
    """
    Atomic parquet writer with deterministic partitioning:

    {kind}/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet

    - Writes temp then rename: part-000.parquet.tmp -> part-000.parquet
    - Uses compression from config (zstd preferred).
    """

    def __init__(self, *, config: AppConfig) -> None:
        self._cfg = config
        self._root = Path(config.storage.root_dir)
        self._compression = config.storage.compression
        self._tz = config.session.exchange_timezone

        # part counters per partition key
        self._parts: dict[PartitionKey, int] = {}

    def _next_part_path(self, *, key: PartitionKey) -> Path:
        n = self._parts.get(key, 0)
        self._parts[key] = n + 1
        d = partition_dir_for(root_dir=self._root, key=key)
        return d / f"part-{n:03d}.parquet"

    def write_snapshots(self, snapshots: list[Snapshot], *, session_id: str, ticker: str) -> list[Path]:
        if not snapshots:
            return []

        # Group by date partition derived from snapshot.timestamp (event-time).
        by_date: dict[str, list[Snapshot]] = {}
        for s in snapshots:
            ds = date_str_from_ts_ms(ts_ms=int(s.timestamp), exchange_timezone=self._tz)
            by_date.setdefault(ds, []).append(s)

        out_paths: list[Path] = []
        for ds, group in by_date.items():
            key = PartitionKey(kind="snapshots", date=ds, ticker=ticker, session_id=session_id)
            records = [_snapshot_record(s) for s in group]
            table = pa.Table.from_pylist(records)
            out_path = self._next_part_path(key=key)
            _atomic_write_parquet(table=table, out_path=out_path, compression=self._compression)
            out_paths.append(out_path)

        return out_paths

    def write_episodes(self, episodes: list[Episode], *, session_id: str, ticker: str) -> list[Path]:
        if not episodes:
            return []

        by_date: dict[str, list[Episode]] = {}
        for e in episodes:
            # Partition by zone_entry_time (deterministic, most meaningful)
            ds = date_str_from_ts_ms(ts_ms=int(e.zone_entry_time), exchange_timezone=self._tz)
            by_date.setdefault(ds, []).append(e)

        out_paths: list[Path] = []
        for ds, group in by_date.items():
            key = PartitionKey(kind="episodes", date=ds, ticker=ticker, session_id=session_id)
            records = [_episode_record(e) for e in group]
            table = pa.Table.from_pylist(records)
            out_path = self._next_part_path(key=key)
            _atomic_write_parquet(table=table, out_path=out_path, compression=self._compression)
            out_paths.append(out_path)

        return out_paths

    def write_session_stream(self, snapshots: list[Snapshot], *, session_id: str, ticker: str) -> list[Path]:
        """
        Continuous per-ticker stream (Addendum H).
        Uses the same Snapshot schema, but stored under session_stream/.
        """
        if not snapshots:
            return []
        by_date: dict[str, list[Snapshot]] = {}
        for s in snapshots:
            ds = date_str_from_ts_ms(ts_ms=int(s.timestamp), exchange_timezone=self._tz)
            by_date.setdefault(ds, []).append(s)

        out_paths: list[Path] = []
        for ds, group in by_date.items():
            key = PartitionKey(kind="session_stream", date=ds, ticker=ticker, session_id=session_id)
            records = [_snapshot_record(s) for s in group]
            table = pa.Table.from_pylist(records)
            out_path = self._next_part_path(key=key)
            _atomic_write_parquet(table=table, out_path=out_path, compression=self._compression)
            out_paths.append(out_path)
        return out_paths

    def write_markers(self, markers: list[Marker], *, session_id: str, ticker: str) -> list[Path]:
        if not markers:
            return []
        by_date: dict[str, list[Marker]] = {}
        for m in markers:
            ds = date_str_from_ts_ms(ts_ms=int(m.ts_ms), exchange_timezone=self._tz)
            by_date.setdefault(ds, []).append(m)

        out_paths: list[Path] = []
        for ds, group in by_date.items():
            key = PartitionKey(kind="markers", date=ds, ticker=ticker, session_id=session_id)
            records = [_to_primitive(x) for x in group]
            table = pa.Table.from_pylist(records)  # type: ignore[arg-type]
            out_path = self._next_part_path(key=key)
            _atomic_write_parquet(table=table, out_path=out_path, compression=self._compression)
            out_paths.append(out_path)
        return out_paths

    def write_tf_indicators(self, rows: list[dict[str, Any]], *, session_id: str, ticker: str) -> list[Path]:
        """
        Multi-timeframe indicator rows (Addendum F) stored under tf_indicators/.
        Each row must already be a primitive dict.
        """
        if not rows:
            return []
        by_date: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            ds = date_str_from_ts_ms(ts_ms=int(r["timestamp"]), exchange_timezone=self._tz)
            by_date.setdefault(ds, []).append(r)

        out_paths: list[Path] = []
        for ds, group in by_date.items():
            key = PartitionKey(kind="tf_indicators", date=ds, ticker=ticker, session_id=session_id)
            table = pa.Table.from_pylist(group)
            out_path = self._next_part_path(key=key)
            _atomic_write_parquet(table=table, out_path=out_path, compression=self._compression)
            out_paths.append(out_path)
        return out_paths


