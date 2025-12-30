from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from godmode.webapp.store import Store
from godmode.reports.paths import ReportKey, report_path


def _atomic_write_json(*, obj: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    import json

    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out_path)


def _find_session_dates(*, root_dir: Path, ticker: str, session_id: str) -> list[str]:
    """
    Sessions are partitioned by date in parquet output. We find all matching date partitions
    for (ticker, session_id) by scanning episodes/, session_stream/, or markers/.
    """
    dates: set[str] = set()
    
    # Check episodes (original behavior)
    episodes_root = root_dir / "episodes"
    if episodes_root.exists():
        for date_dir in sorted(episodes_root.glob("date=*")):
            date = date_dir.name.split("=", 1)[-1]
            sess_dir = date_dir / f"ticker={ticker}" / f"session={session_id}"
            if sess_dir.exists():
                dates.add(date)
    
    # Also check session_stream (for sessions without episodes)
    stream_root = root_dir / "session_stream"
    if stream_root.exists():
        for date_dir in sorted(stream_root.glob("date=*")):
            date = date_dir.name.split("=", 1)[-1]
            sess_dir = date_dir / f"ticker={ticker}" / f"session={session_id}"
            if sess_dir.exists():
                dates.add(date)
    
    # Also check markers (for sessions that only have markers)
    markers_root = root_dir / "markers"
    if markers_root.exists():
        for date_dir in sorted(markers_root.glob("date=*")):
            date = date_dir.name.split("=", 1)[-1]
            sess_dir = date_dir / f"ticker={ticker}" / f"session={session_id}"
            if sess_dir.exists():
                dates.add(date)
    
    return sorted(dates)


def write_session_report(*, root_dir: Path, ticker: str, session_id: str) -> list[Path]:
    """
    Compute and persist a stable report.json for this session.
    Returns list of written report paths (one per date partition, usually one).
    """
    store = Store(root_dir=root_dir)
    dates = _find_session_dates(root_dir=root_dir, ticker=ticker, session_id=session_id)
    out: list[Path] = []
    for date in dates:
        rep = store.compute_session_report(date=date, ticker=ticker, session_id=session_id)
        # include provenance pointers so you always know where the raw truth is
        rep["_meta"] = {
            "schema_version": "report_v1",
            "root_dir": str(root_dir).replace("\\", "/"),
            "date": date,
            "ticker": ticker,
            "session_id": session_id,
            "raw_paths": {
                "episodes": f"episodes/date={date}/ticker={ticker}/session={session_id}/",
                "snapshots": f"snapshots/date={date}/ticker={ticker}/session={session_id}/",
                "markers": f"markers/date={date}/ticker={ticker}/session={session_id}/",
                "tf_indicators": f"tf_indicators/date={date}/ticker={ticker}/session={session_id}/",
                "session_stream": f"session_stream/date={date}/ticker={ticker}/session={session_id}/",
            },
        }
        p = report_path(root_dir=root_dir, key=ReportKey(date=date, ticker=ticker, session_id=session_id))
        _atomic_write_json(obj=rep, out_path=p)
        out.append(p)
    return out


