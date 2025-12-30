from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class PartitionKey:
    kind: str  # snapshots | episodes | labels
    date: str  # YYYY-MM-DD
    ticker: str
    session_id: str


def date_str_from_ts_ms(*, ts_ms: int, exchange_timezone: str) -> str:
    tz = ZoneInfo(exchange_timezone)
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=tz)
    return dt.date().isoformat()


def partition_dir_for(*, root_dir: str | Path, key: PartitionKey) -> Path:
    root = Path(root_dir)
    return (
        root
        / key.kind
        / f"date={key.date}"
        / f"ticker={key.ticker}"
        / f"session={key.session_id}"
    )


