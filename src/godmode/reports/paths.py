from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportKey:
    date: str
    ticker: str
    session_id: str


def report_path(*, root_dir: Path, key: ReportKey) -> Path:
    return (
        root_dir
        / "reports"
        / f"date={key.date}"
        / f"ticker={key.ticker}"
        / f"session={key.session_id}"
        / "report.json"
    )


