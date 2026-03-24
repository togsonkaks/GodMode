from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScannerState:
    day: str
    tickers: dict[str, Any]
    alerts: list[dict[str, Any]]

    @staticmethod
    def default(*, day: str) -> "ScannerState":
        return ScannerState(day=day, tickers={}, alerts=[])


def load_state(path: Path, *, day: str) -> ScannerState:
    if not path.exists():
        return ScannerState.default(day=day)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("day") != day:
            return ScannerState.default(day=day)
        return ScannerState(day=day, tickers=raw.get("tickers", {}) or {}, alerts=raw.get("alerts", []) or [])
    except Exception:
        return ScannerState.default(day=day)


def save_state(path: Path, state: ScannerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"day": state.day, "tickers": state.tickers, "alerts": state.alerts}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

