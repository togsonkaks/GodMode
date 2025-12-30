from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from godmode.core.enums import DirectionBias
from godmode.core.markers import Marker, MarkerType


CommandType = Literal["add_level", "add_marker"]


@dataclass(frozen=True, slots=True)
class AddLevelCmd:
    ts_ms: int
    ticker: str
    level_price: float
    level_type: str
    level_width_atr: float = 0.25
    level_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True, slots=True)
class AddMarkerCmd:
    ts_ms: int
    ticker: str
    marker_type: MarkerType
    marker_id: Optional[str] = None
    direction_bias: Optional[DirectionBias] = None
    notes: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CommandEvent:
    ts_ms: int
    type: CommandType
    seq: int
    obj: object


def load_commands_csv(path: Path) -> list[CommandEvent]:
    """
    Load deterministic event-time commands from CSV.

    Columns:
    - ts_ms (int)
    - ticker (str)
    - type: add_level|add_marker
    For add_level:
      - level_price, level_type, level_width_atr(optional), level_id(optional), notes(optional)
    For add_marker:
      - marker_type, marker_id(optional), direction_bias(optional), notes(optional)
    """
    df = pd.read_csv(path)
    if df.empty:
        return []
    required = {"ts_ms", "ticker", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"commands csv missing columns: {sorted(missing)}")

    evs: list[CommandEvent] = []
    seq = 0
    for row in df.itertuples(index=False):
        tp = str(getattr(row, "type"))
        ts_ms = int(getattr(row, "ts_ms"))
        ticker = str(getattr(row, "ticker"))
        if tp == "add_level":
            evs.append(
                CommandEvent(
                    ts_ms=ts_ms,
                    type="add_level",
                    seq=seq,
                    obj=AddLevelCmd(
                        ts_ms=ts_ms,
                        ticker=ticker,
                        level_price=float(getattr(row, "level_price")),
                        level_type=str(getattr(row, "level_type")),
                        level_width_atr=float(getattr(row, "level_width_atr", 0.25)),
                        level_id=(None if pd.isna(getattr(row, "level_id", None)) else str(getattr(row, "level_id"))),
                        notes=(None if pd.isna(getattr(row, "notes", None)) else str(getattr(row, "notes"))),
                    ),
                )
            )
        elif tp == "add_marker":
            mb = getattr(row, "direction_bias", None)
            db = None
            if mb is not None and not pd.isna(mb):
                db = DirectionBias(str(mb))
            evs.append(
                CommandEvent(
                    ts_ms=ts_ms,
                    type="add_marker",
                    seq=seq,
                    obj=AddMarkerCmd(
                        ts_ms=ts_ms,
                        ticker=ticker,
                        marker_type=MarkerType(str(getattr(row, "marker_type"))),
                        marker_id=(None if pd.isna(getattr(row, "marker_id", None)) else str(getattr(row, "marker_id"))),
                        direction_bias=db,
                        notes=(None if pd.isna(getattr(row, "notes", None)) else str(getattr(row, "notes"))),
                    ),
                )
            )
        else:
            raise ValueError(f"unknown command type: {tp}")
        seq += 1

    # Deterministic stable order by (ts_ms, seq)
    evs.sort(key=lambda e: (e.ts_ms, e.seq))
    return evs


