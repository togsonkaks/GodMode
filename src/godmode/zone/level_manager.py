from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from godmode.core.models import Level


@dataclass(frozen=True, slots=True)
class LevelSet:
    ticker: str
    levels: tuple[Level, ...]


class LevelManager:
    """
    Loads user-defined levels from YAML.

    File format (deterministic, minimal):

    ```yaml
    ticker: AAPL
    levels:
      - level_id: AAPL_2025-01-01_R1
        level_price: 197.25
        level_type: resistance
        level_source: manual
        level_width_atr: 0.25
    ```
    """

    def __init__(self) -> None:
        self._cache: dict[str, LevelSet] = {}

    def load_yaml(self, path: str | Path) -> LevelSet:
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        ticker = str(data.get("ticker", "")).strip()
        if not ticker:
            raise ValueError("levels yaml missing 'ticker'")

        # Levels may be empty (e.g., marker-driven workflows + runtime add_level commands).
        raw_levels = data.get("levels")
        if raw_levels is None:
            raw_levels = []
        if not isinstance(raw_levels, list):
            raise ValueError("levels yaml 'levels' must be a list")

        levels: list[Level] = []
        for i, item in enumerate(raw_levels):
            if not isinstance(item, dict):
                raise ValueError(f"levels[{i}] must be a mapping")

            for req in ["level_id", "level_price", "level_type", "level_source"]:
                if req not in item:
                    raise ValueError(f"levels[{i}] missing '{req}'")

            level_width_atr = float(item.get("level_width_atr", 0.25))
            levels.append(
                Level(
                    level_id=str(item["level_id"]),
                    level_price=float(item["level_price"]),
                    level_type=str(item["level_type"]),
                    level_source=str(item["level_source"]),
                    level_width_atr=level_width_atr,
                    created_ts_ms=int(item.get("created_ts_ms", 0) or 0),
                    notes=item.get("notes"),
                )
            )

        out = LevelSet(ticker=ticker, levels=tuple(levels))
        self._cache[ticker] = out
        return out

    def get(self, ticker: str) -> LevelSet | None:
        return self._cache.get(ticker)


