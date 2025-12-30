from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.models import Level


@dataclass(slots=True)
class LevelBook:
    """
    Runtime level store for a ticker.

    Supports deterministic manual level add (Addendum G) without changing provider/engine parity.
    """

    ticker: str
    _levels: dict[str, Level]

    @staticmethod
    def from_levels(*, ticker: str, levels: list[Level]) -> "LevelBook":
        return LevelBook(ticker=ticker, _levels={lvl.level_id: lvl for lvl in levels})

    def list(self) -> list[Level]:
        # Deterministic ordering by level_id
        return [self._levels[k] for k in sorted(self._levels.keys())]

    def get(self, level_id: str) -> Optional[Level]:
        return self._levels.get(level_id)

    def add(self, level: Level) -> None:
        self._levels[level.level_id] = level


