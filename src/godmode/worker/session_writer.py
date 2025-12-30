from __future__ import annotations

from dataclasses import dataclass, field

from godmode.core.config import AppConfig
from godmode.core.models import Episode, Snapshot
from godmode.storage.writer import ParquetStorageWriter
from godmode.core.markers import Marker


@dataclass(slots=True)
class SessionWriter:
    """
    Minimal integration point for Day 13.

    Higher-level workers/orchestrators will call these methods as episodes/snapshots finalize.
    """

    config: AppConfig
    session_id: str
    ticker: str
    _writer: ParquetStorageWriter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._writer = ParquetStorageWriter(config=self.config)

    def write_snapshots(self, snapshots: list[Snapshot]) -> None:
        self._writer.write_snapshots(snapshots, session_id=self.session_id, ticker=self.ticker)

    def write_episodes(self, episodes: list[Episode]) -> None:
        self._writer.write_episodes(episodes, session_id=self.session_id, ticker=self.ticker)

    def write_tf_indicators(self, rows: list[dict]) -> None:
        self._writer.write_tf_indicators(rows, session_id=self.session_id, ticker=self.ticker)

    def write_session_stream(self, snapshots: list[Snapshot]) -> None:
        self._writer.write_session_stream(snapshots, session_id=self.session_id, ticker=self.ticker)

    def write_markers(self, markers: list[Marker]) -> None:
        self._writer.write_markers(markers, session_id=self.session_id, ticker=self.ticker)


