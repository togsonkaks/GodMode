from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from godmode.core.models import Level


class ZoneEventType(str, Enum):
    ENTER = "enter"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    type: ZoneEventType
    ts_ms: int
    ticker: str
    level_id: str
    level_price: float
    level_type: str
    level_source: str
    level_width_atr: float
    level_entry_side: Optional[str] = None  # above|below set on ENTER


@dataclass(frozen=True, slots=True)
class ZoneSnapshot:
    """
    Per-level zone state at a snapshot timestamp.
    """

    ts_ms: int
    ticker: str
    level_id: str

    in_zone: bool
    signed_distance_to_level: float
    signed_distance_to_level_atr: float
    abs_distance_to_level_atr: float

    zone_entry_time: Optional[int]
    zone_exit_time: Optional[int]
    touch_count: int


@dataclass(slots=True)
class _PerLevelState:
    in_zone: bool = False
    zone_entry_time: Optional[int] = None
    zone_exit_time: Optional[int] = None

    # Touch-count bookkeeping (Addendum C)
    saw_entry: bool = False
    pending_exit_confirm: bool = False
    touch_count: int = 0


class ZoneGate:
    """
    Zone gating around levels using ATR-normalized distance.

    Default zone rule (SPEC): abs(signed_distance_to_level_atr) <= level_width_atr (default 0.25).

    Touch count (Addendum C):
    - touch increments when outside->inside (entry) AND
      inside->outside for >= 1 full snapshot (exit).

    Implementation at snapshot cadence:
    - On inside->outside transition at snapshot t: mark pending exit confirmation.
    - If next snapshot is still outside, confirm exit and increment touch_count.
    - If next snapshot is back inside, exit was not >= 1 full snapshot -> do not count.
    """

    def __init__(self, *, ticker: str) -> None:
        self._ticker = ticker
        self._state: dict[str, _PerLevelState] = {}
        self._last_ts_ms: Optional[int] = None

    @property
    def ticker(self) -> str:
        return self._ticker

    def update(
        self,
        *,
        ts_ms: int,
        last_price: float,
        atr_value: float,
        levels: list[Level],
    ) -> tuple[list[ZoneSnapshot], list[ZoneEvent]]:
        ts_ms = int(ts_ms)
        if self._last_ts_ms is not None and ts_ms < self._last_ts_ms:
            raise ValueError(f"out-of-order snapshot ts_ms={ts_ms} < last_ts_ms={self._last_ts_ms}")
        self._last_ts_ms = ts_ms

        if atr_value <= 0:
            raise ValueError("atr_value must be > 0")

        snapshots: list[ZoneSnapshot] = []
        events: list[ZoneEvent] = []

        for lvl in levels:
            st = self._state.setdefault(lvl.level_id, _PerLevelState())

            signed = float(last_price) - float(lvl.level_price)
            signed_atr = signed / float(atr_value)
            abs_atr = abs(signed_atr)
            width = float(lvl.level_width_atr)
            in_zone = abs_atr <= width

            # Confirm a pending exit if we stayed outside for >=1 full snapshot.
            if st.pending_exit_confirm and (not in_zone):
                st.pending_exit_confirm = False
                st.touch_count += 1

            # Transition logic
            if (not st.in_zone) and in_zone:
                # Outside -> inside
                st.in_zone = True
                st.zone_entry_time = ts_ms
                st.zone_exit_time = None
                st.saw_entry = True
                st.pending_exit_confirm = False

                entry_side = "above" if signed >= 0 else "below"
                events.append(
                    ZoneEvent(
                        type=ZoneEventType.ENTER,
                        ts_ms=ts_ms,
                        ticker=self._ticker,
                        level_id=lvl.level_id,
                        level_price=lvl.level_price,
                        level_type=lvl.level_type,
                        level_source=lvl.level_source,
                        level_width_atr=lvl.level_width_atr,
                        level_entry_side=entry_side,
                    )
                )

            elif st.in_zone and (not in_zone):
                # Inside -> outside (start exit confirmation window)
                st.in_zone = False
                st.zone_exit_time = ts_ms
                if st.saw_entry:
                    st.pending_exit_confirm = True

                events.append(
                    ZoneEvent(
                        type=ZoneEventType.EXIT,
                        ts_ms=ts_ms,
                        ticker=self._ticker,
                        level_id=lvl.level_id,
                        level_price=lvl.level_price,
                        level_type=lvl.level_type,
                        level_source=lvl.level_source,
                        level_width_atr=lvl.level_width_atr,
                        level_entry_side=None,
                    )
                )

            # If we are back inside while pending exit confirm, cancel (not out for >=1 snapshot).
            if st.pending_exit_confirm and in_zone:
                st.pending_exit_confirm = False

            snapshots.append(
                ZoneSnapshot(
                    ts_ms=ts_ms,
                    ticker=self._ticker,
                    level_id=lvl.level_id,
                    in_zone=in_zone,
                    signed_distance_to_level=signed,
                    signed_distance_to_level_atr=signed_atr,
                    abs_distance_to_level_atr=abs_atr,
                    zone_entry_time=st.zone_entry_time,
                    zone_exit_time=st.zone_exit_time,
                    touch_count=st.touch_count,
                )
            )

        return snapshots, events


