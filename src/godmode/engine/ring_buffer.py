from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, Iterator, Optional, Protocol, TypeVar


class HasEventTime(Protocol):
    ts_ms: int


T = TypeVar("T", bound=HasEventTime)


@dataclass(frozen=True, slots=True)
class _Item(Generic[T]):
    """Internal wrapper to provide deterministic ordering on equal ts_ms."""

    ts_ms: int
    seq: int
    value: T


class TimeRingBuffer(Generic[T]):
    """
    Time-bounded ring buffer keyed by event-time (ts_ms).

    Determinism:
    - Accepts non-decreasing event-time. Equal timestamps are allowed.
    - Provides stable ordering via a monotonically increasing sequence counter.
    - Eviction is based solely on event-time and max_age_ms.
    """

    def __init__(self, *, max_age_ms: int, allow_out_of_order: bool = False) -> None:
        if max_age_ms <= 0:
            raise ValueError("max_age_ms must be > 0")
        self._max_age_ms = int(max_age_ms)
        self._allow_out_of_order = allow_out_of_order

        self._items: Deque[_Item[T]] = deque()
        self._seq = 0
        self._last_ts_ms: Optional[int] = None

    @property
    def max_age_ms(self) -> int:
        return self._max_age_ms

    @property
    def last_ts_ms(self) -> Optional[int]:
        return self._last_ts_ms

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        for it in self._items:
            yield it.value

    def items(self) -> Iterator[tuple[int, int, T]]:
        """Iterate (ts_ms, seq, value) in deterministic order."""
        for it in self._items:
            yield (it.ts_ms, it.seq, it.value)

    def append(self, value: T) -> None:
        ts_ms = int(value.ts_ms)

        if (not self._allow_out_of_order) and (self._last_ts_ms is not None) and ts_ms < self._last_ts_ms:
            raise ValueError(
                f"out-of-order event-time: ts_ms={ts_ms} < last_ts_ms={self._last_ts_ms}. "
                "If this is expected, construct TimeRingBuffer(allow_out_of_order=True)."
            )

        item = _Item(ts_ms=ts_ms, seq=self._seq, value=value)
        self._seq += 1
        self._items.append(item)
        self._last_ts_ms = ts_ms

        # Evict based on newest event-time (deterministic in replay/live if event ordering is preserved).
        self.evict_older_than(ts_ms - self._max_age_ms)

        # If out-of-order is allowed, re-sort for deterministic range queries.
        if self._allow_out_of_order:
            self._resort()

    def _resort(self) -> None:
        # Sorting a deque deterministically by (ts_ms, seq).
        if len(self._items) <= 1:
            return
        self._items = deque(sorted(self._items, key=lambda x: (x.ts_ms, x.seq)))

    def evict_older_than(self, cutoff_ts_ms: int) -> int:
        """Evict items with ts_ms < cutoff_ts_ms. Returns number evicted."""
        evicted = 0
        while self._items and self._items[0].ts_ms < cutoff_ts_ms:
            self._items.popleft()
            evicted += 1
        return evicted

    def latest(self) -> Optional[T]:
        return self._items[-1].value if self._items else None

    def window(self, *, start_ts_ms: int, end_ts_ms: int) -> list[T]:
        """
        Return items with start_ts_ms <= ts_ms <= end_ts_ms (inclusive), in deterministic order.
        """
        if end_ts_ms < start_ts_ms:
            raise ValueError("end_ts_ms must be >= start_ts_ms")

        # Linear scan is fine for the small buffer (~5 minutes).
        return [it.value for it in self._items if start_ts_ms <= it.ts_ms <= end_ts_ms]

    def tail(self, n: int) -> list[T]:
        if n <= 0:
            return []
        if n >= len(self._items):
            return [it.value for it in self._items]
        # deque slicing is not supported; convert last n manually.
        out: list[T] = []
        for it in list(self._items)[-n:]:
            out.append(it.value)
        return out


