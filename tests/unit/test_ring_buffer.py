from __future__ import annotations

import pytest

from godmode.engine.ring_buffer import TimeRingBuffer


class _Evt:
    def __init__(self, ts_ms: int, name: str) -> None:
        self.ts_ms = ts_ms
        self.name = name


def test_evict_by_age_on_append() -> None:
    buf = TimeRingBuffer[_Evt](max_age_ms=5_000)

    buf.append(_Evt(0, "a"))
    buf.append(_Evt(2_000, "b"))
    buf.append(_Evt(4_999, "c"))
    assert [e.name for e in buf] == ["a", "b", "c"]

    # Append at t=6_000 -> cutoff is 1_000; "a" should evict (0 < 1_000)
    buf.append(_Evt(6_000, "d"))
    assert [e.name for e in buf] == ["b", "c", "d"]


def test_window_inclusive_bounds() -> None:
    buf = TimeRingBuffer[_Evt](max_age_ms=60_000)
    for t, name in [(1000, "a"), (2000, "b"), (3000, "c")]:
        buf.append(_Evt(t, name))

    out = buf.window(start_ts_ms=2000, end_ts_ms=3000)
    assert [e.name for e in out] == ["b", "c"]


def test_reject_out_of_order_by_default() -> None:
    buf = TimeRingBuffer[_Evt](max_age_ms=60_000)
    buf.append(_Evt(2_000, "b"))
    with pytest.raises(ValueError):
        buf.append(_Evt(1_000, "a"))


def test_allow_out_of_order_with_resort_and_stable_tiebreaker() -> None:
    buf = TimeRingBuffer[_Evt](max_age_ms=60_000, allow_out_of_order=True)

    # Out of order appends are allowed and buffer is resorted deterministically by (ts_ms, seq).
    buf.append(_Evt(2_000, "b"))
    buf.append(_Evt(1_000, "a"))
    buf.append(_Evt(1_000, "a2"))  # equal ts_ms, later seq -> stable order
    buf.append(_Evt(3_000, "c"))

    assert [e.name for e in buf] == ["a", "a2", "b", "c"]


def test_tail() -> None:
    buf = TimeRingBuffer[_Evt](max_age_ms=60_000)
    for t, name in [(1000, "a"), (2000, "b"), (3000, "c")]:
        buf.append(_Evt(t, name))

    assert [e.name for e in buf.tail(2)] == ["b", "c"]
    assert [e.name for e in buf.tail(10)] == ["a", "b", "c"]
    assert buf.tail(0) == []


