from __future__ import annotations

import pytest

from godmode.providers.polygon import _parse_quote_row, _parse_trade_row


def test_parse_trade_row_ns_to_ms() -> None:
    row = {"sip_timestamp": 1_700_000_000_000_000_000, "price": 100.5, "size": 10, "conditions": [1, 2]}
    t = _parse_trade_row("TSLA", row, ts_unit_hint="ns")
    assert t.ts_ms == 1_700_000_000_000
    assert t.price == 100.5
    assert t.size == 10.0
    assert t.conditions == ("1", "2")


def test_parse_quote_row_ms() -> None:
    row = {"t": 1234, "bp": 100.0, "bs": 1, "ap": 100.5, "as": 2}
    q = _parse_quote_row("TSLA", row, ts_unit_hint="ms")
    assert q.ts_ms == 1234
    assert q.bid == 100.0
    assert q.ask == 100.5


def test_parse_trade_missing_fields_raises() -> None:
    with pytest.raises(ValueError):
        _parse_trade_row("TSLA", {"t": 1}, ts_unit_hint="ms")


