"""Unit tests for Alpaca provider parsing logic."""
from godmode.providers.alpaca import _parse_iso_to_ms, _parse_trade, _parse_quote


def test_parse_iso_to_ms_basic() -> None:
    # Standard ISO with Z suffix
    ts = _parse_iso_to_ms("2025-12-23T14:30:00Z")
    # 2025-12-23T14:30:00Z = 1766500200000 ms
    assert ts == 1766500200000


def test_parse_iso_to_ms_with_nanoseconds() -> None:
    # Alpaca returns nanosecond precision - should truncate to ms
    ts = _parse_iso_to_ms("2025-12-23T14:30:00.123456789Z")
    assert ts == 1766500200123


def test_parse_trade_row() -> None:
    row = {
        "t": "2025-12-23T14:30:00.500Z",
        "p": 100.50,
        "s": 200,
        "c": ["@", "F"],
    }
    trade = _parse_trade("TSLA", row)
    assert trade.symbol == "TSLA"
    assert trade.price == 100.50
    assert trade.size == 200
    assert trade.conditions == ("@", "F")
    assert trade.ts_ms == 1766500200500


def test_parse_quote_row() -> None:
    row = {
        "t": "2025-12-23T14:30:01.000Z",
        "bp": 100.45,
        "bs": 100,
        "ap": 100.55,
        "as": 150,
    }
    quote = _parse_quote("TSLA", row)
    assert quote.symbol == "TSLA"
    assert quote.bid == 100.45
    assert quote.bid_size == 100
    assert quote.ask == 100.55
    assert quote.ask_size == 150
    assert quote.ts_ms == 1766500201000

