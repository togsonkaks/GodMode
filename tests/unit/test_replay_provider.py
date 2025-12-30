from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from godmode.providers.replay import ReplayDataProvider


@pytest.mark.asyncio
async def test_get_trades_filters_by_symbol_and_range(tmp_path: Path) -> None:
    trades = pd.DataFrame(
        [
            {"ts_ms": 1000, "symbol": "AAPL", "price": 10.0, "size": 1.0},
            {"ts_ms": 2000, "symbol": "AAPL", "price": 11.0, "size": 1.0},
            {"ts_ms": 3000, "symbol": "TSLA", "price": 20.0, "size": 1.0},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"ts_ms": 1000, "symbol": "AAPL", "bid": 9.9, "bid_size": 1, "ask": 10.1, "ask_size": 1},
        ]
    )
    tpath = tmp_path / "trades.parquet"
    qpath = tmp_path / "quotes.parquet"
    trades.to_parquet(tpath)
    quotes.to_parquet(qpath)

    p = ReplayDataProvider(trades_path=tpath, quotes_path=qpath, fmt="parquet")
    out = await p.getTrades("AAPL", 1500, 2500)
    assert [t.ts_ms for t in out] == [2000]


@pytest.mark.asyncio
async def test_stream_ordering_is_deterministic_with_ties(tmp_path: Path) -> None:
    # Two AAPL trades with same ts_ms but different input row order; stable sort must preserve order.
    trades = pd.DataFrame(
        [
            {"ts_ms": 1000, "symbol": "AAPL", "price": 10.0, "size": 1.0},
            {"ts_ms": 1000, "symbol": "AAPL", "price": 10.1, "size": 1.0},
            {"ts_ms": 900, "symbol": "AAPL", "price": 9.0, "size": 1.0},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"ts_ms": 1000, "symbol": "AAPL", "bid": 9.9, "bid_size": 1, "ask": 10.1, "ask_size": 1},
        ]
    )
    tpath = tmp_path / "trades.parquet"
    qpath = tmp_path / "quotes.parquet"
    trades.to_parquet(tpath)
    quotes.to_parquet(qpath)

    p = ReplayDataProvider(trades_path=tpath, quotes_path=qpath, fmt="parquet")

    got = []
    async for t in p.subscribeTrades(["AAPL"]):
        got.append((t.ts_ms, t.price))

    assert got == [(900, 9.0), (1000, 10.0), (1000, 10.1)]


@pytest.mark.asyncio
async def test_missing_columns_raise(tmp_path: Path) -> None:
    trades = pd.DataFrame([{"ts_ms": 1, "symbol": "AAPL", "price": 1.0}])  # missing size
    quotes = pd.DataFrame(
        [{"ts_ms": 1, "symbol": "AAPL", "bid": 1.0, "bid_size": 1, "ask": 1.1, "ask_size": 1}]
    )
    tpath = tmp_path / "trades.parquet"
    qpath = tmp_path / "quotes.parquet"
    trades.to_parquet(tpath)
    quotes.to_parquet(qpath)

    with pytest.raises(ValueError):
        ReplayDataProvider(trades_path=tpath, quotes_path=qpath, fmt="parquet")


