"""Alpaca data export utility."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from godmode.core.config import AppConfig
from godmode.providers.alpaca import AlpacaProvider


async def export_alpaca_to_replay(
    *,
    config: AppConfig,
    ticker: str,
    start_ts_ms: int,
    end_ts_ms: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Fetch trades/quotes from Alpaca REST and write deterministic replay parquet files:
    - trades.parquet: ts_ms, symbol, price, size, conditions
    - quotes.parquet: ts_ms, symbol, bid, bid_size, ask, ask_size
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.parquet"
    quotes_path = out_dir / "quotes.parquet"

    provider = AlpacaProvider.from_env()
    try:
        trades = await provider.getTrades(ticker, start_ts_ms, end_ts_ms)
        quotes = await provider.getQuotes(ticker, start_ts_ms, end_ts_ms)
    finally:
        await provider.aclose()

    # Deterministic ordering by ts_ms (stable by original order)
    trades_rows = [asdict(t) for t in trades]
    quotes_rows = [asdict(q) for q in quotes]

    tdf = pd.DataFrame(trades_rows)
    qdf = pd.DataFrame(quotes_rows)

    if not tdf.empty:
        tdf = tdf.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)
    if not qdf.empty:
        qdf = qdf.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)

    tdf.to_parquet(trades_path, index=False)
    qdf.to_parquet(quotes_path, index=False)

    return trades_path, quotes_path

