from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from godmode.core.config import AppConfig
from godmode.providers.polygon import PolygonProvider


def parse_dt_to_ts_ms(dt_str: str, *, tz: str) -> int:
    """
    Parse a datetime string into event-time ms.

    Accepted formats:
    - ISO-8601 with timezone: 2025-12-23T09:30:00-05:00
    - ISO-8601 without timezone: 2025-12-23T09:30:00 (interpreted in `tz`)
    - Space separator also accepted: 2025-12-23 09:30:00
    """
    s = dt_str.strip().replace(" ", "T")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return int(dt.timestamp() * 1000)


async def export_polygon_to_replay(
    *,
    config: AppConfig,
    ticker: str,
    start_ts_ms: int,
    end_ts_ms: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Fetch trades/quotes from Polygon REST and write deterministic replay parquet files:
    - trades.parquet: ts_ms, symbol, price, size, conditions(optional)
    - quotes.parquet: ts_ms, symbol, bid, bid_size, ask, ask_size
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.parquet"
    quotes_path = out_dir / "quotes.parquet"

    provider = PolygonProvider.from_env()
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


