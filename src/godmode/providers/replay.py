from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Iterable, Literal, Optional

import pandas as pd

from godmode.core.models import Quote, Trade
from godmode.providers.base import DataProvider


ReplayFormat = Literal["parquet", "csv"]


def _load_df(path: Path, fmt: ReplayFormat) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported fmt: {fmt}")


def _require_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["ts_ms", "symbol", "price", "size"], name="trades")
    # Optional conditions column: a string or list-like; we normalize later.
    out = df.copy()
    out["ts_ms"] = out["ts_ms"].astype("int64")
    out["symbol"] = out["symbol"].astype("string")
    out["price"] = out["price"].astype("float64")
    out["size"] = out["size"].astype("float64")
    if "conditions" not in out.columns:
        out["conditions"] = None
    return out


def _normalize_quotes_df(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["ts_ms", "symbol", "bid", "bid_size", "ask", "ask_size"], name="quotes")
    out = df.copy()
    out["ts_ms"] = out["ts_ms"].astype("int64")
    out["symbol"] = out["symbol"].astype("string")
    out["bid"] = out["bid"].astype("float64")
    out["bid_size"] = out["bid_size"].astype("float64")
    out["ask"] = out["ask"].astype("float64")
    out["ask_size"] = out["ask_size"].astype("float64")
    return out


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    # Stable deterministic ordering by ts_ms with tie-breaker on original row order.
    df = df.reset_index(drop=False).rename(columns={"index": "_row"})
    return df.sort_values(["ts_ms", "_row"], kind="mergesort").reset_index(drop=True)


class ReplayDataProvider(DataProvider):
    """
    Replay provider (Implementation Decisions #4).

    Treats replay as just another data source; consumers use the same engine logic.

    File expectations (minimal, deterministic):
    - trades file: columns ts_ms, symbol, price, size, optional conditions
    - quotes file: columns ts_ms, symbol, bid, bid_size, ask, ask_size
    """

    def __init__(
        self,
        *,
        trades_path: Path,
        quotes_path: Path,
        fmt: ReplayFormat = "parquet",
    ) -> None:
        self._trades_path = Path(trades_path)
        self._quotes_path = Path(quotes_path)
        self._fmt: ReplayFormat = fmt

        trades_df = _normalize_trades_df(_load_df(self._trades_path, fmt))
        quotes_df = _normalize_quotes_df(_load_df(self._quotes_path, fmt))

        self._trades = _stable_sort(trades_df)
        self._quotes = _stable_sort(quotes_df)

    async def getTrades(self, symbol: str, start: int, end: int) -> list[Trade]:
        df = self._trades
        m = (df["symbol"] == symbol) & (df["ts_ms"] >= int(start)) & (df["ts_ms"] <= int(end))
        out: list[Trade] = []
        for row in df.loc[m].itertuples(index=False):
            conditions = ()
            if getattr(row, "conditions", None) is not None:
                c = getattr(row, "conditions")
                if isinstance(c, str):
                    conditions = (c,)
                elif isinstance(c, (list, tuple)):
                    conditions = tuple(str(x) for x in c)
                else:
                    conditions = (str(c),)
            out.append(Trade(ts_ms=int(row.ts_ms), symbol=str(row.symbol), price=float(row.price), size=float(row.size), conditions=conditions))
        return out

    async def getQuotes(self, symbol: str, start: int, end: int) -> list[Quote]:
        df = self._quotes
        m = (df["symbol"] == symbol) & (df["ts_ms"] >= int(start)) & (df["ts_ms"] <= int(end))
        out: list[Quote] = []
        for row in df.loc[m].itertuples(index=False):
            out.append(
                Quote(
                    ts_ms=int(row.ts_ms),
                    symbol=str(row.symbol),
                    bid=float(row.bid),
                    bid_size=float(row.bid_size),
                    ask=float(row.ask),
                    ask_size=float(row.ask_size),
                )
            )
        return out

    async def subscribeTrades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        symset = set(symbols)
        for row in self._trades.itertuples(index=False):
            if str(row.symbol) not in symset:
                continue
            yield Trade(
                ts_ms=int(row.ts_ms),
                symbol=str(row.symbol),
                price=float(row.price),
                size=float(row.size),
                conditions=(),
            )
            await asyncio.sleep(0)

    async def subscribeQuotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        symset = set(symbols)
        for row in self._quotes.itertuples(index=False):
            if str(row.symbol) not in symset:
                continue
            yield Quote(
                ts_ms=int(row.ts_ms),
                symbol=str(row.symbol),
                bid=float(row.bid),
                bid_size=float(row.bid_size),
                ask=float(row.ask),
                ask_size=float(row.ask_size),
            )
            await asyncio.sleep(0)


