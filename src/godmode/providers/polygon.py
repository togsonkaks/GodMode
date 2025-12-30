from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

import httpx
import websockets

from godmode.core.models import Quote, Trade
from godmode.providers.base import DataProvider


PolygonAssetClass = Literal["stocks"]
PolygonTimestampUnit = Literal["ms", "ns"]


def _first_present(d: dict[str, Any], keys: list[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_ts_ms(x: Any, *, unit_hint: PolygonTimestampUnit) -> int:
    """
    Polygon commonly returns timestamps in nanoseconds. We normalize to ms.
    """
    v = int(x)
    if unit_hint == "ns":
        return v // 1_000_000
    return v


def _to_query_ts(x_ms: int, *, unit: PolygonTimestampUnit) -> int:
    x_ms = int(x_ms)
    return x_ms * 1_000_000 if unit == "ns" else x_ms


def _parse_trade_row(symbol: str, row: dict[str, Any], *, ts_unit_hint: PolygonTimestampUnit) -> Trade:
    ts_raw = _first_present(row, ["sip_timestamp", "participant_timestamp", "timestamp", "t"])
    if ts_raw is None:
        raise ValueError("trade row missing timestamp")
    ts_ms = _to_ts_ms(ts_raw, unit_hint=ts_unit_hint)

    price = _first_present(row, ["price", "p"])
    size = _first_present(row, ["size", "s"])
    if price is None or size is None:
        raise ValueError("trade row missing price/size")

    cond = row.get("conditions", None)
    conditions: tuple[str, ...] = ()
    if cond is None:
        conditions = ()
    elif isinstance(cond, list):
        conditions = tuple(str(x) for x in cond)
    else:
        conditions = (str(cond),)

    return Trade(ts_ms=ts_ms, symbol=symbol, price=float(price), size=float(size), conditions=conditions)


def _parse_quote_row(symbol: str, row: dict[str, Any], *, ts_unit_hint: PolygonTimestampUnit) -> Quote:
    ts_raw = _first_present(row, ["sip_timestamp", "participant_timestamp", "timestamp", "t"])
    if ts_raw is None:
        raise ValueError("quote row missing timestamp")
    ts_ms = _to_ts_ms(ts_raw, unit_hint=ts_unit_hint)

    bid = _first_present(row, ["bid_price", "bp", "bid"])
    ask = _first_present(row, ["ask_price", "ap", "ask"])
    bid_size = _first_present(row, ["bid_size", "bs"])
    ask_size = _first_present(row, ["ask_size", "as"])
    if bid is None or ask is None or bid_size is None or ask_size is None:
        raise ValueError("quote row missing bid/ask fields")

    return Quote(
        ts_ms=ts_ms,
        symbol=symbol,
        bid=float(bid),
        bid_size=float(bid_size),
        ask=float(ask),
        ask_size=float(ask_size),
    )


@dataclass(frozen=True, slots=True)
class PolygonConfig:
    api_key: str
    asset_class: PolygonAssetClass = "stocks"
    rest_base_url: str = "https://api.polygon.io"
    ws_base_url: str = "wss://socket.polygon.io"
    timestamp_unit: PolygonTimestampUnit = "ns"  # polygon commonly uses ns on v3


class PolygonProvider(DataProvider):
    """
    Polygon.io provider adapter (V1).

    - REST: getTrades/getQuotes (historical range)
    - WS: subscribeTrades/subscribeQuotes (streaming)

    Determinism:
    - Always normalizes all timestamps to `ts_ms` (ms).
    - WS batches are yielded in stable order by (ts_ms, seq).
    """

    def __init__(self, *, cfg: PolygonConfig) -> None:
        if not cfg.api_key:
            raise ValueError("PolygonConfig.api_key is required")
        self._cfg = cfg
        self._client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "POLYGON_API_KEY",
        timestamp_unit: PolygonTimestampUnit = "ns",
    ) -> "PolygonProvider":
        key = os.environ.get(api_key_env, "")
        return cls(cfg=PolygonConfig(api_key=key, timestamp_unit=timestamp_unit))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _paginate_v3(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        next_url: Optional[str] = url
        next_params = dict(params)
        while next_url is not None:
            r = await self._client.get(next_url, params=next_params)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", []) or []
            if not isinstance(results, list):
                raise ValueError("unexpected polygon response: results is not a list")
            out.extend(results)
            next_url = data.get("next_url")
            next_params = {"apiKey": self._cfg.api_key} if next_url else {}
        return out

    async def getTrades(self, symbol: str, start: int, end: int) -> list[Trade]:
        s = symbol.upper()
        url = f"{self._cfg.rest_base_url}/v3/trades/{s}"
        params = {
            "timestamp.gte": _to_query_ts(int(start), unit=self._cfg.timestamp_unit),
            "timestamp.lte": _to_query_ts(int(end), unit=self._cfg.timestamp_unit),
            "sort": "timestamp",
            "order": "asc",
            "limit": 50000,
            "apiKey": self._cfg.api_key,
        }
        rows = await self._paginate_v3(url, params)
        return [_parse_trade_row(s, row, ts_unit_hint=self._cfg.timestamp_unit) for row in rows]

    async def getQuotes(self, symbol: str, start: int, end: int) -> list[Quote]:
        s = symbol.upper()
        url = f"{self._cfg.rest_base_url}/v3/quotes/{s}"
        params = {
            "timestamp.gte": _to_query_ts(int(start), unit=self._cfg.timestamp_unit),
            "timestamp.lte": _to_query_ts(int(end), unit=self._cfg.timestamp_unit),
            "sort": "timestamp",
            "order": "asc",
            "limit": 50000,
            "apiKey": self._cfg.api_key,
        }
        rows = await self._paginate_v3(url, params)
        return [_parse_quote_row(s, row, ts_unit_hint=self._cfg.timestamp_unit) for row in rows]

    async def _ws(self) -> websockets.WebSocketClientProtocol:
        url = f"{self._cfg.ws_base_url}/{self._cfg.asset_class}"
        ws = await websockets.connect(url, ping_interval=20, ping_timeout=20)
        await ws.send(json.dumps({"action": "auth", "params": self._cfg.api_key}))
        return ws

    async def subscribeTrades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        subs = ",".join([f"T.{s.upper()}" for s in symbols])
        ws = await self._ws()
        await ws.send(json.dumps({"action": "subscribe", "params": subs}))
        seq = 0
        try:
            async for msg in ws:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                batch: list[Trade] = []
                for ev in data:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("ev") not in ("T", "XT"):  # T=trade (stocks)
                        continue
                    sym = str(ev.get("sym") or ev.get("symbol") or "").upper()
                    if not sym:
                        continue
                    row = {
                        "t": ev.get("t"),
                        "p": ev.get("p"),
                        "s": ev.get("s"),
                        "conditions": ev.get("c"),
                    }
                    try:
                        batch.append(_parse_trade_row(sym, row, ts_unit_hint="ms"))
                    except Exception:
                        continue
                # Stable order by (ts_ms, seq) within this message
                batch.sort(key=lambda t: (t.ts_ms, 0))
                for t in batch:
                    yield t
                    seq += 1
                await asyncio.sleep(0)
        finally:
            await ws.close()

    async def subscribeQuotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        subs = ",".join([f"Q.{s.upper()}" for s in symbols])
        ws = await self._ws()
        await ws.send(json.dumps({"action": "subscribe", "params": subs}))
        try:
            async for msg in ws:
                data = json.loads(msg)
                if not isinstance(data, list):
                    continue
                batch: list[Quote] = []
                for ev in data:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("ev") not in ("Q", "XQ"):  # Q=quote (stocks)
                        continue
                    sym = str(ev.get("sym") or ev.get("symbol") or "").upper()
                    if not sym:
                        continue
                    row = {
                        "t": ev.get("t"),
                        "bp": ev.get("bp"),
                        "bs": ev.get("bs"),
                        "ap": ev.get("ap"),
                        "as": ev.get("as"),
                    }
                    try:
                        batch.append(_parse_quote_row(sym, row, ts_unit_hint="ms"))
                    except Exception:
                        continue
                batch.sort(key=lambda q: (q.ts_ms, 0))
                for q in batch:
                    yield q
                await asyncio.sleep(0)
        finally:
            await ws.close()


