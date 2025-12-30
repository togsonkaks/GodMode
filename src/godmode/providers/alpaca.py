"""Alpaca data provider adapter (REST + WebSocket)."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal, Optional

import httpx
import websockets

from godmode.core.models import Quote, Trade
from godmode.providers.base import DataProvider


AlpacaFeed = Literal["iex", "sip"]


def _parse_iso_to_ms(iso_str: str) -> int:
    """Parse ISO-8601 timestamp to milliseconds."""
    # Alpaca returns timestamps like "2025-12-23T14:30:00.123456789Z"
    # Handle nanosecond precision by truncating
    s = iso_str.rstrip("Z")
    if "." in s:
        parts = s.split(".")
        # Keep only first 6 digits of fractional seconds (microseconds)
        frac = parts[1][:6].ljust(6, "0")
        s = f"{parts[0]}.{frac}"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_rfc3339(ts_ms: int) -> str:
    """Convert ms timestamp to RFC3339 for Alpaca API."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_trade(symbol: str, row: dict[str, Any]) -> Trade:
    """Parse Alpaca trade response to Trade model."""
    ts_ms = _parse_iso_to_ms(row["t"])
    return Trade(
        ts_ms=ts_ms,
        symbol=symbol,
        price=float(row["p"]),
        size=float(row["s"]),
        conditions=tuple(row.get("c", [])),
    )


def _parse_quote(symbol: str, row: dict[str, Any]) -> Quote:
    """Parse Alpaca quote response to Quote model."""
    ts_ms = _parse_iso_to_ms(row["t"])
    return Quote(
        ts_ms=ts_ms,
        symbol=symbol,
        bid=float(row["bp"]),
        bid_size=float(row["bs"]),
        ask=float(row["ap"]),
        ask_size=float(row["as"]),
    )


@dataclass(frozen=True, slots=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    feed: AlpacaFeed = "sip"  # "sip" for full data, "iex" for free
    data_base_url: str = "https://data.alpaca.markets"
    ws_base_url: str = "wss://stream.data.alpaca.markets"


class AlpacaProvider(DataProvider):
    """
    Alpaca data provider implementing DataProvider interface.
    
    Supports:
    - Historical trades/quotes via REST
    - Real-time streaming via WebSocket
    """

    def __init__(self, *, cfg: AlpacaConfig) -> None:
        if not cfg.api_key or not cfg.secret_key:
            raise ValueError("AlpacaConfig requires api_key and secret_key")
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "APCA-API-KEY-ID": cfg.api_key,
                "APCA-API-SECRET-KEY": cfg.secret_key,
            },
        )

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "ALPACA_API_KEY",
        secret_key_env: str = "ALPACA_SECRET_KEY",
        feed: AlpacaFeed = "sip",
    ) -> "AlpacaProvider":
        api_key = os.environ.get(api_key_env, "")
        secret_key = os.environ.get(secret_key_env, "")
        return cls(cfg=AlpacaConfig(api_key=api_key, secret_key=secret_key, feed=feed))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _paginate(
        self, url: str, params: dict[str, Any], results_key: str
    ) -> list[dict[str, Any]]:
        """Paginate through Alpaca API results."""
        out: list[dict[str, Any]] = []
        next_token: Optional[str] = None

        while True:
            if next_token:
                params["page_token"] = next_token
            
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get(results_key, [])
            if results:
                out.extend(results)

            next_token = data.get("next_page_token")
            if not next_token:
                break

        return out

    async def getTrades(self, symbol: str, start: int, end: int) -> list[Trade]:
        """Fetch historical trades for symbol in [start, end] ms range."""
        s = symbol.upper()
        url = f"{self._cfg.data_base_url}/v2/stocks/{s}/trades"
        params = {
            "start": _ms_to_rfc3339(start),
            "end": _ms_to_rfc3339(end),
            "feed": self._cfg.feed,
            "limit": 10000,
        }
        rows = await self._paginate(url, params, "trades")
        return [_parse_trade(s, row) for row in rows]

    async def getQuotes(self, symbol: str, start: int, end: int) -> list[Quote]:
        """Fetch historical quotes for symbol in [start, end] ms range."""
        s = symbol.upper()
        url = f"{self._cfg.data_base_url}/v2/stocks/{s}/quotes"
        params = {
            "start": _ms_to_rfc3339(start),
            "end": _ms_to_rfc3339(end),
            "feed": self._cfg.feed,
            "limit": 10000,
        }
        rows = await self._paginate(url, params, "quotes")
        return [_parse_quote(s, row) for row in rows]

    async def subscribeTrades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        """Stream real-time trades via WebSocket."""
        url = f"{self._cfg.ws_base_url}/v2/{self._cfg.feed}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self._cfg.api_key,
                "secret": self._cfg.secret_key,
            }
            await ws.send(json.dumps(auth_msg))
            auth_resp = await ws.recv()
            
            # Subscribe to trades
            sub_msg = {
                "action": "subscribe",
                "trades": [s.upper() for s in symbols],
            }
            await ws.send(json.dumps(sub_msg))
            await ws.recv()  # subscription confirmation

            # Stream trades
            async for msg in ws:
                data = json.loads(msg)
                if isinstance(data, list):
                    for item in data:
                        if item.get("T") == "t":  # trade message
                            yield _parse_trade(item["S"], item)

    async def subscribeQuotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """Stream real-time quotes via WebSocket."""
        url = f"{self._cfg.ws_base_url}/v2/{self._cfg.feed}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self._cfg.api_key,
                "secret": self._cfg.secret_key,
            }
            await ws.send(json.dumps(auth_msg))
            await ws.recv()
            
            # Subscribe to quotes
            sub_msg = {
                "action": "subscribe",
                "quotes": [s.upper() for s in symbols],
            }
            await ws.send(json.dumps(sub_msg))
            await ws.recv()

            # Stream quotes
            async for msg in ws:
                data = json.loads(msg)
                if isinstance(data, list):
                    for item in data:
                        if item.get("T") == "q":  # quote message
                            yield _parse_quote(item["S"], item)

