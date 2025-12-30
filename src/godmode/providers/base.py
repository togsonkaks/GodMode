from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from godmode.core.models import Quote, Trade


class DataProvider(ABC):
    """
    Provider abstraction (Implementation Decisions #1).

    All events are required to include:
    - ts_ms (exchange/SIP event timestamp in milliseconds)
    - symbol

    Live + replay must preserve ordering by ts_ms (stable tie-breakers if needed).
    """

    @abstractmethod
    async def getTrades(self, symbol: str, start: int, end: int) -> list[Trade]:
        raise NotImplementedError

    @abstractmethod
    async def getQuotes(self, symbol: str, start: int, end: int) -> list[Quote]:
        raise NotImplementedError

    @abstractmethod
    async def subscribeTrades(self, symbols: list[str]) -> AsyncIterator[Trade]:
        raise NotImplementedError

    @abstractmethod
    async def subscribeQuotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        raise NotImplementedError


