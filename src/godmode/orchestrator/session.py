from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from godmode.core.config import AppConfig
from godmode.core.enums import DirectionBias
from godmode.orchestrator.commands import AddLevelCmd, AddMarkerCmd, CommandEvent, load_commands_csv
from godmode.providers.base import DataProvider
from godmode.providers.replay import ReplayDataProvider
from godmode.worker.ticker_worker import TickerWorker
from godmode.zone.level_manager import LevelManager


EventType = Literal["quote", "trade"]


@dataclass(frozen=True, slots=True)
class _Ev:
    ts_ms: int
    type: EventType
    seq: int
    obj: object


async def run_replay_session(
    *,
    config: AppConfig,
    ticker: str,
    session_id: str,
    direction_bias: DirectionBias,
    levels_yaml: Optional[Path] = None,
    trades_path: Path,
    quotes_path: Path,
    fmt: Literal["parquet", "csv"] = "parquet",
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
    commands_csv: Optional[Path] = None,
) -> None:
    """
    End-to-end replay runner:
    - loads levels
    - loads trades/quotes from replay provider
    - merges by (ts_ms, type_order, seq) where quotes precede trades on ties
    - drives a per-ticker worker that writes episodes/snapshots
    """
    provider: DataProvider = ReplayDataProvider(trades_path=trades_path, quotes_path=quotes_path, fmt=fmt)

    levels: list = []
    if levels_yaml is not None:
        lm = LevelManager()
        levelset = lm.load_yaml(levels_yaml)
        if levelset.ticker != ticker:
            raise ValueError(f"levels yaml ticker {levelset.ticker} != requested ticker {ticker}")
        levels = list(levelset.levels)

    worker = TickerWorker(
        ticker=ticker,
        session_id=session_id,
        config=config,
        levels=levels,
        direction_bias=direction_bias,
    )

    if start_ts_ms is None or end_ts_ms is None:
        # Determine range from provider files by querying a wide range and using returned min/max.
        # (This keeps the DataProvider interface stable.)
        trades_all = await provider.getTrades(ticker, 0, 2**62)
        quotes_all = await provider.getQuotes(ticker, 0, 2**62)
        ts = [t.ts_ms for t in trades_all] + [q.ts_ms for q in quotes_all]
        if not ts:
            return
        start_ts_ms = min(ts) if start_ts_ms is None else start_ts_ms
        end_ts_ms = max(ts) if end_ts_ms is None else end_ts_ms
    assert start_ts_ms is not None and end_ts_ms is not None

    trades = await provider.getTrades(ticker, start_ts_ms, end_ts_ms)
    quotes = await provider.getQuotes(ticker, start_ts_ms, end_ts_ms)

    evs: list[_Ev] = []
    seq = 0
    for q in quotes:
        evs.append(_Ev(ts_ms=int(q.ts_ms), type="quote", seq=seq, obj=q))
        seq += 1
    for t in trades:
        evs.append(_Ev(ts_ms=int(t.ts_ms), type="trade", seq=seq, obj=t))
        seq += 1

    # Optional runtime commands (manual add_level / markers) merged as another event source.
    cmd_evs: list[CommandEvent] = []
    if commands_csv is not None:
        cmd_evs = load_commands_csv(commands_csv)
        for c in cmd_evs:
            evs.append(_Ev(ts_ms=int(c.ts_ms), type="quote", seq=seq, obj=c))  # type_order handled below
            seq += 1

    def type_order(tp: EventType) -> int:
        return 0 if tp == "quote" else 1  # quotes first on tie so quote-test can see same-ts quote

    # We also want commands processed before trades at the same ts, and after quotes.
    def _kind_order(obj: object) -> int:
        # 0=quote, 1=command, 2=trade (only used within 'quote' bucket hack below)
        if isinstance(obj, CommandEvent):
            return 1
        return 0

    def _order(e: _Ev) -> tuple[int, int, int]:
        if e.type == "quote":
            # quote bucket contains actual quotes and command events
            return (e.ts_ms, _kind_order(e.obj), e.seq)
        return (e.ts_ms, 2, e.seq)

    evs.sort(key=_order)

    # Drive worker in event-time order.
    for e in evs:
        if isinstance(e.obj, CommandEvent):
            if e.obj.type == "add_level":
                worker.on_add_level(e.obj.obj)  # type: ignore[arg-type]
            elif e.obj.type == "add_marker":
                worker.on_add_marker(e.obj.obj)  # type: ignore[arg-type]
            else:
                raise ValueError(f"unknown command: {e.obj.type}")
        elif e.type == "quote":
            worker.on_quote(e.obj)  # type: ignore[arg-type]
        else:
            worker.on_trade(e.obj)  # type: ignore[arg-type]
        worker.on_event_time(e.ts_ms)

    # Flush ticks up to end_ts_ms
    worker.on_event_time(int(end_ts_ms))
    worker.flush()


