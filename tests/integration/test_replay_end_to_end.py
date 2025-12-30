from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from godmode.core.config import AppConfig
from godmode.core.enums import DirectionBias
from godmode.orchestrator.session import run_replay_session


def test_replay_end_to_end_writes_parquet(tmp_path: Path) -> None:
    # Prepare synthetic trades/quotes around a level so an episode starts and resolves.
    # Use UTC timezone for deterministic partitioning.
    cfg = AppConfig()
    cfg.storage.root_dir = str(tmp_path / "out")
    cfg.session.exchange_timezone = "UTC"
    cfg.snapshot.interval_seconds = 10
    cfg.atr.fixed_default_atr = 4.0  # so 0.50 ATR => 2.0 points

    ticker = "AAPL"
    session_id = "TESTSESSION"

    # Levels
    levels_yaml = tmp_path / "levels.yaml"
    levels_yaml.write_text(
        "\n".join(
            [
                "ticker: AAPL",
                "levels:",
                "  - level_id: L1",
                "    level_price: 100.0",
                "    level_type: support",
                "    level_source: manual",
                "    level_width_atr: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    # Quotes every few seconds
    quotes = pd.DataFrame(
        [
            {"ts_ms": 1_000, "symbol": ticker, "bid": 99.0, "bid_size": 1, "ask": 101.0, "ask_size": 1},
            {"ts_ms": 9_000, "symbol": ticker, "bid": 99.0, "bid_size": 1, "ask": 101.0, "ask_size": 1},
            {"ts_ms": 19_000, "symbol": ticker, "bid": 99.0, "bid_size": 1, "ask": 101.0, "ask_size": 1},
            {"ts_ms": 29_000, "symbol": ticker, "bid": 101.0, "bid_size": 1, "ask": 103.0, "ask_size": 1},
        ]
    )

    # Trades: enter zone near 100 around t=9s, exit at 19s, then hit success >= 102 at 29s.
    trades = pd.DataFrame(
        [
            {"ts_ms": 5_000, "symbol": ticker, "price": 98.0, "size": 10},
            {"ts_ms": 9_500, "symbol": ticker, "price": 99.5, "size": 10},   # in-zone (within 1.0)
            {"ts_ms": 19_500, "symbol": ticker, "price": 98.5, "size": 10},  # out-of-zone
            {"ts_ms": 29_500, "symbol": ticker, "price": 102.1, "size": 10}, # success
        ]
    )

    tpath = tmp_path / "trades.parquet"
    qpath = tmp_path / "quotes.parquet"
    trades.to_parquet(tpath)
    quotes.to_parquet(qpath)

    asyncio.run(
        run_replay_session(
            config=cfg,
            ticker=ticker,
            session_id=session_id,
            direction_bias=DirectionBias.LONG,
            levels_yaml=levels_yaml,
            trades_path=tpath,
            quotes_path=qpath,
            fmt="parquet",
        )
    )

    out_root = Path(cfg.storage.root_dir)
    # Expect at least one episodes parquet and one snapshots parquet file
    episodes_files = list(out_root.rglob("episodes/**/*.parquet"))
    snapshots_files = list(out_root.rglob("snapshots/**/*.parquet"))
    tf_files = list(out_root.rglob("tf_indicators/**/*.parquet"))
    stream_files = list(out_root.rglob("session_stream/**/*.parquet"))
    assert episodes_files, "expected episodes parquet output"
    assert snapshots_files, "expected snapshots parquet output"
    assert tf_files, "expected tf_indicators parquet output"
    assert stream_files, "expected session_stream parquet output"

    # Validate schema aliasing exists in snapshots parquet
    table = pq.read_table(snapshots_files[0])
    cols = set(table.column_names)
    assert "%_at_ask" in cols
    assert "%_at_bid" in cols


