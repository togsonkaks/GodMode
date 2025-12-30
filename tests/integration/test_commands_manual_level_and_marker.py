from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from godmode.core.config import AppConfig
from godmode.core.enums import DirectionBias
from godmode.orchestrator.session import run_replay_session


def test_commands_add_level_and_marker_extract(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.storage.root_dir = str(tmp_path / "out")
    cfg.session.exchange_timezone = "UTC"
    cfg.snapshot.interval_seconds = 10
    cfg.atr.fixed_default_atr = 4.0

    ticker = "AAPL"
    session_id = "TESTSESSION"

    # Start with NO levels; we'll add one via command.
    levels_yaml = tmp_path / "levels.yaml"
    levels_yaml.write_text("ticker: AAPL\nlevels:\n  - level_id: PRE\n    level_price: 9999\n    level_type: resistance\n    level_source: manual\n    level_width_atr: 0.25\n", encoding="utf-8")

    quotes = pd.DataFrame(
        [
            {"ts_ms": 9_000, "symbol": ticker, "bid": 99.0, "bid_size": 1, "ask": 101.0, "ask_size": 1},
            {"ts_ms": 19_000, "symbol": ticker, "bid": 99.0, "bid_size": 1, "ask": 101.0, "ask_size": 1},
            {"ts_ms": 29_000, "symbol": ticker, "bid": 101.0, "bid_size": 1, "ask": 103.0, "ask_size": 1},
        ]
    )
    trades = pd.DataFrame(
        [
            {"ts_ms": 9_500, "symbol": ticker, "price": 99.5, "size": 10},
            {"ts_ms": 19_500, "symbol": ticker, "price": 98.5, "size": 10},
            {"ts_ms": 29_500, "symbol": ticker, "price": 102.1, "size": 10},
        ]
    )

    tpath = tmp_path / "trades.parquet"
    qpath = tmp_path / "quotes.parquet"
    trades.to_parquet(tpath)
    quotes.to_parquet(qpath)

    # Commands: add a manual level at 100, and add a marker around 19s.
    commands_csv = tmp_path / "commands.csv"
    pd.DataFrame(
        [
            {
                "ts_ms": 9_999,
                "ticker": ticker,
                "type": "add_level",
                "level_price": 100.0,
                "level_type": "support",
                "level_width_atr": 0.25,
            },
            {
                "ts_ms": 19_999,
                "ticker": ticker,
                "type": "add_marker",
                "marker_type": "downtrend_break",
                "direction_bias": "long",
            },
        ]
    ).to_csv(commands_csv, index=False)

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
            commands_csv=commands_csv,
        )
    )

    out_root = Path(cfg.storage.root_dir)
    marker_files = list(out_root.rglob("markers/**/*.parquet"))
    assert marker_files, "expected markers parquet output"

    # Marker-extracted episode should appear (episode_id == marker_id)
    episodes_files = list(out_root.rglob("episodes/**/*.parquet"))
    assert episodes_files
    dfs = [pq.ParquetFile(p).read().to_pandas() for p in episodes_files]
    ep_table = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    assert not ep_table.empty
    assert (ep_table["episode_source"] == "marker_extract").any()


