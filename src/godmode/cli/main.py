from __future__ import annotations

import asyncio
from pathlib import Path

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """GodMode — Numeric market-state recorder (see SPEC.md)."""
    # Load local .env (if present) so users can store API keys without setx.
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"), override=False)
    except Exception:
        # dotenv is optional at runtime; env vars may already be set via OS.
        pass


@main.command("version")
def version_cmd() -> None:
    """Print version."""
    from godmode import __version__

    click.echo(__version__)


@main.command("doctor")
def doctor_cmd() -> None:
    """Run a quick environment + dependency preflight to avoid repeated setup failures."""
    from godmode.util.doctor import print_doctor, run_doctor

    rc = print_doctor(run_doctor())
    raise SystemExit(rc)


@main.command("replay-run")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("config/default.yaml"))
@click.option("--ticker", type=str, required=True)
@click.option("--session-id", type=str, required=True)
@click.option("--direction-bias", type=click.Choice(["long", "short"]), required=True)
@click.option("--levels-yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False, default=None)
@click.option("--trades-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--quotes-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--fmt", type=click.Choice(["parquet", "csv"]), default="parquet")
@click.option("--commands-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--start-ts-ms", type=int, default=None)
@click.option("--end-ts-ms", type=int, default=None)
@click.option("--out-dir", type=str, default=None, help="Override storage.root_dir")
def replay_run_cmd(
    config_path: Path,
    ticker: str,
    session_id: str,
    direction_bias: str,
    levels_yaml: Path | None,
    trades_path: Path,
    quotes_path: Path,
    fmt: str,
    commands_csv: Path | None,
    start_ts_ms: int | None,
    end_ts_ms: int | None,
    out_dir: str | None,
) -> None:
    """Run the full pipeline in replay mode (provider-agnostic engine)."""
    from godmode.core.config import AppConfig
    from godmode.core.enums import DirectionBias
    from godmode.orchestrator.session import run_replay_session
    from godmode.reports.writer import write_session_report

    cfg = AppConfig.load(config_path)
    if out_dir is not None:
        cfg.storage.root_dir = out_dir

    asyncio.run(
        run_replay_session(
            config=cfg,
            ticker=ticker,
            session_id=session_id,
            direction_bias=DirectionBias(direction_bias),
            levels_yaml=levels_yaml,
            trades_path=trades_path,
            quotes_path=quotes_path,
            fmt=fmt,  # type: ignore[arg-type]
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            commands_csv=commands_csv,
        )
    )
    # Persist a stable report.json so later comparisons don’t depend on re-computing.
    write_session_report(root_dir=Path(cfg.storage.root_dir), ticker=ticker, session_id=session_id)


@main.command("polygon-export")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("config/default.yaml"))
@click.option("--ticker", type=str, required=True)
@click.option("--start", "start_dt", type=str, required=True, help="Start datetime (ISO), interpreted in exchange TZ if no offset.")
@click.option("--end", "end_dt", type=str, required=True, help="End datetime (ISO), interpreted in exchange TZ if no offset.")
@click.option("--out-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True)
def polygon_export_cmd(config_path: Path, ticker: str, start_dt: str, end_dt: str, out_dir: Path) -> None:
    """Fetch Polygon REST trades/quotes for a window and write replay parquet files."""
    from godmode.core.config import AppConfig
    from godmode.orchestrator.polygon_export import export_polygon_to_replay, parse_dt_to_ts_ms

    cfg = AppConfig.load(config_path)
    start_ms = parse_dt_to_ts_ms(start_dt, tz=cfg.session.exchange_timezone)
    end_ms = parse_dt_to_ts_ms(end_dt, tz=cfg.session.exchange_timezone)
    if end_ms < start_ms:
        raise click.UsageError("--end must be >= --start")

    async def _run() -> None:
        tpath, qpath = await export_polygon_to_replay(
            config=cfg, ticker=ticker, start_ts_ms=start_ms, end_ts_ms=end_ms, out_dir=out_dir
        )
        click.echo(f"Wrote: {tpath}")
        click.echo(f"Wrote: {qpath}")

    asyncio.run(_run())


@main.command("web")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("config/default.yaml"))
@click.option("--host", type=str, default="127.0.0.1")
@click.option("--port", type=int, default=8000)
def web_cmd(config_path: Path, host: str, port: int) -> None:
    """Run the GodMode web UI (read-only visualization over Parquet outputs)."""
    from godmode.core.config import AppConfig
    from godmode.webapp.app import create_app

    cfg = AppConfig.load(config_path)
    app = create_app(config=cfg)

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()



