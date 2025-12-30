"""In-memory job queue for V1 (per Addendum J)."""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import json

from godmode.core.config import AppConfig
from godmode.orchestrator.alpaca_export import export_alpaca_to_replay
from godmode.orchestrator.session import run_replay_session
from godmode.reports.writer import write_session_report


@dataclass
class MarkerInput:
    """Single marker from form input."""
    start_ts_ms: int
    marker_type: str
    end_ts_ms: Optional[int] = None  # Optional for single-moment markers
    # Optional support/resistance price tags (emitted as add_level commands)
    price_tags: list[dict[str, Any]] = field(default_factory=list)  # [{price: float, kind: str|None}]
    # Extra marker metadata (will be persisted into marker notes JSON for reports)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """Single job record (Addendum J3)."""

    job_id: str
    ticker: str
    session_id: str
    status: str  # pending | running | done | error
    created_at: int  # ms since epoch
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    error_message: Optional[str] = None

    # Input params (for auditability)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    direction_bias: str = "long"
    markers: list[MarkerInput] = field(default_factory=list)

    # Progress tracking
    current_step: str = ""
    steps_completed: list[str] = field(default_factory=list)
    trade_count: int = 0
    quote_count: int = 0
    first_price: Optional[float] = None
    last_price: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ticker": self.ticker,
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_message": self.error_message,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "direction_bias": self.direction_bias,
            "marker_count": len(self.markers),
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "trade_count": self.trade_count,
            "quote_count": self.quote_count,
            "first_price": self.first_price,
            "last_price": self.last_price,
        }


class JobManager:
    """In-memory job manager (Addendum J7)."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._config = config or AppConfig()
        self._lock = asyncio.Lock()

    def list_jobs(self, *, status: Optional[str] = None) -> list[dict[str, Any]]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        j = self._jobs.get(job_id)
        return j.to_dict() if j else None

    async def submit_record_request(
        self,
        *,
        tickers: list[str],
        start_ms: int,
        end_ms: int,
        direction_bias: str,
        markers: list[MarkerInput],
        session_id_override: Optional[str] = None,
    ) -> list[str]:
        """Create jobs for each ticker and start execution in background.
        
        Args:
            session_id_override: If provided, use this session_id instead of auto-generating.
                               Only valid when len(tickers) == 1.
        """
        job_ids: list[str] = []
        now_ms = int(time.time() * 1000)

        for ticker in tickers:
            job_id = str(uuid.uuid4())
            if session_id_override and len(tickers) == 1:
                session_id = session_id_override
            else:
                # Generate base session_id from start time
                base_session_id = f"{ticker}_{datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                session_id = base_session_id
                
                # Check for existing sessions with same ID and add suffix if needed
                from pathlib import Path
                storage_root = Path(self._config.storage.root_dir)
                date_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                session_stream_dir = storage_root / "session_stream" / f"date={date_str}" / f"ticker={ticker.upper()}"
                
                if session_stream_dir.exists():
                    existing = [d.name for d in session_stream_dir.glob(f"session={base_session_id}*")]
                    if existing:
                        # Find next available suffix
                        suffix = 1
                        while f"session={base_session_id}_{suffix}" in existing:
                            suffix += 1
                        session_id = f"{base_session_id}_{suffix}"

            job = Job(
                job_id=job_id,
                ticker=ticker.upper().strip(),
                session_id=session_id,
                status="pending",
                created_at=now_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                direction_bias=direction_bias,
                markers=markers.copy(),
            )
            self._jobs[job_id] = job
            job_ids.append(job_id)

            # Start job in background
            asyncio.create_task(self._run_job(job))

        return job_ids

    def _update_step(self, job: Job, step: str) -> None:
        """Update job's current step."""
        if job.current_step:
            job.steps_completed.append(job.current_step)
        job.current_step = step

    async def _run_job(self, job: Job) -> None:
        """Execute a single job: Alpaca export → replay run."""
        async with self._lock:
            job.status = "running"
            job.started_at = int(time.time() * 1000)

        try:
            # Step 1: Export from Alpaca
            self._update_step(job, "Fetching trades from Alpaca...")
            
            out_dir = Path(
                self._config.storage.root_dir
            ) / "replay" / f"{job.ticker}_{job.session_id}"

            trades_path, quotes_path = await export_alpaca_to_replay(
                config=self._config,
                ticker=job.ticker,
                start_ts_ms=job.start_ms or 0,
                end_ts_ms=job.end_ms or 0,
                out_dir=out_dir,
            )

            # Read back trade/quote counts and price preview
            self._update_step(job, "Reading data stats...")
            import pandas as pd
            
            try:
                tdf = pd.read_parquet(trades_path)
                job.trade_count = len(tdf)
                if not tdf.empty and "price" in tdf.columns:
                    job.first_price = float(tdf["price"].iloc[0])
                    job.last_price = float(tdf["price"].iloc[-1])
            except Exception:
                pass

            try:
                qdf = pd.read_parquet(quotes_path)
                job.quote_count = len(qdf)
            except Exception:
                pass

            self._update_step(job, f"Fetched {job.trade_count:,} trades, {job.quote_count:,} quotes")

            # Step 2: Build commands CSV for markers
            commands_csv_path: Optional[Path] = None
            if job.markers:
                import csv

                self._update_step(job, f"Writing {len(job.markers)} marker(s) + levels...")
                commands_csv_path = out_dir / "commands.csv"

                # IMPORTANT: must match orchestrator.load_commands_csv schema:
                # required columns: ts_ms,ticker,type
                # add_level: level_price, level_type, level_width_atr(optional), level_id(optional), notes(optional)
                # add_marker: marker_type, marker_id(optional), direction_bias(optional), notes(optional)
                header = [
                    "ts_ms",
                    "ticker",
                    "type",
                    "marker_type",
                    "marker_id",
                    "direction_bias",
                    "notes",
                    "level_price",
                    "level_type",
                    "level_width_atr",
                    "level_id",
                ]

                def _marker_note(marker: MarkerInput) -> str:
                    payload: dict[str, Any] = {
                        "end_ts_ms": marker.end_ts_ms,
                        "price_tags": marker.price_tags,
                    }
                    if marker.notes:
                        # Merge extra metadata (e.g., per-level outcomes) in a backward-compatible way.
                        for k, v in marker.notes.items():
                            payload[k] = v
                    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

                with open(commands_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()

                    # Emit add_level rows first at the marker start time so they become active immediately.
                    for marker in job.markers:
                        for tag in marker.price_tags:
                            price = float(tag["price"])
                            kind = tag.get("kind") or "manual"
                            level_type = str(kind)
                            level_id = f"marker_level:{job.ticker}:{marker.start_ts_ms}:{price}:{level_type}"
                            writer.writerow(
                                {
                                    "ts_ms": int(marker.start_ts_ms),
                                    "ticker": job.ticker,
                                    "type": "add_level",
                                    "marker_type": "",
                                    "marker_id": "",
                                    "direction_bias": "",
                                    "notes": f"from_marker:{marker.marker_type}",
                                    "level_price": price,
                                    "level_type": level_type,
                                    "level_width_atr": 0.25,
                                    "level_id": level_id,
                                }
                            )

                    # Emit add_marker rows
                    for marker in job.markers:
                        marker_id = f"marker:{job.ticker}:{marker.start_ts_ms}:{marker.marker_type}"
                        skip_episode = bool((marker.notes or {}).get("skip_episode", False))
                        writer.writerow(
                            {
                                "ts_ms": int(marker.start_ts_ms),
                                "ticker": job.ticker,
                                "type": "add_marker",
                                "marker_type": marker.marker_type,
                                "marker_id": marker_id,
                                # Leaving direction_bias blank prevents marker-based episode extraction
                                # for marker types that do not have a deterministic inferred bias.
                                "direction_bias": ("" if skip_episode else job.direction_bias),
                                "notes": _marker_note(marker),
                                "level_price": "",
                                "level_type": "",
                                "level_width_atr": "",
                                "level_id": "",
                            }
                        )

            # Step 3: Run replay
            self._update_step(job, "Running pipeline (bars → features → snapshots)...")
            
            await run_replay_session(
                ticker=job.ticker,
                session_id=job.session_id,
                direction_bias=job.direction_bias,
                levels_yaml=None,
                trades_path=trades_path,
                quotes_path=quotes_path,
                commands_csv=commands_csv_path,
                config=self._config,
            )

            self._update_step(job, "Writing report...")
            # Persist a stable report.json so sessions can be compared later.
            try:
                write_session_report(
                    root_dir=Path(self._config.storage.root_dir),
                    ticker=job.ticker,
                    session_id=job.session_id,
                )
            except Exception:
                # Report generation should not fail the job; raw parquet is still written.
                pass

            job.status = "done"
            job.finished_at = int(time.time() * 1000)
            job.current_step = "Complete!"
            job.steps_completed.append("Complete!")

        except Exception as e:
            job.status = "error"
            job.finished_at = int(time.time() * 1000)
            job.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            job.current_step = f"Error: {type(e).__name__}"
