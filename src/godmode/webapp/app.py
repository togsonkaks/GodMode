"""FastAPI web application for GodMode visualization + orchestration."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from godmode.core.config import AppConfig
from godmode.webapp.jobs import JobManager, MarkerInput
from godmode.webapp.store import Store
from godmode.reports.paths import ReportKey, report_path
from godmode.reports.reader import load_report_json, summarize_report_brief
from godmode.reports.writer import write_session_report


def _alpaca_feed() -> str:
    """Choose Alpaca feed for REST endpoints.

    Default to IEX unless explicitly set. Many Alpaca accounts are IEX-only; hardcoding
    SIP causes 403s and makes live pages (and derived summaries) appear broken.
    """
    import os

    f = (os.environ.get("ALPACA_FEED") or "").strip().lower()
    return f if f in {"iex", "sip"} else "iex"


def _parse_local_datetime_to_utc_ms(dt_str: str, tz_name: str) -> int:
    """Parse a local datetime string and convert to UTC milliseconds."""
    # datetime-local format: "2025-12-23T09:30"
    dt_naive = datetime.fromisoformat(dt_str)
    tz = ZoneInfo(tz_name)
    dt_local = dt_naive.replace(tzinfo=tz)
    return int(dt_local.timestamp() * 1000)


def create_app(*, config: Optional[AppConfig] = None) -> FastAPI:
    cfg = config or AppConfig()
    store = Store(root_dir=Path(cfg.storage.root_dir))
    job_manager = JobManager(config=cfg)

    app = FastAPI(title="GodMode UI", version="0.1.0")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    # Custom Jinja filter for Chicago timezone
    from datetime import datetime, timezone as dt_timezone
    try:
        from zoneinfo import ZoneInfo
        CHICAGO_TZ = ZoneInfo("America/Chicago")
    except ImportError:
        from datetime import timedelta
        CHICAGO_TZ = dt_timezone(timedelta(hours=-6))

    def ts_to_chicago(ts_ms: int | float | None) -> str:
        """Convert milliseconds timestamp to Chicago time string."""
        if ts_ms is None:
            return "—"
        try:
            dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=dt_timezone.utc)
            dt_local = dt.astimezone(CHICAGO_TZ)
            return dt_local.strftime('%H:%M:%S')
        except Exception:
            return str(ts_ms)

    templates.env.filters["chicago"] = ts_to_chicago

    # ============ HTML PAGES ============

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        sessions = store.list_sessions()
        root_dir = Path(cfg.storage.root_dir)
        for s in sessions:
            key = ReportKey(date=s["date"], ticker=s["ticker"], session_id=s["session_id"])
            rep = load_report_json(root_dir=root_dir, key=key)
            if rep is not None:
                s["brief"] = summarize_report_brief(rep)
                s["has_report"] = True
            else:
                s["brief"] = None
                s["has_report"] = False
        return templates.TemplateResponse(
            name="home.html",
            context={
                "request": request,
                "sessions": sessions,
                "root_dir": str(cfg.storage.root_dir),
            },
        )

    @app.get("/record", response_class=HTMLResponse)
    def record_page(request: Request) -> HTMLResponse:
        """Record form page (Addendum J1)."""
        return templates.TemplateResponse(name="record.html", context={"request": request})

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request) -> HTMLResponse:
        """Jobs status page with polling (Addendum J5)."""
        jobs = job_manager.list_jobs()
        return templates.TemplateResponse(name="jobs.html", context={"request": request, "jobs": jobs})

    @app.get("/live", response_class=HTMLResponse)
    def live_dashboard(request: Request) -> HTMLResponse:
        """Live dashboard - multi-ticker real-time monitoring."""
        return templates.TemplateResponse(name="live_dashboard.html", context={"request": request})

    @app.get("/zones", response_class=HTMLResponse)
    def zones_page(request: Request) -> HTMLResponse:
        """Zone picker + analysis page (single-zone v1)."""
        return templates.TemplateResponse(name="zones.html", context={"request": request})

    @app.get("/scanner", response_class=HTMLResponse)
    def scanner_page(request: Request) -> HTMLResponse:
        """Downtrend-break scanner page."""
        return templates.TemplateResponse(
            name="scanner.html",
            context={"request": request, "refresh_seconds": 5, "top_n": 12},
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail_page(request: Request, job_id: str) -> HTMLResponse:
        """Single job detail page - shows job status and redirects when complete."""
        job = job_manager.get_job(job_id)
        if job is None:
            # Job not found - maybe it completed and was cleaned up, go to jobs list
            return RedirectResponse(url="/jobs", status_code=302)
        
        jobs = [job]  # Show just this job
        return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs, "highlight_job": job_id})

    @app.get("/sessions/{date}/{ticker}/{session_id}", response_class=HTMLResponse)
    def session_detail(request: Request, date: str, ticker: str, session_id: str) -> HTMLResponse:
        eps = store.read_episodes(date=date, ticker=ticker, session_id=session_id)
        markers = store.read_markers(date=date, ticker=ticker, session_id=session_id)
        return templates.TemplateResponse(
            "session.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "episodes": eps.to_dict(orient="records"),
                "markers": markers.to_dict(orient="records"),
            },
        )

    @app.get("/sessions/{date}/{ticker}/{session_id}/report", response_class=HTMLResponse)
    def session_report(request: Request, date: str, ticker: str, session_id: str) -> HTMLResponse:
        # For LIVE sessions, always compute fresh (no cache)
        is_live = "_LIVE" in session_id
        
        if is_live:
            # Always compute fresh for live sessions
            report = store.compute_session_report(date=date, ticker=ticker, session_id=session_id)
        else:
            # Prefer persisted report.json (stable for later comparisons). Fallback to compute on the fly.
            rp = report_path(root_dir=Path(cfg.storage.root_dir), key=ReportKey(date=date, ticker=ticker, session_id=session_id))
            if rp.exists():
                import json
                report = json.loads(rp.read_text(encoding="utf-8"))
            else:
                report = store.compute_session_report(date=date, ticker=ticker, session_id=session_id)
                # Best-effort persist for consistency.
                try:
                    write_session_report(root_dir=Path(cfg.storage.root_dir), ticker=ticker, session_id=session_id)
                except Exception:
                    pass
        
        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "report": report,
                "now": int(time.time() * 1000),  # For live mode timestamp display
            },
        )

    @app.get("/sessions/{date}/{ticker}/{session_id}/live", response_class=HTMLResponse)
    def session_live(request: Request, date: str, ticker: str, session_id: str) -> HTMLResponse:
        """Simplified live view - just state panel and dot chart."""
        # Always compute fresh for live view
        report = store.compute_session_report(date=date, ticker=ticker, session_id=session_id)
        
        # Extract the first level data for display
        level_data = None
        level_price = 0.0
        level_kind = "support"
        
        for chain in report.get("level_chains", []):
            for lv in chain.get("levels", []):
                level_data = lv
                level_price = lv.get("level_price", 0)
                level_kind = lv.get("level_kind", "support")
                break
            if level_data:
                break
        
        return templates.TemplateResponse(
            "live.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "level_data": level_data,
                "level_price": level_price,
                "level_kind": level_kind,
                "now": int(time.time() * 1000),
            },
        )

    @app.get("/reports/backfill")
    def reports_backfill(force: bool = False) -> RedirectResponse:
        """
        One-click backfill: generate missing report.json files for existing sessions
        (useful for sessions created before report persistence was added).
        """
        root_dir = Path(cfg.storage.root_dir)
        sessions = store.list_sessions()
        for s in sessions:
            key = ReportKey(date=s["date"], ticker=s["ticker"], session_id=s["session_id"])
            if (not force) and report_path(root_dir=root_dir, key=key).exists():
                continue
            try:
                write_session_report(root_dir=root_dir, ticker=s["ticker"], session_id=s["session_id"])
            except Exception:
                continue
        return RedirectResponse(url="/", status_code=303)

    @app.get("/session/{date}/{ticker}/{session_id}/edit", response_class=HTMLResponse)
    def session_edit(request: Request, date: str, ticker: str, session_id: str) -> HTMLResponse:
        """
        Show the record form pre-populated with existing session data for editing.
        Saving will overwrite the existing session.
        """
        # Get existing report to extract level info
        rp = report_path(root_dir=Path(cfg.storage.root_dir), key=ReportKey(date=date, ticker=ticker, session_id=session_id))
        report_data = {}
        if rp.exists():
            import json
            report_data = json.loads(rp.read_text(encoding="utf-8"))
        
        # Extract level chain info for pre-population
        levels = []
        level_chains = report_data.get("level_chains", [])
        if level_chains:
            chain = level_chains[0]
            for lv in chain.get("levels", []):
                levels.append({
                    "price": lv.get("level_price"),
                    "kind": lv.get("level_kind", "support"),
                    "outcome": lv.get("level_outcome", ""),
                    "start_ts_ms": lv.get("watch_start_ts_ms"),
                    "end_ts_ms": lv.get("watch_end_ts_ms"),
                })
        
        return templates.TemplateResponse(
            "edit_session.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "levels": levels,
                "direction_bias": report_data.get("direction_bias", "long"),
                "mode": "edit",
            },
        )

    @app.get("/session/{date}/{ticker}/{session_id}/duplicate", response_class=HTMLResponse)
    def session_duplicate(request: Request, date: str, ticker: str, session_id: str) -> HTMLResponse:
        """
        Show the record form pre-populated with existing session data for duplication.
        Saving will create a new session (original untouched).
        """
        # Get existing report to extract level info
        rp = report_path(root_dir=Path(cfg.storage.root_dir), key=ReportKey(date=date, ticker=ticker, session_id=session_id))
        report_data = {}
        if rp.exists():
            import json
            report_data = json.loads(rp.read_text(encoding="utf-8"))
        
        # Extract level chain info for pre-population
        levels = []
        level_chains = report_data.get("level_chains", [])
        if level_chains:
            chain = level_chains[0]
            for lv in chain.get("levels", []):
                levels.append({
                    "price": lv.get("level_price"),
                    "kind": lv.get("level_kind", "support"),
                    "outcome": lv.get("level_outcome", ""),
                    "start_ts_ms": lv.get("watch_start_ts_ms"),
                    "end_ts_ms": lv.get("watch_end_ts_ms"),
                })
        
        return templates.TemplateResponse(
            "edit_session.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "levels": levels,
                "direction_bias": report_data.get("direction_bias", "long"),
                "mode": "duplicate",
            },
        )

    @app.post("/session/{date}/{ticker}/{session_id}/edit")
    async def session_edit_submit(
        request: Request,
        date: str,
        ticker: str,
        session_id: str,
    ):
        """
        Process edit form submission. Overwrites existing session with new times.
        """
        form_data = await request.form()
        timezone_name = form_data.get("timezone", "America/Chicago")
        direction_bias = form_data.get("direction_bias", "long")
        level_count = int(form_data.get("level_count", 0))
        
        # Collect level data
        levels_data = []
        for i in range(1, level_count + 1):
            price = form_data.get(f"level_price_{i}")
            kind = form_data.get(f"level_kind_{i}", "support")
            outcome = form_data.get(f"level_outcome_{i}", "")
            start_raw = form_data.get(f"level_start_{i}")
            end_raw = form_data.get(f"level_end_{i}")
            
            if price and start_raw and end_raw:
                start_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                end_ms = _parse_local_datetime_to_utc_ms(str(end_raw), str(timezone_name))
                levels_data.append({
                    "price": float(price),
                    "kind": kind,
                    "outcome": outcome,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })
        
        if not levels_data:
            raise HTTPException(
                status_code=400, 
                detail=f"At least one level with price+start+end times is required. Got level_count={level_count}. Make sure start/end times are filled in."
            )
        
        # Delete existing session data
        import shutil
        root = Path(cfg.storage.root_dir)
        
        try:
            # Remove session_stream
            stream_path = root / "session_stream" / f"date={date}" / f"ticker={ticker}" / f"session={session_id}"
            if stream_path.exists():
                shutil.rmtree(stream_path)
            
            # Remove markers
            markers_path = root / "markers" / f"date={date}" / f"ticker={ticker}"
            if markers_path.exists():
                for f in markers_path.glob("*.parquet"):
                    try:
                        import pyarrow.parquet as pq
                        import pyarrow as pa
                        t = pq.read_table(f)
                        df = t.to_pandas()
                        df = df[df["session_id"] != session_id]
                        if len(df) > 0:
                            pq.write_table(pa.Table.from_pandas(df), f)
                        else:
                            f.unlink()
                    except Exception:
                        pass  # Continue even if one marker file fails
            
            # Remove old report
            report_p = root / "reports" / f"date={date}" / f"ticker={ticker}" / f"session={session_id}"
            if report_p.exists():
                shutil.rmtree(report_p)
        except Exception as e:
            # Log but continue - we can still re-create the session
            print(f"Warning: Error cleaning old session data: {e}")
        
        # Now create new markers and fetch data (same as record)
        # Derive session time range from levels
        all_start_ms = min(lv["start_ms"] for lv in levels_data)
        all_end_ms = max(lv["end_ms"] for lv in levels_data)
        
        # Create job to fetch data
        from godmode.webapp.jobs import MarkerInput
        
        chain_id = f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        markers = []
        for i, lv in enumerate(levels_data):
            markers.append(
                MarkerInput(
                    marker_type="consolidation",
                    start_ts_ms=lv["start_ms"],
                    end_ts_ms=lv["end_ms"],
                    price_tags=[{"price": lv["price"], "kind": lv["kind"]}],
                    notes={
                        "schema": "level_chain_v1",
                        "chain_id": chain_id,
                        "level_price": lv["price"],
                        "level_kind": lv["kind"],
                        "watch_start_ts_ms": lv["start_ms"],
                        "watch_end_ts_ms": lv["end_ms"],
                        "direction_bias": direction_bias,
                        "level_index": i,
                        "level_outcome": lv["outcome"] or None,
                        "skip_episode": False,
                    },
                )
            )
        
        job_ids = await job_manager.submit_record_request(
            tickers=[ticker],
            start_ms=all_start_ms,
            end_ms=all_end_ms,
            direction_bias=direction_bias,
            markers=markers,
            session_id_override=session_id,  # Keep same session ID
        )
        
        return RedirectResponse(url=f"/jobs/{job_ids[0]}", status_code=303)

    @app.post("/session/{date}/{ticker}/{session_id}/duplicate")
    async def session_duplicate_submit(
        request: Request,
        date: str,
        ticker: str,
        session_id: str,
    ):
        """
        Process duplicate form submission. Creates a new session (original untouched).
        """
        form_data = await request.form()
        timezone_name = form_data.get("timezone", "America/Chicago")
        direction_bias = form_data.get("direction_bias", "long")
        level_count = int(form_data.get("level_count", 0))
        
        # Collect level data
        levels_data = []
        for i in range(1, level_count + 1):
            price = form_data.get(f"level_price_{i}")
            kind = form_data.get(f"level_kind_{i}", "support")
            outcome = form_data.get(f"level_outcome_{i}", "")
            start_raw = form_data.get(f"level_start_{i}")
            end_raw = form_data.get(f"level_end_{i}")
            
            if price and start_raw and end_raw:
                start_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                end_ms = _parse_local_datetime_to_utc_ms(str(end_raw), str(timezone_name))
                levels_data.append({
                    "price": float(price),
                    "kind": kind,
                    "outcome": outcome,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })
        
        if not levels_data:
            raise HTTPException(
                status_code=400, 
                detail=f"At least one level with price+start+end times is required. Got level_count={level_count}. Make sure start/end times are filled in."
            )
        
        # Derive session time range from levels
        all_start_ms = min(lv["start_ms"] for lv in levels_data)
        all_end_ms = max(lv["end_ms"] for lv in levels_data)
        
        # Create job to fetch data (new session ID will be auto-generated based on times)
        from godmode.webapp.jobs import MarkerInput
        
        chain_id = f"dup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        markers = []
        for i, lv in enumerate(levels_data):
            markers.append(
                MarkerInput(
                    marker_type="consolidation",
                    start_ts_ms=lv["start_ms"],
                    end_ts_ms=lv["end_ms"],
                    price_tags=[{"price": lv["price"], "kind": lv["kind"]}],
                    notes={
                        "schema": "level_chain_v1",
                        "chain_id": chain_id,
                        "level_price": lv["price"],
                        "level_kind": lv["kind"],
                        "watch_start_ts_ms": lv["start_ms"],
                        "watch_end_ts_ms": lv["end_ms"],
                        "direction_bias": direction_bias,
                        "level_index": i,
                        "level_outcome": lv["outcome"] or None,
                        "skip_episode": False,
                    },
                )
            )
        
        # For duplicate, let it auto-generate a new session ID based on the new times
        job_ids = await job_manager.submit_record_request(
            tickers=[ticker],
            start_ms=all_start_ms,
            end_ms=all_end_ms,
            direction_bias=direction_bias,
            markers=markers,
            # No session_id_override - new ID will be generated
        )
        
        return RedirectResponse(url=f"/jobs/{job_ids[0]}", status_code=303)

    @app.get("/api/reports")
    def api_reports() -> JSONResponse:
        """
        Return all persisted report.json objects (stable schema) so clients can aggregate without
        touching raw Parquet (avoids schema drift issues).
        """
        from godmode.reports.reader import list_report_paths
        import json

        root_dir = Path(cfg.storage.root_dir)
        out: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return JSONResponse(out)

    @app.get("/api/reports.csv")
    def api_reports_csv() -> HTMLResponse:
        """
        Export a compact, stable CSV across all sessions based on report.json.
        This is the “compile all sessions” hook.
        """
        from godmode.reports.reader import list_report_paths, summarize_report_brief
        import csv
        import io
        import json

        root_dir = Path(cfg.storage.root_dir)
        rows: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = rep.get("_meta") or {}
            brief = summarize_report_brief(rep)
            rows.append(
                {
                    "date": meta.get("date") or rep.get("date"),
                    "ticker": meta.get("ticker") or rep.get("ticker"),
                    "session_id": meta.get("session_id") or rep.get("session_id"),
                    **brief,
                }
            )
        rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("ticker", "")), str(r.get("session_id", ""))))

        buf = io.StringIO()
        fieldnames = [
            "date",
            "ticker",
            "session_id",
            "episodes_total",
            "win",
            "loss",
            "scratch",
            "win_rate",
            "mfe_p50",
            "mae_p90",
            "top_feature",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/api/watch_windows.csv")
    def api_watch_windows_csv() -> HTMLResponse:
        """
        Export one row per watch-window marker (start+end) using persisted report.json.
        This is the main “compile across sessions” artifact for pre-move analysis.
        """
        import csv
        import io
        import json

        from godmode.reports.reader import list_report_paths

        root_dir = Path(cfg.storage.root_dir)
        rows: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = rep.get("_meta") or {}
            date = meta.get("date") or rep.get("date")
            ticker = meta.get("ticker") or rep.get("ticker")
            session_id = meta.get("session_id") or rep.get("session_id")
            for w in rep.get("watch_windows", []) or []:
                for lv in (w.get("levels", []) or []):
                    between = lv.get("between") or {}
                    checklist = lv.get("checklist") or {}
                    ctx = checklist.get("context") or {}
                    trig = checklist.get("trigger") or {}
                    rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "session_id": session_id,
                            "marker_type": w.get("marker_type"),
                            "watch_start_ts_ms": w.get("watch_start_ts_ms"),
                            "watch_end_ts_ms": w.get("watch_end_ts_ms"),
                            "level_kind": lv.get("level_kind"),
                            "level_price": lv.get("level_price"),
                            "band_pct": w.get("band_pct"),
                            "between_delta_sum": between.get("delta_sum"),
                            "between_band_delta_sum": between.get("band_delta_sum"),
                            "between_time_in_band_s": between.get("time_in_band_s"),
                            "between_touch_count": between.get("touch_count"),
                            "between_reclaim_attempts": between.get("reclaim_attempts"),
                            "between_reclaim_hold_rate": between.get("reclaim_hold_rate"),
                            "between_flag_score": (between.get("flags") or {}).get("score") if isinstance(between.get("flags"), dict) else None,
                            "ctx_compression_level": ctx.get("compression_level"),
                            "ctx_compression_trend": ctx.get("compression_trend_into_T"),
                            "trig_reaction": trig.get("reaction"),
                            "trig_band_delta_last30s": trig.get("band_delta_last30s"),
                            "trig_rel_aggr_last30s": trig.get("rel_aggr_last30s"),
                        }
                    )

        rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("ticker", "")), str(r.get("session_id", "")), str(r.get("watch_start_ts_ms", ""))))

        buf = io.StringIO()
        fieldnames = [
            "date",
            "ticker",
            "session_id",
            "marker_type",
            "watch_start_ts_ms",
            "watch_end_ts_ms",
            "level_kind",
            "level_price",
            "band_pct",
            "between_delta_sum",
            "between_band_delta_sum",
            "between_time_in_band_s",
            "between_touch_count",
            "between_reclaim_attempts",
            "between_reclaim_hold_rate",
            "between_flag_score",
            "ctx_compression_level",
            "ctx_compression_trend",
            "trig_reaction",
            "trig_band_delta_last30s",
            "trig_rel_aggr_last30s",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/api/level_chains.csv")
    def api_level_chains_csv() -> HTMLResponse:
        """
        Export one row per level in the new per-level workflow.
        Includes the automatically computed move segment (end_i → start_{i+1}) as move_from_prev.
        """
        import csv
        import io
        import json

        from godmode.reports.reader import list_report_paths

        root_dir = Path(cfg.storage.root_dir)
        rows: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = rep.get("_meta") or {}
            date = meta.get("date") or rep.get("date")
            ticker = meta.get("ticker") or rep.get("ticker")
            session_id = meta.get("session_id") or rep.get("session_id")

            for ch in rep.get("level_chains", []) or []:
                cid = ch.get("chain_id") or "default"
                for lv in ch.get("levels", []) or []:
                    checklist = lv.get("checklist") or {}
                    ctx = checklist.get("context") or {}
                    trig = checklist.get("trigger") or {}
                    mv = lv.get("move_from_prev") or {}
                    sw = lv.get("session_wide") or {}
                    ts = lv.get("timeline_summary") or {}
                    rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "session_id": session_id,
                            "chain_id": cid,
                            "level_index": lv.get("level_index"),
                            "level_kind": lv.get("level_kind"),
                            "level_price": lv.get("level_price"),
                            "level_outcome": lv.get("level_outcome"),
                            "watch_start_ts_ms": lv.get("watch_start_ts_ms"),
                            "watch_end_ts_ms": lv.get("watch_end_ts_ms"),
                            "ctx_compression_level": ctx.get("compression_level"),
                            "ctx_compression_trend": ctx.get("compression_trend_into_T"),
                            "ctx_compression_mean": ctx.get("compression_mean"),
                            "trig_reaction": trig.get("reaction"),
                            "trig_band_delta_last30s": trig.get("band_delta_last30s"),
                            "trig_rel_aggr_last30s": trig.get("rel_aggr_last30s"),
                            "move_velocity_bucket": mv.get("velocity_bucket"),
                            "move_direction": mv.get("direction"),
                            "move_flow_side": mv.get("flow_side"),
                            "move_flow_strength": mv.get("flow_strength"),
                            "move_flow_imbalance_ratio": mv.get("flow_imbalance_ratio"),
                            "move_verdict": mv.get("verdict"),
                            "move_duration_s": mv.get("duration_s"),
                            "move_return_pct": mv.get("return_pct"),
                            "move_return_abs": mv.get("return_abs"),
                            "move_delta_sum": mv.get("delta_sum"),
                            "move_total_volume_sum": mv.get("total_volume_sum"),
                            "move_abs_return_atr": mv.get("abs_return_atr"),
                            "move_abs_return_atr_per_min": mv.get("abs_return_atr_per_min"),
                            "move_abs_return_pct": mv.get("abs_return_pct"),
                            "move_abs_return_pct_per_min": mv.get("abs_return_pct_per_min"),
                            # Session-wide continuous tracking
                            "session_touches": ts.get("touches"),
                            "session_breaks": ts.get("breaks"),
                            "session_rejects": ts.get("rejects"),
                            "session_reclaim_attempts": ts.get("reclaim_attempts"),
                            "session_cross_throughs": ts.get("cross_throughs"),
                            "session_total_events": ts.get("total_events"),
                            "session_band_delta_sum": sw.get("band_delta_sum"),
                            "session_time_in_band_s": sw.get("time_in_band_s"),
                            "session_touch_count": sw.get("touch_count"),
                            "session_cross_count": sw.get("cross_count"),
                            "session_reclaim_hold_rate": sw.get("reclaim_hold_rate"),
                            "role_flip": lv.get("role_flip"),
                        }
                    )

        rows.sort(
            key=lambda r: (
                str(r.get("date", "")),
                str(r.get("ticker", "")),
                str(r.get("session_id", "")),
                str(r.get("chain_id", "")),
                int(r.get("level_index") or 0),
                int(r.get("watch_start_ts_ms") or 0),
            )
        )

        buf = io.StringIO()
        fieldnames = [
            "date",
            "ticker",
            "session_id",
            "chain_id",
            "level_index",
            "level_kind",
            "level_price",
            "level_outcome",
            "watch_start_ts_ms",
            "watch_end_ts_ms",
            "ctx_compression_level",
            "ctx_compression_trend",
            "ctx_compression_mean",
            "trig_reaction",
            "trig_band_delta_last30s",
            "trig_rel_aggr_last30s",
            "move_velocity_bucket",
            "move_direction",
            "move_flow_side",
            "move_flow_strength",
            "move_flow_imbalance_ratio",
            "move_verdict",
            "move_duration_s",
            "move_return_pct",
            "move_return_abs",
            "move_delta_sum",
            "move_total_volume_sum",
            "move_abs_return_atr",
            "move_abs_return_atr_per_min",
            "move_abs_return_pct",
            "move_abs_return_pct_per_min",
            # Session-wide continuous tracking
            "session_touches",
            "session_breaks",
            "session_rejects",
            "session_reclaim_attempts",
            "session_cross_throughs",
            "session_total_events",
            "session_band_delta_sum",
            "session_time_in_band_s",
            "session_touch_count",
            "session_cross_count",
            "session_reclaim_hold_rate",
            "role_flip",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/api/touch_packets.csv")
    def api_touch_packets_csv() -> HTMLResponse:
        """
        Export one row per Touch Packet across all sessions (Addendum K.2).
        """
        import csv
        import io
        import json

        from godmode.reports.reader import list_report_paths

        root_dir = Path(cfg.storage.root_dir)
        rows: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = rep.get("_meta") or {}
            date = meta.get("date") or rep.get("date")
            ticker = meta.get("ticker") or rep.get("ticker")
            session_id = meta.get("session_id") or rep.get("session_id")

            for ch in rep.get("level_chains", []) or []:
                cid = ch.get("chain_id") or "default"
                for pkt in ch.get("touch_packets", []) or []:
                    rows.append({
                        "date": date,
                        "ticker": ticker,
                        "session_id": session_id,
                        "chain_id": cid,
                        **{k: v for k, v in pkt.items()},
                    })

        rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("ticker", "")), str(r.get("touch_ts_ms", 0))))

        buf = io.StringIO()
        fieldnames = [
            "date", "ticker", "session_id", "chain_id",
            "touch_id", "level_price", "level_kind", "touch_ts_ms", "touch_number",
            "time_since_last_touch_s", "from_side", "reclaim_attempt",
            "approach_velocity_atr_per_min", "approach_type", "approach_delta_60s", "approach_rel_aggr_60s",
            "touch_dwell_s", "bounce_return_30s_pct", "bounce_return_60s_pct",
            "band_delta_0_30s", "band_delta_0_60s", "rel_aggr_0_30s", "rel_aggr_0_60s", "delta_flip_flag",
            "max_penetration_pct", "wick_recovered_flag",
            "touch_volume_z", "large_trade_count_at_touch", "large_trade_buy_ratio",
            "absorption_mean_0_30s", "price_efficiency_0_30s",
            "cvd_60s_at_touch", "cvd_slope_into_touch", "div_flag_at_touch",
            "spread_pct_at_touch", "spread_widening_into_touch", "spread_narrowing_after_touch",
            "compression_at_touch", "compression_trend_into_touch", "price_vs_vwap_at_touch", "ema_stack_at_touch",
            "break_confirmed_30s", "reclaim_after_break", "reclaim_hold_30s", "touch_outcome",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/api/reclaim_packets.csv")
    def api_reclaim_packets_csv() -> HTMLResponse:
        """
        Export one row per Reclaim Packet across all sessions (Addendum K.3).
        """
        import csv
        import io
        import json

        from godmode.reports.reader import list_report_paths

        root_dir = Path(cfg.storage.root_dir)
        rows: list[dict[str, Any]] = []
        for p in list_report_paths(root_dir=root_dir):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = rep.get("_meta") or {}
            date = meta.get("date") or rep.get("date")
            ticker = meta.get("ticker") or rep.get("ticker")
            session_id = meta.get("session_id") or rep.get("session_id")

            for ch in rep.get("level_chains", []) or []:
                cid = ch.get("chain_id") or "default"
                for pkt in ch.get("reclaim_packets", []) or []:
                    rows.append({
                        "date": date,
                        "ticker": ticker,
                        "session_id": session_id,
                        "chain_id": cid,
                        **{k: v for k, v in pkt.items()},
                    })

        rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("ticker", "")), str(r.get("t_reclaim_ms", 0))))

        buf = io.StringIO()
        fieldnames = [
            "date", "ticker", "session_id", "chain_id",
            "reclaim_id", "reclaimed_level_price", "tagged_deeper_level_price",
            "t_tag_deeper_ms", "t_reclaim_ms",
            "time_to_reclaim_ms", "time_to_reclaim_s", "snapback_flag",
            "reclaim_band_delta_30s", "reclaim_band_delta_60s",
            "reclaim_rel_aggr_30s", "reclaim_rel_aggr_60s",
            "reclaim_hold_30s", "reclaim_hold_60s",
            "drop_pct_to_deeper", "drop_atr_to_deeper", "flush_flag",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/api/positive_patterns.csv")
    def api_positive_patterns_csv() -> HTMLResponse:
        """
        Export global positive pattern counts by scenario_key (long/short × win/loss).
        """
        import csv
        import io
        import json

        root_dir = Path(cfg.storage.root_dir)
        p = root_dir / "reports_index" / "positive_patterns.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="positive_patterns.json not found; run scripts/compile_positive_patterns.py")

        d = json.loads(p.read_text(encoding="utf-8"))
        scenarios = d.get("scenarios") or {}

        buf = io.StringIO()
        fieldnames = [
            "scenario_key",
            "positive_touch_count",
            "sessions",
            "tickers",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for sk, v in sorted(scenarios.items()):
            w.writerow(
                {
                    "scenario_key": sk,
                    "positive_touch_count": v.get("positive_touch_count"),
                    "sessions": v.get("sessions"),
                    "tickers": v.get("tickers"),
                }
            )

        return HTMLResponse(content=buf.getvalue(), media_type="text/csv")

    @app.get("/episodes/{date}/{ticker}/{session_id}/{episode_id}", response_class=HTMLResponse)
    def episode_detail(
        request: Request, date: str, ticker: str, session_id: str, episode_id: str
    ) -> HTMLResponse:
        ep = store.read_episode_row(date=date, ticker=ticker, session_id=session_id, episode_id=episode_id)
        snaps = store.read_snapshots_for_episode(date=date, ticker=ticker, session_id=session_id, episode_id=episode_id)
        tf = store.read_tf_indicators_for_episode(date=date, ticker=ticker, session_id=session_id, episode_id=episode_id)
        return templates.TemplateResponse(
            "episode.html",
            {
                "request": request,
                "date": date,
                "ticker": ticker,
                "session_id": session_id,
                "episode": ep,
                "snapshots": snaps.to_dict(orient="records"),
                "tf_rows": tf.to_dict(orient="records"),
            },
        )

    # ============ API: READ-ONLY ============

    @app.get("/api/sessions")
    def api_sessions() -> JSONResponse:
        return JSONResponse(store.list_sessions())

    @app.get("/api/episodes")
    def api_episodes(date: str, ticker: str, session_id: str) -> JSONResponse:
        df = store.read_episodes(date=date, ticker=ticker, session_id=session_id)
        return JSONResponse(df.to_dict(orient="records"))

    @app.get("/api/snapshots")
    def api_snapshots(date: str, ticker: str, session_id: str, episode_id: str) -> JSONResponse:
        df = store.read_snapshots_for_episode(date=date, ticker=ticker, session_id=session_id, episode_id=episode_id)
        return JSONResponse(df.to_dict(orient="records"))

    @app.get("/api/candle")
    async def api_candle(ticker: str, timestamp: str, timezone: str = "America/Chicago") -> JSONResponse:
        """
        Fetch the 1-minute candle for a given ticker and timestamp.
        Used by the record form to verify support/resistance prices.
        
        Args:
            ticker: Stock symbol (e.g., "TSLA")
            timestamp: Local datetime string in format "YYYY-MM-DDTHH:MM" or "YYYY-MM-DDTHH:MM:SS"
            timezone: Timezone name (default: America/Chicago)
            
        Returns:
            JSON with {close, high, low, open, volume, vwap} or {error: message}
        """
        import os
        import httpx
        from datetime import timedelta
        
        try:
            # Parse local time to UTC
            dt_naive = datetime.fromisoformat(timestamp)
            tz = ZoneInfo(timezone)
            dt_local = dt_naive.replace(tzinfo=tz)
            dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
            
            # Format for Alpaca API (RFC3339)
            start_str = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_dt = dt_utc + timedelta(minutes=1)
            end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Get API keys from environment
            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            
            if not api_key or not secret_key:
                return JSONResponse({"error": "Alpaca API keys not configured"})
            
            # Fetch 1-minute bar from Alpaca
            url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/bars"
            params = {
                "start": start_str,
                "end": end_str,
                "timeframe": "1Min",
                "feed": _alpaca_feed(),
                "limit": 1,
            }
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            
            bars = data.get("bars", [])
            if not bars:
                return JSONResponse({"error": f"No data for {ticker} at {timestamp}"})
            
            bar = bars[0]
            return JSONResponse({
                "ticker": ticker.upper(),
                "timestamp": timestamp,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
            })
            
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/chart_candles")
    async def api_chart_candles(
        ticker: str, 
        timeframe: str = "1m",
        timezone: str = "America/Chicago"
    ) -> JSONResponse:
        """
        Fetch 1 week of candle data for charting.
        
        Args:
            ticker: Stock symbol (e.g., "TSLA")
            timeframe: "30s", "1m", "2m", or "3m"
            timezone: Timezone name for display (data is returned as UTC timestamps)
            
        Returns:
            JSON with {candles: [{time, open, high, low, close}, ...]} or {error: message}
        """
        import os
        import httpx
        from datetime import timedelta
        import pandas as pd
        
        try:
            # Get 1 week of data (30s is special-cased below)
            now = datetime.now(ZoneInfo("UTC"))
            end_dt = now
            start_dt = now - timedelta(days=7)
            
            start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Get API keys from environment
            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            
            if not api_key or not secret_key:
                return JSONResponse({"error": "Alpaca API keys not configured"})

            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            }

            # ---------------------------------------------------------
            # 30s candles: Alpaca bars endpoint may not support seconds.
            # Build 30s OHLCV from trades instead.
            # Keep the lookback shorter to avoid huge trade volumes.
            # ---------------------------------------------------------
            if timeframe == "30s":
                start_dt = now - timedelta(hours=12)
                start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                trades_url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/trades"
                trades_params = {
                    "start": start_str,
                    "end": end_str,
                    "limit": 10000,
                    "feed": _alpaca_feed(),
                    "sort": "asc",
                }

                all_trades: list[dict[str, Any]] = []
                next_page_token = None
                max_trades = 200_000  # hard cap to keep response bounded

                async with httpx.AsyncClient(timeout=30.0) as client:
                    while True:
                        if next_page_token:
                            trades_params["page_token"] = next_page_token
                        resp = await client.get(trades_url, params=trades_params, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()
                        trades = data.get("trades", [])
                        if trades:
                            all_trades.extend(trades)
                            if len(all_trades) >= max_trades:
                                all_trades = all_trades[:max_trades]
                                break
                        next_page_token = data.get("next_page_token")
                        if not next_page_token:
                            break

                if not all_trades:
                    return JSONResponse({"error": f"No trades for {ticker} in the last 12 hours"})

                tdf = pd.DataFrame(all_trades)
                if "t" not in tdf.columns or "p" not in tdf.columns:
                    return JSONResponse({"error": f"Unexpected trade schema for {ticker}"})

                tdf["ts_ms"] = pd.to_datetime(tdf["t"]).astype("int64") // 10**6
                tdf["price"] = tdf["p"]
                tdf["size"] = tdf["s"] if "s" in tdf.columns else 0
                tdf = tdf.sort_values("ts_ms").reset_index(drop=True)

                bucket_ms = 30_000
                tdf["bucket"] = (tdf["ts_ms"] // bucket_ms).astype("int64")

                agg = tdf.groupby("bucket").agg(
                    open=("price", "first"),
                    high=("price", "max"),
                    low=("price", "min"),
                    close=("price", "last"),
                    volume=("size", "sum"),
                    start_ts_ms=("ts_ms", "min"),
                ).reset_index()

                candles = []
                for _, row in agg.iterrows():
                    # Lightweight Charts expects seconds; we use candle END time
                    end_sec = int(row["start_ts_ms"] // 1000) + 30
                    candles.append(
                        {
                            "time": end_sec,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        }
                    )

                return JSONResponse(
                    {
                        "ticker": ticker.upper(),
                        "timeframe": timeframe,
                        "candles": candles,
                        "count": len(candles),
                        "_meta": {"source": "trades_resample_30s", "lookback_hours": 12, "trades": len(all_trades)},
                    }
                )

            # ---------------------------------------------------------
            # 1m/2m/3m candles: use Alpaca bars endpoint
            # ---------------------------------------------------------
            tf_map = {
                "1m": "1Min",
                "2m": "2Min",
                "3m": "3Min",
            }
            alpaca_tf = tf_map.get(timeframe, "1Min")

            url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/bars"
            params = {
                "start": start_str,
                "end": end_str,
                "timeframe": alpaca_tf,
                "feed": _alpaca_feed(),
                "limit": 10000,  # Max allowed
            }

            all_bars = []
            next_page_token = None

            async with httpx.AsyncClient(timeout=30.0) as client:
                while True:
                    if next_page_token:
                        params["page_token"] = next_page_token

                    resp = await client.get(url, params=params, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                    bars = data.get("bars", [])
                    all_bars.extend(bars)

                    next_page_token = data.get("next_page_token")
                    if not next_page_token:
                        break

            if not all_bars:
                return JSONResponse({"error": f"No data for {ticker} in the last week"})

            # Convert to Lightweight Charts format (use candle END time)
            tf_seconds = {"1m": 60, "2m": 120, "3m": 180}
            duration_sec = tf_seconds.get(timeframe, 60)

            candles = []
            for bar in all_bars:
                ts_str = bar.get("t", "")
                if ts_str:
                    ts_str = ts_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts_str)
                    unix_ts = int(dt.timestamp()) + duration_sec
                    candles.append(
                        {
                            "time": unix_ts,
                            "open": bar.get("o"),
                            "high": bar.get("h"),
                            "low": bar.get("l"),
                            "close": bar.get("c"),
                            "volume": bar.get("v"),
                        }
                    )

            return JSONResponse({"ticker": ticker.upper(), "timeframe": timeframe, "candles": candles, "count": len(candles)})
            
        except Exception as e:
            return JSONResponse({"error": str(e)})

    # ============ API: JOBS (Addendum J4) ============

    @app.get("/api/jobs")
    def api_jobs(status: Optional[str] = None) -> JSONResponse:
        """List all jobs, optionally filtered by status."""
        return JSONResponse(job_manager.list_jobs(status=status))

    @app.get("/api/jobs/{job_id}")
    def api_job_detail(job_id: str) -> JSONResponse:
        """Get single job status."""
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(job)

    @app.get("/api/jobs/stop")
    def api_stop_live_job(session_id: str) -> RedirectResponse:
        """Stop a live job by session_id and redirect back."""
        # Find the job with this session_id
        jobs = job_manager.list_jobs(status="live")
        for j in jobs:
            if j.get("session_id") == session_id:
                job_manager.stop_live_job(j["job_id"])
                break
        # Redirect to the report page
        # Need to extract date from session_id (format: TICKER_YYYYMMDD_HHMMSS_LIVE)
        parts = session_id.split("_")
        if len(parts) >= 2:
            date_part = parts[1]  # YYYYMMDD
            date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            ticker = parts[0]
            return RedirectResponse(url=f"/sessions/{date_str}/{ticker}/{session_id}/report", status_code=303)
        return RedirectResponse(url="/jobs", status_code=303)

    # ============ API: SCANNER ============
    from godmode.scan.downtrend_break_scanner import DowntrendBreakScanner, ScannerConfig

    _scanner_cfg = ScannerConfig(top_n=12, refresh_seconds=5)
    _scanner = DowntrendBreakScanner(cfg=_scanner_cfg)
    _scanner_cache: dict[str, Any] = {"last_run_ms": 0, "payload": None}
    _scanner_state: dict[str, Any] = {"day": None, "tickers": {}, "alerts": []}

    async def _alpaca_top_gainers(top_n: int) -> list[str]:
        import os
        import httpx

        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not api_secret:
            return []
        url = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        params = {"top": int(top_n)}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        movers = data.get("movers", []) if isinstance(data, dict) else []
        syms: list[str] = []
        for m in movers:
            if not isinstance(m, dict) or not m.get("symbol"):
                continue
            # Prefer gainers only when the payload includes change fields.
            ch = m.get("percent_change", m.get("change", None))
            try:
                if ch is not None and float(ch) <= 0:
                    continue
            except Exception:
                pass
            syms.append(str(m["symbol"]).upper())
        return syms[: int(top_n)]

    def _ct_now_str() -> str:
        from datetime import datetime, timezone
        import pytz

        dt = datetime.now(tz=timezone.utc)
        chicago = pytz.timezone("America/Chicago")
        return dt.astimezone(chicago).strftime("%Y-%m-%d %H:%M:%S CT")

    def _ct_day_key() -> str:
        from datetime import datetime, timezone
        import pytz

        dt = datetime.now(tz=timezone.utc)
        chicago = pytz.timezone("America/Chicago")
        return dt.astimezone(chicago).strftime("%Y-%m-%d")

    def _ct_hm() -> float:
        from datetime import datetime, timezone
        import pytz

        dt = datetime.now(tz=timezone.utc)
        chicago = pytz.timezone("America/Chicago")
        local = dt.astimezone(chicago)
        return float(local.hour) + float(local.minute) / 60.0

    @app.get("/api/scanner")
    async def api_scanner() -> JSONResponse:
        import time

        hm = _ct_hm()
        paused = hm < 3.0 or hm >= 15.0
        now_ms = int(time.time() * 1000)
        last_run = int(_scanner_cache.get("last_run_ms") or 0)
        if _scanner_cache.get("payload") is not None and (now_ms - last_run) < int(
            _scanner_cfg.refresh_seconds * 1000
        ):
            payload = dict(_scanner_cache["payload"])
            payload["paused"] = paused
            payload["now_ct"] = _ct_now_str()
            return JSONResponse(payload)

        day_key = _ct_day_key()
        if _scanner_state.get("day") != day_key:
            _scanner_state["day"] = day_key
            _scanner_state["tickers"] = {}
            _scanner_state["alerts"] = []

        tickers = await _alpaca_top_gainers(_scanner_cfg.top_n)
        results = await _scanner.scan(tickers, now_ms=now_ms) if not paused else []

        tf_order = ["30s", "1m", "2m", "3m", "5m"]
        new_alerts: list[dict[str, Any]] = []

        for r in results:
            t = r.ticker
            st = _scanner_state["tickers"].setdefault(
                t,
                {"cycle": 0, "fired": set(), "noted_align": False, "armed": False, "low_at_cycle": None},
            )

            if st["low_at_cycle"] is not None and r.session_low is not None:
                prev_low = float(st["low_at_cycle"])
                cur_low = float(r.session_low)
                if cur_low < prev_low * (1.0 - float(_scanner_cfg.new_low_realert_min_drop_pct)):
                    st["armed"] = True

            if st["armed"]:
                any_positive = any(s.ema_ok and s.macd_ok for s in (r.signals or []))
                if any_positive:
                    st["cycle"] += 1
                    st["fired"] = set()
                    st["noted_align"] = False
                    st["armed"] = False
                    st["low_at_cycle"] = r.session_low

            if st["low_at_cycle"] is None:
                st["low_at_cycle"] = r.session_low

            sig_by_tf = {s.timeframe: s for s in (r.signals or [])}
            for tf in tf_order:
                s = sig_by_tf.get(tf)
                if s is None:
                    continue
                if not (s.ema_ok and s.macd_ok):
                    continue
                key = f"{tf}:cycle{st['cycle']}"
                if key in st["fired"]:
                    continue
                st["fired"].add(key)
                new_alerts.append(
                    {"ticker": t, "kind": f"{tf} EMA+MACD", "when": _ct_now_str(), "detail": f"MACD={s.macd_tier}"}
                )

            for s in (r.signals or []):
                if not s.ema200_touch_regain:
                    continue
                key = f"ema200:{s.timeframe}:cycle{st['cycle']}"
                if key in st["fired"]:
                    continue
                st["fired"].add(key)
                new_alerts.append(
                    {"ticker": t, "kind": f"{s.timeframe} EMA200 regain", "when": _ct_now_str(), "detail": s.ema200_reason}
                )

            if not st["noted_align"]:
                need = {f"{tf}:cycle{st['cycle']}" for tf in ("30s", "1m", "2m", "3m")}
                if need.issubset(set(st["fired"])):
                    st["noted_align"] = True
                    new_alerts.append(
                        {"ticker": t, "kind": "ALIGN 30s+1m+2m+3m", "when": _ct_now_str(), "detail": "multi-timeframe alignment"}
                    )

        if new_alerts:
            _scanner_state["alerts"] = (new_alerts + _scanner_state["alerts"])[:200]

        payload = {
            "paused": paused,
            "now_ct": _ct_now_str(),
            "results": [
                {**r.__dict__, "signals": [s.__dict__ for s in (r.signals or [])]} for r in results
            ],
            "alerts": _scanner_state["alerts"],
        }
        _scanner_cache["last_run_ms"] = now_ms
        _scanner_cache["payload"] = payload
        return JSONResponse(payload)

    async def _scanner_tick(*, now_ms: int) -> dict[str, Any]:
        day_key = _ct_day_key()
        if _scanner_state.get("day") != day_key:
            _scanner_state["day"] = day_key
            _scanner_state["tickers"] = {}
            _scanner_state["alerts"] = []

        tickers = await _alpaca_top_gainers(_scanner_cfg.top_n)
        results = await _scanner.scan(tickers, now_ms=now_ms)

        tf_order = ["30s", "1m", "2m", "3m", "5m"]
        new_alerts: list[dict[str, Any]] = []

        for r in results:
            t = r.ticker
            st = _scanner_state["tickers"].setdefault(
                t,
                {"cycle": 0, "fired": set(), "noted_align": False, "armed": False, "low_at_cycle": None},
            )

            if st["low_at_cycle"] is not None and r.session_low is not None:
                prev_low = float(st["low_at_cycle"])
                cur_low = float(r.session_low)
                if cur_low < prev_low * (1.0 - float(_scanner_cfg.new_low_realert_min_drop_pct)):
                    st["armed"] = True

            if st["armed"]:
                any_positive = any(s.ema_ok and s.macd_ok for s in (r.signals or []))
                if any_positive:
                    st["cycle"] += 1
                    st["fired"] = set()
                    st["noted_align"] = False
                    st["armed"] = False
                    st["low_at_cycle"] = r.session_low

            if st["low_at_cycle"] is None:
                st["low_at_cycle"] = r.session_low

            sig_by_tf = {s.timeframe: s for s in (r.signals or [])}
            for tf in tf_order:
                s = sig_by_tf.get(tf)
                if s is None or not (s.ema_ok and s.macd_ok):
                    continue
                key = f"{tf}:cycle{st['cycle']}"
                if key in st["fired"]:
                    continue
                st["fired"].add(key)
                new_alerts.append(
                    {"ticker": t, "kind": f"{tf} EMA+MACD", "when": _ct_now_str(), "detail": f"MACD={s.macd_tier}"}
                )

            for s in (r.signals or []):
                if not s.ema200_touch_regain:
                    continue
                key = f"ema200:{s.timeframe}:cycle{st['cycle']}"
                if key in st["fired"]:
                    continue
                st["fired"].add(key)
                new_alerts.append(
                    {"ticker": t, "kind": f"{s.timeframe} EMA200 regain", "when": _ct_now_str(), "detail": s.ema200_reason}
                )

            if not st["noted_align"]:
                need = {f"{tf}:cycle{st['cycle']}" for tf in ("30s", "1m", "2m", "3m")}
                if need.issubset(set(st["fired"])):
                    st["noted_align"] = True
                    new_alerts.append(
                        {"ticker": t, "kind": "ALIGN 30s+1m+2m+3m", "when": _ct_now_str(), "detail": "multi-timeframe alignment"}
                    )

        if new_alerts:
            _scanner_state["alerts"] = (new_alerts + _scanner_state["alerts"])[:200]

        payload = {
            "paused": False,
            "now_ct": _ct_now_str(),
            "results": [{**r.__dict__, "signals": [s.__dict__ for s in (r.signals or [])]} for r in results],
            "alerts": _scanner_state["alerts"],
            "new_alerts": new_alerts,
        }
        _scanner_cache["last_run_ms"] = now_ms
        _scanner_cache["payload"] = payload
        return payload

    async def _scanner_loop() -> None:
        import os
        import time
        from godmode.scan.expo_push import ExpoPushMessage, send_expo_push

        if (os.environ.get("SCANNER_ENABLED") or "").strip() != "1":
            return
        interval_s = int((os.environ.get("SCANNER_INTERVAL_SECONDS") or "840").strip() or "840")
        tokens_raw = (os.environ.get("EXPO_PUSH_TOKENS") or "").strip()
        tokens = [t.strip() for t in tokens_raw.split(",") if t.strip()]

        while True:
            try:
                hm = _ct_hm()
                if hm < 3.0 or hm >= 15.0:
                    await asyncio.sleep(interval_s)
                    continue

                now_ms = int(time.time() * 1000)
                payload = await _scanner_tick(now_ms=now_ms)
                new_alerts = payload.get("new_alerts") or []
                if tokens and new_alerts:
                    messages: list[ExpoPushMessage] = []
                    for a in new_alerts[:25]:
                        title = f"{a['ticker']} {a['kind']}"
                        body = a.get("detail") or ""
                        data = {
                            "type": "alert",
                            "symbol": a.get("ticker"),
                            "kind": a.get("kind"),
                            "detail": a.get("detail"),
                            "when": a.get("when"),
                        }
                        for tok in tokens:
                            messages.append(ExpoPushMessage(to=tok, title=title, body=body, data=data))
                    await send_expo_push(messages=messages)
            except Exception:
                pass
            await asyncio.sleep(interval_s)

    @app.on_event("startup")
    async def _start_scanner_loop() -> None:
        # Fire-and-forget background scanner loop for hosted deploys (Render).
        asyncio.create_task(_scanner_loop())

    @app.get("/api/live/fast")
    async def api_live_fast(ticker: str, level: float, start_ms: int, end_ms: int) -> JSONResponse:
        """
        FAST real-time analysis - no file I/O, no full pipeline.
        Fetches trades AND quotes from Alpaca, uses Lee-Ready for buy/sell classification.
        """
        import os
        import asyncio
        import httpx
        from datetime import datetime
        from godmode.webapp.store import compute_fast_zones_with_quotes, _compute_ema_macd_setup
        
        try:
            api_key = os.environ.get("ALPACA_API_KEY", "")
            api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
            
            if not api_key or not api_secret:
                return JSONResponse({"error": "Alpaca API not configured", "level_data": None})
            
            # Convert timestamps to RFC3339
            start_dt = datetime.utcfromtimestamp(start_ms / 1000)
            end_dt = datetime.utcfromtimestamp(end_ms / 1000)
            start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            }
            
            # Fetch BOTH trades and quotes for Lee-Ready classification.
            # IMPORTANT: we must paginate for longer windows or liquid tickers, otherwise we only get the tail.
            MAX_PAGES = 50
            MAX_TRADES = 250_000
            MAX_QUOTES = 250_000

            async with httpx.AsyncClient(timeout=60.0) as client:
                async def _fetch_all(url: str, params: dict, key: str, max_pages: int, max_rows: int):
                    rows: list[dict] = []
                    next_page_token = None
                    truncated = False

                    for _ in range(max_pages):
                        if next_page_token:
                            params["page_token"] = next_page_token
                        else:
                            params.pop("page_token", None)

                        resp = await client.get(url, params=params, headers=headers)
                        if resp.status_code != 200:
                            raise RuntimeError(f"Alpaca {key}: {resp.status_code}")

                        data = resp.json()
                        batch = data.get(key, []) or []
                        rows.extend(batch)

                        if len(rows) >= max_rows:
                            truncated = True
                            break

                        next_page_token = data.get("next_page_token")
                        if not next_page_token:
                            break

                    if next_page_token:
                        truncated = True

                    return rows, truncated

                trades_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades"
                feed = _alpaca_feed()
                trades_params = {"start": start_str, "end": end_str, "limit": 10000, "feed": feed, "sort": "desc"}

                quotes_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes"
                quotes_params = {"start": start_str, "end": end_str, "limit": 10000, "feed": feed, "sort": "desc"}

                try:
                    trades, trades_truncated = await _fetch_all(
                        trades_url, trades_params, "trades", MAX_PAGES, MAX_TRADES
                    )
                except RuntimeError as e:
                    return JSONResponse({"error": str(e), "level_data": None})

                # Quotes are helpful; if they error/truncate, we can still run with empty/partial quotes.
                quotes = []
                quotes_truncated = False
                try:
                    quotes, quotes_truncated = await _fetch_all(
                        quotes_url, quotes_params, "quotes", MAX_PAGES, MAX_QUOTES
                    )
                except Exception:
                    quotes = []
                    quotes_truncated = True
            
            if not trades:
                return JSONResponse({"error": f"No trades for {ticker}", "level_data": None})
            
            # Convert to DataFrames
            import pandas as pd
            tdf = pd.DataFrame(trades)
            qdf = pd.DataFrame(quotes) if quotes else pd.DataFrame()
            
            # Parse trade timestamps and prices
            if "t" in tdf.columns:
                tdf["ts_ms"] = pd.to_datetime(tdf["t"]).astype("int64") // 10**6
            if "p" in tdf.columns:
                tdf["price"] = tdf["p"]
            if "s" in tdf.columns:
                tdf["size"] = tdf["s"]
            
            # Re-sort to chronological order (we fetched desc to get latest)
            if "ts_ms" in tdf.columns:
                tdf = tdf.sort_values("ts_ms").reset_index(drop=True)
            
            # Parse quote timestamps and bid/ask
            if not qdf.empty and "t" in qdf.columns:
                qdf["ts_ms"] = pd.to_datetime(qdf["t"]).astype("int64") // 10**6
                if "bp" in qdf.columns:
                    qdf["bid"] = qdf["bp"]
                if "ap" in qdf.columns:
                    qdf["ask"] = qdf["ap"]
                qdf["midpoint"] = (qdf["bid"] + qdf["ask"]) / 2
                # Re-sort to chronological order
                qdf = qdf.sort_values("ts_ms").reset_index(drop=True)

            coverage = {
                "requested_start_ms": start_ms,
                "requested_end_ms": end_ms,
                "trades_truncated": bool(trades_truncated),
                "quotes_truncated": bool(quotes_truncated),
                "trades_count": int(len(tdf)),
                "quotes_count": int(len(qdf)) if not qdf.empty else 0,
            }
            try:
                if "ts_ms" in tdf.columns and len(tdf) > 0:
                    coverage["trades_start_ms"] = int(tdf["ts_ms"].iloc[0])
                    coverage["trades_end_ms"] = int(tdf["ts_ms"].iloc[-1])
                    coverage["covers_start"] = int(tdf["ts_ms"].iloc[0]) <= start_ms
                    coverage["covers_end"] = int(tdf["ts_ms"].iloc[-1]) >= end_ms
            except Exception:
                pass
            try:
                if not qdf.empty and "ts_ms" in qdf.columns:
                    coverage["quotes_start_ms"] = int(qdf["ts_ms"].iloc[0])
                    coverage["quotes_end_ms"] = int(qdf["ts_ms"].iloc[-1])
            except Exception:
                pass
            
            # Show time range of data we actually have
            if "ts_ms" in tdf.columns and len(tdf) > 0:
                data_start = datetime.utcfromtimestamp(tdf["ts_ms"].iloc[0] / 1000).strftime("%H:%M:%S")
                data_end = datetime.utcfromtimestamp(tdf["ts_ms"].iloc[-1] / 1000).strftime("%H:%M:%S")
                print(f"[FAST] Fetched {len(tdf)} trades ({data_start}-{data_end}), {len(qdf)} quotes for {ticker}")
            else:
                print(f"[FAST] Fetched {len(tdf)} trades, {len(qdf)} quotes for {ticker}")
            
            # Compute fast zones with Lee-Ready classification
            result = compute_fast_zones_with_quotes(tdf, qdf, level, interval_s=30)
            
            # Debug: print price range and buy/sell info
            price_min = float(tdf["price"].min()) if "price" in tdf.columns else 0
            price_max = float(tdf["price"].max()) if "price" in tdf.columns else 0
            last_price = float(tdf["price"].iloc[-1]) if "price" in tdf.columns and not tdf.empty else 0
            
            # Summarize buy/sell from segments
            segs = result['zone_segments']
            total_buy = sum(s.get('buy_vol', 0) for s in segs)
            total_sell = sum(s.get('sell_vol', 0) for s in segs)
            total_inst_buy = sum(s.get('inst_buy', 0) for s in segs)
            total_inst_sell = sum(s.get('inst_sell', 0) for s in segs)
            
            print(f"[FAST] {ticker} @ ${level:.4f} | price: ${price_min:.4f}-${price_max:.4f} | last: ${last_price:.4f}")
            print(f"[FAST] {len(segs)} segs | Buy: {total_buy:,.0f} Sell: {total_sell:,.0f} | INST: {total_inst_buy:,.0f}B / {total_inst_sell:,.0f}S")
            
            # Compute EMA/MACD
            setup_10s = _compute_ema_macd_setup(tdf, bar_seconds=10)
            setup_30s = _compute_ema_macd_setup(tdf, bar_seconds=30)
            setup_60s = _compute_ema_macd_setup(tdf, bar_seconds=60)
            setup_120s = _compute_ema_macd_setup(tdf, bar_seconds=120)
            setup_180s = _compute_ema_macd_setup(tdf, bar_seconds=180)

            def _ema_stack(setup: dict) -> dict:
                e9 = float(setup.get("ema9", 0) or 0)
                e20 = float(setup.get("ema20", 0) or 0)
                e30 = float(setup.get("ema30", 0) or 0)
                if e9 > e20 and e20 > e30:
                    stack = "Bull (9>20>30)"
                elif e9 < e20 and e20 < e30:
                    stack = "Bear (9<20<30)"
                else:
                    stack = "Mixed"
                return {
                    "stack": stack,
                    "ema9": round(e9, 4),
                    "ema20": round(e20, 4),
                    "ema30": round(e30, 4),
                    "bars": int(setup.get("bars", 0) or 0),
                    "setup_on": bool(setup.get("setup_on", False)),
                }

            ema_stack = {
                "10s": _ema_stack(setup_10s),
                "30s": _ema_stack(setup_30s),
                "1m": _ema_stack(setup_60s),
                "2m": _ema_stack(setup_120s),
                "3m": _ema_stack(setup_180s),
            }
            ema_macd = {
                "setup_1s": {"setup_on": False},
                # Backward compat: keep setup_10s, but setup_30s is now true 30s
                "setup_10s": setup_10s,
                "setup_30s": setup_30s,
                "setup_1m": setup_60s,
                "setup_2m": setup_120s,
                "setup_3m": setup_180s,
                "ema_stack": ema_stack,
                "setup_count": int(setup_30s.get("setup_on", False)),
                "last_price": last_price,
            }
            
            # Format as level_data for frontend compatibility
            level_data = {
                "zone_segments": result["zone_segments"],
                "defense_score": result["defense_score"],
                "aggression_score": result["aggression_score"],
                "market_state": result["market_state"],
                "distribution_pattern": result.get("distribution_pattern", {}),
                "late_momentum": result.get("late_momentum", {}),
                "zone_analysis": result.get("zone_analysis", {}),
                "_debug": {
                    "price_min": price_min,
                    "price_max": price_max,
                    "last_price": last_price,
                    "trades_count": len(tdf),
                }
            }
            return JSONResponse({"level_data": level_data, "ema_macd": ema_macd, "coverage": coverage})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)[:100], "level_data": None})

    @app.get("/api/live/analyze")
    async def api_live_analyze(ticker: str, level: float, start_ms: int, end_ms: int) -> JSONResponse:
        """
        SLOW real-time analysis for live dashboard (deprecated - use /api/live/fast).
        Uses EXACT SAME PIPELINE as report page.
        """
        import os
        import shutil
        import csv
        import json
        from godmode.orchestrator.alpaca_export import export_alpaca_to_replay
        from godmode.orchestrator.session import run_replay_session
        
        try:
            # Check Alpaca credentials
            api_key = os.environ.get("ALPACA_API_KEY", "")
            api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
            
            if not api_key or not api_secret:
                return JSONResponse({"error": "Alpaca API credentials not configured", "level_data": None})
            
            # Create temp session for this analysis
            temp_id = f"LIVE_{int(time.time())}"
            temp_session_id = f"{ticker}_{temp_id}"
            out_dir = Path(cfg.storage.root_dir) / "replay" / f"{ticker}_{temp_session_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Export from Alpaca (SAME AS REPORT)
            try:
                trades_path, quotes_path = await export_alpaca_to_replay(
                    config=cfg,
                    ticker=ticker,
                    start_ts_ms=start_ms,
                    end_ts_ms=end_ms,
                    out_dir=out_dir,
                )
            except Exception as fetch_err:
                error_msg = str(fetch_err)
                if "Timeout" in error_msg or "timeout" in error_msg:
                    return JSONResponse({"error": "Alpaca timeout - try shorter time range", "level_data": None})
                return JSONResponse({"error": f"Alpaca error: {error_msg[:100]}", "level_data": None})
            
            # Check if we got any data
            import pandas as pd
            try:
                tdf = pd.read_parquet(trades_path)
                if tdf.empty or "ts_ms" not in tdf.columns:
                    return JSONResponse({"error": f"No trades for {ticker} in this time range", "level_data": None})
            except Exception:
                return JSONResponse({"error": f"No data for {ticker}", "level_data": None})
            
            # Step 2: Create commands.csv with the level (SAME AS REPORT)
            commands_csv_path = out_dir / "commands.csv"
            header = ["ts_ms", "ticker", "type", "marker_type", "marker_id", "direction_bias", 
                      "notes", "level_price", "level_type", "level_width_atr", "level_id"]
            
            with open(commands_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                # Add the level
                writer.writerow({
                    "ts_ms": start_ms,
                    "ticker": ticker,
                    "type": "add_level",
                    "marker_type": "",
                    "marker_id": "",
                    "direction_bias": "",
                    "notes": json.dumps({"end_ts_ms": end_ms}),
                    "level_price": level,
                    "level_type": "support",
                    "level_width_atr": 0.25,
                    "level_id": f"live_level:{ticker}:{level}",
                })
                # Add marker for the watch window (use support_bounce as the type)
                # Notes must have schema, watch times, and price_tags for level_chain detection
                marker_notes = {
                    "schema": "level_chain_v1",
                    "chain_id": f"live_{ticker}",
                    "end_ts_ms": end_ms,
                    "watch_start_ts_ms": start_ms,
                    "watch_end_ts_ms": end_ms,
                    "price_tags": [{"price": level, "kind": "support"}],
                    "direction_bias": "long",
                    "band_pct": 0.0015,  # 0.15% band
                }
                writer.writerow({
                    "ts_ms": start_ms,
                    "ticker": ticker,
                    "type": "add_marker",
                    "marker_type": "support_bounce",
                    "marker_id": f"live_watch:{ticker}",
                    "direction_bias": "long",
                    "notes": json.dumps(marker_notes),
                    "level_price": "",
                    "level_type": "",
                    "level_width_atr": "",
                    "level_id": "",
                })
            
            # Step 3: Run replay session (SAME AS REPORT)
            await run_replay_session(
                ticker=ticker,
                session_id=temp_session_id,
                direction_bias="long",
                levels_yaml=None,
                trades_path=trades_path,
                quotes_path=quotes_path,
                commands_csv=commands_csv_path,
                config=cfg,
            )
            
            # Step 4: Compute report (SAME AS REPORT)
            # Get date from the actual trade data timestamps
            date_str = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d")
            print(f"[LIVE] Computing report for date={date_str}, ticker={ticker}, session={temp_session_id}")
            
            report = store.compute_session_report(
                date=date_str,
                ticker=ticker,
                session_id=temp_session_id,
            )
            
            print(f"[LIVE] Report keys: {list(report.keys())}")
            print(f"[LIVE] Level chains count: {len(report.get('level_chains', []))}")
            
            # Extract level data from report
            level_data = None
            for chain in report.get("level_chains", []):
                print(f"[LIVE] Chain has {len(chain.get('levels', []))} levels")
                for lv in chain.get("levels", []):
                    level_data = lv
                    break
                if level_data:
                    break
            
            # Compute EMA/MACD setup signals from trades using 10s bars (matches refresh interval)
            from godmode.webapp.store import _compute_ema_macd_setup
            
            # Use 10-second bars to match the 10-second refresh interval
            setup_10s = _compute_ema_macd_setup(tdf, bar_seconds=10)
            
            ema_macd = {
                "setup_1s": {"setup_on": False},  # Skipped for performance
                "setup_30s": setup_10s,  # Named 30s for backward compat but uses 10s bars
                "setup_count": int(setup_10s.get("setup_on", False)),
                "last_price": float(tdf["price"].iloc[-1]) if "price" in tdf.columns and not tdf.empty else 0,
            }
            
            # Cleanup temp files
            try:
                shutil.rmtree(out_dir)
                # Also clean up the episodes/snapshots/session_stream dirs
                for kind in ["episodes", "snapshots", "session_stream", "tf_indicators", "markers"]:
                    p = Path(cfg.storage.root_dir) / kind / f"date={date_str}" / f"ticker={ticker}" / f"session={temp_session_id}"
                    if p.exists():
                        shutil.rmtree(p)
            except Exception:
                pass
            
            return JSONResponse({"level_data": level_data, "ema_macd": ema_macd})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e), "level_data": None})

    @app.post("/record", response_class=RedirectResponse)
    async def record_form_submit(request: Request) -> RedirectResponse:
        """HTML form submission handler - redirects to /jobs after submitting."""
        form = await request.form()
        
        # Extract basic fields
        tickers_raw = form.get("tickers", "")
        timezone_name = form.get("timezone", "America/Chicago")
        start_raw = str(form.get("start", "") or "").strip()
        end_raw = str(form.get("end", "") or "").strip()
        direction_bias = form.get("direction_bias", "long")

        # Parse tickers
        ticker_list = [t.strip().upper() for t in str(tickers_raw).split(",") if t.strip()]
        if not ticker_list:
            raise HTTPException(status_code=400, detail="No tickers provided")

        # Extract per-level watch windows (preferred UI)
        # Each level has: price, kind, start, end, optional outcome.
        markers: list[MarkerInput] = []
        levels_chain_id = str(form.get("levels_chain_id", "") or "").strip() or None
        level_time_ranges_ms: list[tuple[int, int]] = []

        # Collect level indices from both price-based and indicator-based levels
        level_indices: set[str] = set()
        for key in form.keys():
            if key.startswith("level_price_") or key.startswith("level_type_"):
                level_indices.add(key.split("_")[-1])

        # Track if we're in live mode based on levels (start provided but no end)
        level_live_mode = False
        
        for idx in sorted(level_indices, key=lambda x: int(x) if str(x).isdigit() else 999999):
            level_type = form.get(f"level_type_{idx}") or "price"
            k_raw = form.get(f"level_kind_{idx}") or "support"
            s_raw = form.get(f"level_start_{idx}")
            e_raw = form.get(f"level_end_{idx}")
            o_raw = form.get(f"level_outcome_{idx}") or ""

            # Validate start time (required)
            if s_raw is None or str(s_raw).strip() == "":
                continue
            
            # End time is optional - if missing, we're in live mode
            try:
                start_ts_ms = _parse_local_datetime_to_utc_ms(str(s_raw), str(timezone_name))
                if e_raw is None or str(e_raw).strip() == "":
                    # LIVE MODE: No end time, use current time
                    end_ts_ms = int(time.time() * 1000)
                    level_live_mode = True
                else:
                    end_ts_ms = _parse_local_datetime_to_utc_ms(str(e_raw), str(timezone_name))
            except Exception:
                continue
            if end_ts_ms <= start_ts_ms:
                continue

            outcome = str(o_raw).strip().lower()
            if outcome not in {"win", "loss"}:
                outcome = ""

            if level_type == "indicator":
                # Indicator mode: collect selected indicators
                selected_indicators: list[str] = []
                if form.get(f"level_ind_vwap_{idx}"):
                    selected_indicators.append("vwap")
                if form.get(f"level_ind_ema9_{idx}"):
                    selected_indicators.append("ema9")
                if form.get(f"level_ind_ema20_{idx}"):
                    selected_indicators.append("ema20")
                if form.get(f"level_ind_ema30_{idx}"):
                    selected_indicators.append("ema30")
                if form.get(f"level_ind_ema200_{idx}"):
                    selected_indicators.append("ema200")

                if not selected_indicators:
                    continue  # No indicators selected, skip

                # Get the selected chart timeframe for indicator analysis
                indicator_tf = str(form.get(f"indicator_timeframe_{idx}", "1m") or "1m").strip()
                if indicator_tf not in ("30s", "1m", "2m", "3m"):
                    indicator_tf = "1m"  # Default to 1m if invalid

                level_time_ranges_ms.append((int(start_ts_ms), int(end_ts_ms)))

                # Create a marker for indicator-based level
                markers.append(
                    MarkerInput(
                        start_ts_ms=start_ts_ms,
                        marker_type="consolidation",
                        end_ts_ms=end_ts_ms,
                        price_tags=[],  # No fixed price for indicators
                        notes={
                            "schema": "level_chain_v1",
                            "chain_id": levels_chain_id,
                            "level_index": int(idx) if str(idx).isdigit() else None,
                            "level_outcome": (outcome or None),
                            "level_type": "indicator",
                            "indicators": selected_indicators,
                            "indicator_timeframe": indicator_tf,  # Chart timeframe for EMA calculation
                            "band_pct": 0.0015,  # 0.15% band for touch detection
                            "direction_bias": direction_bias,
                            "skip_episode": False,
                        },
                    )
                )
            else:
                # Price mode: existing behavior
                p_raw = form.get(f"level_price_{idx}")
                if p_raw is None or str(p_raw).strip() == "":
                    continue
                try:
                    price = float(str(p_raw))
                except Exception:
                    continue

                level_time_ranges_ms.append((int(start_ts_ms), int(end_ts_ms)))

                markers.append(
                    MarkerInput(
                        start_ts_ms=start_ts_ms,
                        marker_type="consolidation",
                        end_ts_ms=end_ts_ms,
                        price_tags=[{"price": float(price), "kind": str(k_raw).strip().lower() or None}],
                        notes={
                            "schema": "level_chain_v1",
                            "chain_id": levels_chain_id,
                            "level_index": int(idx) if str(idx).isdigit() else None,
                            "level_outcome": (outcome or None),
                            "level_type": "price",
                            "direction_bias": direction_bias,
                            "skip_episode": False,
                        },
                    )
                )

        # Parse/derive session start/end with timezone conversion.
        # If user left session start/end blank, derive from levels (recommended workflow).
        # If start is provided but end is blank → LIVE MODE
        pad_ms = 0  # No padding - use exact times as entered
        is_live = False
        
        if start_raw and end_raw:
            # Both provided - normal replay mode
            try:
                start_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                end_ms = _parse_local_datetime_to_utc_ms(str(end_raw), str(timezone_name))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")
        elif start_raw and not end_raw:
            # Start provided but no end → LIVE MODE
            try:
                start_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                end_ms = None  # Will be set to current time in jobs.py
                is_live = True
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")
        else:
            # Derive from levels
            if not level_time_ranges_ms:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either Session Start/End or at least one Level with Start+End (so the session range can be derived).",
                )
            min_start = min(s for s, _ in level_time_ranges_ms)
            max_end = max(e for _, e in level_time_ranges_ms)
            start_ms = int(max(0, min_start - pad_ms))
            end_ms = int(max_end + pad_ms)
            
            # If levels triggered live mode, propagate it
            if level_live_mode:
                is_live = True
                end_ms = None  # Will be set to current time in jobs.py

        if not is_live and end_ms is not None and end_ms <= start_ms:
            raise HTTPException(status_code=400, detail="Session end must be after session start.")

        # Extract legacy markers from form (backward-compatible)
        # Each marker has start time, optional end time, multiple types, and optional price tags.
        processed_indices: set[str] = set()
        
        for key in form.keys():
            if key.startswith("marker_start_"):
                idx = key.split("_")[-1]
                if idx in processed_indices:
                    continue
                processed_indices.add(idx)
                
                start_raw = form.get(f"marker_start_{idx}")
                end_raw = form.get(f"marker_end_{idx}")
                # Get all checked types for this marker (multiple checkboxes)
                type_values = form.getlist(f"marker_type_{idx}")
                # Optional price tags (up to 5)
                price_tags: list[dict[str, Any]] = []
                for j in range(1, 6):
                    p_raw = form.get(f"marker_price_{idx}_{j}")
                    if p_raw is None or str(p_raw).strip() == "":
                        continue
                    kind = form.get(f"marker_price_kind_{idx}_{j}") or ""
                    try:
                        price_tags.append({"price": float(str(p_raw)), "kind": (str(kind) if str(kind).strip() else None)})
                    except Exception:
                        continue
                
                if start_raw and type_values:
                    try:
                        start_ts_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                        end_ts_ms = None
                        if end_raw and str(end_raw).strip():
                            end_ts_ms = _parse_local_datetime_to_utc_ms(str(end_raw), str(timezone_name))
                        # Emit one MarkerInput per type (no composite enums); each carries the same end_ts_ms and tags.
                        for t in type_values:
                            markers.append(
                                MarkerInput(
                                    start_ts_ms=start_ts_ms,
                                    marker_type=str(t),
                                    end_ts_ms=end_ts_ms,
                                    price_tags=price_tags,
                                    notes={},
                                )
                            )
                    except Exception:
                        pass  # Skip invalid markers

        # Submit jobs
        await job_manager.submit_record_request(
            tickers=ticker_list,
            start_ms=start_ms,
            end_ms=end_ms,
            direction_bias=str(direction_bias),
            markers=markers,
            is_live=is_live,
        )

        return RedirectResponse(url="/jobs", status_code=303)

    @app.post("/api/record")
    async def api_record(
        tickers: str = Form(...),
        start: str = Form(...),
        end: str = Form(...),
        timezone: str = Form("America/Chicago"),
        direction_bias: str = Form("long"),
    ) -> JSONResponse:
        """
        Submit a record request via API.
        """
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            raise HTTPException(status_code=400, detail="No tickers provided")

        try:
            start_ms = _parse_local_datetime_to_utc_ms(start, timezone)
            end_ms = _parse_local_datetime_to_utc_ms(end, timezone)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")

        job_ids = await job_manager.submit_record_request(
            tickers=ticker_list,
            start_ms=start_ms,
            end_ms=end_ms,
            direction_bias=direction_bias,
            markers=[],
        )

        return JSONResponse({"job_ids": job_ids, "count": len(job_ids)})

    return app
