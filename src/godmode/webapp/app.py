"""FastAPI web application for GodMode visualization + orchestration."""
from __future__ import annotations

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
            "home.html",
            {
                "request": request,
                "sessions": sessions,
                "root_dir": str(cfg.storage.root_dir),
            },
        )

    @app.get("/record", response_class=HTMLResponse)
    def record_page(request: Request) -> HTMLResponse:
        """Record form page (Addendum J1)."""
        return templates.TemplateResponse("record.html", {"request": request})

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request) -> HTMLResponse:
        """Jobs status page with polling (Addendum J5)."""
        jobs = job_manager.list_jobs()
        return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})

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
                "feed": "sip",
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
        
        try:
            # Map timeframe to Alpaca format
            tf_map = {
                "30s": "30Sec",
                "1m": "1Min",
                "2m": "2Min",
                "3m": "3Min",
            }
            alpaca_tf = tf_map.get(timeframe, "1Min")
            
            # Get 1 week of data
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
            
            # Fetch bars from Alpaca
            url = f"https://data.alpaca.markets/v2/stocks/{ticker.upper()}/bars"
            params = {
                "start": start_str,
                "end": end_str,
                "timeframe": alpaca_tf,
                "feed": "sip",
                "limit": 10000,  # Max allowed
            }
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
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
            
            # Convert to Lightweight Charts format
            # Time must be Unix timestamp in seconds
            # Map timeframe to seconds for END time calculation
            tf_seconds = {
                "30s": 30,
                "1m": 60,
                "2m": 120,
                "3m": 180,
            }
            duration_sec = tf_seconds.get(timeframe, 60)
            
            candles = []
            for bar in all_bars:
                # Parse Alpaca timestamp (RFC3339)
                ts_str = bar.get("t", "")
                if ts_str:
                    # Remove 'Z' and parse
                    ts_str = ts_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts_str)
                    # Use candle END time to match Webull display
                    # Alpaca returns START time, so we add the candle duration
                    unix_ts = int(dt.timestamp()) + duration_sec
                    
                    candles.append({
                        "time": unix_ts,
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                    })
            
            return JSONResponse({
                "ticker": ticker.upper(),
                "timeframe": timeframe,
                "candles": candles,
                "count": len(candles),
            })
            
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

        for idx in sorted(level_indices, key=lambda x: int(x) if str(x).isdigit() else 999999):
            level_type = form.get(f"level_type_{idx}") or "price"
            k_raw = form.get(f"level_kind_{idx}") or "support"
            s_raw = form.get(f"level_start_{idx}")
            e_raw = form.get(f"level_end_{idx}")
            o_raw = form.get(f"level_outcome_{idx}") or ""

            # Validate start/end times
            if s_raw is None or str(s_raw).strip() == "" or e_raw is None or str(e_raw).strip() == "":
                continue
            try:
                start_ts_ms = _parse_local_datetime_to_utc_ms(str(s_raw), str(timezone_name))
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
        pad_ms = 0  # No padding - use exact times as entered
        if start_raw and end_raw:
            try:
                start_ms = _parse_local_datetime_to_utc_ms(str(start_raw), str(timezone_name))
                end_ms = _parse_local_datetime_to_utc_ms(str(end_raw), str(timezone_name))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid datetime: {e}")
        else:
            if not level_time_ranges_ms:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either Session Start/End or at least one Level with Start+End (so the session range can be derived).",
                )
            min_start = min(s for s, _ in level_time_ranges_ms)
            max_end = max(e for _, e in level_time_ranges_ms)
            start_ms = int(max(0, min_start - pad_ms))
            end_ms = int(max_end + pad_ms)

        if end_ms <= start_ms:
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
