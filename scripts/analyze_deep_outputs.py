from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


@dataclass(frozen=True)
class SessionKey:
    date: str
    ticker: str
    session: str


KINDS = ["snapshots", "episodes", "markers", "tf_indicators", "session_stream"]


def _find_sessions(root: Path) -> list[SessionKey]:
    sessions: set[SessionKey] = set()
    for kind in KINDS:
        base = root / kind
        if not base.exists():
            continue
        for date_dir in base.glob("date=*"):
            date = date_dir.name.split("=", 1)[-1]
            for ticker_dir in date_dir.glob("ticker=*"):
                ticker = ticker_dir.name.split("=", 1)[-1]
                for session_dir in ticker_dir.glob("session=*"):
                    session = session_dir.name.split("=", 1)[-1]
                    sessions.add(SessionKey(date=date, ticker=ticker, session=session))
    return sorted(sessions, key=lambda s: (s.date, s.ticker, s.session))


def _dataset_for(root: Path, kind: str, sk: SessionKey) -> ds.Dataset | None:
    base = root / kind / f"date={sk.date}" / f"ticker={sk.ticker}" / f"session={sk.session}"
    if not base.exists():
        return None
    return ds.dataset(str(base), format="parquet", partitioning="hive")


def _to_df(dset: ds.Dataset, columns: list[str] | None = None) -> pd.DataFrame:
    # Read fragment-by-fragment to avoid schema drift issues across parquet parts
    dfs: list[pd.DataFrame] = []
    for frag in dset.get_fragments():
        try:
            tbl = frag.to_table(columns=columns)
        except Exception:
            # If a fragment is unreadable, skip it but keep going
            continue
        dfs.append(tbl.to_pandas())

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True, sort=False)

    # Avoid categorical surprises / normalize common string-like columns
    for col in ["ticker", "phase", "marker_type", "direction_bias", "timeframe", "episode_source", "resolution_trigger", "outcome", "resolution_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _diff_stats(ts: pd.Series) -> dict[str, Any]:
    ts = ts.dropna().astype("int64").sort_values()
    if len(ts) < 2:
        return {"n": int(len(ts))}
    d = ts.diff().dropna()
    return {
        "n": int(len(ts)),
        "diff_min": int(d.min()),
        "diff_p50": int(d.median()),
        "diff_p90": int(d.quantile(0.9)),
        "diff_max": int(d.max()),
        "pct_eq_10s": float((d == 10_000).mean()),
        "gaps_gt_20s": int((d > 20_000).sum()),
        "gaps_gt_60s": int((d > 60_000).sum()),
    }


def _vc_dict(s: pd.Series) -> dict[str, int]:
    vc = s.value_counts(dropna=False)
    out: dict[str, int] = {}
    for k, v in vc.items():
        key = "<NA>" if pd.isna(k) else str(k)
        out[key] = int(v)
    return out


def _top_outliers(df: pd.DataFrame, col: str, n: int = 10) -> list[dict[str, Any]]:
    if col not in df.columns or df.empty:
        return []
    extra = [c for c in ["return_10s", "return_norm", "delta", "delta_norm"] if c in df.columns]
    cols: list[str] = []
    for c in ["timestamp", col] + extra:
        if c not in cols:
            cols.append(c)
    d = df[cols].copy()
    d = d.dropna(subset=[col])
    if d.empty:
        return []
    d = d.sort_values(col, ascending=False).head(n)
    return json.loads(d.to_json(orient="records"))


def _parse_commands_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    markers = 0
    levels = 0
    marker_types: dict[str, int] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("type") or "").strip()
            if t == "add_marker":
                markers += 1
                mt = (row.get("marker_type") or "").strip()
                if mt:
                    marker_types[mt] = marker_types.get(mt, 0) + 1
            elif t == "add_level":
                levels += 1
    return {"exists": True, "marker_cmds": markers, "level_cmds": levels, "marker_types": marker_types}


def analyze_one(root: Path, replay_root: Path, sk: SessionKey) -> dict[str, Any]:
    rep: dict[str, Any] = {"session": sk.__dict__.copy(), "checks": {}, "snapshots": {}, "episodes": {}, "markers": {}, "tf_indicators": {}, "session_stream": {}}

    # Commands.csv (best-effort)
    commands_csv = replay_root / f"{sk.ticker}_{sk.session}" / "commands.csv"
    rep["checks"]["commands_csv"] = _parse_commands_csv(commands_csv)

    # SNAPSHOTS
    snap_ds = _dataset_for(root, "snapshots", sk)
    if snap_ds is not None:
        snap_df = _to_df(snap_ds)
        rep["snapshots"]["rows"] = int(len(snap_df))
        rep["snapshots"]["time_min"] = int(snap_df["timestamp"].min())
        rep["snapshots"]["time_max"] = int(snap_df["timestamp"].max())
        # IMPORTANT: snapshots are stored PER EPISODE (episode_id). Many episodes share the same timestamps.
        # Global timestamp diffs will show 0ms and huge gaps when you interleave multiple episodes.
        if "episode_id" in snap_df.columns:
            rep["snapshots"]["episodes_in_snapshots"] = int(snap_df["episode_id"].nunique(dropna=True))
            per = []
            for ep_id, g in snap_df.groupby("episode_id", dropna=True):
                ts_stats = _diff_stats(g["timestamp"])
                item: dict[str, Any] = {
                    "episode_id": str(ep_id),
                    "timestamp_diffs": ts_stats,
                }
                if "sequence_id" in g.columns:
                    seq = g["sequence_id"].dropna().astype("int64")
                    item["sequence_monotonic"] = bool(seq.is_monotonic_increasing)
                    item["sequence_duplicates"] = int(seq.duplicated().sum())
                per.append(item)

            # Summarize per-episode cadence
            if per:
                pct10 = [float(x["timestamp_diffs"].get("pct_eq_10s", 0.0)) for x in per if "timestamp_diffs" in x]
                gaps20 = [int(x["timestamp_diffs"].get("gaps_gt_20s", 0)) for x in per if "timestamp_diffs" in x]
                rep["snapshots"]["per_episode_cadence_summary"] = {
                    "episodes": int(len(per)),
                    "pct_eq_10s_min": float(min(pct10)) if pct10 else None,
                    "pct_eq_10s_p50": float(pd.Series(pct10).median()) if pct10 else None,
                    "pct_eq_10s_max": float(max(pct10)) if pct10 else None,
                    "episodes_with_gap_gt_20s": int(sum(1 for g in gaps20 if g > 0)),
                }
                # keep a small sample (first 5) for debugging
                rep["snapshots"]["per_episode_sample"] = per[:5]
        else:
            rep["snapshots"]["timestamp_diffs"] = _diff_stats(snap_df["timestamp"])
            if "sequence_id" in snap_df.columns:
                seq = snap_df["sequence_id"].dropna().astype("int64")
                rep["snapshots"]["sequence_monotonic"] = bool(seq.is_monotonic_increasing)
                rep["snapshots"]["sequence_duplicates"] = int(seq.duplicated().sum())
        if "phase" in snap_df.columns:
            rep["snapshots"]["phase_counts"] = _vc_dict(snap_df["phase"])

        # Zone gating sanity: how much is inside the canonical zone?
        if "signed_distance_to_level_atr" in snap_df.columns:
            s = snap_df["signed_distance_to_level_atr"].astype("float64")
            rep["snapshots"]["pct_within_zone_0p25_atr"] = float((s.abs() <= 0.25).mean())
            rep["snapshots"]["pct_within_zone_0p50_atr"] = float((s.abs() <= 0.50).mean())

        # Smart-money + divergence sanity
        if "divergence_flag" in snap_df.columns:
            rep["snapshots"]["pct_divergence_flag_true"] = float((snap_df["divergence_flag"] == 1).mean())
        rep["snapshots"]["top_div_score"] = _top_outliers(snap_df, "div_score", n=10)
        rep["snapshots"]["top_delta"] = _top_outliers(snap_df, "delta", n=10)

    # EPISODES
    ep_ds = _dataset_for(root, "episodes", sk)
    if ep_ds is not None:
        ep_df = _to_df(ep_ds)
        rep["episodes"]["rows"] = int(len(ep_df))
        rep["episodes"]["start_min"] = int(ep_df["start_time"].min())
        rep["episodes"]["start_max"] = int(ep_df["start_time"].max())
        for c in ["episode_source", "marker_type", "resolution_trigger", "outcome", "resolution_type"]:
            if c in ep_df.columns:
                rep["episodes"][f"{c}_counts"] = _vc_dict(ep_df[c])

        # Time sanity
        for a, b in [
            ("start_time", "zone_entry_time"),
            ("zone_entry_time", "zone_exit_time"),
            ("zone_entry_time", "resolution_time"),
            ("start_time", "end_time"),
        ]:
            if a in ep_df.columns and b in ep_df.columns:
                dd = (ep_df[b].astype("float64") - ep_df[a].astype("float64")).dropna()
                if len(dd):
                    rep["episodes"][f"dt_{a}_to_{b}_ms"] = {
                        "min": float(dd.min()),
                        "p50": float(dd.median()),
                        "p90": float(dd.quantile(0.9)),
                        "max": float(dd.max()),
                    }

        # Marker-extract consistency
        if "episode_source" in ep_df.columns:
            mex = ep_df[ep_df["episode_source"] == "marker_extract"]
            rep["episodes"]["marker_extract_rows"] = int(len(mex))
            if len(mex):
                rep["episodes"]["marker_extract_missing_marker_id"] = int(mex["marker_id"].isna().sum()) if "marker_id" in mex.columns else None
                rep["episodes"]["marker_extract_missing_marker_ts_ms"] = int(mex["marker_ts_ms"].isna().sum()) if "marker_ts_ms" in mex.columns else None

    # MARKERS
    mk_ds = _dataset_for(root, "markers", sk)
    if mk_ds is not None:
        mk_df = _to_df(mk_ds)
        rep["markers"]["rows"] = int(len(mk_df))
        rep["markers"]["ts_min"] = int(mk_df["ts_ms"].min())
        rep["markers"]["ts_max"] = int(mk_df["ts_ms"].max())
        for c in ["marker_type", "direction_bias"]:
            if c in mk_df.columns:
                rep["markers"][f"{c}_counts"] = _vc_dict(mk_df[c])
        rep["markers"]["marker_id_duplicates"] = int(mk_df["marker_id"].duplicated().sum()) if "marker_id" in mk_df.columns else None

    # TF INDICATORS
    tf_ds = _dataset_for(root, "tf_indicators", sk)
    if tf_ds is not None:
        tf_df = _to_df(tf_ds)
        rep["tf_indicators"]["rows"] = int(len(tf_df))
        rep["tf_indicators"]["time_min"] = int(tf_df["timestamp"].min())
        rep["tf_indicators"]["time_max"] = int(tf_df["timestamp"].max())
        if "timeframe" in tf_df.columns:
            rep["tf_indicators"]["timeframe_counts"] = _vc_dict(tf_df["timeframe"])
            # IMPORTANT: tf_indicators are also per-episode (episode_id) + timeframe.
            if "episode_id" in tf_df.columns:
                rep["tf_indicators"]["episodes_in_tf"] = int(tf_df["episode_id"].nunique(dropna=True))
                per_tf: dict[str, Any] = {}
                for tf in tf_df["timeframe"].dropna().unique().tolist():
                    sub = tf_df[tf_df["timeframe"] == tf]
                    per = []
                    for ep_id, g in sub.groupby("episode_id", dropna=True):
                        per.append(
                            {
                                "episode_id": str(ep_id),
                                "timestamp_diffs": _diff_stats(g["timestamp"]),
                            }
                        )
                    pct10 = [float(x["timestamp_diffs"].get("pct_eq_10s", 0.0)) for x in per]
                    gaps20 = [int(x["timestamp_diffs"].get("gaps_gt_20s", 0)) for x in per]
                    per_tf[str(tf)] = {
                        "episodes": int(len(per)),
                        "pct_eq_10s_min": float(min(pct10)) if pct10 else None,
                        "pct_eq_10s_p50": float(pd.Series(pct10).median()) if pct10 else None,
                        "pct_eq_10s_max": float(max(pct10)) if pct10 else None,
                        "episodes_with_gap_gt_20s": int(sum(1 for g in gaps20 if g > 0)),
                        "sample": per[:5],
                    }
                rep["tf_indicators"]["per_episode_by_timeframe"] = per_tf
            else:
                # Cadence per timeframe (fallback)
                for tf in tf_df["timeframe"].dropna().unique().tolist():
                    sub = tf_df[tf_df["timeframe"] == tf]
                    rep["tf_indicators"][f"timestamp_diffs_{tf}"] = _diff_stats(sub["timestamp"])

    # SESSION STREAM
    ss_ds = _dataset_for(root, "session_stream", sk)
    if ss_ds is not None:
        ss_df = _to_df(ss_ds)
        rep["session_stream"]["rows"] = int(len(ss_df))
        rep["session_stream"]["time_min"] = int(ss_df["timestamp"].min()) if "timestamp" in ss_df.columns else None
        rep["session_stream"]["time_max"] = int(ss_df["timestamp"].max()) if "timestamp" in ss_df.columns else None
        if "timestamp" in ss_df.columns:
            rep["session_stream"]["timestamp_diffs"] = _diff_stats(ss_df["timestamp"])

    # Cross-kind coverage checks
    if rep["snapshots"].get("time_min") and rep["session_stream"].get("time_min"):
        rep["checks"]["snapshots_within_session_stream"] = bool(
            rep["snapshots"]["time_min"] >= rep["session_stream"]["time_min"]
            and rep["snapshots"]["time_max"] <= rep["session_stream"]["time_max"]
        )

    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default="data/output")
    ap.add_argument("--replay_root", default="data/output/replay")
    ap.add_argument("--format", choices=["json"], default="json")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    replay_root = Path(args.replay_root)

    sessions = _find_sessions(out_root)
    reports = [analyze_one(out_root, replay_root, sk) for sk in sessions]

    # Also list replay exports that do NOT have an output session written (useful to spot failed jobs)
    replay_exports: list[str] = []
    if replay_root.exists():
        for d in replay_root.iterdir():
            if d.is_dir():
                replay_exports.append(d.name)

    written_sessions = {f"{r['session']['ticker']}_{r['session']['session']}" for r in reports}
    missing_outputs = sorted([x for x in replay_exports if x not in written_sessions])

    blob = {
        "sessions_found": len(sessions),
        "written_sessions": sorted(list(written_sessions)),
        "replay_exports_found": len(replay_exports),
        "replay_exports_missing_outputs": missing_outputs,
        "reports": reports,
    }
    print(json.dumps(blob, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


