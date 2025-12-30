from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
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


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _table_row_count(dataset: ds.Dataset) -> int:
    # Fast row count: zero columns table still has row count.
    return int(dataset.to_table(columns=[]).num_rows)


def _minmax_ms(dataset: ds.Dataset, col: str) -> tuple[int | None, int | None]:
    n = _table_row_count(dataset)
    if n == 0:
        return None, None
    arr = dataset.to_table(columns=[col])[col]
    return int(pc.min(arr).as_py()), int(pc.max(arr).as_py())


def _null_rate(dataset: ds.Dataset, col: str) -> float | None:
    n = _table_row_count(dataset)
    if n == 0:
        return None
    arr = dataset.to_table(columns=[col])[col]
    return float(arr.null_count) / float(n)


def _value_counts(dataset: ds.Dataset, col: str, limit: int = 50) -> list[dict[str, Any]]:
    # Robust against per-file schema drift where some fragments have NULL type for this column.
    arrays: list[pa.Array] = []
    for frag in dataset.get_fragments():
        try:
            t = frag.to_table(columns=[col])
        except Exception:
            continue
        if col not in t.column_names:
            continue
        a = t[col]  # ChunkedArray
        if pa.types.is_null(a.type):
            continue
        a = pc.cast(a, pa.large_string())
        if isinstance(a, pa.ChunkedArray):
            arrays.extend([chunk for chunk in a.chunks if chunk is not None])
        else:
            arrays.append(a)

    if not arrays:
        return []
    arr = pa.concat_arrays(arrays)
    arr = pc.drop_null(arr)
    if len(arr) == 0:
        return []

    vc = pc.value_counts(arr)
    values = vc.field("values")
    counts = vc.field("counts")
    out: list[dict[str, Any]] = []
    for i in range(min(len(vc), limit)):
        out.append({"value": values[i].as_py(), "count": int(counts[i].as_py())})
    return out


def _quantiles(dataset: ds.Dataset, col: str, qs: Iterable[float] = (0.5, 0.9)) -> dict[str, Any] | None:
    if col not in dataset.schema.names:
        return None
    arr = dataset.to_table(columns=[col])[col]
    arr = pc.drop_null(arr)
    if len(arr) == 0:
        return None
    q = pc.quantile(arr, q=list(qs), interpolation="linear")
    q_vals = [q[i].as_py() for i in range(len(q))]
    return {
        "min": pc.min(arr).as_py(),
        "max": pc.max(arr).as_py(),
        "mean": pc.mean(arr).as_py(),
        **{f"p{int(qs_i*100)}": q_vals[idx] for idx, qs_i in enumerate(list(qs))},
    }


def analyze_session(root: Path, sk: SessionKey) -> dict[str, Any]:
    report: dict[str, Any] = {"session": {"date": sk.date, "ticker": sk.ticker, "session": sk.session}, "kinds": {}}

    for kind in KINDS:
        dset = _dataset_for(root, kind, sk)
        if dset is None:
            continue
        cols = dset.schema.names

        time_col = None
        if kind == "snapshots":
            time_col = _first_existing(cols, ["timestamp", "ts_ms"])
        elif kind == "episodes":
            time_col = _first_existing(cols, ["start_time", "zone_entry_time", "resolution_time", "end_time"])
        elif kind == "markers":
            time_col = _first_existing(cols, ["start_ts_ms", "ts_ms", "timestamp"])
        else:
            time_col = _first_existing(cols, ["ts_ms", "timestamp"])

        kind_rep: dict[str, Any] = {"rows": _table_row_count(dset), "columns": len(cols), "time_col": time_col}
        if time_col:
            tmin, tmax = _minmax_ms(dset, time_col)
            kind_rep["time_min"] = tmin
            kind_rep["time_max"] = tmax

        # Per-kind breakdowns
        if kind == "markers":
            for c in ["marker_type", "direction_bias"]:
                if c in cols:
                    kind_rep[f"counts_by_{c}"] = _value_counts(dset, c)
        if kind == "episodes":
            for c in ["phase", "outcome", "resolution_trigger", "resolution_type", "episode_source", "marker_type"]:
                if c in cols:
                    kind_rep[f"counts_by_{c}"] = _value_counts(dset, c)

            # Metrics sanity: min/p50/p90/max/mean for common episode metrics
            metric_cols = ["mfe", "mae", "time_to_mfe_ms", "time_to_failure_ms"]
            metric_cols = [c for c in metric_cols if c in cols]
            if metric_cols:
                stats: dict[str, Any] = {}
                # Read fragment-by-fragment to avoid schema drift (some parts may have NULL type for these columns).
                for c in metric_cols:
                    arrays: list[pa.Array] = []
                    for frag in dset.get_fragments():
                        try:
                            t = frag.to_table(columns=[c])
                        except Exception:
                            continue
                        if c not in t.column_names:
                            continue
                        a = t[c]
                        if pa.types.is_null(a.type):
                            continue
                        if isinstance(a, pa.ChunkedArray):
                            arrays.extend([chunk for chunk in a.chunks if chunk is not None])
                        else:
                            arrays.append(a)
                    if not arrays:
                        continue
                    arr = pa.concat_arrays(arrays)
                    arr = pc.drop_null(arr)
                    if len(arr) == 0:
                        continue
                    q = pc.quantile(arr, q=[0.5, 0.9], interpolation="linear")
                    stats[c] = {
                        "min": pc.min(arr).as_py(),
                        "p50": q[0].as_py(),
                        "p90": q[1].as_py(),
                        "max": pc.max(arr).as_py(),
                        "mean": pc.mean(arr).as_py(),
                    }
                if stats:
                    kind_rep["metric_sanity"] = stats
        if kind == "tf_indicators" and "timeframe" in cols:
            kind_rep["counts_by_timeframe"] = _value_counts(dset, "timeframe")

        # Snapshots: “feature health” scan for key fields
        if kind == "snapshots":
            key_cols = [
                "last_price",
                "bid",
                "ask",
                "mid_price",
                "spread_pct",
                "atr_1m",
                "atr_status",
                "vwap_session",
                "ema9",
                "ema20",
                "ema30",
                "ema200",
                "trade_count",
                "total_volume",
                "buy_volume",
                "sell_volume",
                "unknown_volume",
                "delta",
                "cvd_10s",
                "cvd_30s",
                "cvd_60s",
                "divergence_flag",
                "div_score",
                "buy_on_red",
                "sell_on_green",
                "large_trade_threshold_size",
                "large_trade_count_10s",
                "large_trade_delta",
                "large_trade_share_of_total_vol_10s",
                "vol_z_60s",
                "trade_rate_z_60s",
                "signed_distance_to_level_atr",
                "touch_count",
            ]
            key_cols = [c for c in key_cols if c in cols]
            kind_rep["null_rates"] = {c: _null_rate(dset, c) for c in key_cols}
            dist_cols = [c for c in ["spread_pct", "atr_1m", "delta", "div_score", "vol_z_60s", "signed_distance_to_level_atr"] if c in cols]
            kind_rep["dists"] = {c: _quantiles(dset, c) for c in dist_cols}

        report["kinds"][kind] = kind_rep

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/output", help="Output root directory (default: data/output)")
    ap.add_argument("--date", default=None, help="Filter: date=YYYY-MM-DD")
    ap.add_argument("--ticker", default=None, help="Filter: ticker=XYZ")
    ap.add_argument("--session", default=None, help="Filter: session=...")
    ap.add_argument("--format", choices=["json"], default="json")
    args = ap.parse_args()

    root = Path(args.root)
    sessions = _find_sessions(root)
    if args.date:
        sessions = [s for s in sessions if s.date == args.date]
    if args.ticker:
        sessions = [s for s in sessions if s.ticker == args.ticker]
    if args.session:
        sessions = [s for s in sessions if s.session == args.session]

    out = {"sessions_found": len(sessions), "reports": [analyze_session(root, s) for s in sessions]}
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


