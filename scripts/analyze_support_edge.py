from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def read_dir(path: Path) -> pd.DataFrame:
    dset = ds.dataset(str(path), format="parquet", partitioning="hive")
    dfs: list[pd.DataFrame] = []
    for frag in dset.get_fragments():
        try:
            dfs.append(frag.to_table().to_pandas())
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def _pstats(x: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {}
    return {
        "p10": float(x.quantile(0.1)),
        "p50": float(x.quantile(0.5)),
        "p90": float(x.quantile(0.9)),
        "mean": float(x.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/output_clean")
    ap.add_argument("--date", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--marker-ts-ms", type=int, required=True, help="Edge decision marker timestamp (ms)")
    ap.add_argument("--support-price", type=float, required=True)
    ap.add_argument("--edge-price", type=float, required=True)
    ap.add_argument("--window-minutes", type=int, default=5)
    ap.add_argument("--trace-seconds", type=int, default=120, help="How many seconds before marker to print trace")
    args = ap.parse_args()

    root = Path(args.root)
    date = args.date
    ticker = args.ticker
    session = args.session
    M = int(args.marker_ts_ms)

    ep_dir = root / "episodes" / f"date={date}" / f"ticker={ticker}" / f"session={session}"
    sn_dir = root / "snapshots" / f"date={date}" / f"ticker={ticker}" / f"session={session}"
    ss_dir = root / "session_stream" / f"date={date}" / f"ticker={ticker}" / f"session={session}"

    eps = read_dir(ep_dir)
    snaps = read_dir(sn_dir)
    stream = read_dir(ss_dir)

    if eps.empty or snaps.empty or stream.empty:
        raise SystemExit("Missing episodes, snapshots, or session_stream for session")

    # Note: support levels can be emitted at multiple timestamps (e.g. consolidation marker time and downtrend_break time).
    # For analysis we want "all interactions at this price", so match by suffix token rather than exact level_id.
    support_token = f":{args.support_price}:support"
    edge_token = f":{args.edge_price}:resistance"

    eps["zone_entry_time"] = pd.to_numeric(eps.get("zone_entry_time"), errors="coerce")
    snaps["timestamp"] = pd.to_numeric(snaps.get("timestamp"), errors="coerce")
    stream["timestamp"] = pd.to_numeric(stream.get("timestamp"), errors="coerce")
    stream["atr_value"] = pd.to_numeric(stream.get("atr_value"), errors="coerce")

    lvl_id = eps["level_id"].astype(str)
    lg_support = eps[(eps["episode_source"] == "level_gate") & (lvl_id.str.contains(support_token, regex=False))].copy()
    lg_edge = eps[(eps["episode_source"] == "level_gate") & (lvl_id.str.contains(edge_token, regex=False))].copy()
    marker_dt = eps[(eps["episode_source"] == "marker_extract") & (eps.get("marker_type").astype("string") == "downtrend_break")].copy()

    print(f"session={session} episodes={len(eps)} snapshots={len(snaps)}")
    print(f"support_token={support_token}")
    print(f"edge_token={edge_token}")
    print(f"level_gate episodes: support={len(lg_support)} edge={len(lg_edge)}")
    if not marker_dt.empty:
        print(f"marker_extract downtrend_break episode_id={marker_dt.iloc[0]['episode_id']}")

    # Window into edge break
    w0 = M - int(args.window_minutes) * 60_000
    w1 = M

    support_ep_ids = set(lg_support["episode_id"].astype(str))
    edge_ep_ids = set(lg_edge["episode_id"].astype(str))

    support_sn = snaps[(snaps["episode_id"].astype(str).isin(support_ep_ids)) & (snaps["timestamp"] >= w0) & (snaps["timestamp"] <= w1)].copy()
    edge_sn = snaps[(snaps["episode_id"].astype(str).isin(edge_ep_ids)) & (snaps["timestamp"] >= w0) & (snaps["timestamp"] <= w1)].copy()

    print(f"\nWindow [{w0} .. {w1}] ({args.window_minutes}m into edge)")
    print(f"rows: support={len(support_sn)} edge={len(edge_sn)}")

    key = [
        "last_price",
        "signed_distance_to_level_atr",
        "touch_count",
        "cross_count_60s",
        "time_in_zone_rolling",
        "delta",
        "cvd_60s",
        "divergence_flag",
        "div_score",
        "buy_on_red",
        "sell_on_green",
        "large_trade_share_of_total_vol_10s",
        "spread_pct",
        "vol_z_60s",
        "trade_rate_z_60s",
    ]

    def summarize(df: pd.DataFrame, label: str) -> None:
        if df.empty:
            print(f"\n{label}: no rows")
            return
        print(f"\n{label} stats (p10/p50/p90/mean):")
        for c in key:
            if c not in df.columns:
                continue
            st = _pstats(df[c])
            if not st:
                continue
            print(f"- {c}: p50={st['p50']:.4g} p90={st['p90']:.4g} mean={st['mean']:.4g}")

    summarize(support_sn, f"SUPPORT({args.support_price}) before edge")
    summarize(edge_sn, f"EDGE({args.edge_price}) before edge")

    # Last 60s traces
    lastN = M - int(args.trace_seconds) * 1000
    for df, label in [
        (support_sn[support_sn["timestamp"] >= lastN].sort_values("timestamp"), f"SUPPORT last {args.trace_seconds}s"),
        (edge_sn[edge_sn["timestamp"] >= lastN].sort_values("timestamp"), f"EDGE last {args.trace_seconds}s"),
    ]:
        if df.empty:
            print(f"\n{label}: none")
            continue
        cols = [c for c in ["timestamp", "last_price", "signed_distance_to_level_atr", "delta", "cvd_60s", "divergence_flag", "buy_on_red", "sell_on_green", "spread_pct"] if c in df.columns]
        print(f"\n{label} (tail 12 rows):")
        print(df[cols].tail(12).to_string(index=False))

    # Marker_extract downtrend_break episode trace (snapshots) for the same last-N window
    if not marker_dt.empty:
        m_ep_id = str(marker_dt.iloc[0]["episode_id"])
        m_sn = snaps[(snaps["episode_id"].astype(str) == m_ep_id) & (snaps["timestamp"] >= lastN) & (snaps["timestamp"] <= M)].copy()
        m_sn = m_sn.sort_values("timestamp")
        if not m_sn.empty:
            # Derive distances to support/edge in ATR units using snapshot atr_value
            atr = pd.to_numeric(m_sn.get("atr_value"), errors="coerce")
            lp = pd.to_numeric(m_sn.get("last_price"), errors="coerce")
            ok = (atr > 0) & lp.notna()
            m_sn = m_sn.loc[ok].copy()
            if not m_sn.empty:
                m_sn["dist_support_atr"] = (lp.loc[ok] - float(args.support_price)) / atr.loc[ok]
                m_sn["dist_edge_atr"] = (lp.loc[ok] - float(args.edge_price)) / atr.loc[ok]

                cols = [c for c in [
                    "timestamp",
                    "last_price",
                    "dist_support_atr",
                    "dist_edge_atr",
                    "phase",
                    "delta",
                    "cvd_60s",
                    "divergence_flag",
                    "div_score",
                    "buy_on_red",
                    "sell_on_green",
                    "large_trade_share_of_total_vol_10s",
                    "spread_pct",
                    "vol_z_60s",
                ] if c in m_sn.columns]
                print(f"\nMARKER_EXTRACT downtrend_break snapshots trace (last {args.trace_seconds}s into marker):")
                print(m_sn[cols].to_string(index=False))

    # --- Continuous stream view (independent of level IDs) ---
    # Compute distances to the support/edge prices in ATR units, using stream atr_value.
    # This gives a consistent picture of where price was relative to those anchors even before edge level was added.
    st = stream[(stream["timestamp"] >= w0) & (stream["timestamp"] <= w1)].copy()
    if not st.empty:
        st["last_price"] = pd.to_numeric(st.get("last_price"), errors="coerce")
        st = st.dropna(subset=["timestamp", "last_price", "atr_value"])
        st = st[st["atr_value"] > 0]
        if not st.empty:
            st["dist_support_atr"] = (st["last_price"] - float(args.support_price)) / st["atr_value"]
            st["dist_edge_atr"] = (st["last_price"] - float(args.edge_price)) / st["atr_value"]

            def summarize_stream(df: pd.DataFrame, label: str) -> None:
                if df.empty:
                    print(f"\n{label}: no rows")
                    return
                cols = [
                    "dist_support_atr",
                    "dist_edge_atr",
                    "delta",
                    "cvd_60s",
                    "divergence_flag",
                    "div_score",
                    "buy_on_red",
                    "sell_on_green",
                    "large_trade_share_of_total_vol_10s",
                    "spread_pct",
                    "vol_z_60s",
                ]
                print(f"\n{label} (session_stream) stats:")
                for c in cols:
                    if c not in df.columns:
                        continue
                    stc = _pstats(df[c])
                    if not stc:
                        continue
                    print(f"- {c}: p50={stc['p50']:.4g} p90={stc['p90']:.4g} mean={stc['mean']:.4g}")

            summarize_stream(st, "STREAM window into edge")
            summarize_stream(st[st["timestamp"] >= lastN], f"STREAM last {args.trace_seconds}s into edge")


if __name__ == "__main__":
    main()


