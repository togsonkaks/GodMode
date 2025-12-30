from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


@dataclass(frozen=True)
class Session:
    root: Path
    date: str
    ticker: str
    session_id: str


def _load_parquet_dir(path: Path) -> pd.DataFrame:
    # Read fragment-by-fragment to avoid schema drift issues where some parts wrote a column as NULL type.
    dset = ds.dataset(str(path), format="parquet", partitioning="hive")
    dfs: list[pd.DataFrame] = []
    for frag in dset.get_fragments():
        try:
            tbl = frag.to_table()
        except Exception:
            continue
        dfs.append(tbl.to_pandas())
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def _episode_window_agg(snaps: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    window:
      - baseline: phase == baseline
      - stress: phase == stress
      - resolution: phase == resolution
      - early_stress_60s: first 60s (6 snapshots) of stress per episode
    """
    s = snaps.copy()
    s["timestamp"] = pd.to_numeric(s["timestamp"], errors="coerce")
    s = s.dropna(subset=["episode_id", "timestamp"])

    if window in {"baseline", "stress", "resolution"}:
        s = s[s["phase"] == window]
    elif window == "early_stress_60s":
        ss = s[s["phase"] == "stress"].copy()
        # first 6 rows by timestamp per episode
        ss = ss.sort_values(["episode_id", "timestamp"])
        ss["rank"] = ss.groupby("episode_id").cumcount()
        s = ss[ss["rank"] < 6].drop(columns=["rank"])
    else:
        raise ValueError(f"unknown window: {window}")

    # Numeric columns only (exclude ids)
    ignore = {"episode_id", "timestamp", "sequence_id"}
    num_cols = [c for c in s.columns if c not in ignore and pd.api.types.is_numeric_dtype(s[c])]
    # Aggregate: mean, std (optional), last
    g = s.groupby("episode_id", dropna=False)
    out = g[num_cols].mean().add_suffix(f"_{window}_mean")
    # Keep a compact set; std is helpful for “stability” but can explode column count.
    out_std = g[num_cols].std(ddof=0).fillna(0.0).add_suffix(f"_{window}_std")
    return pd.concat([out, out_std], axis=1)


def _rank_features(df: pd.DataFrame, label_col: str, top_k: int = 15) -> pd.DataFrame:
    """
    Returns a table with:
      - mean(win), mean(loss), diff, abs_diff, corr with label
    """
    y = df[label_col].astype(int)
    feats = df.drop(columns=[label_col]).select_dtypes(include=["number"])

    rows = []
    for c in feats.columns:
        x = pd.to_numeric(feats[c], errors="coerce")
        if x.isna().all():
            continue
        win = x[y == 1].dropna()
        loss = x[y == 0].dropna()
        if len(win) < 2 or len(loss) < 2:
            continue
        mw = float(win.mean())
        ml = float(loss.mean())
        diff = mw - ml
        corr = float(np.corrcoef(x.fillna(x.median()), y)[0, 1]) if x.notna().sum() >= 2 else 0.0
        rows.append(
            {
                "feature": c,
                "mean_win": mw,
                "mean_loss": ml,
                "diff": diff,
                "abs_diff": abs(diff),
                "corr": corr,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_diff", "corr"], ascending=[False, False]).head(top_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="data/output_rerun2/PCLA_20251223_140000")
    args = ap.parse_args()

    base = Path(args.base)
    episodes = _load_parquet_dir(base / "episodes")
    snaps = _load_parquet_dir(base / "snapshots")

    print(f"Loaded episodes={len(episodes):,} snapshots={len(snaps):,} from {base.as_posix()}")

    if "outcome" not in episodes.columns:
        raise SystemExit("episodes.outcome missing")

    print("\nOutcome counts:")
    print(episodes["outcome"].value_counts(dropna=False).to_string())

    # Binary label: win=1, loss=0 (ignore scratch/no-trade for now)
    ep = episodes[episodes["outcome"].isin(["win", "loss"])].copy()
    ep["y_win"] = (ep["outcome"] == "win").astype(int)

    # Build episode-level feature table: baseline vs early stress
    snaps = snaps.merge(ep[["episode_id", "y_win"]], on="episode_id", how="inner")

    baseline = _episode_window_agg(snaps, "baseline")
    early_stress = _episode_window_agg(snaps, "early_stress_60s")
    stress = _episode_window_agg(snaps, "stress")

    feat = ep.set_index("episode_id")[["y_win"]].join([baseline, early_stress, stress], how="left")

    # Add deltas: early_stress_mean - baseline_mean for a small set of “high signal” fields if present
    candidates = [
        "delta",
        "cvd_60s",
        "cvd_30s",
        "relative_aggression",
        "spread_pct",
        "signed_distance_to_level_atr",
        "vol_z_60s",
        "trade_rate_z_60s",
        "compression_index",
        "ema_confluence_score",
    ]
    for c in candidates:
        b = f"{c}_baseline_mean"
        e = f"{c}_early_stress_60s_mean"
        if b in feat.columns and e in feat.columns:
            feat[f"{c}_early_minus_base"] = feat[e] - feat[b]

    # Rank features (small sample; use as directional guidance)
    print("\nTop feature separators (episode-level; based on abs mean difference):")
    ranked = _rank_features(feat.dropna(axis=1, how="all"), label_col="y_win", top_k=20)
    if ranked.empty:
        print("(not enough data to rank)")
    else:
        with pd.option_context("display.max_colwidth", 120):
            print(ranked.to_string(index=False))

    # Metrics sanity (if present)
    for c in ["mfe", "mae", "time_to_mfe_ms", "time_to_failure_ms"]:
        if c in episodes.columns:
            s = pd.to_numeric(episodes[c], errors="coerce").dropna()
            if len(s):
                print(f"\n{c}: min={s.min():.4g} p50={s.quantile(0.5):.4g} p90={s.quantile(0.9):.4g} max={s.max():.4g} mean={s.mean():.4g}")


if __name__ == "__main__":
    main()


