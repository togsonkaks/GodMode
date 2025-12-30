from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x  # not NaN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to report.json")
    args = ap.parse_args()

    p = Path(args.report)
    d = json.loads(p.read_text(encoding="utf-8"))

    chains = d.get("level_chains") or []
    if not chains:
        print("No level_chains found in report.")
        return 2

    ch = chains[0]
    ms = ch.get("main_support") or {}
    tps = ch.get("touch_packets") or []
    rps = ch.get("reclaim_packets") or []

    print("=== Support Chain Analysis (Addendum K) ===")
    print(f"report: {p.as_posix()}")
    print(f"schema: {ch.get('schema')}")
    print(f"touch_packets: {len(tps)}")
    print(f"reclaim_packets: {len(rps)}")
    print(
        "main_support:",
        ms.get("main_support_id"),
        "score:",
        ms.get("main_support_score"),
        "focus:",
        ms.get("focus_level"),
    )

    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tp in tps:
        by_level[str(tp.get("level_price"))].append(tp)

    for lvl in sorted(by_level.keys(), key=lambda x: float(x), reverse=True):
        rows = by_level[lvl]
        outs = Counter([r.get("touch_outcome") for r in rows])
        flips = sum(1 for r in rows if r.get("delta_flip_flag") is True)
        comps = [r.get("compression_at_touch") for r in rows if _is_num(r.get("compression_at_touch"))]
        bounces = [r.get("bounce_return_30s_pct") for r in rows if _is_num(r.get("bounce_return_30s_pct"))]
        pos = sum(1 for x in bounces if x > 0)

        print("")
        print(f"LEVEL {lvl}  touches={len(rows)}")
        print("  outcomes:", dict(outs))
        print("  delta_flip_rate:", round(flips / len(rows), 3))
        if comps:
            print(
                "  compression_at_touch:",
                "mean",
                round(mean(comps), 3),
                "p50",
                round(median(comps), 3),
                "max",
                round(max(comps), 3),
            )
        if bounces:
            print(
                "  bounce_30s_pct:",
                "mean",
                round(mean(bounces), 4),
                "p50",
                round(median(bounces), 4),
                "pos_rate",
                round(pos / len(bounces), 3),
            )

        top = sorted(
            [r for r in rows if _is_num(r.get("bounce_return_30s_pct"))],
            key=lambda r: float(r.get("bounce_return_30s_pct") or 0.0),
            reverse=True,
        )[:3]

        for r in top:
            print(
                "  TOP bounce_30s_pct=",
                round(float(r.get("bounce_return_30s_pct") or 0.0), 6),
                "ts=",
                r.get("touch_ts_ms"),
                "out=",
                r.get("touch_outcome"),
                "approach=",
                r.get("approach_type"),
                "preDelta60=",
                r.get("approach_delta_60s"),
                "postDelta30=",
                r.get("band_delta_0_30s"),
                "rel30=",
                r.get("rel_aggr_0_30s"),
                "comp=",
                r.get("compression_at_touch"),
            )

    if rps:
        print("")
        print("RECLAIMS")
        for rp in rps:
            print(
                "  reclaimed=",
                rp.get("reclaimed_level_price"),
                "after_tag=",
                rp.get("tagged_deeper_level_price"),
                "t_reclaim_s=",
                rp.get("time_to_reclaim_s"),
                "snapback=",
                rp.get("snapback_flag"),
                "hold30=",
                rp.get("reclaim_hold_30s"),
                "delta30=",
                rp.get("reclaim_band_delta_30s"),
                "aggr30=",
                rp.get("reclaim_rel_aggr_30s"),
                "drop_atr=",
                rp.get("drop_atr_to_deeper"),
                "flush=",
                rp.get("flush_flag"),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


