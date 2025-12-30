from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_reports(root_dir: Path) -> list[Path]:
    reports_root = root_dir / "reports"
    if not reports_root.exists():
        return []
    return sorted(reports_root.rglob("report.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True, help="Root output dir (e.g. data/output_clean)")
    ap.add_argument(
        "--out-dir",
        default="reports_index",
        help="Subdir under root-dir to write compiled outputs",
    )
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = root_dir / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregations:
    # - per scenario_key: counts of flags and combos
    # - map where each flag/combo occurred (sessions)
    scenario_flag_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_combo_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_sessions: dict[str, set[str]] = defaultdict(set)
    scenario_tickers: dict[str, set[str]] = defaultdict(set)

    where_flag: dict[tuple[str, str], set[str]] = defaultdict(set)  # (scenario, flag) -> set(session)
    where_combo: dict[tuple[str, str], set[str]] = defaultdict(set)  # (scenario, combo) -> set(session)

    total_positive_by_scenario: Counter[str] = Counter()

    report_paths = _iter_reports(root_dir)
    if not report_paths:
        print("No report.json files found.")
        return 2

    for rp in report_paths:
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = rep.get("_meta") or {}
        date = meta.get("date") or rep.get("date")
        ticker = meta.get("ticker") or rep.get("ticker")
        session_id = meta.get("session_id") or rep.get("session_id")
        sess_key = f"{date}|{ticker}|{session_id}"

        for ch in rep.get("level_chains", []) or []:
            ledger = ch.get("positive_touch_ledger") or {}
            if ledger.get("schema") != "positive_touch_ledger_v1":
                continue
            for sc in ledger.get("scenarios", []) or []:
                sk = sc.get("scenario_key") or "unknown_pending"
                pts = sc.get("positive_touches") or []
                total_positive_by_scenario[sk] += len(pts)
                scenario_sessions[sk].add(sess_key)
                if ticker:
                    scenario_tickers[sk].add(str(ticker))

                for e in pts:
                    flags = (e.get("flags") or {})
                    # flags
                    for fk, fv in flags.items():
                        if fv:
                            scenario_flag_counts[sk][fk] += 1
                            where_flag[(sk, fk)].add(sess_key)
                    # combo
                    combo = "+".join(sorted([k for k, v in flags.items() if v]))
                    if combo:
                        scenario_combo_counts[sk][combo] += 1
                        where_combo[(sk, combo)].add(sess_key)

    # Write JSON summary
    summary = {
        "schema": "positive_patterns_index_v1",
        "root_dir": str(root_dir),
        "scenarios": {},
    }
    for sk in sorted(total_positive_by_scenario.keys()):
        summary["scenarios"][sk] = {
            "positive_touch_count": int(total_positive_by_scenario[sk]),
            "sessions": len(scenario_sessions.get(sk, set())),
            "tickers": len(scenario_tickers.get(sk, set())),
            "flag_counts": dict(scenario_flag_counts.get(sk, Counter())),
            "combo_top": [
                {"combo": c, "count": int(n)}
                for c, n in scenario_combo_counts.get(sk, Counter()).most_common(20)
            ],
        }

    (out_dir / "positive_patterns.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write CSV for flags
    with (out_dir / "positive_flags.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_key", "flag", "count", "sessions_with_flag"])
        for (sk, fk), sess in sorted(where_flag.items(), key=lambda x: (x[0][0], -len(x[1]), x[0][1])):
            w.writerow([sk, fk, int(scenario_flag_counts[sk][fk]), len(sess)])

    # Write CSV for combos
    with (out_dir / "positive_combos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_key", "combo", "count", "sessions_with_combo"])
        for (sk, ck), sess in sorted(where_combo.items(), key=lambda x: (x[0][0], -len(x[1]), x[0][1])):
            w.writerow([sk, ck, int(scenario_combo_counts[sk][ck]), len(sess)])

    print("Wrote:")
    print(" -", (out_dir / "positive_patterns.json").as_posix())
    print(" -", (out_dir / "positive_flags.csv").as_posix())
    print(" -", (out_dir / "positive_combos.csv").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


