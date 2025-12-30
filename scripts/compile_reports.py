from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from godmode.reports.reader import list_report_paths, summarize_report_brief


def _write_csv(*, rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile stable per-session report.json into one CSV.")
    ap.add_argument("--root", type=str, required=True, help="storage.root_dir (e.g., data/output_clean)")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: <root>/reports_index.csv)")
    args = ap.parse_args()

    root_dir = Path(args.root)
    out_path = Path(args.out) if args.out else (root_dir / "reports_index.csv")

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
    _write_csv(rows=rows, out_path=out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


