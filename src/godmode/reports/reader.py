from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from godmode.reports.paths import ReportKey, report_path


def load_report_json(*, root_dir: Path, key: ReportKey) -> dict[str, Any] | None:
    p = report_path(root_dir=root_dir, key=key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_report_paths(*, root_dir: Path) -> list[Path]:
    base = root_dir / "reports"
    if not base.exists():
        return []
    return sorted(base.glob("date=*/ticker=*/session=*/report.json"))


def summarize_report_brief(report: dict[str, Any]) -> dict[str, Any]:
    """
    Converts report.json into a compact, UI-friendly summary.
    This is intentionally small and stable; raw Parquet remains the source of truth.
    """
    episodes_total = int(report.get("episodes_total") or report.get("episodes") or 0)
    oc = report.get("outcome_counts") or {}
    win = int(oc.get("win", 0) or 0)
    loss = int(oc.get("loss", 0) or 0)
    scratch = int(oc.get("scratch", 0) or 0)
    denom = win + loss
    win_rate = (win / denom) if denom > 0 else None

    m = report.get("metrics") or {}
    mfe = m.get("mfe") or {}
    mae = m.get("mae") or {}

    def _get(d: dict[str, Any], k: str) -> float | None:
        try:
            v = d.get(k, None)
            return float(v) if v is not None else None
        except Exception:
            return None

    top_feature = None
    fr = report.get("feature_ranking") or []
    if isinstance(fr, list) and fr:
        f0 = fr[0] or {}
        top_feature = f0.get("feature", None)

    return {
        "episodes_total": episodes_total,
        "win": win,
        "loss": loss,
        "scratch": scratch,
        "win_rate": win_rate,  # 0..1 or None
        "mfe_p50": _get(mfe, "p50"),
        "mae_p90": _get(mae, "p90"),
        "top_feature": top_feature,
    }


