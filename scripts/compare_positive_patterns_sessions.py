from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_chain(rep: dict[str, Any]) -> dict[str, Any] | None:
    chains = rep.get("level_chains") or []
    return chains[0] if chains else None


def _scenario_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sc in ledger.get("scenarios", []) or []:
        key = sc.get("scenario_key") or "unknown_pending"
        pts = sc.get("positive_touches") or []
        n = len(pts)
        # rates for a stable subset of flags
        flags = ["pos_aggr", "pos_delta", "high_compression", "delta_flip", "div", "controlled_approach", "from_above"]
        counts = Counter()
        for e in pts:
            fl = e.get("flags") or {}
            for f in flags:
                if fl.get(f) is True:
                    counts[f] += 1
        rates = {f: (counts[f] / n if n else None) for f in flags}
        out[key] = {
            "n": n,
            "flag_rates": rates,
            "flag_counts": dict(counts),
            "combo_top": sc.get("combo_top") or [],
        }
    return out


def _fmt_rate(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{int(round(x * 100))}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="report.json path A")
    ap.add_argument("--b", required=True, help="report.json path B")
    ap.add_argument("--c", required=True, help="report.json path C")
    args = ap.parse_args()

    reps = [(Path(args.a), "A"), (Path(args.b), "B"), (Path(args.c), "C")]
    loaded: list[tuple[str, dict[str, Any]]] = []
    for p, tag in reps:
        rep = _load_report(p)
        meta = rep.get("_meta") or {}
        name = f"{meta.get('ticker')}|{meta.get('session_id')}"
        loaded.append((name, rep))

    chains = []
    for name, rep in loaded:
        ch = _find_chain(rep)
        if not ch:
            chains.append((name, None, {}))
            continue
        ledger = ch.get("positive_touch_ledger") or {}
        chains.append((name, ch, _scenario_summary(ledger)))

    print("=== 3-way Positive Pattern Comparison (scenario-stratified) ===")
    for name, ch, summ in chains:
        if not ch:
            print(f"- {name}: no level_chains (no comparison possible)")
            continue
        print(f"- {name}: schema={ch.get('schema')} scenarios={list(summ.keys())}")

    # Compare only shared scenario keys across those that exist
    scenario_sets = [set(s.keys()) for _, _, s in chains if s]
    shared = set.intersection(*scenario_sets) if scenario_sets else set()
    if not shared:
        # fallback: union, but mark missing
        shared = set.union(*scenario_sets) if scenario_sets else set()

    print("")
    print("Shared scenario keys:", ", ".join(sorted(shared)) if shared else "(none)")
    print("")

    flags = ["pos_aggr", "pos_delta", "high_compression", "delta_flip", "div", "controlled_approach", "from_above"]

    for sk in sorted(shared):
        print(f"--- Scenario: {sk} ---")
        # sample sizes
        sizes = []
        for name, _, summ in chains:
            n = (summ.get(sk) or {}).get("n") if summ else None
            sizes.append((name, n))
        print("sample_sizes:", ", ".join([f"{name}={n if n is not None else 'n/a'}" for name, n in sizes]))

        # per-flag rates
        for f in flags:
            row = []
            for name, _, summ in chains:
                r = ((summ.get(sk) or {}).get("flag_rates") or {}).get(f) if summ else None
            row.append(f"{name}:{_fmt_rate(r)}")
            print(f"  {f}: " + " | ".join(row))

        # top combos (top 3 each)
        print("  top_combos:")
        for name, _, summ in chains:
            combos = (summ.get(sk) or {}).get("combo_top") or []
            top3 = ", ".join([f"{c.get('combo')}({c.get('count')})" for c in combos[:3]]) if combos else "n/a"
            print(f"    {name}: {top3}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


