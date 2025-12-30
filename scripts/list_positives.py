#!/usr/bin/env python3
"""List all positive touches from multiple SOBR sessions and find similarities."""

import json
from pathlib import Path
from collections import Counter

def main():
    sessions = [
        ("MWG_161030", "data/output_clean/reports/date=2025-12-24/ticker=MWG/session=MWG_20251224_161030/report.json"),
        ("MWG_160300", "data/output_clean/reports/date=2025-12-24/ticker=MWG/session=MWG_20251224_160300/report.json"),
        ("SOBR_180430", "data/output_clean/reports/date=2025-12-24/ticker=SOBR/session=SOBR_20251224_180430/report.json"),
        ("SOBR_172530", "data/output_clean/reports/date=2025-12-24/ticker=SOBR/session=SOBR_20251224_172530/report.json"),
        ("SOBR_144130", "data/output_clean/reports/date=2025-12-24/ticker=SOBR/session=SOBR_20251224_144130/report.json"),
    ]

    all_session_flags = {}
    all_session_combos = {}
    
    for name, path in sessions:
        p = Path(path)
        if not p.exists():
            print(f"{name}: report not found")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ch = (d.get("level_chains") or [{}])[0]
        ledger = ch.get("positive_touch_ledger") or {}
        all_pos = ledger.get("all_positive_touches") or []

        print(f"\n{'='*60}")
        print(f"  {name}  |  {len(all_pos)} positive touches")
        print(f"{'='*60}")

        session_flags = Counter()
        session_combos = Counter()

        for i, tp in enumerate(all_pos, 1):
            flags = tp.get("flags") or {}
            flag_list = sorted([k for k, v in flags.items() if v])
            combo = "+".join(flag_list) if flag_list else "none"

            for f in flag_list:
                session_flags[f] += 1
            session_combos[combo] += 1

            bounce = round((tp.get("bounce_return_30s_pct") or 0) * 100, 3)
            delta = tp.get("band_delta_0_30s")
            aggr = tp.get("rel_aggr_0_30s")
            comp = tp.get("compression_at_touch")
            approach = tp.get("approach_type")
            from_side = tp.get("from_side")

            print(f"  #{i}: bounce={bounce}%  delta30={delta}  aggr30={round(aggr,3) if aggr else 'n/a'}  comp={round(comp,3) if comp else 'n/a'}")
            print(f"       approach={approach}  from={from_side}")
            print(f"       flags: {combo}")
            print()

        all_session_flags[name] = session_flags
        all_session_combos[name] = session_combos

    # Find similarities
    print("\n" + "="*60)
    print("  SIMILARITIES ACROSS ALL 3 SESSIONS")
    print("="*60)

    # Flag frequency across sessions
    print("\nFlag frequency (how many positives had each flag):")
    all_flags = set()
    for sf in all_session_flags.values():
        all_flags.update(sf.keys())
    
    session_names = ["MWG_161030", "MWG_160300", "SOBR_180430", "SOBR_172530", "SOBR_144130"]
    for flag in sorted(all_flags):
        counts = []
        for name in session_names:
            c = all_session_flags.get(name, {}).get(flag, 0)
            counts.append(f"{name}:{c}")
        print(f"  {flag}: {', '.join(counts)}")

    # Combo frequency
    print("\nTop combos per session:")
    for name in session_names:
        combos = all_session_combos.get(name, {})
        top = combos.most_common(3)
        print(f"  {name}:")
        for combo, count in top:
            print(f"    - {combo}: {count}")

    # Shared combos (appear in all sessions)
    print(f"\nShared combos (appear in ALL {len(session_names)} sessions):")
    combo_sets = [set(all_session_combos.get(name, {}).keys()) for name in session_names]
    combo_sets = [cs for cs in combo_sets if cs]  # filter empty
    if combo_sets:
        shared = combo_sets[0]
        for cs in combo_sets[1:]:
            shared = shared.intersection(cs)
        for combo in sorted(shared):
            counts = [all_session_combos.get(name, {}).get(combo, 0) for name in session_names]
            print(f"  {combo}: {' + '.join(str(c) for c in counts)} = {sum(counts)} total")
        if not shared:
            print("  (no combos shared by ALL sessions)")
    else:
        print("  (no combo data)")


if __name__ == "__main__":
    main()

