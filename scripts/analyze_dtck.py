#!/usr/bin/env python3
"""Analyze DTCK_20251226_173400 session with touch scoring."""
import sys
sys.path.insert(0, "src")

from godmode.webapp.store import Store
from godmode.reports.writer import write_session_report
from pathlib import Path
import json

def main():
    store = Store(Path("data/output_clean"))
    report = store.compute_session_report(date="2025-12-26", ticker="DTCK", session_id="DTCK_20251226_191000")

    # Write the updated report
    write_session_report(root_dir=Path("data/output_clean"), ticker="DTCK", session_id="DTCK_20251226_191000")

    print("=" * 60)
    print("DTCK_20251226_173400 ANALYSIS")
    print("=" * 60)
    
    # Basic info
    print(f"\nEpisodes: {report.get('episodes_total', 0)}")
    print(f"Outcomes: {report.get('outcome_counts', {})}")
    
    # Level chains
    if report.get("level_chains"):
        for ch in report["level_chains"]:
            print(f"\n--- Chain: {ch.get('chain_id')} ---")
            
            for lv in ch.get("levels", []):
                price = lv.get("level_price")
                kind = lv.get("level_kind")
                outcome = lv.get("level_outcome")
                
                print(f"\n[{kind.upper()} @ {price}] outcome={outcome}")
                
                # Session-wide band stats
                sw = lv.get("session_wide", {})
                print(f"  Band Delta: {sw.get('band_delta_sum', 0):,.0f}")
                print(f"  Band Aggression: {sw.get('band_relative_aggression_mean', 0):.3f}")
                print(f"  Large Trade Count: {sw.get('band_large_trade_count', 0)}")
                large_buy = sw.get("band_large_buy_volume", 0) or 0
                large_sell = sw.get("band_large_sell_volume", 0) or 0
                if large_buy + large_sell > 0:
                    sell_pct = large_sell / (large_buy + large_sell) * 100
                    print(f"  Large Sell %: {sell_pct:.1f}%")
                
                # Touch scoring
                ts = lv.get("touch_scoring")
                if ts:
                    print(f"\n  === TOUCH SCORING ===")
                    print(f"  Verdict: {ts.get('verdict')}")
                    print(f"  Touches: {ts.get('touches')}")
                    print(f"  Avg Score: {ts.get('avg_score')}")
                    print(f"  Peak Score: {ts.get('peak_score')} (touch #{ts.get('peak_touch_number')})")
                    print(f"  Cum Final: {ts.get('cum_final_score')}")
                    if ts.get("notes"):
                        print(f"  Notes: {', '.join(ts['notes'])}")
                    
                    print(f"\n  Top Bullish Drivers:")
                    for r in ts.get("top_plus", [])[:5]:
                        print(f"    + {r['reason']} ({r['count']}x)")
                    
                    print(f"\n  Top Bearish Drivers:")
                    for r in ts.get("top_minus", [])[:5]:
                        print(f"    - {r['reason']} ({r['count']}x)")
                else:
                    print("  (no touch scoring)")
                
                # Level flags
                flags = lv.get("level_flags", {})
                if flags:
                    print(f"\n  Green Flags: {flags.get('green', [])}")
                    print(f"  Red Flags: {flags.get('red', [])}")
                
                # Touch packets summary
                tps = lv.get("touch_packets", [])
                if tps:
                    print(f"\n  Touch Packets ({len(tps)} touches):")
                    for tp in tps:
                        tn = tp.get("touch_number", "?")
                        approach = tp.get("approach_type", "?")
                        delta_30 = tp.get("band_delta_0_30s", 0)
                        aggr_30 = tp.get("rel_aggr_0_30s", 0)
                        bounce = tp.get("bounce_return_30s_pct", 0) or 0
                        inst_buy = tp.get("institutional_buying_flag", False)
                        inst_sell = tp.get("institutional_selling_flag", False)
                        buy_ratio = tp.get("large_trade_buy_ratio_30s", 0) or 0
                        score = tp.get("score_v1", "?")
                        cum = tp.get("cum_score_v1", "?")
                        
                        flags_str = ""
                        if inst_buy:
                            flags_str += " INST_BUY"
                        if inst_sell:
                            flags_str += " INST_SELL"
                        
                        print(f"    #{tn}: approach={approach}, delta_30s={delta_30:,.0f}, aggr={aggr_30:.2f}, "
                              f"bounce={bounce*100:.2f}%, buy_ratio={buy_ratio:.1%}{flags_str}")
                        print(f"        -> score={score}, cumulative={cum}")
                        
                        plus = tp.get("reasons_plus", [])
                        minus = tp.get("reasons_minus", [])
                        if plus:
                            print(f"        + {', '.join(plus[:4])}")
                        if minus:
                            print(f"        - {', '.join(minus[:4])}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
if __name__ == "__main__":
    main()

