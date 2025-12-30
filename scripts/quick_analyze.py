#!/usr/bin/env python3
"""Quick session analysis."""
import sys
import json
from pathlib import Path

def main(session_id: str):
    parts = session_id.split("_")
    ticker = parts[0]
    date_str = parts[1]
    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    report = json.loads(report_path.read_text())
    
    print("SESSION INFO:")
    print(f"Outcome counts: {report.get('outcome_counts', {})}")
    print()
    
    for ch in report.get("level_chains", []):
        for lv in ch.get("levels", []):
            print(f"Level: {lv.get('level_kind')} @ {lv.get('level_price')}")
            print(f"Outcome: {lv.get('level_outcome')}")
            
            sw = lv.get("session_wide", {})
            print(f"Band Delta: {sw.get('band_delta_sum', 0):+,.0f}")
            print(f"Large Trade Count: {sw.get('band_large_trade_count', 0)}")
            
            large_buy = sw.get("band_large_buy_volume", 0) or 0
            large_sell = sw.get("band_large_sell_volume", 0) or 0
            if large_buy + large_sell > 0:
                sell_pct = large_sell / (large_buy + large_sell) * 100
                print(f"Large Sell %: {sell_pct:.1f}%")
            
            print()
            print("TOUCH PACKETS:")
            for tp in lv.get("touch_packets", []):
                tn = tp.get("touch_number")
                approach = tp.get("approach_type", "?")
                delta = tp.get("band_delta_0_30s", 0) or 0
                aggr = tp.get("rel_aggr_0_30s", 0) or 0
                bounce = (tp.get("bounce_return_30s_pct", 0) or 0) * 100
                buy_ratio = (tp.get("large_trade_buy_ratio_30s", 0) or 0) * 100
                large_count = tp.get("large_trade_count_30s", 0) or 0
                inst_buy = tp.get("institutional_buying_flag", False)
                inst_sell = tp.get("institutional_selling_flag", False)
                
                flags = ""
                if inst_buy:
                    flags += " INST_BUY"
                if inst_sell:
                    flags += " INST_SELL"
                
                print(f"  #{tn}: approach={approach}, delta={delta:+,.0f}, aggr={aggr:+.2f}, "
                      f"bounce={bounce:+.2f}%, buy_ratio={buy_ratio:.0f}%, large_count={large_count:.0f}{flags}")

if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "DTCK_20251226_162200"
    main(session_id)

