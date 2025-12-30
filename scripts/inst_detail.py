#!/usr/bin/env python3
"""Show detailed institutional activity for a session."""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from datetime import datetime
import pytz

CHICAGO_TZ = pytz.timezone('America/Chicago')

def format_ts(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.utc)
    return dt.astimezone(CHICAGO_TZ).strftime('%H:%M:%S')

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
    
    for ch in report.get("level_chains", []):
        for lv in ch.get("levels", []):
            tps = lv.get("touch_packets", [])
            sw = lv.get("session_wide", {})
            
            print("=" * 70)
            print(f"{session_id} - DETAILED INSTITUTIONAL ACTIVITY")
            print("=" * 70)
            print()
            
            # Overall totals
            large_buy = sw.get("band_large_buy_volume", 0) or 0
            large_sell = sw.get("band_large_sell_volume", 0) or 0
            
            print("OVERALL TOTALS (in band):")
            print(f"  Total Large Buy Shares:  {large_buy:,}")
            print(f"  Total Large Sell Shares: {large_sell:,}")
            print(f"  Net:                     {large_buy - large_sell:+,}")
            print()
            
            # Count INST trades and collect timing
            inst_buys = []
            inst_sells = []
            
            for tp in tps:
                ts = tp.get("ts_ms", 0)
                time_str = format_ts(ts) if ts else "N/A"
                touch_num = tp.get("touch_number", 0)
                
                if tp.get("institutional_buying_flag"):
                    buy_vol = tp.get("large_buy_volume_30s", 0) or 0
                    inst_buys.append({
                        "touch": touch_num,
                        "time": time_str,
                        "volume": buy_vol,
                        "bounce": (tp.get("bounce_return_30s_pct", 0) or 0) * 100
                    })
                
                if tp.get("institutional_selling_flag"):
                    sell_vol = tp.get("large_sell_volume_30s", 0) or 0
                    inst_sells.append({
                        "touch": touch_num,
                        "time": time_str,
                        "volume": sell_vol,
                        "bounce": (tp.get("bounce_return_30s_pct", 0) or 0) * 100
                    })
            
            print(f"INSTITUTIONAL BUYS: {len(inst_buys)} trades")
            print("-" * 50)
            if inst_buys:
                total_buy_vol = 0
                for ib in inst_buys:
                    print(f"  Touch #{ib['touch']:2} @ {ib['time']} | Vol: {ib['volume']:>8,} | Bounce: {ib['bounce']:+.1f}%")
                    total_buy_vol += ib["volume"]
                print(f"  TOTAL INST_BUY VOLUME: {total_buy_vol:,}")
            else:
                print("  None")
            
            print()
            print(f"INSTITUTIONAL SELLS: {len(inst_sells)} trades")
            print("-" * 50)
            if inst_sells:
                total_sell_vol = 0
                for iss in inst_sells:
                    bounce_status = "ABSORBED" if iss["bounce"] > 0 else "CONFIRMED"
                    print(f"  Touch #{iss['touch']:2} @ {iss['time']} | Vol: {iss['volume']:>8,} | Bounce: {iss['bounce']:+.1f}% ({bounce_status})")
                    total_sell_vol += iss["volume"]
                print(f"  TOTAL INST_SELL VOLUME: {total_sell_vol:,}")
            else:
                print("  None")
            
            print()
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"  INST_BUY:  {len(inst_buys)} trades")
            print(f"  INST_SELL: {len(inst_sells)} trades")
            absorbed = sum(1 for s in inst_sells if s["bounce"] > 0)
            confirmed = sum(1 for s in inst_sells if s["bounce"] <= 0)
            print(f"  Absorbed:  {absorbed} / {len(inst_sells)}")
            print(f"  Confirmed: {confirmed} / {len(inst_sells)}")

if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "ASPC_20251226_162000"
    main(session_id)

