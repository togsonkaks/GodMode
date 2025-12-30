#!/usr/bin/env python
"""Quick script to check session report."""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from datetime import datetime

session_id = sys.argv[1] if len(sys.argv) > 1 else "ASPC_20251226_200000"
ticker = session_id.split("_")[0]
date_part = session_id.split("_")[1]
date = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"

report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
print(f"Reading: {report_path}\n")

report = json.loads(report_path.read_text())

print(f"Level Chains: {len(report.get('level_chains', []))}")
for chain in report.get('level_chains', []):
    print(f"\n=== Chain: {chain.get('chain_id')} ===")
    for lv in chain.get('levels', []):
        ind_name = lv.get('indicator_name')
        level_price = lv.get('level_price')
        
        if ind_name:
            print(f"\n  INDICATOR: {ind_name} (mean price: {level_price:.4f})")
        else:
            print(f"\n  PRICE LEVEL: {level_price} ({lv.get('level_kind')})")
        
        touches = lv.get('touch_packets', [])
        print(f"  Touches: {len(touches)}")
        
        if touches:
            first = touches[0]
            last = touches[-1]
            
            first_time = datetime.utcfromtimestamp(first.get('touch_ts_ms', 0) / 1000)
            last_time = datetime.utcfromtimestamp(last.get('touch_ts_ms', 0) / 1000)
            
            # For indicator levels, show the indicator value at that touch
            first_ind_val = first.get('indicator_value_at_touch', first.get('level_price', 0))
            last_ind_val = last.get('indicator_value_at_touch', last.get('level_price', 0))
            
            print(f"  First touch: {first_time.strftime('%H:%M:%S')} @ indicator={first_ind_val:.4f}")
            print(f"  Last touch:  {last_time.strftime('%H:%M:%S')} @ indicator={last_ind_val:.4f}")
            
            # Price change
            first_price = first.get('last_price', 0)
            last_price = last.get('last_price', 0)
            if first_price:
                pct_change = (last_price - first_price) / first_price * 100
                print(f"  Price change: {pct_change:+.2f}%")
            
            # Count INST flags
            inst_buy = sum(1 for t in touches if t.get('institutional_buying_flag'))
            inst_sell = sum(1 for t in touches if t.get('institutional_selling_flag'))
            print(f"  INST_BUY: {inst_buy}, INST_SELL: {inst_sell}")

