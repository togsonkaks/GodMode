#!/usr/bin/env python3
"""Analyze ECDA session."""

import sys
sys.path.insert(0, 'src')

from godmode.webapp.store import Store
from pathlib import Path
import json

store = Store(root_dir=Path('data/output_clean'))
report = store.compute_session_report(date='2025-12-26', ticker='PCLA', session_id='PCLA_20251226_162000')

# Save report
report_dir = Path('data/output_clean/reports/date=2025-12-26/ticker=PCLA/session=PCLA_20251226_162000')
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')

print('Report generated!')
print()

# Print key stats
if 'level_chains' in report and report['level_chains']:
    chain = report['level_chains'][0]
    
    # Volume summary first
    for lv in chain.get('levels', []):
        sw = lv.get('session_wide', {})
        price = lv.get('level_price')
        
        print('=== VOLUME SUMMARY ===')
        print(f'Level: ${price}')
        
        total_vol = sw.get('total_volume_sum', 0) or 0
        band_vol = sw.get('band_total_volume_sum', 0) or 0
        band_time_s = sw.get('time_in_band_s', 0) or 1
        trade_count = sw.get('trade_count_sum', 0) or 0
        band_trades = sw.get('band_trade_count_sum', 0) or 0
        
        # Calculate averages
        duration_min = band_time_s / 60
        avg_vol_per_min = band_vol / duration_min if duration_min > 0 else 0
        avg_trades_per_min = band_trades / duration_min if duration_min > 0 else 0
        avg_shares_per_trade = band_vol / band_trades if band_trades > 0 else 0
        
        # Large trade stats
        large_count = sw.get('band_large_trade_count', 0) or 0
        large_buy = sw.get('band_large_buy_volume', 0) or 0
        large_sell = sw.get('band_large_sell_volume', 0) or 0
        large_total = large_buy + large_sell
        sell_pct = (large_sell / large_total * 100) if large_total > 0 else 0
        
        print(f'  Total Session Volume: {total_vol:,.0f} shares')
        print(f'  Band Volume: {band_vol:,.0f} shares')
        print(f'  Time in Band: {band_time_s:.0f}s ({duration_min:.1f} min)')
        print(f'  Avg Volume/min: {avg_vol_per_min:,.0f} shares')
        print(f'  Avg Trades/min: {avg_trades_per_min:.0f}')
        print(f'  Avg Shares/Trade: {avg_shares_per_trade:.0f}')
        print()
        print(f'  Large Trades: {large_count}')
        print(f'  Large Buy Vol: {large_buy:,.0f}')
        print(f'  Large Sell Vol: {large_sell:,.0f}')
        print(f'  Large Sell %: {sell_pct:.1f}%')
        print()
    
if 'level_chains' in report and report['level_chains']:
    chain = report['level_chains'][0]
    print('=== LEVEL CHAIN ANALYSIS ===')
    print()
    
    for lv in chain.get('levels', []):
        price = lv.get('level_price')
        kind = lv.get('level_kind')
        print(f'Level: ${price} ({kind})')
        print(f'  Outcome: {lv.get("level_outcome")}')
        
        # Session-wide stats
        sw = lv.get('session_wide', {})
        print(f'  Band Delta: {sw.get("band_delta_sum", 0):,.0f}')
        print(f'  Time in Band: {sw.get("time_in_band_s", 0):.0f}s')
        print(f'  Touches: {sw.get("touches", 0)}')
        print(f'  Breaks: {sw.get("breaks", 0)}')
        print(f'  Rejects: {sw.get("rejects", 0)}')
        
        # Large trade stats
        print(f'  Large Trade Count: {sw.get("band_large_trade_count", 0)}')
        print(f'  Large Buy Vol: {sw.get("band_large_buy_volume_sum", 0):,.0f}')
        print(f'  Large Sell Vol: {sw.get("band_large_sell_volume_sum", 0):,.0f}')
        buy_ratio = sw.get('band_large_trade_buy_ratio', 0) or 0
        print(f'  Large Trade Buy Ratio: {buy_ratio*100:.1f}%')
        print()
    
    # Touch packets
    packets = chain.get('touch_packets', [])
    if packets:
        print(f'=== TOUCH PACKETS ({len(packets)} touches) ===')
        for i, tp in enumerate(packets):
            print(f'Touch {i+1}:')
            print(f'  Compression: {tp.get("compression_at_touch", 0):.3f}')
            print(f'  Delta 0-30s: {tp.get("band_delta_0_30s", 0):,.0f}')
            print(f'  Rel Aggr 0-30s: {tp.get("rel_aggr_0_30s", 0):.3f}')
            bounce = tp.get("bounce_return_30s_pct", 0) or 0
            print(f'  Bounce 30s: {bounce*100:.2f}%')
            print(f'  INST_BUY: {tp.get("institutional_buying_flag", False)}')
            print(f'  INST_SELL: {tp.get("institutional_selling_flag", False)}')
            
            # Large trade details
            large_count = tp.get("large_trade_count_30s", 0) or 0
            large_buy = tp.get("large_buy_volume_30s", 0) or 0
            large_sell = tp.get("large_sell_volume_30s", 0) or 0
            if large_count > 0:
                print(f'  Large Trades: {large_count}')
                print(f'  Large Buy/Sell: {large_buy:,.0f} / {large_sell:,.0f}')
            print()
    
    # Level stats
    lstats = chain.get('level_stats', {})
    if lstats:
        print('=== LEVEL STATS ===')
        for level_id, stats in lstats.items():
            print(f'{level_id}:')
            print(f'  Touch count: {stats.get("touch_count", 0)}')
            pos_rate = stats.get("positive_touch_rate", 0) or 0
            print(f'  Positive touch rate: {pos_rate*100:.0f}%')
            comp_mean = stats.get("compression_mean", 0) or 0
            print(f'  Compression mean: {comp_mean:.3f}')
            delta_mean = stats.get("delta_mean", 0) or 0
            print(f'  Delta mean: {delta_mean:,.0f}')
            inst_buy = stats.get("inst_buy_rate", 0) or 0
            inst_sell = stats.get("inst_sell_rate", 0) or 0
            print(f'  INST_BUY rate: {inst_buy*100:.0f}%')
            print(f'  INST_SELL rate: {inst_sell*100:.0f}%')
            print()
    
    # Flags
    lflags = chain.get('level_flags', {})
    if lflags:
        print('=== FLAGS ===')
        for level_id, flags in lflags.items():
            green = flags.get('green', [])
            red = flags.get('red', [])
            print(f'{level_id}:')
            if green:
                print(f'  GREEN: {green}')
            if red:
                print(f'  RED: {red}')
            print()

# Positive touch ledger
if 'positive_touch_ledger' in report:
    ledger = report['positive_touch_ledger']
    print('=== POSITIVE TOUCH LEDGER ===')
    for scenario, data in ledger.items():
        touches = data.get('touches', [])
        if touches:
            print(f'{scenario}: {len(touches)} positive touches')
            for t in touches[:3]:  # Show first 3
                flags = t.get('flags', {})
                active = [k for k, v in flags.items() if v]
                print(f'  Flags: {active}')
    print()

print('Full report saved to reports directory.')

