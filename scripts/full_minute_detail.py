"""Full minute breakdown including above/below band activity."""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone

CHI = pytz.timezone('America/Chicago')
session = 'OMER_OMER_20251226_150400_4'
level = 14.42
band_lo = level * 0.9985  # 14.3984
band_hi = level * 1.0015  # 14.4416

start_local = CHI.localize(datetime(2025, 12, 26, 9, 13, 0))
end_local = CHI.localize(datetime(2025, 12, 26, 9, 15, 0))
start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
end_ms = int(end_local.astimezone(timezone.utc).timestamp() * 1000)

def has_W(x):
    if x is None: return False
    if isinstance(x, np.ndarray): return 'W' in x.tolist()
    if isinstance(x, (list, tuple, set)): return 'W' in x
    return 'W' in str(x)

tr = pd.read_parquet(f'data/output_clean/replay/{session}/trades.parquet').sort_values('ts_ms')
qt = pd.read_parquet(f'data/output_clean/replay/{session}/quotes.parquet').sort_values('ts_ms')

tr = tr[(tr.ts_ms >= start_ms) & (tr.ts_ms < end_ms)].copy()
tr = tr[~tr['conditions'].apply(has_W)].copy()

tr['time_chicago'] = pd.to_datetime(tr['ts_ms'], unit='ms', utc=True).dt.tz_convert('America/Chicago').dt.strftime('%H:%M:%S.%f')
tr['minute'] = pd.to_datetime(tr['ts_ms'], unit='ms', utc=True).dt.tz_convert('America/Chicago').dt.strftime('%H:%M')

qt_sub = qt[(qt.ts_ms >= start_ms - 60000) & (qt.ts_ms < end_ms)].copy()
j = pd.merge_asof(tr.sort_values('ts_ms'), qt_sub[['ts_ms', 'bid', 'ask']].sort_values('ts_ms'), on='ts_ms', direction='backward')
j = j.sort_values('ts_ms').reset_index(drop=True)

# Lee-Ready classification
sides = []
last_side = 'unknown'
last_price = None
for i, row in j.iterrows():
    p = float(row['price'])
    bid = row['bid'] if pd.notna(row['bid']) else None
    ask = row['ask'] if pd.notna(row['ask']) else None
    if bid is not None and ask is not None:
        if p >= float(ask):
            side = 'BUY'
        elif p <= float(bid):
            side = 'SELL'
        else:
            if last_price is not None:
                if p > last_price: side = 'BUY'
                elif p < last_price: side = 'SELL'
                else: side = last_side if last_side in ['BUY','SELL'] else 'unknown'
            else: side = 'unknown'
    else:
        if last_price is not None:
            if p > last_price: side = 'BUY'
            elif p < last_price: side = 'SELL'
            else: side = last_side if last_side in ['BUY','SELL'] else 'unknown'
        else: side = 'unknown'
    sides.append(side)
    last_price = p
    if side in ['BUY','SELL']: last_side = side

j['side'] = sides

# Zone classification
def zone(p):
    if p < band_lo: return 'BELOW'
    elif p > band_hi: return 'ABOVE'
    else: return 'IN_BAND'

j['zone'] = j['price'].apply(zone)

INST = 500

for minute in ['09:13', '09:14']:
    m = j[j['minute'] == minute]
    print('='*80)
    print(f'MINUTE: {minute} - ALL TRADES (not just in-band)')
    print(f'Band: {band_lo:.4f} to {band_hi:.4f}')
    print('='*80)
    
    for z in ['ABOVE', 'IN_BAND', 'BELOW']:
        zm = m[m['zone'] == z]
        if len(zm) == 0:
            continue
        buys = zm[zm['side'] == 'BUY']
        sells = zm[zm['side'] == 'SELL']
        buy_sh = int(buys['size'].sum())
        sell_sh = int(sells['size'].sum())
        delta = buy_sh - sell_sh
        
        inst_b = buys[buys['size'] >= INST]
        inst_s = sells[sells['size'] >= INST]
        inst_b_sh = int(inst_b['size'].sum())
        inst_s_sh = int(inst_s['size'].sum())
        
        label = {'ABOVE': 'ABOVE band (>14.44)', 'IN_BAND': 'IN BAND (14.40-14.44)', 'BELOW': 'BELOW band (<14.40)'}[z]
        print(f'  {label}')
        print(f'    Trades: {len(buys)} buy, {len(sells)} sell')
        print(f'    Shares: {buy_sh:,} buy, {sell_sh:,} sell  (delta: {delta:+,})')
        print(f'    INST:   {len(inst_b)} buy ({inst_b_sh:,}sh), {len(inst_s)} sell ({inst_s_sh:,}sh)')
        
        # Show large trades
        for _, r in inst_b.iterrows():
            t = r['time_chicago'][:12]
            sz = int(r['size'])
            px = r['price']
            print(f'      -> BUY  {t}  {sz:>5,} @ ${px:.4f}')
        for _, r in inst_s.iterrows():
            t = r['time_chicago'][:12]
            sz = int(r['size'])
            px = r['price']
            print(f'      -> SELL {t}  {sz:>5,} @ ${px:.4f}')
        print()
    
    # Totals
    buys = m[m['side'] == 'BUY']
    sells = m[m['side'] == 'SELL']
    total_buy = int(buys['size'].sum())
    total_sell = int(sells['size'].sum())
    print(f'  TOTAL: {total_buy:,} buy vs {total_sell:,} sell  (delta: {total_buy - total_sell:+,})')
    print()

