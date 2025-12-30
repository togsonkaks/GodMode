"""Simple totals for 09:13 and 09:14."""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone

CHI = pytz.timezone('America/Chicago')
session = 'OMER_OMER_20251226_150400_4'

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
        if p >= float(ask): side = 'BUY'
        elif p <= float(bid): side = 'SELL'
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
INST = 500

print('='*70)
print('09:13 + 09:14 - ALL TRADES (SIMPLE TOTALS)')
print('='*70)
print()

for minute in ['09:13', '09:14']:
    m = j[j['minute'] == minute]
    buys = m[m['side'] == 'BUY']
    sells = m[m['side'] == 'SELL']
    buy_sh = int(buys['size'].sum())
    sell_sh = int(sells['size'].sum())
    inst_b = buys[buys['size'] >= INST]
    inst_s = sells[sells['size'] >= INST]
    inst_b_sh = int(inst_b['size'].sum()) if len(inst_b) > 0 else 0
    inst_s_sh = int(inst_s['size'].sum()) if len(inst_s) > 0 else 0
    
    winner = "BUYERS" if buy_sh > sell_sh else "SELLERS"
    
    print(f'{minute}:')
    print(f'  BUY:       {len(buys):>4} trades    {buy_sh:>8,} shares')
    print(f'  SELL:      {len(sells):>4} trades    {sell_sh:>8,} shares')
    print(f'  DELTA:                    {buy_sh - sell_sh:>+8,} --> {winner}')
    print()
    print(f'  INST_BUY:  {len(inst_b):>4} trades    {inst_b_sh:>8,} shares')
    print(f'  INST_SELL: {len(inst_s):>4} trades    {inst_s_sh:>8,} shares')
    print(f'  INST DELTA:               {inst_b_sh - inst_s_sh:>+8,}')
    print()
    print('-'*70)
    print()

# Combined
m = j[j['minute'].isin(['09:13', '09:14'])]
buys = m[m['side'] == 'BUY']
sells = m[m['side'] == 'SELL']
buy_sh = int(buys['size'].sum())
sell_sh = int(sells['size'].sum())
inst_b = buys[buys['size'] >= INST]
inst_s = sells[sells['size'] >= INST]
inst_b_sh = int(inst_b['size'].sum()) if len(inst_b) > 0 else 0
inst_s_sh = int(inst_s['size'].sum()) if len(inst_s) > 0 else 0

winner = "BUYERS" if buy_sh > sell_sh else "SELLERS"

print('COMBINED (09:13 + 09:14):')
print(f'  BUY:       {len(buys):>4} trades    {buy_sh:>8,} shares')
print(f'  SELL:      {len(sells):>4} trades    {sell_sh:>8,} shares')
print(f'  DELTA:                    {buy_sh - sell_sh:>+8,} --> {winner}')
print()
print(f'  INST_BUY:  {len(inst_b):>4} trades    {inst_b_sh:>8,} shares')
print(f'  INST_SELL: {len(inst_s):>4} trades    {inst_s_sh:>8,} shares')
print(f'  INST DELTA:               {inst_b_sh - inst_s_sh:>+8,}')
print('='*70)

