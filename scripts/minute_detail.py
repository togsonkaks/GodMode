"""Detailed buy/sell breakdown for specific minutes."""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone

CHI = pytz.timezone('America/Chicago')
session = 'OMER_OMER_20251226_150400_4'
level = 14.42
band_pct = 0.0015
lo = level * (1 - band_pct)
hi = level * (1 + band_pct)

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
tr = tr[(tr.price >= lo) & (tr.price <= hi)].copy()

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
                if p > last_price:
                    side = 'BUY'
                elif p < last_price:
                    side = 'SELL'
                else:
                    side = last_side if last_side in ['BUY', 'SELL'] else 'unknown'
            else:
                side = 'unknown'
    else:
        if last_price is not None:
            if p > last_price:
                side = 'BUY'
            elif p < last_price:
                side = 'SELL'
            else:
                side = last_side if last_side in ['BUY', 'SELL'] else 'unknown'
        else:
            side = 'unknown'
    sides.append(side)
    last_price = p
    if side in ['BUY', 'SELL']:
        last_side = side

j['side'] = sides

INST = 500

for minute in ['09:13', '09:14']:
    m = j[j['minute'] == minute]
    print('='*70)
    print(f'MINUTE: {minute} (In-Band at 14.42 +/- 0.15%)')
    print('='*70)
    
    buys = m[m['side'] == 'BUY']
    sells = m[m['side'] == 'SELL']
    
    buy_shares = int(buys['size'].sum())
    sell_shares = int(sells['size'].sum())
    
    print(f'  BUY trades:  {len(buys):>4}   shares: {buy_shares:>8,}')
    print(f'  SELL trades: {len(sells):>4}   shares: {sell_shares:>8,}')
    print(f'  DELTA:              shares: {buy_shares - sell_shares:>+8,}')
    print()
    
    inst_buys = buys[buys['size'] >= INST]
    inst_sells = sells[sells['size'] >= INST]
    
    print(f'  INST_BUY:  {len(inst_buys)} trades, {int(inst_buys["size"].sum()):,} shares')
    if len(inst_buys) > 0:
        for _, r in inst_buys.iterrows():
            t = r['time_chicago'][:12]
            sz = int(r['size'])
            px = r['price']
            print(f'    -> {t}  {sz:>5,} shares @ ${px:.4f}')
    
    print()
    print(f'  INST_SELL: {len(inst_sells)} trades, {int(inst_sells["size"].sum()):,} shares')
    if len(inst_sells) > 0:
        for _, r in inst_sells.iterrows():
            t = r['time_chicago'][:12]
            sz = int(r['size'])
            px = r['price']
            print(f'    -> {t}  {sz:>5,} shares @ ${px:.4f}')
    
    print()

