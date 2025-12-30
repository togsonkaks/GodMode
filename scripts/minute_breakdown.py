"""Per-minute breakdown of in-band trades with Lee-Ready classification."""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timezone

CHI = pytz.timezone('America/Chicago')
session = 'OMER_OMER_20251226_150400_5'
level = 14.42
band_pct = 0.0015
lo = level * (1 - band_pct)
hi = level * (1 + band_pct)

# Extended time range: 09:04 to 09:17 (to include 09:14, 09:15, 09:16)
start_local = CHI.localize(datetime(2025, 12, 26, 9, 4, 0))
end_local = CHI.localize(datetime(2025, 12, 26, 9, 17, 0))
start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
end_ms = int(end_local.astimezone(timezone.utc).timestamp() * 1000)

def has_W(x):
    if x is None: return False
    if isinstance(x, np.ndarray): return 'W' in x.tolist()
    if isinstance(x, (list, tuple, set)): return 'W' in x
    return 'W' in str(x)

# Load trades + quotes
tr = pd.read_parquet(f'data/output_clean/replay/{session}/trades.parquet').sort_values('ts_ms')
qt = pd.read_parquet(f'data/output_clean/replay/{session}/quotes.parquet').sort_values('ts_ms')

# Filter time
tr = tr[(tr.ts_ms >= start_ms) & (tr.ts_ms < end_ms)].copy()
tr = tr[~tr['conditions'].apply(has_W)].copy()

# Filter in-band
tr = tr[(tr.price >= lo) & (tr.price <= hi)].copy()

# Add Chicago time
tr['time_chicago'] = pd.to_datetime(tr['ts_ms'], unit='ms', utc=True).dt.tz_convert('America/Chicago').dt.strftime('%H:%M:%S.%f')

# Asof merge for bid/ask
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
df = j

# Extract minute from time
df['minute'] = df['time_chicago'].str[:5]  # HH:MM

# Define institutional threshold
INST_THRESHOLD = 500

# Per-minute breakdown
print('='*100)
print('PER-MINUTE BREAKDOWN (In-Band Trades at 14.42 +/- 0.15%)')
print('='*100)
header = f"{'Minute':<8} {'BUY':>6} {'SELL':>6} {'BuyShares':>10} {'SellShares':>10} {'INST_BUY':>9} {'INST_SELL':>10} {'InstBuySh':>10} {'InstSellSh':>11} {'Winner':>8}"
print(header)
print('-'*100)

totals = {'buy_trades':0, 'sell_trades':0, 'buy_shares':0, 'sell_shares':0, 
          'inst_buy':0, 'inst_sell':0, 'inst_buy_shares':0, 'inst_sell_shares':0}

for minute in sorted(df['minute'].unique()):
    m_df = df[df['minute'] == minute]
    
    buys = m_df[m_df['side'] == 'BUY']
    sells = m_df[m_df['side'] == 'SELL']
    
    buy_trades = len(buys)
    sell_trades = len(sells)
    buy_shares = int(buys['size'].sum())
    sell_shares = int(sells['size'].sum())
    
    inst_buys = buys[buys['size'] >= INST_THRESHOLD]
    inst_sells = sells[sells['size'] >= INST_THRESHOLD]
    
    inst_buy_count = len(inst_buys)
    inst_sell_count = len(inst_sells)
    inst_buy_shares = int(inst_buys['size'].sum())
    inst_sell_shares = int(inst_sells['size'].sum())
    
    # Determine winner
    if buy_shares > sell_shares * 1.2:
        winner = 'BUYERS'
    elif sell_shares > buy_shares * 1.2:
        winner = 'SELLERS'
    else:
        winner = 'EVEN'
    
    row = f"{minute:<8} {buy_trades:>6} {sell_trades:>6} {buy_shares:>10} {sell_shares:>10} {inst_buy_count:>9} {inst_sell_count:>10} {inst_buy_shares:>10} {inst_sell_shares:>11} {winner:>8}"
    print(row)
    
    totals['buy_trades'] += buy_trades
    totals['sell_trades'] += sell_trades
    totals['buy_shares'] += buy_shares
    totals['sell_shares'] += sell_shares
    totals['inst_buy'] += inst_buy_count
    totals['inst_sell'] += inst_sell_count
    totals['inst_buy_shares'] += inst_buy_shares
    totals['inst_sell_shares'] += inst_sell_shares

print('-'*100)
total_row = f"{'TOTAL':<8} {totals['buy_trades']:>6} {totals['sell_trades']:>6} {totals['buy_shares']:>10} {totals['sell_shares']:>10} {totals['inst_buy']:>9} {totals['inst_sell']:>10} {totals['inst_buy_shares']:>10} {totals['inst_sell_shares']:>11}"
print(total_row)
print('='*100)

# Summary
print()
print('SUMMARY:')
print(f"  Total Buy Shares:  {totals['buy_shares']:,}")
print(f"  Total Sell Shares: {totals['sell_shares']:,}")
delta = totals['buy_shares'] - totals['sell_shares']
print(f"  Delta:             {delta:+,}")
print()
print(f"  INST_BUY:  {totals['inst_buy']} trades, {totals['inst_buy_shares']:,} shares")
print(f"  INST_SELL: {totals['inst_sell']} trades, {totals['inst_sell_shares']:,} shares")
inst_delta = totals['inst_buy_shares'] - totals['inst_sell_shares']
print(f"  INST Delta: {inst_delta:+,} shares")

