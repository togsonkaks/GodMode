#!/usr/bin/env python
"""Quick script to check session stream data."""
import sys
sys.path.insert(0, "src")

import pandas as pd
from datetime import datetime

session_id = sys.argv[1] if len(sys.argv) > 1 else "ASPC_20251226_200000"
ticker = session_id.split("_")[0]
date_part = session_id.split("_")[1]
date = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"

path = f"data/output_clean/session_stream/date={date}/ticker={ticker}/session={session_id}/part-000.parquet"
print(f"Reading: {path}\n")

df = pd.read_parquet(path)

print("Key columns:", [c for c in df.columns if c in ['timestamp', 'last_price', 'ema9', 'ema20', 'ema30', 'ema200', 'vwap_session']])
print()

print("Price range:")
print(f"  Min: {df['last_price'].min():.4f}")
print(f"  Max: {df['last_price'].max():.4f}")
print()

# Time range
df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"Time range:")
print(f"  Start: {df['time'].min()}")
print(f"  End:   {df['time'].max()}")
print()

# Check EMAs
for col in ['ema9', 'ema20', 'ema30', 'ema200']:
    if col in df.columns:
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

print()

# Key price points
print("=== PRICE TIMELINE ===")
print(f"Start (20:00):   ${df.iloc[0]['last_price']:.2f}")

# Find peak
peak_idx = df['last_price'].idxmax()
peak_time = df.loc[peak_idx, 'time']
peak_price = df.loc[peak_idx, 'last_price']
print(f"Peak ({peak_time.strftime('%H:%M')}):     ${peak_price:.2f}")

# Find low
low_idx = df['last_price'].idxmin()
low_time = df.loc[low_idx, 'time']
low_price = df.loc[low_idx, 'last_price']
print(f"Low ({low_time.strftime('%H:%M')}):      ${low_price:.2f}")

print(f"End (20:53):     ${df.iloc[-1]['last_price']:.2f}")

# Calculate drop
drop_pct = (low_price - peak_price) / peak_price * 100
print()
print(f"Peak-to-Low drop: {drop_pct:.1f}%")

print()
print("First 5 rows (time, price, ema9, ema20):")
cols = ['time', 'last_price'] + [c for c in ['ema9', 'ema20', 'ema30'] if c in df.columns]
print(df[cols].head(5).to_string())
print()
print("Last 5 rows:")
print(df[cols].tail(5).to_string())

