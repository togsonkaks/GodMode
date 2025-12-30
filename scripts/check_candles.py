#!/usr/bin/env python3
"""Check actual candle OHLC from raw trades."""

import sys
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime, timezone

CHICAGO = pytz.timezone('America/Chicago')

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_candles.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Find replay trades
    base = Path("data/output_clean/replay")
    for sess_dir in base.glob(f"*{session_id}*"):
        trades_file = sess_dir / "trades.parquet"
        if trades_file.exists():
            df = pd.read_parquet(trades_file)
            df = df.sort_values('ts_ms')
            
            print(f"Session: {sess_dir.name}")
            print(f"Total trades: {len(df)}")
            
            # Get time range
            first_ts = df['ts_ms'].min()
            last_ts = df['ts_ms'].max()
            first_dt = datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
            last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
            print(f"Time range: {first_dt.strftime('%H:%M:%S.%f')} - {last_dt.strftime('%H:%M:%S.%f')}")
            print()
            
            # Build 1-minute candles
            print("=== 1-MINUTE CANDLES ===")
            df['minute'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.floor('1min')
            
            for minute, group in df.groupby('minute'):
                local_min = minute.tz_convert(CHICAGO)
                ohlc = {
                    'open': group.iloc[0]['price'],
                    'high': group['price'].max(),
                    'low': group['price'].min(),
                    'close': group.iloc[-1]['price'],
                    'trades': len(group)
                }
                print(f"{local_min.strftime('%H:%M')} | O: {ohlc['open']:.4f} H: {ohlc['high']:.4f} L: {ohlc['low']:.4f} C: {ohlc['close']:.4f} | {ohlc['trades']} trades")
            
            return
    
    print(f"Session not found: {session_id}")

if __name__ == "__main__":
    main()

