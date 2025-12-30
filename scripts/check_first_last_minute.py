#!/usr/bin/env python3
"""Check high/low for first and last minute of a session."""

import sys
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime, timezone

CHICAGO = pytz.timezone('America/Chicago')

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_first_last_minute.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Find session stream
    base = Path("data/output_clean/session_stream")
    for date_dir in base.glob("date=*"):
        for ticker_dir in date_dir.glob("ticker=*"):
            for sess_dir in ticker_dir.glob(f"session=*{session_id}*"):
                parts = list(sess_dir.glob("part-*.parquet"))
                if parts:
                    df = pd.concat([pd.read_parquet(p) for p in parts])
                    df = df.sort_values('timestamp')
                    
                    print(f"Session: {sess_dir.name}")
                    print(f"Total snapshots: {len(df)}")
                    print()
                    
                    print("=== FIRST 6 snapshots (first minute) ===")
                    first_6 = df.head(6)
                    for _, row in first_6.iterrows():
                        dt = datetime.fromtimestamp(row['timestamp']/1000, tz=timezone.utc).astimezone(CHICAGO)
                        print(f"  {dt.strftime('%H:%M:%S')} | Close: {row['last_price']:.4f} | High: {row['high_10s']:.4f} | Low: {row['low_10s']:.4f}")
                    
                    print(f"\n  First minute HIGH: {first_6['high_10s'].max():.4f}")
                    print(f"  First minute LOW:  {first_6['low_10s'].min():.4f}")
                    
                    print("\n=== LAST 6 snapshots (last minute) ===")
                    last_6 = df.tail(6)
                    for _, row in last_6.iterrows():
                        dt = datetime.fromtimestamp(row['timestamp']/1000, tz=timezone.utc).astimezone(CHICAGO)
                        print(f"  {dt.strftime('%H:%M:%S')} | Close: {row['last_price']:.4f} | High: {row['high_10s']:.4f} | Low: {row['low_10s']:.4f}")
                    
                    print(f"\n  Last minute HIGH: {last_6['high_10s'].max():.4f}")
                    print(f"  Last minute LOW:  {last_6['low_10s'].min():.4f}")
                    return
    
    print(f"Session not found: {session_id}")

if __name__ == "__main__":
    main()

