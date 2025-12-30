#!/usr/bin/env python3
"""Debug script to check what time ranges were recorded for a session."""

import sys
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime, timezone

CHICAGO = pytz.timezone('America/Chicago')

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_session_times.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    # Find the session
    base_path = Path("data/output_clean")
    
    # Check session_stream for raw trades
    for date_dir in base_path.glob("session_stream/date=*"):
        for ticker_dir in date_dir.glob("ticker=*"):
            for sess_dir in ticker_dir.glob(f"session=*{session_id}*"):
                print(f"\n{'='*60}")
                print(f"SESSION STREAM: {sess_dir.name}")
                print(f"{'='*60}")
                
                # Trades
                trades_dir = sess_dir / "trades"
                if trades_dir.exists():
                    parts = list(trades_dir.glob("part-*.parquet"))
                    if parts:
                        df = pd.concat([pd.read_parquet(p) for p in parts])
                        first_ts = df['ts_ms'].min()
                        last_ts = df['ts_ms'].max()
                        first_dt = datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
                        last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
                        print(f"\nRAW TRADES FROM ALPACA:")
                        print(f"  Records: {len(df):,}")
                        print(f"  Time range: {first_dt.strftime('%H:%M:%S')} - {last_dt.strftime('%H:%M:%S')} Chicago")
                        print(f"  Price range: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
                
                # Markers
                markers_file = sess_dir / "markers.parquet"
                if markers_file.exists():
                    mdf = pd.read_parquet(markers_file)
                    print(f"\nMARKERS:")
                    for _, row in mdf.iterrows():
                        start_dt = datetime.fromtimestamp(row['start_ts_ms']/1000, tz=timezone.utc).astimezone(CHICAGO)
                        end_dt = datetime.fromtimestamp(row['end_ts_ms']/1000, tz=timezone.utc).astimezone(CHICAGO) if pd.notna(row.get('end_ts_ms')) else None
                        print(f"  Type: {row.get('marker_type', 'N/A')}")
                        print(f"  Start: {start_dt.strftime('%H:%M:%S')} Chicago")
                        if end_dt:
                            print(f"  End: {end_dt.strftime('%H:%M:%S')} Chicago")
                        if 'notes' in row and row['notes']:
                            print(f"  Notes: {row['notes']}")
    
    # Check snapshots
    for date_dir in base_path.glob("snapshots/date=*"):
        for ticker_dir in date_dir.glob("ticker=*"):
            for sess_dir in ticker_dir.glob(f"session=*{session_id}*"):
                print(f"\n{'='*60}")
                print(f"SNAPSHOTS: {sess_dir.name}")
                print(f"{'='*60}")
                
                parts = list(sess_dir.glob("part-*.parquet"))
                if parts:
                    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_values('timestamp')
                    first_ts = df['timestamp'].min()
                    last_ts = df['timestamp'].max()
                    first_dt = datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
                    last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).astimezone(CHICAGO)
                    print(f"\nSNAPSHOTS (processed):")
                    print(f"  Records: {len(df):,}")
                    print(f"  Time range: {first_dt.strftime('%H:%M:%S')} - {last_dt.strftime('%H:%M:%S')} Chicago")
                    print(f"  Price range: ${df['last_price'].min():.4f} - ${df['last_price'].max():.4f}")
                    
                    # Check if high_10s/low_10s are present
                    has_hl = 'high_10s' in df.columns and 'low_10s' in df.columns
                    print(f"  Has high_10s/low_10s: {has_hl}")
                    if has_hl:
                        print(f"  High range: ${df['high_10s'].min():.4f} - ${df['high_10s'].max():.4f}")
                        print(f"  Low range: ${df['low_10s'].min():.4f} - ${df['low_10s'].max():.4f}")

if __name__ == "__main__":
    main()

