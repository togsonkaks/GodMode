#!/usr/bin/env python3
"""Debug stream data for a session."""
import sys
import json
import pandas as pd
from pathlib import Path

def main(session_id: str):
    parts = session_id.split("_")
    ticker = parts[0]
    date_str = parts[1]
    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    stream_path = Path(f"data/output_clean/session_stream/date={date}/ticker={ticker}/session={session_id}")
    files = list(stream_path.glob("*.parquet"))
    
    if not files:
        print("No stream data found")
        return
    
    df = pd.read_parquet(files[0])
    df = df.sort_values("timestamp")
    
    # Try to get level price from report
    report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
    level_price = None
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for ch in report.get("level_chains", []):
            for lv in ch.get("levels", []):
                level_price = lv.get("level_price")
                break
    
    if not level_price:
        # Try markers
        markers_path = Path(f"data/output_clean/markers/date={date}/ticker={ticker}/session={session_id}")
        for f in markers_path.glob("*.parquet"):
            mdf = pd.read_parquet(f)
            if not mdf.empty and "notes" in mdf.columns:
                notes_str = mdf.iloc[0]["notes"]
                if notes_str:
                    try:
                        notes = json.loads(notes_str) if isinstance(notes_str, str) else notes_str
                        price_tags = notes.get("price_tags", [])
                        if price_tags:
                            level_price = price_tags[0].get("price")
                    except:
                        pass
    
    band_pct = 0.0015
    lower = level_price * (1 - band_pct) if level_price else 0
    upper = level_price * (1 + band_pct) if level_price else 0
    
    print(f"Total snapshots: {len(df)}")
    print(f"Level: {level_price}")
    print(f"Band: {lower:.4f} - {upper:.4f} (0.15%)")
    print()
    
    # Show snapshots IN BAND only
    print("Snapshots IN BAND:")
    print("-" * 100)
    
    in_band_count = 0
    for i in range(len(df)):
        row = df.iloc[i]
        ts = pd.to_datetime(row["timestamp"], unit="ms", utc=True)
        price = row.get("last_price", 0)
        
        if lower <= price <= upper:
            in_band_count += 1
            total_vol = row.get("total_volume", 0)
            buy_vol = row.get("buy_volume", 0)
            sell_vol = row.get("sell_volume", 0)
            delta = row.get("delta", 0)
            
            print(f"{i+1:3}: {ts.strftime('%H:%M:%S')} | Price: {price:.4f} | Vol: {total_vol:>8,.0f} | Buy: {buy_vol:>8,.0f} | Sell: {sell_vol:>8,.0f} | Delta: {delta:>+8,.0f}")
    
    print()
    print(f"Total in band: {in_band_count}")

if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "OMER_20251226_150400"
    main(session_id)

