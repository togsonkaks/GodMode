#!/usr/bin/env python3
"""
Unified Analysis System - Combines checklist, phase progression, and institutional detail.
Provides phase-weighted verdicts with trend-based scoring.
"""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pytz

CHICAGO_TZ = pytz.timezone('America/Chicago')

def format_ts(ts_ms):
    """Format timestamp to Chicago time."""
    if not ts_ms:
        return "N/A"
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.utc)
    return dt.astimezone(CHICAGO_TZ).strftime('%H:%M:%S')

def get_marker_info(session_id: str, date: str, ticker: str) -> dict:
    """Read marker notes to detect support/resistance."""
    import pandas as pd
    markers_path = Path(f"data/output_clean/markers/date={date}/ticker={ticker}/session={session_id}")
    
    info = {"level_kind": "unknown", "direction_bias": "unknown", "level_price": None}
    
    for f in markers_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        if not df.empty and "notes" in df.columns:
            notes_str = df.iloc[0]["notes"]
            if notes_str:
                try:
                    notes = json.loads(notes_str) if isinstance(notes_str, str) else notes_str
                    info["direction_bias"] = notes.get("direction_bias", "unknown")
                    price_tags = notes.get("price_tags", [])
                    if price_tags:
                        info["level_kind"] = price_tags[0].get("kind", "unknown")
                        info["level_price"] = price_tags[0].get("price")
                except:
                    pass
    return info

def compute_phases(tps: list, min_gap_seconds: int = 180) -> list:
    """Group touches into phases based on time gaps or touch distribution."""
    if not tps:
        return []
    
    # Check if we have valid timestamps
    has_timestamps = any(tp.get("ts_ms", 0) for tp in tps)
    
    if has_timestamps:
        # Use timestamp-based phase detection
        phases = []
        current_phase = {"touches": [], "start_ts": None, "end_ts": None}
        
        for tp in tps:
            ts = tp.get("ts_ms", 0)
            
            if current_phase["touches"]:
                last_ts = current_phase["touches"][-1].get("ts_ms", 0)
                gap_seconds = (ts - last_ts) / 1000 if ts and last_ts else 0
                
                if gap_seconds >= min_gap_seconds:
                    phases.append(current_phase)
                    current_phase = {"touches": [], "start_ts": None, "end_ts": None}
            
            current_phase["touches"].append(tp)
            if not current_phase["start_ts"]:
                current_phase["start_ts"] = ts
            current_phase["end_ts"] = ts
        
        if current_phase["touches"]:
            phases.append(current_phase)
        
        return phases
    else:
        # Fallback: Split touches into 4 equal phases by touch number
        n = len(tps)
        if n <= 4:
            # Each touch is its own phase
            return [{"touches": [tp], "start_ts": None, "end_ts": None} for tp in tps]
        
        phase_size = max(1, n // 4)
        phases = []
        
        for i in range(0, n, phase_size):
            end_idx = min(i + phase_size, n)
            phase_touches = tps[i:end_idx]
            phases.append({
                "touches": phase_touches,
                "start_ts": None,
                "end_ts": None
            })
        
        return phases

def analyze_phase(phase: dict) -> dict:
    """Analyze a single phase."""
    touches = phase["touches"]
    
    buy_vol = sum(tp.get("large_buy_volume_30s", 0) or 0 for tp in touches)
    sell_vol = sum(tp.get("large_sell_volume_30s", 0) or 0 for tp in touches)
    delta = buy_vol - sell_vol
    
    inst_buy = sum(1 for tp in touches if tp.get("institutional_buying_flag"))
    inst_sell = sum(1 for tp in touches if tp.get("institutional_selling_flag"))
    
    absorbed = sum(1 for tp in touches if tp.get("institutional_selling_flag") and (tp.get("bounce_return_30s_pct", 0) or 0) > 0)
    confirmed = sum(1 for tp in touches if tp.get("institutional_selling_flag") and (tp.get("bounce_return_30s_pct", 0) or 0) <= 0)
    
    if delta > 0:
        winner = "BUYERS"
    elif delta < 0:
        winner = "SELLERS"
    else:
        winner = "CONTESTED"
    
    # Get touch numbers for display
    first_touch_num = touches[0].get("touch_number", 1) if touches else 1
    last_touch_num = touches[-1].get("touch_number", len(touches)) if touches else 1
    
    return {
        "touch_count": len(touches),
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "delta": delta,
        "inst_buy": inst_buy,
        "inst_sell": inst_sell,
        "absorbed": absorbed,
        "confirmed": confirmed,
        "winner": winner,
        "start_ts": phase["start_ts"],
        "end_ts": phase["end_ts"],
        "first_touch": first_touch_num,
        "last_touch": last_touch_num
    }

def compute_trend_score(phase_analyses: list) -> tuple:
    """Compute trend score based on phase progression."""
    if len(phase_analyses) < 2:
        return 0, []
    
    signals = []
    score = 0
    
    # Split into early and late
    mid = len(phase_analyses) // 2
    early = phase_analyses[:mid] if mid > 0 else phase_analyses[:1]
    late = phase_analyses[mid:] if mid > 0 else phase_analyses[1:]
    
    early_inst_buy = sum(p["inst_buy"] for p in early)
    late_inst_buy = sum(p["inst_buy"] for p in late)
    early_inst_sell = sum(p["inst_sell"] for p in early)
    late_inst_sell = sum(p["inst_sell"] for p in late)
    early_delta = sum(p["delta"] for p in early)
    late_delta = sum(p["delta"] for p in late)
    early_buy_vol = sum(p["buy_vol"] for p in early)
    late_buy_vol = sum(p["buy_vol"] for p in late)
    early_sell_vol = sum(p["sell_vol"] for p in early)
    late_sell_vol = sum(p["sell_vol"] for p in late)
    
    # INST_BUY increasing
    if late_inst_buy > early_inst_buy:
        score += 1
        signals.append(f"+1 INST_BUY increasing ({early_inst_buy} -> {late_inst_buy})")
    elif late_inst_buy < early_inst_buy:
        score -= 1
        signals.append(f"-1 INST_BUY decreasing ({early_inst_buy} -> {late_inst_buy})")
    
    # INST_SELL decreasing (bullish)
    if late_inst_sell < early_inst_sell:
        score += 1
        signals.append(f"+1 INST_SELL exhausting ({early_inst_sell} -> {late_inst_sell})")
    elif late_inst_sell > early_inst_sell:
        score -= 1
        signals.append(f"-1 INST_SELL increasing ({early_inst_sell} -> {late_inst_sell})")
    
    # Delta improving
    if late_delta > early_delta:
        score += 1
        signals.append(f"+1 Delta improving ({early_delta:+,} -> {late_delta:+,})")
    elif late_delta < early_delta:
        score -= 1
        signals.append(f"-1 Delta worsening ({early_delta:+,} -> {late_delta:+,})")
    
    # Buy volume increasing
    if late_buy_vol > early_buy_vol * 1.2:
        score += 1
        signals.append(f"+1 Buy volume UP ({early_buy_vol:,} -> {late_buy_vol:,})")
    elif late_buy_vol < early_buy_vol * 0.8:
        score -= 1
        signals.append(f"-1 Buy volume DOWN ({early_buy_vol:,} -> {late_buy_vol:,})")
    
    # Sell volume decreasing (bullish)
    if late_sell_vol < early_sell_vol * 0.8:
        score += 1
        signals.append(f"+1 Sell volume DOWN ({early_sell_vol:,} -> {late_sell_vol:,})")
    elif late_sell_vol > early_sell_vol * 1.2:
        score -= 1
        signals.append(f"-1 Sell volume UP ({early_sell_vol:,} -> {late_sell_vol:,})")
    
    # Last phase winner
    last_phase = phase_analyses[-1]
    if last_phase["winner"] == "BUYERS":
        score += 2
        signals.append(f"+2 Last phase: BUYERS won (delta {last_phase['delta']:+,})")
    elif last_phase["winner"] == "SELLERS":
        score -= 2
        signals.append(f"-2 Last phase: SELLERS won (delta {last_phase['delta']:+,})")
    
    # INST balance in late phases
    if late_inst_buy > late_inst_sell:
        score += 1
        signals.append(f"+1 Late INST_BUY dominates ({late_inst_buy}B > {late_inst_sell}S)")
    elif late_inst_sell > late_inst_buy:
        score -= 1
        signals.append(f"-1 Late INST_SELL dominates ({late_inst_sell}S > {late_inst_buy}B)")
    
    return score, signals

def compute_phase_weighted_score(phase_analyses: list) -> tuple:
    """Apply phase weighting: late=2x, mid=1x, early=0.5x."""
    if not phase_analyses:
        return 0, 0, 0
    
    n = len(phase_analyses)
    early_end = max(1, int(n * 0.3))
    late_start = max(early_end, int(n * 0.7))
    
    early_score = 0
    mid_score = 0
    late_score = 0
    
    for i, p in enumerate(phase_analyses):
        phase_delta = p["delta"]
        inst_balance = p["inst_buy"] - p["inst_sell"]
        phase_score = (1 if phase_delta > 0 else -1 if phase_delta < 0 else 0) + inst_balance
        
        if i < early_end:
            early_score += phase_score * 0.5  # 0.5x weight
        elif i >= late_start:
            late_score += phase_score * 2.0   # 2x weight
        else:
            mid_score += phase_score * 1.0    # 1x weight
    
    return early_score, mid_score, late_score

def get_verdict(trend_score: int, total_inst_buy: int, total_inst_sell: int, 
                total_buy_vol: float, total_sell_vol: float, late_score: float,
                inst_buy_vol: float = 0, inst_sell_vol: float = 0,
                phase_analyses: list = None) -> tuple:
    """Determine direction and confidence based on all factors."""
    
    # === HARD GATES: These override everything ===
    
    # 1. Volume imbalance (>1.5:1 ratio = significant, >2:1 = strong)
    if total_sell_vol > total_buy_vol * 2:
        return "SHORT", "AVOID"
    if total_buy_vol > total_sell_vol * 2:
        return "LONG", "TAKE"
    
    # 2. Institutional shares imbalance (>2:1 shares with meaningful volume)
    if inst_sell_vol > inst_buy_vol * 2 and inst_sell_vol > 5000:
        return "SHORT", "AVOID"
    if inst_buy_vol > inst_sell_vol * 2 and inst_buy_vol > 5000:
        return "LONG", "TAKE"
    
    # 3. Institutional trade count imbalance (>2:1 trades with >=3 trades)
    if total_inst_sell > total_inst_buy * 2 and total_inst_sell >= 3:
        return "SHORT", "AVOID"
    if total_inst_buy > total_inst_sell * 2 and total_inst_buy >= 3:
        return "LONG", "TAKE"
    
    # === LATE PHASE MAGNITUDE CHECK ===
    # Don't let a thin late phase override everything
    adjusted_late_score = late_score
    if phase_analyses and len(phase_analyses) >= 2:
        # Calculate average phase volume
        phase_vols = [(p["buy_vol"] + p["sell_vol"]) for p in phase_analyses]
        avg_phase_vol = sum(phase_vols) / len(phase_vols) if phase_vols else 0
        
        # Get last phase volume
        last_phase_vol = phase_vols[-1] if phase_vols else 0
        
        # If last phase volume is < 30% of average, discount its score heavily
        if avg_phase_vol > 0 and last_phase_vol < avg_phase_vol * 0.3:
            adjusted_late_score = late_score * 0.1  # Almost ignore it
        elif avg_phase_vol > 0 and last_phase_vol < avg_phase_vol * 0.5:
            adjusted_late_score = late_score * 0.3  # Heavy discount
    
    # === STANDARD LOGIC (only if no hard gates triggered) ===
    
    # Combine trend and late score, but late score is NOT absolute
    combined_score = trend_score + (adjusted_late_score * 0.3)  # Reduced late impact
    
    if combined_score >= 2:
        direction = "LONG"
    elif combined_score <= -2:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    
    # Determine confidence
    volume_favors_buyers = total_buy_vol > total_sell_vol
    inst_favors_buyers = total_inst_buy > total_inst_sell
    
    # Count cautions (factors against the direction)
    cautions = 0
    if direction == "LONG":
        if not volume_favors_buyers:
            cautions += 1
        if not inst_favors_buyers:
            cautions += 1
    elif direction == "SHORT":
        if volume_favors_buyers:
            cautions += 1
        if inst_favors_buyers:
            cautions += 1
    
    if direction == "LONG":
        if trend_score >= 5 and cautions == 0:
            confidence = "TAKE"
        elif trend_score >= 3 and cautions <= 1:
            confidence = "WAIT"
        elif trend_score >= 2:
            confidence = "LEAN" if cautions > 0 else "WAIT"
        else:
            confidence = "LEAN"
    elif direction == "SHORT":
        if trend_score <= -5 and cautions == 0:
            confidence = "AVOID"
        elif trend_score <= -3 and cautions <= 1:
            confidence = "WAIT"
        elif trend_score <= -2:
            confidence = "LEAN" if cautions > 0 else "WAIT"
        else:
            confidence = "LEAN"
    else:
        confidence = "WAIT"
    
    return direction, confidence

def main(session_id: str):
    """Main analysis function."""
    # Parse session_id
    parts = session_id.split("_")
    ticker = parts[0]
    date_str = parts[1]
    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Load report
    report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    report = json.loads(report_path.read_text())
    
    # Get marker info
    marker_info = get_marker_info(session_id, date, ticker)
    
    print("=" * 70)
    print(f"UNIFIED ANALYSIS: {session_id}")
    print("=" * 70)
    print()
    print(f"Level Type:  {marker_info['level_kind'].upper()}")
    print(f"Level Price: ${marker_info['level_price']}" if marker_info['level_price'] else "Level Price: N/A")
    print(f"Bias:        {marker_info['direction_bias'].upper()}")
    print()
    
    for ch in report.get("level_chains", []):
        for lv in ch.get("levels", []):
            tps = lv.get("touch_packets", [])
            sw = lv.get("session_wide", {})
            
            if not tps:
                print("No touch data available.")
                continue
            
            # === SECTION 1: OVERALL BALANCE ===
            print("=" * 70)
            print("SECTION 1: OVERALL BALANCE")
            print("=" * 70)
            
            large_buy = sw.get("band_large_buy_volume", 0) or 0
            large_sell = sw.get("band_large_sell_volume", 0) or 0
            
            total_inst_buy = sum(1 for tp in tps if tp.get("institutional_buying_flag"))
            total_inst_sell = sum(1 for tp in tps if tp.get("institutional_selling_flag"))
            
            inst_buy_vol = sum(tp.get("large_buy_volume_30s", 0) or 0 for tp in tps if tp.get("institutional_buying_flag"))
            inst_sell_vol = sum(tp.get("large_sell_volume_30s", 0) or 0 for tp in tps if tp.get("institutional_selling_flag"))
            
            print()
            print("INSTITUTIONAL TRADES:")
            print(f"  INST_BUY:  {total_inst_buy:3} trades | {inst_buy_vol:>10,.0f} shares")
            print(f"  INST_SELL: {total_inst_sell:3} trades | {inst_sell_vol:>10,.0f} shares")
            print(f"  Balance:   {total_inst_buy - total_inst_sell:+3} trades | {inst_buy_vol - inst_sell_vol:>+10,.0f} shares")
            print()
            print("LARGE ORDER VOLUME (in band):")
            print(f"  Large Buy:  {large_buy:>10,.0f} shares")
            print(f"  Large Sell: {large_sell:>10,.0f} shares")
            print(f"  Net:        {large_buy - large_sell:>+10,.0f} shares")
            print()
            
            # === SECTION 2A: MINUTE-BY-MINUTE BREAKDOWN ===
            print("=" * 70)
            print("SECTION 2A: MINUTE-BY-MINUTE BREAKDOWN")
            print("=" * 70)
            print()
            
            # Group touches by minute
            from collections import defaultdict
            minute_data = defaultdict(lambda: {"touches": 0, "buy_vol": 0, "sell_vol": 0, "inst_buy": 0, "inst_sell": 0})
            
            for tp in tps:
                ts = tp.get("touch_ts_ms") or tp.get("ts_ms", 0)
                if ts:
                    minute_key = format_ts(ts)[:5]  # HH:MM
                    minute_data[minute_key]["touches"] += 1
                    minute_data[minute_key]["buy_vol"] += tp.get("large_buy_volume_30s", 0) or 0
                    minute_data[minute_key]["sell_vol"] += tp.get("large_sell_volume_30s", 0) or 0
                    if tp.get("institutional_buying_flag"):
                        minute_data[minute_key]["inst_buy"] += 1
                    if tp.get("institutional_selling_flag"):
                        minute_data[minute_key]["inst_sell"] += 1
            
            if minute_data:
                print(f"{'Minute':<8} {'Touches':<8} {'Buy Vol':<12} {'Sell Vol':<12} {'Delta':<12} {'INST':<10} {'Winner':<10}")
                print("-" * 80)
                
                minute_progression = []
                for minute in sorted(minute_data.keys()):
                    m = minute_data[minute]
                    delta = m["buy_vol"] - m["sell_vol"]
                    inst_str = f"{m['inst_buy']}B/{m['inst_sell']}S"
                    
                    if m["buy_vol"] > m["sell_vol"] * 1.1:
                        winner = "BUYERS"
                        minute_progression.append("[+]")
                    elif m["sell_vol"] > m["buy_vol"] * 1.1:
                        winner = "SELLERS"
                        minute_progression.append("[-]")
                    else:
                        winner = "EVEN"
                        minute_progression.append("[=]")
                    
                    print(f"{minute:<8} {m['touches']:<8} {m['buy_vol']:<12,.0f} {m['sell_vol']:<12,.0f} {delta:>+11,.0f} {inst_str:<10} {winner}")
                
                print()
                print(f"Minute Progression: {''.join(minute_progression)}")
            else:
                print("No minute data available.")
            print()
            
            # === SECTION 2B: PHASE PROGRESSION ===
            print("=" * 70)
            print("SECTION 2B: PHASE PROGRESSION")
            print("=" * 70)
            print()
            
            phases = compute_phases(tps)
            phase_analyses = [analyze_phase(p) for p in phases]
            
            print(f"Phases detected: {len(phases)} (3-min gap threshold)")
            print()
            
            # Phase table header
            print(f"{'Phase':<6} {'Time':<15} {'Touches':<8} {'Buy Vol':<12} {'Sell Vol':<12} {'Delta':<12} {'INST':<8} {'Winner':<10}")
            print("-" * 95)
            
            for i, pa in enumerate(phase_analyses):
                if pa["start_ts"] and pa["end_ts"]:
                    start = format_ts(pa["start_ts"])
                    end = format_ts(pa["end_ts"])
                    time_range = f"{start[:5]}-{end[:5]}" if start != end else start[:5]
                else:
                    # Show touch range instead
                    first_touch = pa.get("first_touch", "?")
                    last_touch = pa.get("last_touch", "?")
                    time_range = f"T{first_touch}-T{last_touch}" if first_touch != last_touch else f"T{first_touch}"
                
                inst_str = f"{pa['inst_buy']}B/{pa['inst_sell']}S"
                winner_symbol = "+" if pa["winner"] == "BUYERS" else "-" if pa["winner"] == "SELLERS" else " "
                
                print(f"P{i+1:<5} {time_range:<15} {pa['touch_count']:<8} {pa['buy_vol']:<12,.0f} {pa['sell_vol']:<12,.0f} {pa['delta']:>+11,.0f} {inst_str:<8} {winner_symbol} {pa['winner']}")
            
            print()
            
            # Phase progression visual
            progression = "".join("[+]" if pa["winner"] == "BUYERS" else "[-]" if pa["winner"] == "SELLERS" else "[ ]" for pa in phase_analyses)
            print(f"Progression: {progression}")
            print()
            
            # === SECTION 3: INSTITUTIONAL DETAIL ===
            print("=" * 70)
            print("SECTION 3: INSTITUTIONAL DETAIL")
            print("=" * 70)
            print()
            
            # INST_BUY detail
            inst_buys = [(i+1, tp) for i, tp in enumerate(tps) if tp.get("institutional_buying_flag")]
            print(f"INST_BUY: {len(inst_buys)} trades")
            print("-" * 50)
            for touch_num, tp in inst_buys:
                vol = tp.get("large_buy_volume_30s", 0) or 0
                bounce = (tp.get("bounce_return_30s_pct", 0) or 0) * 100
                result = "OK" if bounce > 0 else "FAILED"
                print(f"  Touch #{touch_num:2} | Vol: {vol:>8,.0f} | Bounce: {bounce:+5.1f}% ({result})")
            print()
            
            # INST_SELL detail
            inst_sells = [(i+1, tp) for i, tp in enumerate(tps) if tp.get("institutional_selling_flag")]
            print(f"INST_SELL: {len(inst_sells)} trades")
            print("-" * 50)
            absorbed_count = 0
            for touch_num, tp in inst_sells:
                vol = tp.get("large_sell_volume_30s", 0) or 0
                bounce = (tp.get("bounce_return_30s_pct", 0) or 0) * 100
                status = "ABSORBED" if bounce > 0 else "CONFIRMED"
                if bounce > 0:
                    absorbed_count += 1
                print(f"  Touch #{touch_num:2} | Vol: {vol:>8,.0f} | Bounce: {bounce:+5.1f}% ({status})")
            print()
            print(f"Absorption Rate: {absorbed_count}/{len(inst_sells)} ({absorbed_count/len(inst_sells)*100:.0f}%)" if inst_sells else "")
            print()
            
            # === SECTION 4: TREND ANALYSIS ===
            print("=" * 70)
            print("SECTION 4: TREND ANALYSIS")
            print("=" * 70)
            print()
            
            trend_score, trend_signals = compute_trend_score(phase_analyses)
            early_score, mid_score, late_score = compute_phase_weighted_score(phase_analyses)
            
            print("TREND SIGNALS:")
            for sig in trend_signals:
                print(f"  {sig}")
            print()
            print(f"TREND SCORE: {trend_score:+d}")
            print()
            
            print("PHASE-WEIGHTED SCORES:")
            print(f"  Early phases (0.5x): {early_score:+.1f}")
            print(f"  Mid phases (1.0x):   {mid_score:+.1f}")
            print(f"  Late phases (2.0x):  {late_score:+.1f}")
            print(f"  TOTAL:               {early_score + mid_score + late_score:+.1f}")
            print()
            
            # === SECTION 5: VERDICT ===
            print("=" * 70)
            print("VERDICT")
            print("=" * 70)
            print()
            
            # Compute total buy/sell volume from all phases
            total_buy_vol = sum(pa["buy_vol"] for pa in phase_analyses)
            total_sell_vol = sum(pa["sell_vol"] for pa in phase_analyses)
            
            direction, confidence = get_verdict(
                trend_score, total_inst_buy, total_inst_sell,
                total_buy_vol, total_sell_vol, late_score,
                inst_buy_vol, inst_sell_vol, phase_analyses
            )
            
            # Color-code the verdict (using text symbols)
            if confidence == "TAKE":
                verdict_symbol = "[+++]"
            elif confidence == "WAIT":
                verdict_symbol = "[~~~]"
            elif confidence == "LEAN":
                verdict_symbol = "[+/-]"
            else:  # AVOID
                verdict_symbol = "[---]"
            
            print(f"  {verdict_symbol} {direction} - {confidence}")
            print()
            
            # Reasoning
            print("REASONING:")
            if direction == "LONG":
                if trend_score > 0:
                    print(f"  + Trend score positive ({trend_score:+d})")
                if late_score > 0:
                    print(f"  + Late phases favor buyers ({late_score:+.1f})")
                if total_inst_buy > total_inst_sell:
                    print(f"  + INST_BUY dominates ({total_inst_buy} vs {total_inst_sell})")
                if inst_buy_vol > inst_sell_vol:
                    print(f"  + INST buy volume higher ({inst_buy_vol:,.0f} vs {inst_sell_vol:,.0f})")
                
                # Cautions
                if inst_sell_vol > inst_buy_vol:
                    print(f"  ! Caution: INST sell volume higher ({inst_sell_vol:,.0f} vs {inst_buy_vol:,.0f})")
                if total_inst_sell > total_inst_buy:
                    print(f"  ! Caution: More INST_SELL trades ({total_inst_sell} vs {total_inst_buy})")
            elif direction == "SHORT":
                if trend_score < 0:
                    print(f"  - Trend score negative ({trend_score:+d})")
                if late_score < 0:
                    print(f"  - Late phases favor sellers ({late_score:+.1f})")
                if total_inst_sell > total_inst_buy:
                    print(f"  - INST_SELL dominates ({total_inst_sell} vs {total_inst_buy})")
            else:
                print("  = Contested - no clear direction")
            
            print()

if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "ASPC_20251226_162000"
    main(session_id)

