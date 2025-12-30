#!/usr/bin/env python3
"""
Resistance Breakout Analysis - Phase-based evaluation.

Detects rejection episodes (3+ min gaps) and tracks trend across phases
to identify breakout signals.
"""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


@dataclass
class Phase:
    """One rejection episode at resistance."""
    phase_num: int
    start_time: str
    end_time: str
    touches: int
    buy_volume: float
    sell_volume: float
    delta: float
    inst_buy: int
    inst_sell: int
    absorbed: int  # INST_SELL with positive bounce
    confirmed: int  # INST_SELL with negative bounce
    buy_ratio: float
    winner: str  # BUYERS / SELLERS / CONTESTED


try:
    from zoneinfo import ZoneInfo
    CHICAGO_TZ = ZoneInfo("America/Chicago")
except ImportError:
    # Fallback for older Python
    CHICAGO_TZ = timezone(timedelta(hours=-6))


def ms_to_time(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_local = dt.astimezone(CHICAGO_TZ)
    return dt_local.strftime('%H:%M:%S')


def ms_to_short(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_local = dt.astimezone(CHICAGO_TZ)
    return dt_local.strftime('%H:%M')


def detect_phases(touch_packets: list, gap_ms: int = 180000) -> list[Phase]:
    """
    Segment touches into phases based on time gaps.
    gap_ms = 180000 = 3 minutes (default threshold for new phase)
    """
    if not touch_packets:
        return []
    
    phases = []
    current_touches = []
    prev_ts = None
    
    for tp in touch_packets:
        ts = tp.get("touch_ts_ms", 0)
        
        # Check if gap > threshold = new phase
        if prev_ts and (ts - prev_ts) > gap_ms:
            if current_touches:
                phases.append(current_touches)
            current_touches = []
        
        current_touches.append(tp)
        prev_ts = ts
    
    # Don't forget last phase
    if current_touches:
        phases.append(current_touches)
    
    # Convert to Phase objects
    result = []
    for i, phase_touches in enumerate(phases):
        buy_vol = sum((tp.get("large_buy_volume_30s", 0) or 0) for tp in phase_touches)
        sell_vol = sum((tp.get("large_sell_volume_30s", 0) or 0) for tp in phase_touches)
        delta = buy_vol - sell_vol
        
        inst_buy = sum(1 for tp in phase_touches if tp.get("institutional_buying_flag"))
        inst_sell = sum(1 for tp in phase_touches if tp.get("institutional_selling_flag"))
        
        # Count absorbed vs confirmed sells
        absorbed = 0
        confirmed = 0
        for tp in phase_touches:
            if tp.get("institutional_selling_flag"):
                bounce = tp.get("bounce_return_30s_pct", 0) or 0
                if bounce > 0:
                    absorbed += 1
                else:
                    confirmed += 1
        
        total_vol = buy_vol + sell_vol
        buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        
        # Determine winner
        if delta > 500:
            winner = "BUYERS"
        elif delta < -500:
            winner = "SELLERS"
        else:
            winner = "CONTESTED"
        
        t0 = phase_touches[0].get("touch_ts_ms", 0)
        t1 = phase_touches[-1].get("touch_ts_ms", 0)
        
        result.append(Phase(
            phase_num=i + 1,
            start_time=ms_to_short(t0),
            end_time=ms_to_short(t1),
            touches=len(phase_touches),
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            delta=delta,
            inst_buy=inst_buy,
            inst_sell=inst_sell,
            absorbed=absorbed,
            confirmed=confirmed,
            buy_ratio=buy_ratio,
            winner=winner,
        ))
    
    return result


def analyze_trend(phases: list[Phase]) -> dict:
    """
    Analyze trend across phases to detect breakout signals.
    """
    if len(phases) < 2:
        # Single phase - check if it's a clean breakout
        if len(phases) == 1:
            p = phases[0]
            if p.delta > 0 and p.inst_sell == 0:
                return {
                    "trend": "SINGLE_PHASE_BULLISH",
                    "score": 1,
                    "signals": ["Single phase - buyers won, no sellers"],
                    "verdict": "BREAKOUT POSSIBLE (low data)",
                    "early_summary": {"phases": 1, "inst_buy": p.inst_buy, "inst_sell": p.inst_sell, "delta": p.delta},
                    "late_summary": {"phases": 0, "inst_buy": 0, "inst_sell": 0, "delta": 0},
                }
            elif p.delta < 0:
                return {
                    "trend": "SINGLE_PHASE_BEARISH",
                    "score": -1,
                    "signals": ["Single phase - sellers won"],
                    "verdict": "RESISTANCE HOLDING (low data)",
                    "early_summary": {"phases": 1, "inst_buy": p.inst_buy, "inst_sell": p.inst_sell, "delta": p.delta},
                    "late_summary": {"phases": 0, "inst_buy": 0, "inst_sell": 0, "delta": 0},
                }
        return {
            "trend": "INSUFFICIENT_DATA",
            "score": 0,
            "signals": ["Need at least 2 phases to analyze trend"],
            "verdict": "WAIT - insufficient data",
            "early_summary": {"phases": 0, "inst_buy": 0, "inst_sell": 0, "delta": 0},
            "late_summary": {"phases": 0, "inst_buy": 0, "inst_sell": 0, "delta": 0},
        }
    
    signals = []
    score = 0
    
    # Split into early (first half) vs late (second half)
    mid = len(phases) // 2
    if mid == 0:
        mid = 1
    
    early = phases[:mid]
    late = phases[mid:]
    
    # Aggregate early phase metrics
    early_inst_buy = sum(p.inst_buy for p in early)
    early_inst_sell = sum(p.inst_sell for p in early)
    early_buy_vol = sum(p.buy_volume for p in early)
    early_sell_vol = sum(p.sell_volume for p in early)
    early_delta = sum(p.delta for p in early)
    early_absorbed = sum(p.absorbed for p in early)
    
    # Aggregate late phase metrics
    late_inst_buy = sum(p.inst_buy for p in late)
    late_inst_sell = sum(p.inst_sell for p in late)
    late_buy_vol = sum(p.buy_volume for p in late)
    late_sell_vol = sum(p.sell_volume for p in late)
    late_delta = sum(p.delta for p in late)
    late_absorbed = sum(p.absorbed for p in late)
    
    # 1. INST_BUY increasing?
    if late_inst_buy > early_inst_buy:
        signals.append(f"INST_BUY increasing ({early_inst_buy} -> {late_inst_buy})")
        score += 1
    elif late_inst_buy < early_inst_buy:
        signals.append(f"INST_BUY decreasing ({early_inst_buy} -> {late_inst_buy})")
        score -= 1
    
    # 2. INST_SELL decreasing?
    if late_inst_sell < early_inst_sell:
        signals.append(f"INST_SELL exhausting ({early_inst_sell} -> {late_inst_sell})")
        score += 1
    elif late_inst_sell > early_inst_sell:
        signals.append(f"INST_SELL increasing ({early_inst_sell} -> {late_inst_sell})")
        score -= 1
    
    # 3. Buy volume increasing?
    if late_buy_vol > early_buy_vol * 1.2:  # 20% threshold
        signals.append(f"Buy volume UP ({early_buy_vol:,.0f} -> {late_buy_vol:,.0f})")
        score += 1
    elif late_buy_vol < early_buy_vol * 0.8:
        signals.append(f"Buy volume DOWN ({early_buy_vol:,.0f} -> {late_buy_vol:,.0f})")
        score -= 1
    
    # 4. Sell volume decreasing?
    if late_sell_vol < early_sell_vol * 0.8:
        signals.append(f"Sell volume DOWN ({early_sell_vol:,.0f} -> {late_sell_vol:,.0f})")
        score += 1
    elif late_sell_vol > early_sell_vol * 1.2:
        signals.append(f"Sell volume UP ({early_sell_vol:,.0f} -> {late_sell_vol:,.0f})")
        score -= 1
    
    # 5. Delta improving?
    if late_delta > early_delta + 1000:
        signals.append(f"Delta improving ({early_delta:+,.0f} -> {late_delta:+,.0f})")
        score += 1
    elif late_delta < early_delta - 1000:
        signals.append(f"Delta worsening ({early_delta:+,.0f} -> {late_delta:+,.0f})")
        score -= 1
    
    # 6. More absorption in late phases?
    if late_absorbed > early_absorbed:
        signals.append(f"Absorption increasing ({early_absorbed} -> {late_absorbed})")
        score += 1
    
    # 7. Late phase INST_BUY > INST_SELL?
    if late_inst_buy > late_inst_sell:
        signals.append(f"Late phase: INST_BUY dominates ({late_inst_buy}B > {late_inst_sell}S)")
        score += 2  # Strong signal
    elif late_inst_sell > late_inst_buy and late_inst_sell > 0:
        signals.append(f"Late phase: INST_SELL still active ({late_inst_sell}S > {late_inst_buy}B)")
        score -= 1
    
    # 8. Last phase winner?
    last_phase = phases[-1]
    if last_phase.winner == "BUYERS":
        signals.append(f"Last phase: BUYERS won (delta {last_phase.delta:+,.0f})")
        score += 1
    elif last_phase.winner == "SELLERS":
        signals.append(f"Last phase: SELLERS won (delta {last_phase.delta:+,.0f})")
        score -= 1
    
    # Check for "clean breakout" pattern: buyers dominant, no/few sellers
    total_inst_buy = early_inst_buy + late_inst_buy
    total_inst_sell = early_inst_sell + late_inst_sell
    total_delta = early_delta + late_delta
    
    # Clean breakout: positive delta + buyers > sellers (or no sellers)
    if total_delta > 0 and total_inst_buy >= total_inst_sell:
        if total_inst_sell == 0:
            return {
                "trend": "CLEAN_BREAKOUT",
                "score": score,
                "signals": signals + ["NO SELLER DEFENSE - clean breakout"],
                "verdict": "BREAKOUT - TAKE",
                "early_summary": {
                    "phases": len(early),
                    "inst_buy": early_inst_buy,
                    "inst_sell": early_inst_sell,
                    "delta": early_delta,
                },
                "late_summary": {
                    "phases": len(late),
                    "inst_buy": late_inst_buy,
                    "inst_sell": late_inst_sell,
                    "delta": late_delta,
                },
            }
        elif total_inst_buy > total_inst_sell:
            signals.append(f"INST_BUY dominates ({total_inst_buy}B > {total_inst_sell}S)")
            score += 2
    
    # Determine trend and verdict
    if score >= 4:
        trend = "STRONG_BULLISH"
        verdict = "BREAKOUT LIKELY"
    elif score >= 2:
        trend = "BULLISH"
        verdict = "BREAKOUT POSSIBLE"
    elif score >= 0:
        trend = "NEUTRAL"
        verdict = "CONTESTED"
    elif score >= -2:
        trend = "BEARISH"
        verdict = "RESISTANCE HOLDING"
    else:
        trend = "STRONG_BEARISH"
        verdict = "STRONG RESISTANCE"
    
    return {
        "trend": trend,
        "score": score,
        "signals": signals,
        "verdict": verdict,
        "early_summary": {
            "phases": len(early),
            "inst_buy": early_inst_buy,
            "inst_sell": early_inst_sell,
            "delta": early_delta,
        },
        "late_summary": {
            "phases": len(late),
            "inst_buy": late_inst_buy,
            "inst_sell": late_inst_sell,
            "delta": late_delta,
        },
    }


def main(session_id: str):
    # Parse session_id
    parts = session_id.split("_")
    ticker = parts[0]
    date_str = parts[1]
    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    report = json.loads(report_path.read_text())
    
    print("=" * 70)
    print(f"RESISTANCE BREAKOUT ANALYSIS: {session_id}")
    print("=" * 70)
    
    for ch in report.get("level_chains", []):
        for lv in ch.get("levels", []):
            tps = lv.get("touch_packets", [])
            sw = lv.get("session_wide", {})
            
            if not tps:
                print("\nNo touch packets found.")
                continue
            
            print(f"\nTotal Touches: {len(tps)}")
            print(f"Buy Volume: {sw.get('band_large_buy_volume', 0):,.0f}")
            print(f"Sell Volume: {sw.get('band_large_sell_volume', 0):,.0f}")
            
            # Detect phases
            phases = detect_phases(tps, gap_ms=180000)  # 3-minute threshold
            
            print(f"\n{'=' * 70}")
            print(f"REJECTION PHASES ({len(phases)} detected, 3-min gap threshold)")
            print("=" * 70)
            
            for p in phases:
                icon = "+" if p.winner == "BUYERS" else ("-" if p.winner == "SELLERS" else " ")
                print(f"{icon} Phase {p.phase_num} ({p.start_time}-{p.end_time})")
                print(f"    Touches: {p.touches} | Buy: {p.buy_volume:,.0f} | Sell: {p.sell_volume:,.0f} | Delta: {p.delta:+,.0f}")
                print(f"    INST: {p.inst_buy}B / {p.inst_sell}S | Absorbed: {p.absorbed} | Confirmed: {p.confirmed}")
                print(f"    --> {p.winner}")
                print()
            
            # Visual progression
            print("PHASE PROGRESSION:")
            progression = ""
            for p in phases:
                if p.winner == "BUYERS":
                    progression += "[+] "
                elif p.winner == "SELLERS":
                    progression += "[-] "
                else:
                    progression += "[ ] "
            print(f"  {progression}")
            print()
            
            # Build-up tracker
            print("BUILD-UP TRACKER:")
            running_delta = 0
            running_inst_buy = 0
            running_inst_sell = 0
            for p in phases:
                running_delta += p.delta
                running_inst_buy += p.inst_buy
                running_inst_sell += p.inst_sell
                delta_bar = "+" * min(int(abs(running_delta) / 2000), 10) if running_delta > 0 else "-" * min(int(abs(running_delta) / 2000), 10)
                trend_arrow = "^" if p.delta > 0 else ("v" if p.delta < 0 else "-")
                print(f"  P{p.phase_num}: {trend_arrow} {p.delta:+6,.0f} | Running: {running_delta:+8,.0f} | {running_inst_buy}B/{running_inst_sell}S | {delta_bar}")
            print()
            
            # Analyze trend
            trend = analyze_trend(phases)
            
            print("=" * 70)
            print("TREND ANALYSIS")
            print("=" * 70)
            print()
            print(f"Early Phases (1-{trend['early_summary']['phases']}):")
            print(f"  INST: {trend['early_summary']['inst_buy']}B / {trend['early_summary']['inst_sell']}S")
            print(f"  Delta: {trend['early_summary']['delta']:+,.0f}")
            print()
            print(f"Late Phases ({trend['early_summary']['phases']+1}-{trend['early_summary']['phases']+trend['late_summary']['phases']}):")
            print(f"  INST: {trend['late_summary']['inst_buy']}B / {trend['late_summary']['inst_sell']}S")
            print(f"  Delta: {trend['late_summary']['delta']:+,.0f}")
            print()
            print("SIGNALS:")
            for sig in trend["signals"]:
                print(f"  - {sig}")
            print()
            print(f"TREND: {trend['trend']} (score: {trend['score']:+d})")
            print()
            print("=" * 70)
            print(f"VERDICT: {trend['verdict']}")
            print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resistance_eval.py <session_id>")
        print("Example: python resistance_eval.py OMER_20251226_145000")
        sys.exit(1)
    
    session_id = sys.argv[1]
    main(session_id)

