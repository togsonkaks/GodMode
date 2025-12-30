#!/usr/bin/env python3
"""Evaluate session using the checklist system."""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from datetime import datetime, timezone

def ms_to_time(ms):
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime('%H:%M:%S')

def main(session_id: str):
    # Parse session_id to get date and ticker
    parts = session_id.split("_")
    ticker = parts[0]
    date_str = parts[1]
    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    report_path = Path(f"data/output_clean/reports/date={date}/ticker={ticker}/session={session_id}/report.json")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    report = json.loads(report_path.read_text())
    
    print("=" * 60)
    print(f"{session_id} CHECKLIST ANALYSIS")
    print("=" * 60)
    
    # Evaluate checklist - per level (no summing across levels)
    for ch in report.get("level_chains", []):
        for lv in ch.get("levels", []):
            sw = lv.get("session_wide", {})
            tps = lv.get("touch_packets", [])
            
            if not tps:
                continue
            
            band_delta = sw.get("band_delta_sum", 0)
            large_buy = sw.get("band_large_buy_volume", 0) or 0
            large_sell = sw.get("band_large_sell_volume", 0) or 0
            sell_pct = large_sell / (large_buy + large_sell) * 100 if (large_buy + large_sell) > 0 else 0
            
            # Count INST flags for this level
            level_inst_buy = sum(1 for tp in tps if tp.get("institutional_buying_flag"))
            level_inst_sell = sum(1 for tp in tps if tp.get("institutional_selling_flag"))
            
            # Get level name
            level_name = lv.get("level_name", "Unknown")
            
            # Find breakout touch (highest positive bounce > 2%)
            breakout_touch_idx = None
            max_bounce = 0
            for i, tp in enumerate(tps):
                bounce = (tp.get("bounce_return_30s_pct", 0) or 0)
                if bounce > 0.02 and bounce > max_bounce:
                    max_bounce = bounce
                    breakout_touch_idx = i
            
            print("\n" + "=" * 60)
            print(f"LEVEL: {level_name}")
            print("=" * 60)
            print(f"  Buy Trades:   {level_inst_buy:3}      |  Buy Shares:   {large_buy:>10,.0f}")
            print(f"  Sell Trades:  {level_inst_sell:3}      |  Sell Shares:  {large_sell:>10,.0f}")
            print("-" * 60)
            
            # Pre-calculate VOLUME_ACCUMULATION for gate softening
            total_band_volume = large_buy + large_sell
            volume_ratio = large_buy / large_sell if large_sell > 0 else float('inf')
            has_volume_accumulation = (volume_ratio >= 1.5) and (total_band_volume >= 5000)
            
            print("HARD GATES:")
            
            # Late INST_SELL - but only if bounce is NEGATIVE (confirmed breakdown)
            late_inst_sell_confirmed = False
            late_inst_sell_absorbed = False
            for tp in tps[-2:]:
                if tp.get("institutional_selling_flag"):
                    bounce = (tp.get("bounce_return_30s_pct", 0) or 0)
                    if bounce < 0:
                        late_inst_sell_confirmed = True
                    else:
                        late_inst_sell_absorbed = True
            
            status = "X TRIPPED" if late_inst_sell_confirmed else "OK"
            extra = ""
            if late_inst_sell_absorbed:
                extra = " (INST_SELL absorbed -> shakeout)"
            print(f"  LATE_INST_SELL_CONFIRMED:        [{status}]{extra}")
            
            # No buyers - can be softened by VOLUME_ACCUMULATION
            no_buyers = level_inst_buy == 0
            if no_buyers and has_volume_accumulation:
                status = "~ SOFTENED"
                extra_note = " (volume accumulation present)"
            elif no_buyers:
                status = "X TRIPPED"
                extra_note = ""
            else:
                status = "OK"
                extra_note = ""
            print(f"  NO_BUYERS (zero INST_BUY):       [{status}]{extra_note}")
            
            # Seller dominated
            seller_dom = sell_pct > 70
            status = "X TRIPPED" if seller_dom else "OK"
            print(f"  SELLER_DOMINATED (>70%):         [{status}] ({sell_pct:.1f}%)")
            
            # First touch dump
            first_br = (tps[0].get("large_trade_buy_ratio_30s", 0) or 0)
            first_delta = tps[0].get("band_delta_0_30s", 0) or 0
            first_dump = first_br < 0.30 and first_delta < -50000
            status = "X TRIPPED" if first_dump else "OK"
            print(f"  FIRST_TOUCH_DUMP (<30% + <-50k): [{status}]")
            
            # SELLERS OUTNUMBER BUYERS (trade count) - if more inst sells than buys, it's a red flag
            sellers_outnumber = level_inst_sell > level_inst_buy
            status = "X TRIPPED" if sellers_outnumber else "OK"
            print(f"  SELLERS_OUTNUMBER (trades):      [{status}] ({level_inst_buy} buy vs {level_inst_sell} sell)")
            
            # SHARES OUTNUMBER - if sell shares > buy shares, sellers moving more volume
            shares_outnumber = large_sell > large_buy
            status = "X TRIPPED" if shares_outnumber else "OK"
            print(f"  SHARES_OUTNUMBER (volume):       [{status}] ({large_buy:,.0f} buy vs {large_sell:,.0f} sell)")
            
            # LOW SAMPLE SIZE - not enough touches for reliable signal
            low_sample = len(tps) < 10
            status = "? INSUFFICIENT" if low_sample else "OK"
            print(f"  LOW_SAMPLE_SIZE (< 10 touches):  [{status}] ({len(tps)} touches)")
            
            # Determine hard gates - NO_BUYERS is softened if VOLUME_ACCUMULATION is present
            no_buyers_hard = no_buyers and not has_volume_accumulation
            
            any_gate = late_inst_sell_confirmed or no_buyers_hard or seller_dom or first_dump or sellers_outnumber or shares_outnumber
            insufficient_data = low_sample
            
            # Track if we have a softened gate (for WAIT verdict)
            softened_no_buyers = no_buyers and has_volume_accumulation
            
            print("\n" + "=" * 60)
            print("GREEN FLAGS:")
            print("=" * 60)
            
            greens = []
            
            # INST_BUY_EARLY
            early_buy = any(tp.get("institutional_buying_flag") for tp in tps[:2])
            if early_buy:
                greens.append("INST_BUY_EARLY")
            print(f"  INST_BUY_EARLY (touch #1 or #2):   [{'Y' if early_buy else ' '}]")
            
            # INST_HOLD
            inst_hold = level_inst_buy > 0 and level_inst_sell == 0
            if inst_hold:
                greens.append("INST_HOLD")
            print(f"  INST_HOLD (INST_BUY + no SELL):    [{'Y' if inst_hold else ' '}]")
            
            # HIGH_BUY_RATIO - but only if large count >= 5 (otherwise noise)
            MIN_COUNT = 5
            high_br = any(
                (tp.get("large_trade_buy_ratio_30s", 0) or 0) >= 0.55 and
                (tp.get("large_trade_count_30s", 0) or 0) >= MIN_COUNT
                for tp in tps
            )
            if high_br:
                greens.append("HIGH_BUY_RATIO")
            print(f"  HIGH_BUY_RATIO (>=55% + count>=5): [{'Y' if high_br else ' '}]")
            
            # KILLER_COMBO - but only if large count >= 5
            killer = any(
                (tp.get("large_trade_buy_ratio_30s", 0) or 0) >= 0.55 and
                (tp.get("rel_aggr_0_30s", 0) or 0) > 0 and
                (tp.get("large_trade_count_30s", 0) or 0) >= MIN_COUNT
                for tp in tps
            )
            if killer:
                greens.append("KILLER_COMBO")
            print(f"  KILLER_COMBO (high_br+aggr+cnt>=5):[{'Y' if killer else ' '}]")
            
            # COMPRESSION_HIGH
            comp_high = any((tp.get("compression_at_touch", 0) or 0) >= 1.0 for tp in tps)
            if comp_high:
                greens.append("COMPRESSION_HIGH")
            print(f"  COMPRESSION_HIGH (any >= 1.0):     [{'Y' if comp_high else ' '}]")
            
            # SELLER_EXHAUSTION
            if len(tps) >= 4:
                early_count = sum((tp.get("large_trade_count_30s", 0) or 0) for tp in tps[:3]) / 3
                late_count = sum((tp.get("large_trade_count_30s", 0) or 0) for tp in tps[-3:]) / 3
                exhaustion = late_count < early_count * 0.5 and level_inst_sell == 0
            else:
                exhaustion = False
            if exhaustion:
                greens.append("SELLER_EXHAUSTION")
            print(f"  SELLER_EXHAUSTION (50% drop):      [{'Y' if exhaustion else ' '}]")
            
            # ABSORPTION - count how many INST_SELL got absorbed vs confirmed
            # BUT absorption only counts if there's buying conviction (buy shares > sell OR inst_buy > inst_sell)
            inst_sell_absorbed_count = 0
            inst_sell_confirmed_count = 0
            for tp in tps:
                if tp.get("institutional_selling_flag"):
                    bounce = (tp.get("bounce_return_30s_pct", 0) or 0)
                    if bounce > 0:
                        inst_sell_absorbed_count += 1
                    else:
                        inst_sell_confirmed_count += 1
            
            # Absorption only counts as green flag if backed by buying conviction
            has_buying_conviction = (large_buy > large_sell) or (level_inst_buy > level_inst_sell)
            absorption = inst_sell_absorbed_count > 0 and has_buying_conviction
            absorption_no_conviction = inst_sell_absorbed_count > 0 and not has_buying_conviction
            
            if absorption:
                greens.append("ABSORPTION")
            
            absorption_note = ""
            if absorption_no_conviction:
                absorption_note = " (no conviction - sellers paused, not defeated)"
            print(f"  ABSORPTION (INST_SELL + bounce):   [{'Y' if absorption else ' '}] ({inst_sell_absorbed_count} absorbed, {inst_sell_confirmed_count} confirmed){absorption_note}")
            
            # SHAKEOUT_ABSORBED - late INST_SELL that got absorbed is actually bullish
            shakeout = late_inst_sell_absorbed
            if shakeout:
                greens.append("SHAKEOUT_ABSORBED")
            print(f"  SHAKEOUT_ABSORBED (late absorbed): [{'Y' if shakeout else ' '}]")
            
            # BAND_DELTA_POSITIVE
            delta_pos = band_delta > 0
            if delta_pos:
                greens.append("BAND_DELTA_POSITIVE")
            print(f"  BAND_DELTA_POSITIVE:               [{'Y' if delta_pos else ' '}] ({band_delta:+,.0f})")
            
            # SELL_PCT_LOW
            sell_low = sell_pct < 40
            if sell_low:
                greens.append("SELL_PCT_LOW")
            print(f"  SELL_PCT_LOW (< 40%):              [{'Y' if sell_low else ' '}] ({sell_pct:.1f}%)")
            
            # ENDING_STRONG
            if len(tps) >= 3:
                ending_strong = sum(
                    1 for tp in tps[-3:]
                    if tp.get("institutional_buying_flag") or (tp.get("large_trade_buy_ratio_30s", 0) or 0) > 0.6
                ) >= 2
            else:
                ending_strong = False
            if ending_strong:
                greens.append("ENDING_STRONG")
            print(f"  ENDING_STRONG (last 3 touches):    [{'Y' if ending_strong else ' '}]")
            
            # BREAKOUT_IDENTIFIED - clear breakout touch with > 2% bounce
            breakout_identified = breakout_touch_idx is not None
            if breakout_identified:
                greens.append("BREAKOUT_IDENTIFIED")
                bt = tps[breakout_touch_idx]
                bt_num = bt.get("touch_number", breakout_touch_idx + 1)
                bt_bounce = (bt.get("bounce_return_30s_pct", 0) or 0) * 100
            print(f"  BREAKOUT_IDENTIFIED (bounce > 2%): [{'Y' if breakout_identified else ' '}]" + 
                  (f" (touch #{bt_num}: +{bt_bounce:.1f}%)" if breakout_identified else ""))
            
            # VOLUME_ACCUMULATION - buy shares significantly outweigh sell shares (even without INST_BUY)
            # Requires: buy shares > 1.5x sell shares AND total volume > 5000
            # (volume_ratio and total_band_volume already calculated above for gate softening)
            volume_accumulation = has_volume_accumulation
            if volume_accumulation:
                greens.append("VOLUME_ACCUMULATION")
            print(f"  VOLUME_ACCUMULATION (buy>=1.5x):   [{'Y' if volume_accumulation else ' '}] ({volume_ratio:.2f}x, {total_band_volume:,.0f} shares)")
            
            print(f"\n  Total Green Flags: {len(greens)}/12")
            if greens:
                print(f"  Active: {', '.join(greens)}")
            
            print("\n" + "=" * 60)
            print("CAUTION FLAGS:")
            print("=" * 60)
            
            cautions = []
            
            # All negative aggr
            all_neg_aggr = all((tp.get("rel_aggr_0_30s", 0) or 0) < 0 for tp in tps)
            if all_neg_aggr:
                cautions.append("AGGR_NEGATIVE_ALL")
            print(f"  AGGR_NEGATIVE_ALL:                 [{'!' if all_neg_aggr else ' '}]")
            
            # No bounce
            no_bounce = not any((tp.get("bounce_return_30s_pct", 0) or 0) > 0.005 for tp in tps)
            if no_bounce:
                cautions.append("NO_BOUNCE")
            print(f"  NO_BOUNCE (none > 0.5%):           [{'!' if no_bounce else ' '}]")
            
            # Exhausted level
            exhausted_level = len(tps) > 40
            if exhausted_level:
                cautions.append("EXHAUSTED_LEVEL")
            print(f"  EXHAUSTED_LEVEL (> 40 touches):    [{'!' if exhausted_level else ' '}] ({len(tps)} touches)")
            
            # Contested sell %
            contested = 45 <= sell_pct <= 55
            if contested:
                cautions.append("SELL_PCT_CONTESTED")
            print(f"  SELL_PCT_CONTESTED (45-55%):       [{'!' if contested else ' '}]")
            
            print(f"\n  Total Caution Flags: {len(cautions)}")
            if cautions:
                print(f"  Active: {', '.join(cautions)}")
            
            print("\n" + "=" * 60)
            print("VERDICT")
            print("=" * 60)
            
            # Confidence modifier based on sample size
            confidence = ""
            if insufficient_data:
                confidence = " (LOW CONFIDENCE - only {len(tps)} touches)"
            
            if any_gate:
                if insufficient_data:
                    print(f"  ==> LEAN AVOID (hard gate tripped, but only {len(tps)} touches)")
                    print(f"      Gate: Would be AVOID with more data.")
                else:
                    print("  ==> AVOID (hard gate tripped)")
            elif softened_no_buyers:
                # NO_BUYERS was tripped but VOLUME_ACCUMULATION softened it
                print(f"  ==> WAIT - ACCUMULATION (no INST_BUY but {volume_ratio:.1f}x buy volume)")
                print(f"      Buyers present but no institutional confirmation.")
                print(f"      Consider smaller position or wait for INST_BUY.")
            elif len(greens) >= 6:
                if insufficient_data:
                    print(f"  ==> LEAN TAKE (excellent signals: {len(greens)}/12 flags, but only {len(tps)} touches)")
                    print(f"      Consider smaller position due to low sample size.")
                else:
                    print(f"  ==> STRONG TAKE (excellent setup: {len(greens)}/12 green flags)")
            elif len(greens) >= 4:
                if insufficient_data:
                    print(f"  ==> LEAN TAKE (good signals: {len(greens)} green, but only {len(tps)} touches)")
                    print(f"      Consider smaller position due to low sample size.")
                else:
                    print(f"  ==> TAKE (good setup: {len(greens)} green, {len(cautions)} caution)")
            elif len(greens) >= 2:
                if insufficient_data:
                    print(f"  ==> WAIT (some signals: {len(greens)} green, only {len(tps)} touches)")
                else:
                    print(f"  ==> WAIT (watch for confirmation: {len(greens)} green)")
            else:
                if insufficient_data:
                    print(f"  ==> LEAN PASS (weak signals: {len(greens)} green, only {len(tps)} touches)")
                else:
                    print(f"  ==> PASS (not enough edge: {len(greens)} green)")

if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "DTCK_20251226_191000"
    main(session_id)

