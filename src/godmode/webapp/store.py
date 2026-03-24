from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from math import asinh, sqrt
import json


def _read_all_parquet_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    for p in paths:
        dfs.append(pq.ParquetFile(p).read().to_pandas())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _compute_ema(prices: list[float], period: int) -> list[float]:
    """Compute EMA for a list of prices."""
    if not prices or period <= 0:
        return []
    ema = []
    multiplier = 2.0 / (period + 1)
    # Initialize with SMA for first `period` values
    if len(prices) < period:
        return [sum(prices) / len(prices)] * len(prices)
    
    sma = sum(prices[:period]) / period
    ema = [0.0] * (period - 1) + [sma]
    
    for i in range(period, len(prices)):
        ema_val = (prices[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)
    
    return ema


def _compute_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float]]:
    """Compute MACD line, signal line, and histogram."""
    if len(prices) < slow:
        return {"macd": [], "signal": [], "histogram": []}
    
    ema_fast = _compute_ema(prices, fast)
    ema_slow = _compute_ema(prices, slow)
    
    # MACD line = EMA_fast - EMA_slow
    macd_line = []
    for i in range(len(prices)):
        if i < slow - 1:
            macd_line.append(0.0)
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])
    
    # Signal line = EMA of MACD line
    macd_values_for_signal = macd_line[slow - 1:]  # Only valid MACD values
    signal_line_partial = _compute_ema(macd_values_for_signal, signal)
    signal_line = [0.0] * (slow - 1) + signal_line_partial
    
    # Histogram = MACD - Signal
    histogram = [macd_line[i] - signal_line[i] for i in range(len(prices))]
    
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def _compute_ema_macd_setup(trades_df: pd.DataFrame, bar_seconds: int) -> dict[str, Any]:
    """
    Compute EMA/MACD setup from trades.
    
    Returns:
        {
            "ema9": float, "ema20": float, "ema30": float,
            "macd": float, "signal": float,
            "ema_ok": bool, "macd_ok": bool, "setup_on": bool,
            "prev_ema9": float, "prev_ema30": float  # for crossover detection
        }
    """
    if trades_df.empty:
        return {"setup_on": False, "ema_ok": False, "macd_ok": False}
    
    # Get timestamp and price columns
    ts_col = "ts_ms" if "ts_ms" in trades_df.columns else "timestamp"
    price_col = "price" if "price" in trades_df.columns else "last_price"
    
    if ts_col not in trades_df.columns or price_col not in trades_df.columns:
        return {"setup_on": False, "ema_ok": False, "macd_ok": False}
    
    df = trades_df.copy()
    df = df.sort_values(ts_col)
    
    # Create bars of specified duration
    df["bar"] = (df[ts_col] // (bar_seconds * 1000)).astype(int)
    
    # OHLC per bar (use last price as close)
    bars = df.groupby("bar").agg({
        price_col: ["first", "max", "min", "last"]
    }).reset_index()
    bars.columns = ["bar", "open", "high", "low", "close"]
    
    if len(bars) < 30:
        return {"setup_on": False, "ema_ok": False, "macd_ok": False, "bars": len(bars)}
    
    closes = bars["close"].tolist()
    
    # Compute EMAs
    ema9 = _compute_ema(closes, 9)
    ema20 = _compute_ema(closes, 20)
    ema30 = _compute_ema(closes, 30)
    
    # Compute MACD
    macd_data = _compute_macd(closes, 12, 26, 9)
    
    # Get current and previous values
    curr_ema9 = ema9[-1] if ema9 else 0
    curr_ema20 = ema20[-1] if ema20 else 0
    curr_ema30 = ema30[-1] if ema30 else 0
    prev_ema9 = ema9[-2] if len(ema9) >= 2 else curr_ema9
    prev_ema30 = ema30[-2] if len(ema30) >= 2 else curr_ema30
    
    curr_macd = macd_data["macd"][-1] if macd_data["macd"] else 0
    curr_signal = macd_data["signal"][-1] if macd_data["signal"] else 0
    
    # Check crossover: EMA9 was below EMA30 and now above
    crossover = (prev_ema9 <= prev_ema30) and (curr_ema9 > curr_ema30)
    
    # emaOK = (EMA9 > EMA20) AND (EMA9 > EMA30 OR crossover)
    ema_ok = (curr_ema9 > curr_ema20) and (curr_ema9 > curr_ema30 or crossover)
    
    # macdOK = MACD > Signal
    macd_ok = curr_macd > curr_signal
    
    # setupON = emaOK AND macdOK
    setup_on = ema_ok and macd_ok
    
    return {
        "ema9": round(curr_ema9, 4),
        "ema20": round(curr_ema20, 4),
        "ema30": round(curr_ema30, 4),
        "macd": round(curr_macd, 6),
        "signal": round(curr_signal, 6),
        "ema_ok": ema_ok,
        "macd_ok": macd_ok,
        "setup_on": setup_on,
        "crossover": crossover,
        "bars": len(bars),
        "last_close": closes[-1] if closes else 0,
    }


def compute_fast_zones_with_quotes(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    level_price: float,
    interval_s: int = 30,
    above_pct: float = 5.0,
    below_pct: float = 5.0,
    on_pct: float = 1.5,
    on_pct_down: float = 2.0,
) -> dict[str, Any]:
    """
    FAST zone computation with LEE-READY buy/sell classification.
    
    LEE-READY ALGORITHM:
    ====================
    For each trade, we find the most recent quote (bid/ask) and calculate:
    
    1. Midpoint = (Bid + Ask) / 2
    
    2. Compare trade price to midpoint:
       - Trade price > Midpoint → BUY (buyer was aggressive, paid above mid)
       - Trade price < Midpoint → SELL (seller was aggressive, accepted below mid)
       - Trade price = Midpoint → Use TICK RULE (compare to previous trade)
    
    3. Tick Rule (fallback when price = midpoint):
       - Price went UP from last trade → BUY
       - Price went DOWN from last trade → SELL
       - Price SAME → Use previous classification
    
    WHY THIS WORKS:
    - In a trade, someone is "aggressor" (crosses the spread to get filled)
    - If trade happens at/above ask → buyer crossed spread → BUY
    - If trade happens at/below bid → seller crossed spread → SELL
    - Midpoint comparison approximates this without exact bid/ask match
    
    ACCURACY: ~85-95% (vs ~70% for tick rule alone)
    """
    if trades_df.empty:
        return {"zone_segments": [], "defense_score": {}, "aggression_score": {}, "market_state": {}}
    
    # Normalize columns
    ts_col = "ts_ms" if "ts_ms" in trades_df.columns else "timestamp"
    price_col = "price" if "price" in trades_df.columns else "last_price"
    size_col = "size" if "size" in trades_df.columns else "qty"
    
    if ts_col not in trades_df.columns or price_col not in trades_df.columns:
        return {"zone_segments": [], "defense_score": {}, "aggression_score": {}, "market_state": {}}
    
    df = trades_df.copy()
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    # =========================================
    # LEE-READY CLASSIFICATION
    # =========================================
    
    has_quotes = (
        not quotes_df.empty 
        and "ts_ms" in quotes_df.columns 
        and "midpoint" in quotes_df.columns
    )
    
    if has_quotes:
        # Sort quotes by timestamp
        qdf = quotes_df.copy().sort_values("ts_ms").reset_index(drop=True)
        quote_ts = qdf["ts_ms"].values
        quote_mid = qdf["midpoint"].values
        
        # For each trade, find the most recent quote using binary search
        import numpy as np
        
        is_buy_list = []
        last_tick_direction = 0.5  # For tick rule fallback
        prev_price = None
        
        for idx, row in df.iterrows():
            trade_ts = row[ts_col]
            trade_price = row[price_col]
            
            # Binary search: find the latest quote before this trade
            quote_idx = np.searchsorted(quote_ts, trade_ts, side='right') - 1
            
            if quote_idx >= 0:
                midpoint = quote_mid[quote_idx]
                
                # Lee-Ready: compare trade price to midpoint
                if trade_price > midpoint:
                    # Trade above midpoint = buyer aggression
                    is_buy_list.append(True)
                    last_tick_direction = 1.0
                elif trade_price < midpoint:
                    # Trade below midpoint = seller aggression
                    is_buy_list.append(False)
                    last_tick_direction = 0.0
                else:
                    # Trade at midpoint - use tick rule
                    if prev_price is not None:
                        if trade_price > prev_price:
                            is_buy_list.append(True)
                            last_tick_direction = 1.0
                        elif trade_price < prev_price:
                            is_buy_list.append(False)
                            last_tick_direction = 0.0
                        else:
                            # Same price - use last direction
                            is_buy_list.append(last_tick_direction >= 0.5)
                    else:
                        is_buy_list.append(last_tick_direction >= 0.5)
            else:
                # No quote before this trade - use tick rule only
                if prev_price is not None:
                    if trade_price > prev_price:
                        is_buy_list.append(True)
                        last_tick_direction = 1.0
                    elif trade_price < prev_price:
                        is_buy_list.append(False)
                        last_tick_direction = 0.0
                    else:
                        is_buy_list.append(last_tick_direction >= 0.5)
                else:
                    is_buy_list.append(True)  # Default first trade to buy
            
            prev_price = trade_price
        
        df["is_buy"] = is_buy_list
        print(f"[LEE-READY] Classified {len(df)} trades using {len(qdf)} quotes")
        
    else:
        # Fallback to tick rule if no quotes available
        print(f"[TICK-RULE] No quotes available, using tick rule for {len(df)} trades")
        df["price_change"] = df[price_col].diff()
        
        is_buy = []
        last_direction = 0.5
        
        for i, row in df.iterrows():
            change = row["price_change"]
            if pd.isna(change) or change == 0:
                is_buy.append(last_direction >= 0.5)
            elif change > 0:
                is_buy.append(True)
                last_direction = 1.0
            else:
                is_buy.append(False)
                last_direction = 0.0
        
        df["is_buy"] = is_buy
    
    # Count buys vs sells for debug
    buy_count = df["is_buy"].sum()
    sell_count = len(df) - buy_count
    print(f"[CLASSIFY] {buy_count} buys, {sell_count} sells ({100*buy_count/len(df):.1f}% buy)")
    
    # =========================================
    # REST OF ZONE COMPUTATION (same as before)
    # =========================================
    
    # Create interval bars
    interval_ms = interval_s * 1000
    df["bar"] = (df[ts_col] // interval_ms).astype(int)
    
    if size_col not in df.columns:
        df[size_col] = 100  # Default size
    
    # Zone classification
    support = float(level_price)
    above_ceil = support * (1.0 + above_pct / 100.0)
    above_floor = support * (1.0 + on_pct / 100.0)
    on_ceil = above_floor
    on_floor = support * (1.0 - on_pct_down / 100.0)
    below_ceil = on_floor
    below_floor = support * (1.0 - below_pct / 100.0)
    
    def classify_zone(p: float) -> str:
        if p >= above_floor and p <= above_ceil:
            return "ABOVE"
        if p >= on_floor and p < on_ceil:
            return "ON"
        if p >= below_floor and p < below_ceil:
            return "BELOW"
        return "OUT"
    
    df["zone"] = df[price_col].apply(classify_zone)
    
    # Count trades per bar for buy ratio calculation
    trade_counts = df.groupby("bar").size().reset_index(name="trade_count")
    
    # Aggregate by bar
    agg = df.groupby("bar").agg({
        ts_col: ["min", "max"],
        price_col: ["first", "last", "min", "max"],
        size_col: "sum",
        "is_buy": "sum",
        "zone": "last",  # Zone at end of bar
    }).reset_index()
    
    # Flatten columns
    agg.columns = ["bar", "start_ts", "end_ts", "open", "close", "low", "high", "total_vol", "buy_count", "zone"]
    
    # Merge trade counts
    agg = agg.merge(trade_counts, on="bar", how="left")
    agg["trade_count"] = agg["trade_count"].fillna(1)
    
    # Estimate buy/sell volume (buy_count / trade_count × total_vol)
    agg["buy_ratio"] = agg["buy_count"] / agg["trade_count"].clip(1)
    agg["buy_ratio"] = agg["buy_ratio"].fillna(0.5).clip(0, 1)
    agg["buy_vol"] = agg["total_vol"] * agg["buy_ratio"]
    agg["sell_vol"] = agg["total_vol"] * (1 - agg["buy_ratio"])
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    
    # Large trades (> 5000 shares) - institutional activity
    large_threshold = 5000
    large_df = df[df[size_col] >= large_threshold].copy() if size_col in df.columns else pd.DataFrame()
    
    if not large_df.empty:
        # Separate buy and sell volumes for large trades
        large_df["inst_buy_vol"] = large_df.apply(
            lambda r: r[size_col] if r["is_buy"] else 0, axis=1
        )
        large_df["inst_sell_vol"] = large_df.apply(
            lambda r: r[size_col] if not r["is_buy"] else 0, axis=1
        )
        
        large_agg = large_df.groupby("bar").agg({
            "inst_buy_vol": "sum",
            "inst_sell_vol": "sum",
        }).reset_index()
        large_agg.columns = ["bar", "inst_buy", "inst_sell"]
        agg = agg.merge(large_agg, on="bar", how="left")
    
    agg["inst_buy"] = agg.get("inst_buy", pd.Series([0] * len(agg))).fillna(0)
    agg["inst_sell"] = agg.get("inst_sell", pd.Series([0] * len(agg))).fillna(0)

    # ---------------------------------------------------------
    # Per-band totals inside each bar (ABOVE/ON/BELOW/OUT)
    # This answers "are there buyers/sellers ON my level band?"
    # without relying on the bar's last-trade zone label.
    # ---------------------------------------------------------
    try:
        size_series = df[size_col].astype(float)
        is_buy_series = df["is_buy"].astype(bool)
        large_threshold = 5000.0
        is_large = size_series >= large_threshold

        df["_buy_sz"] = size_series.where(is_buy_series, 0.0)
        df["_sell_sz"] = size_series.where(~is_buy_series, 0.0)
        df["_inst_buy_sz"] = size_series.where(is_large & is_buy_series, 0.0)
        df["_inst_sell_sz"] = size_series.where(is_large & (~is_buy_series), 0.0)

        band_agg = (
            df.groupby(["bar", "zone"])
            .agg(
                band_total=(size_col, "sum"),
                band_buy=("_buy_sz", "sum"),
                band_sell=("_sell_sz", "sum"),
                band_inst_buy=("_inst_buy_sz", "sum"),
                band_inst_sell=("_inst_sell_sz", "sum"),
            )
            .unstack(fill_value=0.0)
        )

        # Flatten to: buy_vol_on, sell_vol_above, inst_buy_below, etc.
        rename: dict[tuple[str, str], str] = {}
        for (metric, zone) in band_agg.columns:
            zl = str(zone).lower()
            if metric == "band_total":
                rename[(metric, zone)] = f"total_vol_{zl}"
            elif metric == "band_buy":
                rename[(metric, zone)] = f"buy_vol_{zl}"
            elif metric == "band_sell":
                rename[(metric, zone)] = f"sell_vol_{zl}"
            elif metric == "band_inst_buy":
                rename[(metric, zone)] = f"inst_buy_{zl}"
            elif metric == "band_inst_sell":
                rename[(metric, zone)] = f"inst_sell_{zl}"
            else:
                rename[(metric, zone)] = f"{metric}_{zl}"

        band_agg = band_agg.rename(columns=rename).reset_index()
        agg = agg.merge(band_agg, on="bar", how="left").fillna(0.0)
    except Exception:
        # Keep compatibility if anything goes wrong; "on the line" fields will be absent.
        pass
    
    # Build segments (each bar is a segment for simplicity)
    segments = []
    for _, row in agg.iterrows():
        buy_vol = float(row["buy_vol"])
        sell_vol = float(row["sell_vol"])
        delta = float(row["delta"])
        total_vol = float(row["total_vol"])
        inst_buy = float(row.get("inst_buy", 0))
        inst_sell = float(row.get("inst_sell", 0))
        tc = int(row.get("trade_count", 1))
        
        # Use VWAP-like price based on delta direction
        # If buyers won, weight toward high; if sellers won, weight toward low
        open_p = float(row["open"])
        close_p = float(row["close"])
        high_p = float(row["high"])
        low_p = float(row["low"])
        
        if delta > 0:
            # Buyers won - use close or weighted toward high
            display_price = close_p
        elif delta < 0:
            # Sellers won - use close or weighted toward low
            display_price = close_p
        else:
            display_price = (open_p + close_p) / 2
        
        segments.append({
            "zone": row["zone"],
            "start_ts_ms": int(row["start_ts"]),
            "end_ts_ms": int(row["end_ts"]),
            "start_price": open_p,
            "end_price": close_p,
            "display_price": display_price,
            "low": low_p,
            "high": high_p,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "delta": delta,
            "inst_buy": inst_buy,
            "inst_sell": inst_sell,
            "inst_trades": 0,
            "total_vol": total_vol,
            "trade_count": tc,
            "duration_s": interval_s,
            "avg_trade_size": total_vol / tc if tc > 0 else 0,
            "points": 1,
            # Per-band totals inside the segment (may be all zeros if not computed)
            "total_vol_above": float(row.get("total_vol_above", 0.0)),
            "buy_vol_above": float(row.get("buy_vol_above", 0.0)),
            "sell_vol_above": float(row.get("sell_vol_above", 0.0)),
            "inst_buy_above": float(row.get("inst_buy_above", 0.0)),
            "inst_sell_above": float(row.get("inst_sell_above", 0.0)),
            "total_vol_on": float(row.get("total_vol_on", 0.0)),
            "buy_vol_on": float(row.get("buy_vol_on", 0.0)),
            "sell_vol_on": float(row.get("sell_vol_on", 0.0)),
            "inst_buy_on": float(row.get("inst_buy_on", 0.0)),
            "inst_sell_on": float(row.get("inst_sell_on", 0.0)),
            "total_vol_below": float(row.get("total_vol_below", 0.0)),
            "buy_vol_below": float(row.get("buy_vol_below", 0.0)),
            "sell_vol_below": float(row.get("sell_vol_below", 0.0)),
            "inst_buy_below": float(row.get("inst_buy_below", 0.0)),
            "inst_sell_below": float(row.get("inst_sell_below", 0.0)),
            "total_vol_out": float(row.get("total_vol_out", 0.0)),
            "buy_vol_out": float(row.get("buy_vol_out", 0.0)),
            "sell_vol_out": float(row.get("sell_vol_out", 0.0)),
            "inst_buy_out": float(row.get("inst_buy_out", 0.0)),
            "inst_sell_out": float(row.get("inst_sell_out", 0.0)),
        })
    
    # Compute scores (simplified versions)
    # Defense Score: how well is support defended?
    below_segs = [s for s in segments if s["zone"] == "BELOW"]
    on_segs = [s for s in segments if s["zone"] == "ON"]
    above_segs = [s for s in segments if s["zone"] == "ABOVE"]
    
    # Defense = ON/BELOW have positive delta or low selling
    below_buy = sum(s["buy_vol"] for s in below_segs)
    below_sell = sum(s["sell_vol"] for s in below_segs)
    on_buy = sum(s["buy_vol"] for s in on_segs)
    on_sell = sum(s["sell_vol"] for s in on_segs)
    
    defense_score = 50  # Start neutral
    if below_buy + on_buy > 0:
        defense_score = min(100, int(100 * (below_buy + on_buy) / (below_buy + on_buy + below_sell + on_sell + 1)))
    
    # Aggression = ABOVE has positive delta
    above_buy = sum(s["buy_vol"] for s in above_segs)
    above_sell = sum(s["sell_vol"] for s in above_segs)
    aggression_score = 50
    if above_buy + above_sell > 0:
        aggression_score = min(100, int(100 * above_buy / (above_buy + above_sell + 1)))
    
    # Market state
    state = "MIXED"
    action = "WAIT"
    if defense_score >= 65 and aggression_score >= 70:
        state = "BREAK_LIKELY"
        action = "STARTER"
    elif defense_score >= 65 and aggression_score < 50:
        state = "DEFENSE_ONLY"
        action = "WAIT"
    elif defense_score >= 50:
        state = "RECLAIM_DEVELOPING"
        action = "WATCH"
    elif defense_score < 40:
        state = "BREAKDOWN_RISK"
        action = "AVOID"
    
    return {
        "zone_segments": segments,
        "defense_score": {
            "value": defense_score,
            "reasons": [],
        },
        "aggression_score": {
            "value": aggression_score,
            "reasons": [],
        },
        "market_state": {
            "state": state,
            "action": action,
            "reasons": [],
        },
        "distribution_pattern": {"detected": False, "reasons": []},
        "late_momentum": {"bias": "neutral", "score": 0, "reasons": []},
        "zone_analysis": {
            "support_price": support,
            "above_ceiling": above_ceil,
            "below_floor": below_floor,
        },
    }


@dataclass(frozen=True, slots=True)
class Store:
    root_dir: Path

    _DEFAULT_BAND_PCT: float = 0.0015  # 0.15% band for touch detection
    _DEFAULT_SNAPSHOT_INTERVAL_S: int = 10
    _DEFAULT_ZONE_ABOVE_PCT: float = 5.0
    _DEFAULT_ZONE_BELOW_PCT: float = 5.0
    _DEFAULT_ZONE_ON_PCT: float = 1.5
    _DEFAULT_ZONE_ON_PCT_DOWN: float = 2.0

    def _glob(self, kind: str, date: str, ticker: str, session_id: str) -> list[Path]:
        base = (
            self.root_dir
            / kind
            / f"date={date}"
            / f"ticker={ticker}"
            / f"session={session_id}"
        )
        if not base.exists():
            return []
        return sorted(base.glob("part-*.parquet"))

    def list_sessions(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        # Walk episodes/ directory because it is the most "canonical" artifact.
        episodes_root = self.root_dir / "episodes"
        if not episodes_root.exists():
            return out

        # Collect all sessions
        for date_dir in episodes_root.glob("date=*"):
            date = date_dir.name.split("=", 1)[-1]
            for ticker_dir in date_dir.glob("ticker=*"):
                ticker = ticker_dir.name.split("=", 1)[-1]
                for sess_dir in ticker_dir.glob("session=*"):
                    session_id = sess_dir.name.split("=", 1)[-1]
                    out.append({"date": date, "ticker": ticker, "session_id": session_id})
        
        # Sort by date descending, then by session_id descending (which contains timestamp)
        # Session IDs like "OMER_20251226_201100" - the timestamp part determines order
        def sort_key(s):
            # Extract timestamp from session_id (format: TICKER_YYYYMMDD_HHMMSS)
            parts = s["session_id"].split("_")
            if len(parts) >= 3:
                # Combine date and time parts for sorting
                ts_part = parts[1] + parts[2].split("_")[0]  # Handle _1, _2 suffixes
            else:
                ts_part = s["session_id"]
            return (s["date"], ts_part)
        
        out.sort(key=sort_key, reverse=True)
        return out

    def read_episodes(self, *, date: str, ticker: str, session_id: str) -> pd.DataFrame:
        paths = self._glob("episodes", date, ticker, session_id)
        df = _read_all_parquet_files(paths)
        if not df.empty and "zone_entry_time" in df.columns:
            df = df.sort_values(["zone_entry_time"], kind="mergesort").reset_index(drop=True)
        return df

    def read_episode_row(
        self, *, date: str, ticker: str, session_id: str, episode_id: str
    ) -> dict[str, Any]:
        df = self.read_episodes(date=date, ticker=ticker, session_id=session_id)
        if df.empty:
            raise FileNotFoundError("no episodes found for session")
        m = df["episode_id"].astype(str) == str(episode_id)
        sub = df.loc[m]
        if sub.empty:
            raise FileNotFoundError("episode_id not found")
        return sub.iloc[0].to_dict()

    def read_markers(self, *, date: str, ticker: str, session_id: str) -> pd.DataFrame:
        paths = self._glob("markers", date, ticker, session_id)
        df = _read_all_parquet_files(paths)
        if not df.empty and "ts_ms" in df.columns:
            df = df.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)
        return df

    def read_snapshots_for_episode(
        self, *, date: str, ticker: str, session_id: str, episode_id: str
    ) -> pd.DataFrame:
        paths = self._glob("snapshots", date, ticker, session_id)
        df = _read_all_parquet_files(paths)
        if df.empty:
            return df
        df = df[df["episode_id"].astype(str) == str(episode_id)]
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)
        return df

    def read_tf_indicators_for_episode(
        self, *, date: str, ticker: str, session_id: str, episode_id: str
    ) -> pd.DataFrame:
        paths = self._glob("tf_indicators", date, ticker, session_id)
        df = _read_all_parquet_files(paths)
        if df.empty:
            return df
        df = df[df["episode_id"].astype(str) == str(episode_id)]
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values(["timestamp", "timeframe"], kind="mergesort").reset_index(drop=True)
        return df

    def read_session_stream(self, *, date: str, ticker: str, session_id: str) -> pd.DataFrame:
        paths = self._glob("session_stream", date, ticker, session_id)
        df = _read_all_parquet_files(paths)
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)
        return df

    @staticmethod
    def _parse_notes(notes: Any) -> dict[str, Any]:
        if notes is None:
            return {}
        if isinstance(notes, dict):
            return notes
        s = str(notes).strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            # Some paths may store notes already JSON-ish; fail closed.
            return {}

    @staticmethod
    def _choose_level_tag(price_tags: Any) -> tuple[float | None, str | None]:
        """
        Choose the most relevant price tag deterministically.
        Priority: support, then resistance, then first.
        """
        if not isinstance(price_tags, list) or not price_tags:
            return None, None
        # normalize
        norm: list[dict[str, Any]] = []
        for t in price_tags:
            if not isinstance(t, dict):
                continue
            if "price" not in t:
                continue
            try:
                price = float(t["price"])
            except Exception:
                continue
            kind = t.get("kind", None)
            kind_s = str(kind).strip().lower() if kind is not None else None
            norm.append({"price": price, "kind": kind_s})
        if not norm:
            return None, None
        for k in ("support", "resistance"):
            for t in norm:
                if t.get("kind") == k:
                    return float(t["price"]), k
        return float(norm[0]["price"]), (norm[0].get("kind") or None)

    @staticmethod
    def _all_level_tags(price_tags: Any) -> list[dict[str, Any]]:
        """
        Return all normalized price tags deterministically (stable order).
        Each entry: {price: float, kind: str|None}.
        """
        if not isinstance(price_tags, list) or not price_tags:
            return []
        out: list[dict[str, Any]] = []
        for t in price_tags:
            if not isinstance(t, dict) or "price" not in t:
                continue
            try:
                price = float(t["price"])
            except Exception:
                continue
            kind = t.get("kind", None)
            kind_s = str(kind).strip().lower() if kind is not None else None
            out.append({"price": price, "kind": kind_s or None})
        # stable sort: support first, then resistance, then unknown; then by price
        def _rank(k: str | None) -> int:
            if k == "support":
                return 0
            if k == "resistance":
                return 1
            return 2

        out.sort(key=lambda x: (_rank(x.get("kind")), float(x.get("price", 0.0))))
        # de-dupe exact (price, kind)
        seen: set[tuple[float, str | None]] = set()
        uniq: list[dict[str, Any]] = []
        for x in out:
            key = (float(x["price"]), x.get("kind"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(x)
        return uniq

    def _infer_interval_s(self, df: pd.DataFrame) -> int:
        if df.empty or "timestamp" not in df.columns:
            return self._DEFAULT_SNAPSHOT_INTERVAL_S
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().astype("int64")
        if len(ts) < 3:
            return self._DEFAULT_SNAPSHOT_INTERVAL_S
        diffs = ts.diff().dropna()
        # median spacing in seconds
        med_ms = float(diffs.median())
        if med_ms <= 0:
            return self._DEFAULT_SNAPSHOT_INTERVAL_S
        return int(max(1, round(med_ms / 1000.0)))

    def _window_df(self, df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        # inclusive bounds (deterministic)
        return df[(df["timestamp"] >= int(start_ms)) & (df["timestamp"] <= int(end_ms))].copy()

    def _compute_zone_metrics(
        self,
        *,
        df: pd.DataFrame,
        level_price: float,
        start_ms: int,
        end_ms: int,
        interval_s: int,
        above_pct: float | None = None,
        below_pct: float | None = None,
        on_pct: float | None = None,
        on_pct_down: float | None = None,
    ) -> dict[str, Any]:
        win = self._window_df(df, start_ms, end_ms)
        if win.empty:
            return {"error": "no data in window"}
        if "last_price" not in win.columns:
            return {"error": "missing last_price"}

        win = win.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)
        price = pd.to_numeric(win["last_price"], errors="coerce")
        ts = pd.to_numeric(win["timestamp"], errors="coerce")

        a = float(above_pct if above_pct is not None else self._DEFAULT_ZONE_ABOVE_PCT)
        b = float(below_pct if below_pct is not None else self._DEFAULT_ZONE_BELOW_PCT)
        o_up = float(on_pct if on_pct is not None else self._DEFAULT_ZONE_ON_PCT)
        o_dn = float(on_pct_down if on_pct_down is not None else self._DEFAULT_ZONE_ON_PCT_DOWN)

        above_ceiling = level_price * (1.0 + a / 100.0)
        above_floor = level_price * (1.0 + o_up / 100.0)
        on_ceiling = above_floor
        on_floor = level_price * (1.0 - o_dn / 100.0)
        below_ceiling = on_floor
        below_floor = level_price * (1.0 - b / 100.0)

        zone = pd.Series("OUT", index=win.index)
        mask_above = (price >= above_floor) & (price <= above_ceiling)
        mask_on = (price >= on_floor) & (price < on_ceiling)
        mask_below = (price >= below_floor) & (price < below_ceiling)
        zone = zone.where(~mask_above, "ABOVE")
        zone = zone.where(~mask_on, "ON")
        zone = zone.where(~mask_below, "BELOW")

        def _sum_col(col: str, mask: pd.Series) -> float:
            if col not in win.columns:
                return 0.0
            vals = pd.to_numeric(win[col], errors="coerce").fillna(0)
            return float(vals[mask].sum())

        def _zone_stats(z: str) -> dict[str, Any]:
            mask = zone == z
            points = int(mask.sum())
            return {
                "points": points,
                "duration_s": int(points) * int(interval_s),
                "total_volume_sum": _sum_col("total_volume", mask),
                "buy_volume_sum": _sum_col("buy_volume", mask),
                "sell_volume_sum": _sum_col("sell_volume", mask),
                "delta_sum": _sum_col("delta", mask),
                "inst_buy_volume_sum": _sum_col("large_buy_volume_10s", mask),
                "inst_sell_volume_sum": _sum_col("large_sell_volume_10s", mask),
                "inst_trade_count_sum": _sum_col("large_trade_count_10s", mask),
            }

        zones = {z: _zone_stats(z) for z in ("BELOW", "ON", "ABOVE", "OUT")}
        below_inst_buy = zones["BELOW"]["inst_buy_volume_sum"]
        below_inst_sell = zones["BELOW"]["inst_sell_volume_sum"]
        below_inst_ratio = (
            float(below_inst_buy) / float(below_inst_buy + below_inst_sell)
            if (below_inst_buy + below_inst_sell) > 0
            else None
        )

        # Zone transitions and velocity
        zone_shift = zone != zone.shift(1)
        transitions = int(zone_shift.fillna(False).sum() - 1) if len(zone) > 0 else 0

        def _first_ts(mask: pd.Series) -> int | None:
            if not mask.any():
                return None
            return int(ts[mask].iloc[0])

        first_below_ts = _first_ts(zone == "BELOW")
        first_on_after_below_ts = None
        first_above_after_on_ts = None
        first_above_after_below_ts = None

        if first_below_ts is not None:
            after_below = ts > first_below_ts
            first_on_after_below_ts = _first_ts((zone == "ON") & after_below)
            first_above_after_below_ts = _first_ts((zone == "ABOVE") & after_below)
            if first_on_after_below_ts is not None:
                after_on = ts > first_on_after_below_ts
                first_above_after_on_ts = _first_ts((zone == "ABOVE") & after_on)

        time_below_to_on_s = (
            (first_on_after_below_ts - first_below_ts) / 1000.0
            if first_below_ts is not None and first_on_after_below_ts is not None
            else None
        )
        time_on_to_above_s = (
            (first_above_after_on_ts - first_on_after_below_ts) / 1000.0
            if first_on_after_below_ts is not None and first_above_after_on_ts is not None
            else None
        )
        time_below_to_above_s = (
            (first_above_after_below_ts - first_below_ts) / 1000.0
            if first_below_ts is not None and first_above_after_below_ts is not None
            else None
        )

        # SOAK: deep below then recover to ON or above
        deep_below_threshold = level_price * 0.98
        recover_threshold = level_price * 0.998
        deep_mask = price <= deep_below_threshold
        deep_ts = _first_ts(deep_mask)
        recover_ts = None
        if deep_ts is not None:
            recover_ts = _first_ts((price >= recover_threshold) & (ts > deep_ts))
        soak_confirmed = deep_ts is not None and recover_ts is not None
        soak_time_to_recover_s = (
            (recover_ts - deep_ts) / 1000.0 if soak_confirmed else None
        )

        return {
            "bounds": {
                "support": float(level_price),
                "above_floor": float(above_floor),
                "above_ceiling": float(above_ceiling),
                "on_floor": float(on_floor),
                "on_ceiling": float(on_ceiling),
                "below_floor": float(below_floor),
                "below_ceiling": float(below_ceiling),
            },
            "zones": zones,
            "below_inst_buy_ratio": below_inst_ratio,
            "transitions": transitions,
            "velocity": {
                "first_below_ts_ms": first_below_ts,
                "first_on_ts_ms": first_on_after_below_ts,
                "first_above_ts_ms": first_above_after_on_ts,
                "first_above_after_below_ts_ms": first_above_after_below_ts,
                "time_below_to_on_s": time_below_to_on_s,
                "time_on_to_above_s": time_on_to_above_s,
                "time_below_to_above_s": time_below_to_above_s,
            },
            "soak": {
                "confirmed": bool(soak_confirmed),
                "deep_below_threshold": float(deep_below_threshold),
                "recover_threshold": float(recover_threshold),
                "first_deep_ts_ms": deep_ts,
                "recover_ts_ms": recover_ts,
                "time_to_recover_s": soak_time_to_recover_s,
            },
        }

    def _compute_last_minutes_bias(
        self,
        *,
        df: pd.DataFrame,
        end_ms: int,
        minutes: int,
    ) -> dict[str, Any]:
        if df.empty or "timestamp" not in df.columns:
            return {"error": "missing data"}
        start_ms = int(end_ms) - int(minutes * 60_000)
        win = self._window_df(df, start_ms, end_ms)
        if win.empty:
            return {"error": "no data in window"}

        def _sum(col: str) -> float:
            if col not in win.columns:
                return 0.0
            return float(pd.to_numeric(win[col], errors="coerce").fillna(0).sum())

        buy = _sum("buy_volume")
        sell = _sum("sell_volume")
        inst_buy = _sum("large_buy_volume_10s")
        inst_sell = _sum("large_sell_volume_10s")
        delta = buy - sell
        if buy > sell * 1.1:
            winner = "BUYERS"
        elif sell > buy * 1.1:
            winner = "SELLERS"
        else:
            winner = "EVEN"

        return {
            "window_start_ts_ms": int(start_ms),
            "window_end_ts_ms": int(end_ms),
            "buy_vol": buy,
            "sell_vol": sell,
            "delta": delta,
            "inst_buy_vol": inst_buy,
            "inst_sell_vol": inst_sell,
            "winner": winner,
        }

    def _classify_zone(
        self,
        *,
        price: float,
        support: float,
        above_pct: float,
        below_pct: float,
        on_pct: float,
        on_pct_down: float,
    ) -> str:
        above_ceiling = support * (1.0 + above_pct / 100.0)
        above_floor = support * (1.0 + on_pct / 100.0)
        on_ceiling = above_floor
        on_floor = support * (1.0 - on_pct_down / 100.0)
        below_ceiling = on_floor
        below_floor = support * (1.0 - below_pct / 100.0)

        if price >= above_floor and price <= above_ceiling:
            return "ABOVE"
        if price >= on_floor and price < on_ceiling:
            return "ON"
        if price >= below_floor and price < below_ceiling:
            return "BELOW"
        return "OUT"

    def _compute_zone_segments(
        self,
        *,
        df: pd.DataFrame,
        level_price: float,
        start_ms: int,
        end_ms: int,
        interval_s: int,
        above_pct: float,
        below_pct: float,
        on_pct: float,
        on_pct_down: float,
    ) -> list[dict[str, Any]]:
        win = self._window_df(df, start_ms, end_ms)
        if win.empty or "timestamp" not in win.columns or "last_price" not in win.columns:
            return []
        win = win.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)

        def _num(col: str) -> pd.Series:
            if col not in win.columns:
                return pd.Series([0.0] * len(win))
            return pd.to_numeric(win[col], errors="coerce").fillna(0.0)

        price = pd.to_numeric(win["last_price"], errors="coerce")
        ts = pd.to_numeric(win["timestamp"], errors="coerce")
        buy = _num("buy_volume")
        sell = _num("sell_volume")
        delta = _num("delta")
        inst_buy = _num("large_buy_volume_10s")
        inst_sell = _num("large_sell_volume_10s")
        inst_trades = _num("large_trade_count_10s")
        total_vol = _num("total_volume")
        trade_count = _num("trade_count")
        low_col = (
            pd.to_numeric(win["low_10s"], errors="coerce")
            if "low_10s" in win.columns
            else price
        )
        high_col = (
            pd.to_numeric(win["high_10s"], errors="coerce")
            if "high_10s" in win.columns
            else price
        )

        zones = []
        for i in range(len(win)):
            p = float(price.iloc[i]) if pd.notna(price.iloc[i]) else None
            if p is None:
                zones.append("OUT")
            else:
                zones.append(
                    self._classify_zone(
                        price=p,
                        support=float(level_price),
                        above_pct=float(above_pct),
                        below_pct=float(below_pct),
                        on_pct=float(on_pct),
                        on_pct_down=float(on_pct_down),
                    )
                )

        segments: list[dict[str, Any]] = []
        if not zones:
            return segments

        def _start_segment(idx: int, zone: str) -> dict[str, Any]:
            p = float(price.iloc[idx]) if pd.notna(price.iloc[idx]) else 0.0
            b = float(buy.iloc[idx])
            s = float(sell.iloc[idx])
            return {
                "zone": zone,
                "start_ts_ms": int(ts.iloc[idx]) if pd.notna(ts.iloc[idx]) else None,
                "end_ts_ms": int(ts.iloc[idx]) if pd.notna(ts.iloc[idx]) else None,
                "start_price": p if p else None,
                "end_price": p if p else None,
                "low": float(low_col.iloc[idx]) if pd.notna(low_col.iloc[idx]) else None,
                "high": float(high_col.iloc[idx]) if pd.notna(high_col.iloc[idx]) else None,
                "buy_vol": b,
                "sell_vol": s,
                "delta": float(delta.iloc[idx]),
                "inst_buy": float(inst_buy.iloc[idx]),
                "inst_sell": float(inst_sell.iloc[idx]),
                "inst_trades": float(inst_trades.iloc[idx]),
                "total_vol": float(total_vol.iloc[idx]),
                "trade_count": float(trade_count.iloc[idx]),
                "points": 1,
                # Track individual rows for Winner's Price calculation
                "_rows": [(p, b, s)] if p else [],
            }

        def _add_row(seg: dict[str, Any], idx: int) -> None:
            p = float(price.iloc[idx]) if pd.notna(price.iloc[idx]) else None
            b = float(buy.iloc[idx])
            s = float(sell.iloc[idx])
            seg["end_ts_ms"] = int(ts.iloc[idx]) if pd.notna(ts.iloc[idx]) else seg["end_ts_ms"]
            seg["end_price"] = p if p else seg["end_price"]
            if pd.notna(low_col.iloc[idx]):
                seg["low"] = (
                    float(low_col.iloc[idx])
                    if seg["low"] is None
                    else min(seg["low"], float(low_col.iloc[idx]))
                )
            if pd.notna(high_col.iloc[idx]):
                seg["high"] = (
                    float(high_col.iloc[idx])
                    if seg["high"] is None
                    else max(seg["high"], float(high_col.iloc[idx]))
                )
            seg["buy_vol"] += b
            seg["sell_vol"] += s
            seg["delta"] += float(delta.iloc[idx])
            seg["inst_buy"] += float(inst_buy.iloc[idx])
            seg["inst_sell"] += float(inst_sell.iloc[idx])
            seg["inst_trades"] += float(inst_trades.iloc[idx])
            seg["total_vol"] += float(total_vol.iloc[idx])
            seg["trade_count"] += float(trade_count.iloc[idx])
            seg["points"] += 1
            # Track for Winner's Price
            if p:
                seg["_rows"].append((p, b, s))

        # Create segments based on zone transitions AND time intervals
        # Each segment is max interval_s seconds (default 10s), or until zone changes
        interval_ms = interval_s * 1000
        current = _start_segment(0, zones[0])
        for i in range(1, len(zones)):
            current_ts = int(ts.iloc[i]) if pd.notna(ts.iloc[i]) else 0
            start_ts = current.get("start_ts_ms") or 0
            time_elapsed = current_ts - start_ts
            
            # Start new segment if zone changes OR time exceeds interval
            if zones[i] != current["zone"] or time_elapsed >= interval_ms:
                segments.append(current)
                current = _start_segment(i, zones[i])
            else:
                _add_row(current, i)
        segments.append(current)

        def _compute_winner_price(seg: dict[str, Any]) -> float:
            """
            Compute display price based on winning side's activity.
            Rule: If #1 >= #2 × 1.5 → leader's price, else weighted avg of top 3.
            """
            rows = seg.get("_rows", [])
            if not rows:
                return seg.get("end_price") or seg.get("start_price") or 0.0
            
            # Determine winner: buyers (delta > 0) or sellers (delta < 0)
            seg_delta = seg.get("delta", 0)
            is_buyer_win = seg_delta >= 0
            
            # Extract (price, winning_vol) pairs
            if is_buyer_win:
                # Sort by buy volume descending
                ranked = sorted([(p, b) for p, b, s in rows if b > 0], key=lambda x: -x[1])
            else:
                # Sort by sell volume descending
                ranked = sorted([(p, s) for p, b, s in rows if s > 0], key=lambda x: -x[1])
            
            if not ranked:
                return seg.get("end_price") or seg.get("start_price") or 0.0
            
            if len(ranked) == 1:
                return ranked[0][0]  # Only one row, use its price
            
            top1_price, top1_vol = ranked[0]
            top2_price, top2_vol = ranked[1]
            
            # Leader rule: #1 >= #2 × 1.5
            if top1_vol >= top2_vol * 1.5:
                return top1_price
            
            # Otherwise: weighted average of top 3
            top3 = ranked[:3]
            total_vol = sum(v for _, v in top3)
            if total_vol == 0:
                return top1_price
            
            weighted_price = sum(p * v for p, v in top3) / total_vol
            return weighted_price

        for seg in segments:
            seg["duration_s"] = int(seg.get("points", 0)) * int(interval_s)
            tc = float(seg.get("trade_count") or 0)
            seg["avg_trade_size"] = (
                float(seg["total_vol"]) / tc if tc > 0 else None
            )
            # Compute Winner's Price for dot positioning
            seg["display_price"] = _compute_winner_price(seg)
            # Clean up internal tracking
            seg.pop("_rows", None)

        return segments

    def _annotate_zone_segments(
        self,
        *,
        segments: list[dict[str, Any]],
        zone_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not segments:
            return []
        annotations: list[list[str]] = [[] for _ in segments]

        for i, seg in enumerate(segments):
            zone = seg.get("zone")
            buy = float(seg.get("buy_vol") or 0)
            sell = float(seg.get("sell_vol") or 0)
            delta = float(seg.get("delta") or 0)
            inst_buy = float(seg.get("inst_buy") or 0)
            inst_sell = float(seg.get("inst_sell") or 0)
            total = float(seg.get("total_vol") or 0)

            if zone == "BELOW" and sell > buy * 2:
                annotations[i].append(
                    "Heavy selling below support at ${}".format(
                        f"{seg.get('start_price'):.4f}" if seg.get("start_price") else "?"
                    )
                )
            if zone == "BELOW" and buy > sell:
                annotations[i].append(
                    "Buyers accumulating at ${}".format(
                        f"{seg.get('start_price'):.4f}" if seg.get("start_price") else "?"
                    )
                )
            if zone == "ABOVE" and delta > 0 and inst_sell < inst_buy * 1.2:
                annotations[i].append("Light resistance above, clean air")
            if zone == "ABOVE" and delta < 0 and inst_sell > inst_buy * 1.2:
                annotations[i].append("Heavy resistance above, sellers defending")

            if total > 0 and i > 0:
                prev_total = float(segments[i - 1].get("total_vol") or 0)
                if total < prev_total * 0.8:
                    annotations[i].append("Volume declining vs prior segment")

            if i > 0:
                prev_delta = float(segments[i - 1].get("delta") or 0)
                if prev_delta < 0 and delta > 0:
                    annotations[i].append("Momentum shifted to buyers")

        soak = zone_analysis.get("soak", {}) if isinstance(zone_analysis, dict) else {}
        if soak.get("confirmed"):
            for i, seg in enumerate(segments):
                if seg.get("zone") in {"ON", "ABOVE"}:
                    annotations[i].append("SOAK confirmed - reclaimed support")
                    break

        return [
            {**seg, "annotations": annotations[idx]}
            for idx, seg in enumerate(segments)
        ]

    def _compute_additional_signals(
        self,
        *,
        segments: list[dict[str, Any]],
        zone_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        if not segments:
            return {}

        # INST timing: leading if INST_BUY in BELOW before first ON
        first_on_idx = next((i for i, s in enumerate(segments) if s.get("zone") == "ON"), None)
        below_before_on = []
        below_after_on = []
        for i, s in enumerate(segments):
            if s.get("zone") != "BELOW":
                continue
            if first_on_idx is not None and i < first_on_idx:
                below_before_on.append(s)
            else:
                below_after_on.append(s)
        inst_leading = any((s.get("inst_buy") or 0) > 0 for s in below_before_on)
        inst_chasing = not inst_leading and any((s.get("inst_buy") or 0) > 0 for s in below_after_on)
        inst_timing = "leading" if inst_leading else "chasing" if inst_chasing else "neutral"

        # Volume trend per zone (compare first to last)
        def _trend(zone: str) -> str:
            zsegs = [s for s in segments if s.get("zone") == zone]
            if len(zsegs) < 2:
                return "n/a"
            first = float(zsegs[0].get("total_vol") or 0)
            last = float(zsegs[-1].get("total_vol") or 0)
            if last > first * 1.2:
                return "increasing"
            if last < first * 0.8:
                return "decreasing"
            return "flat"

        volume_trend = {
            "BELOW": _trend("BELOW"),
            "ON": _trend("ON"),
            "ABOVE": _trend("ABOVE"),
        }

        # Price velocity from zone_analysis
        velocity = zone_analysis.get("velocity", {}) if isinstance(zone_analysis, dict) else {}
        t_below_on = velocity.get("time_below_to_on_s")
        t_on_above = velocity.get("time_on_to_above_s")
        price_velocity = "n/a"
        if isinstance(t_below_on, (int, float)) and isinstance(t_on_above, (int, float)):
            if t_below_on <= 60 and t_on_above <= 60:
                price_velocity = "fast"
            elif t_below_on >= 180 or t_on_above >= 180:
                price_velocity = "slow"
            else:
                price_velocity = "mixed"

        # Repeat tests of BELOW
        below_tests = [s for s in segments if s.get("zone") == "BELOW"]
        repeat_tests = {
            "count": len(below_tests),
            "trend": "n/a",
        }
        if len(below_tests) >= 2:
            deltas = [float(s.get("delta") or 0) for s in below_tests]
            if deltas[-1] > deltas[0]:
                repeat_tests["trend"] = "stronger"
            elif deltas[-1] < deltas[0]:
                repeat_tests["trend"] = "weaker"
            else:
                repeat_tests["trend"] = "flat"

        # Delta acceleration overall
        delta_accel = "n/a"
        if len(segments) >= 2:
            deltas = [float(s.get("delta") or 0) for s in segments[-3:]]
            if len(deltas) >= 2 and deltas[-1] > deltas[-2]:
                delta_accel = "improving"
            elif len(deltas) >= 2 and deltas[-1] < deltas[-2]:
                delta_accel = "worsening"
            else:
                delta_accel = "flat"

        # Trade size trend
        avg_sizes = [
            s.get("avg_trade_size")
            for s in segments
            if s.get("avg_trade_size") is not None
        ]
        trade_size_trend = "n/a"
        if len(avg_sizes) >= 2:
            if avg_sizes[-1] > avg_sizes[0] * 1.2:
                trade_size_trend = "larger"
            elif avg_sizes[-1] < avg_sizes[0] * 0.8:
                trade_size_trend = "smaller"
            else:
                trade_size_trend = "flat"

        # Time of day (Chicago) from first segment start
        time_of_day = "n/a"
        try:
            from datetime import datetime, timezone
            import pytz

            ts0 = segments[0].get("start_ts_ms")
            if ts0:
                dt = datetime.fromtimestamp(int(ts0) / 1000, tz=timezone.utc)
                chicago = pytz.timezone("America/Chicago")
                local = dt.astimezone(chicago)
                hm = local.hour + local.minute / 60.0
                if 9.5 <= hm < 10.5:
                    time_of_day = "open"
                elif 10.5 <= hm < 14.5:
                    time_of_day = "midday"
                elif 14.5 <= hm <= 16.0:
                    time_of_day = "power_hour"
                else:
                    time_of_day = "off_hours"
        except Exception:
            time_of_day = "n/a"

        return {
            "inst_timing": inst_timing,
            "volume_trend": volume_trend,
            "price_velocity": price_velocity,
            "repeat_tests": repeat_tests,
            "time_of_day": time_of_day,
            "delta_acceleration": delta_accel,
            "trade_size_trend": trade_size_trend,
        }

    def _compute_risk_reward(
        self,
        *,
        level_price: float,
        zone_analysis: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        below_lows = [s.get("low") for s in segments if s.get("zone") == "BELOW" and s.get("low") is not None]
        stop_level = min(below_lows) if below_lows else None
        above_ceiling = None
        try:
            above_ceiling = float(zone_analysis.get("bounds", {}).get("above_ceiling"))
        except Exception:
            above_ceiling = None
        risk = float(level_price - stop_level) if stop_level is not None else None
        reward = float(above_ceiling - level_price) if above_ceiling is not None else None
        rr = None
        if risk is not None and reward is not None and risk > 0:
            rr = float(reward / risk)
        return {
            "stop_level": stop_level,
            "target": above_ceiling,
            "risk": risk,
            "reward": reward,
            "rr": rr,
        }

    def _compute_final_verdict(
        self,
        *,
        zone_analysis: dict[str, Any],
        additional: dict[str, Any],
        segments: list[dict[str, Any]],
        risk_reward: dict[str, Any],
    ) -> dict[str, Any]:
        soak = bool(zone_analysis.get("soak", {}).get("confirmed"))
        zones = zone_analysis.get("zones", {})
        above = zones.get("ABOVE", {})
        above_delta = float(above.get("delta_sum") or 0)
        above_inst_sell = float(above.get("inst_sell_volume_sum") or 0)
        above_inst_buy = float(above.get("inst_buy_volume_sum") or 0)
        above_total = float(above.get("total_volume_sum") or 0)
        clean_above = above_delta > 0 and above_inst_sell < max(above_inst_buy * 1.2, above_total * 0.2)
        heavy_above = above_delta < 0 and above_inst_sell > above_inst_buy * 1.2

        below_inst_buy = float(zones.get("BELOW", {}).get("inst_buy_volume_sum") or 0)
        below_inst_sell = float(zones.get("BELOW", {}).get("inst_sell_volume_sum") or 0)
        on_inst_buy = float(zones.get("ON", {}).get("inst_buy_volume_sum") or 0)
        on_inst_sell = float(zones.get("ON", {}).get("inst_sell_volume_sum") or 0)
        inst_buy_below = below_inst_buy > below_inst_sell * 1.1
        inst_buy_on = on_inst_buy > on_inst_sell * 1.1
        inst_buy_above = above_inst_buy > above_inst_sell * 1.1
        inst_buy_across_zones = inst_buy_below and inst_buy_on and inst_buy_above

        last_3m = zone_analysis.get("last_minutes", {}).get("last_3m", {})
        last_winner = last_3m.get("winner")
        inst_timing = additional.get("inst_timing")

        confidence = "WAIT"
        reason: list[str] = []

        def _downgrade(current: str, target: str) -> str:
            order = ["AVOID", "WAIT", "LEAN_LONG", "STRONG_LONG"]
            if current not in order or target not in order:
                return current
            return target if order.index(current) > order.index(target) else current

        if soak and clean_above and last_winner == "BUYERS" and inst_timing == "leading":
            confidence = "STRONG_LONG"
            reason.append("SOAK + clean ABOVE + buyers late + INST leading")
        elif soak and clean_above and last_winner in {"EVEN", "BUYERS"}:
            confidence = "LEAN_LONG"
            reason.append("SOAK + clean ABOVE + mixed/positive late flow")
        elif soak and last_winner == "SELLERS":
            confidence = "WAIT"
            reason.append("SOAK but sellers still active late")
        elif (not soak or heavy_above or inst_timing == "chasing") and not inst_buy_across_zones:
            confidence = "AVOID"
            reason.append("No SOAK or heavy ABOVE resistance / INST selling")

        if inst_buy_across_zones and last_winner != "SELLERS":
            confidence = _downgrade(confidence, "LEAN_LONG")
            reason.append("INST buying across zones")
        elif confidence == "WAIT" and last_winner != "SELLERS" and (inst_buy_on or inst_buy_above):
            confidence = _downgrade(confidence, "LEAN_LONG")
            reason.append("INST buying in ON/ABOVE")

        rr = risk_reward.get("rr")
        rr_blocked = False
        if isinstance(rr, (int, float)) and rr < 1.0:
            confidence = _downgrade(confidence, "WAIT")
            rr_blocked = True
            reason.append("R:R below 1.0")

        repeat = additional.get("repeat_tests", {})
        if repeat.get("count", 0) >= 10 and repeat.get("trend") == "weaker":
            confidence = _downgrade(confidence, "WAIT")
            reason.append("Level exhausted after 10+ weaker tests")

        for seg in segments[-10:]:
            inst_sell = float(seg.get("inst_sell") or 0)
            inst_buy = float(seg.get("inst_buy") or 0)
            if inst_sell > 10000 and inst_buy == 0:
                confidence = _downgrade(confidence, "WAIT")
                reason.append("Late INST sell dump detected")
                break

        if additional.get("delta_acceleration") == "worsening":
            if confidence == "STRONG_LONG":
                confidence = "LEAN_LONG"
            reason.append("Delta acceleration worsening")

        vol_trend = additional.get("volume_trend", {})
        trends = [v for v in vol_trend.values() if v != "n/a"]
        if trends and all(v == "decreasing" for v in trends):
            confidence = _downgrade(confidence, "WAIT")
            reason.append("Volume declining across all zones")

        if segments and segments[-1].get("zone") == "OUT" and float(segments[-1].get("delta") or 0) < 0:
            reason.append("Ended outside zones with negative delta")

        # Late momentum override (flag only if R:R blocked)
        late_momentum = False
        if len(segments) >= 3:
            last_3 = segments[-3:]
            late_delta = sum(float(s.get("delta") or 0) for s in last_3)
            if late_delta > 200000 and all(float(s.get("delta") or 0) > 0 for s in last_3):
                late_momentum = True
                if not rr_blocked:
                    confidence = _downgrade(confidence, "LEAN_LONG")
                reason.append("Strong late buyer momentum")
                if rr_blocked and isinstance(rr, (int, float)):
                    reason.append(f"Late momentum detected, but R:R is {rr:.2f}")

        # Accumulation counter: do not allow AVOID if repeated accumulation
        accumulation_count = 0
        for s in segments:
            anns = s.get("annotations") or []
            if any("Buyers accumulating" in a for a in anns):
                accumulation_count += 1
        if accumulation_count >= 3:
            reason.append(f"Buyers accumulated {accumulation_count}x")
            if confidence == "AVOID":
                confidence = "WAIT"

        # Tighter stop suggestion if R:R blocked and late momentum present
        rr_suggestion = None
        if rr_blocked and late_momentum:
            target = risk_reward.get("target")
            if isinstance(target, (int, float)):
                desired_rr = 1.5
                stop_tighter = float(target) - (float(target) - float(zone_analysis.get("bounds", {}).get("support", 0.0))) / desired_rr
                rr_suggestion = {
                    "desired_rr": desired_rr,
                    "suggested_stop": stop_tighter,
                }

        return {
            "confidence": confidence,
            "reasons": reason,
            "rr_blocked": rr_blocked,
            "rr_suggestion": rr_suggestion,
        }

    def _compute_trade_trigger(
        self,
        *,
        level_price: float,
        risk_reward: dict[str, Any],
        zone_analysis: dict[str, Any],
    ) -> str | None:
        stop = risk_reward.get("stop_level")
        target = risk_reward.get("target")
        reclaim = None
        try:
            reclaim = float(zone_analysis.get("bounds", {}).get("on_floor"))
        except Exception:
            reclaim = None
        if stop is None or target is None or reclaim is None:
            return None
        return (
            "Enter when price reclaims ${} with buyer delta, stop at ${}, target ${}".format(
                f"{reclaim:.4f}", f"{float(stop):.4f}", f"{float(target):.4f}"
            )
        )

    def _compute_role_behavior(
        self,
        *,
        zone_analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        level_kind: str | None,
    ) -> dict[str, Any]:
        zones = zone_analysis.get("zones", {}) if isinstance(zone_analysis, dict) else {}
        below = zones.get("BELOW", {})
        above = zones.get("ABOVE", {})
        last_3m = zone_analysis.get("last_minutes", {}).get("last_3m", {})
        bounds = zone_analysis.get("bounds", {})

        support_score = 0
        resistance_score = 0
        reasons_support: list[str] = []
        reasons_resistance: list[str] = []

        below_delta = float(below.get("delta_sum") or 0)
        below_inst_buy = float(below.get("inst_buy_volume_sum") or 0)
        below_inst_sell = float(below.get("inst_sell_volume_sum") or 0)
        above_delta = float(above.get("delta_sum") or 0)
        above_inst_buy = float(above.get("inst_buy_volume_sum") or 0)
        above_inst_sell = float(above.get("inst_sell_volume_sum") or 0)

        if below_delta > 0:
            support_score += 1
            reasons_support.append("below_delta_positive")
        if below_inst_buy > below_inst_sell:
            support_score += 1
            reasons_support.append("below_inst_buying")
        if bool(zone_analysis.get("soak", {}).get("confirmed")):
            support_score += 1
            reasons_support.append("soak_confirmed")
        if last_3m.get("winner") == "BUYERS":
            support_score += 1
            reasons_support.append("late_buyers")

        if above_delta < 0:
            resistance_score += 1
            reasons_resistance.append("above_delta_negative")
        if above_inst_sell > above_inst_buy:
            resistance_score += 1
            reasons_resistance.append("above_inst_selling")
        if last_3m.get("winner") == "SELLERS":
            resistance_score += 1
            reasons_resistance.append("late_sellers")

        above_ceiling = None
        below_floor = None
        try:
            above_ceiling = float(bounds.get("above_ceiling"))
            below_floor = float(bounds.get("below_floor"))
        except Exception:
            above_ceiling = None
            below_floor = None

        # Failed breakout: OUT above ceiling then back inside
        if above_ceiling is not None and segments:
            saw_out_above = False
            for seg in segments:
                if seg.get("zone") == "OUT" and seg.get("high") is not None and float(seg.get("high")) >= above_ceiling:
                    saw_out_above = True
                    continue
                if saw_out_above and seg.get("zone") in {"ABOVE", "ON", "BELOW"}:
                    resistance_score += 1
                    reasons_resistance.append("failed_breakout")
                    break

        # Failed breakdown: OUT below floor then back inside
        if below_floor is not None and segments:
            saw_out_below = False
            for seg in segments:
                if seg.get("zone") == "OUT" and seg.get("low") is not None and float(seg.get("low")) <= below_floor:
                    saw_out_below = True
                    continue
                if saw_out_below and seg.get("zone") in {"BELOW", "ON", "ABOVE"}:
                    support_score += 1
                    reasons_support.append("failed_breakdown")
                    break

        if support_score > resistance_score:
            role = "support"
        elif resistance_score > support_score:
            role = "resistance"
        else:
            role = "mixed"

        diff = abs(support_score - resistance_score)
        if diff >= 2:
            confidence = "high"
        elif diff >= 1:
            confidence = "med"
        else:
            confidence = "low"

        mismatch = False
        if level_kind and role in {"support", "resistance"}:
            mismatch = str(level_kind).lower() != role

        return {
            "role": role,
            "confidence": confidence,
            "support_score": support_score,
            "resistance_score": resistance_score,
            "support_reasons": reasons_support,
            "resistance_reasons": reasons_resistance,
            "mismatch": mismatch,
        }

    # =========================================================================
    # DEFENSE SCORE (DS) / AGGRESSION SCORE (AS) / STATE MACHINE
    # =========================================================================

    def _compute_defense_score(
        self,
        *,
        segments: list[dict[str, Any]],
        zone_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Defense Score (DS): How well is support being defended?
        
        Signals that increase DS:
        - ON zone has more BUY than SELL
        - BELOW zone is quiet (low volume, low count)
        - INST buying at ON
        - Absorption: SELL at ON gets absorbed (followed by BUY)
        
        Returns score 0-100 with breakdown.
        """
        score = 0
        max_score = 100
        reasons: list[str] = []
        
        zones = zone_analysis.get("zones", {}) if isinstance(zone_analysis, dict) else {}
        on_zone = zones.get("ON", {})
        below_zone = zones.get("BELOW", {})
        
        on_buy = float(on_zone.get("buy_volume_sum") or 0)
        on_sell = float(on_zone.get("sell_volume_sum") or 0)
        on_delta = float(on_zone.get("delta_sum") or 0)
        on_inst_buy = float(on_zone.get("inst_buy_volume_sum") or 0)
        on_inst_sell = float(on_zone.get("inst_sell_volume_sum") or 0)
        
        below_buy = float(below_zone.get("buy_volume_sum") or 0)
        below_sell = float(below_zone.get("sell_volume_sum") or 0)
        below_delta = float(below_zone.get("delta_sum") or 0)
        below_total = float(below_zone.get("total_volume_sum") or 0)
        below_inst_sell = float(below_zone.get("inst_sell_volume_sum") or 0)
        
        # 1. ON zone delta positive (+20)
        if on_delta > 0:
            score += 20
            reasons.append("ON delta positive")
        elif on_delta == 0:
            score += 10
            reasons.append("ON delta neutral")
        
        # 2. ON zone INST buying > selling (+15)
        if on_inst_buy > on_inst_sell * 1.2 and on_inst_buy > 1000:
            score += 15
            reasons.append("INST buying at support")
        elif on_inst_sell > on_inst_buy * 1.5 and on_inst_sell > 1000:
            score -= 10
            reasons.append("INST selling at support")
        
        # 3. BELOW zone quiet (+15) or active selling (-15)
        total_volume = sum(float(z.get("total_volume_sum") or 0) for z in zones.values())
        below_pct = (below_total / total_volume * 100) if total_volume > 0 else 0
        if below_pct < 20:
            score += 15
            reasons.append("BELOW quiet ({}%)".format(int(below_pct)))
        elif below_pct > 40:
            score -= 10
            reasons.append("Heavy BELOW activity ({}%)".format(int(below_pct)))
        
        # 4. BELOW zone delta positive (+15) - buyers accumulating
        if below_delta > 0:
            score += 15
            reasons.append("Buyers accumulating BELOW")
        elif below_delta < 0 and abs(below_delta) > below_buy * 0.3:
            score -= 10
            reasons.append("Sellers dominating BELOW")
        
        # 5. No INST selling BELOW (+10) or heavy INST selling (-15)
        if below_inst_sell < 1000:
            score += 10
            reasons.append("No INST selling BELOW")
        elif below_inst_sell > 10000:
            score -= 15
            reasons.append("Heavy INST selling BELOW")
        
        # 6. SOAK confirmed (+15)
        if zone_analysis.get("soak", {}).get("confirmed"):
            score += 15
            reasons.append("SOAK confirmed")
        
        # 7. Absorption detection: look for SELL segments followed by BUY segments at ON
        absorption_count = 0
        for i, seg in enumerate(segments[:-1]):
            if seg.get("zone") == "ON" and float(seg.get("delta") or 0) < 0:
                next_seg = segments[i + 1]
                if next_seg.get("zone") == "ON" and float(next_seg.get("delta") or 0) > 0:
                    absorption_count += 1
        if absorption_count >= 2:
            score += 10
            reasons.append("Absorption detected ({} times)".format(absorption_count))
        
        # 8. ACCUMULATION PATTERN: INST buying when price dips BELOW
        # Look for: price enters BELOW → INST buying appears → price recovers to ON
        below_inst_buy = float(below_zone.get("inst_buy_volume_sum") or 0)
        accumulation_detected = False
        accumulation_count = 0
        
        # Count sequences: BELOW segment with INST buying followed by ON/ABOVE segment
        for i, seg in enumerate(segments[:-1]):
            if seg.get("zone") == "BELOW":
                seg_inst_buy = float(seg.get("inst_buy") or 0)
                seg_inst_sell = float(seg.get("inst_sell") or 0)
                # INST buying the dip
                if seg_inst_buy > seg_inst_sell and seg_inst_buy > 1000:
                    next_seg = segments[i + 1]
                    # And price recovers to ON or ABOVE
                    if next_seg.get("zone") in {"ON", "ABOVE"}:
                        accumulation_count += 1
        
        # Also check: INST buy volume in BELOW > INST sell volume in BELOW
        if below_inst_buy > below_inst_sell * 1.3 and below_inst_buy > 3000:
            accumulation_detected = True
            score += 15
            reasons.append("INST accumulating BELOW ({}K bought)".format(int(below_inst_buy / 1000)))
        
        if accumulation_count >= 2:
            accumulation_detected = True
            score += 10
            reasons.append("Dip-buying pattern ({} times)".format(accumulation_count))
        elif accumulation_count == 1:
            score += 5
            reasons.append("Single dip-buy detected")
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "max": max_score,
            "pct": int(score),
            "reasons": reasons,
            "grade": "HIGH" if score >= 65 else "MEDIUM" if score >= 40 else "LOW",
            "accumulation_detected": accumulation_detected,
            "accumulation_count": accumulation_count,
        }

    def _compute_aggression_score(
        self,
        *,
        segments: list[dict[str, Any]],
        zone_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Aggression Score (AS): Are buyers pressing into ABOVE zone?
        
        Signals that increase AS:
        - ABOVE zone has BUY dominance
        - INST buying ABOVE
        - Volume ABOVE is growing (late segments bigger than early)
        - Price migrating from ON → ABOVE
        - Seller response ABOVE failing (no price drop back to ON)
        
        Returns score 0-100 with breakdown.
        """
        score = 0
        max_score = 100
        reasons: list[str] = []
        
        zones = zone_analysis.get("zones", {}) if isinstance(zone_analysis, dict) else {}
        above_zone = zones.get("ABOVE", {})
        
        above_buy = float(above_zone.get("buy_volume_sum") or 0)
        above_sell = float(above_zone.get("sell_volume_sum") or 0)
        above_delta = float(above_zone.get("delta_sum") or 0)
        above_inst_buy = float(above_zone.get("inst_buy_volume_sum") or 0)
        above_inst_sell = float(above_zone.get("inst_sell_volume_sum") or 0)
        above_total = float(above_zone.get("total_volume_sum") or 0)
        above_time = float(above_zone.get("duration_s") or 0)
        
        # Get total session time
        total_time = sum(float(z.get("duration_s") or 0) for z in zones.values())
        above_time_pct = (above_time / total_time * 100) if total_time > 0 else 0
        
        # 1. ABOVE zone delta positive (+20)
        if above_delta > 0:
            score += 20
            reasons.append("ABOVE delta positive")
        elif above_delta < 0 and abs(above_delta) > above_buy * 0.3:
            score -= 15
            reasons.append("Sellers dominating ABOVE")
        
        # 2. INST buying ABOVE (+15)
        if above_inst_buy > above_inst_sell * 1.2 and above_inst_buy > 1000:
            score += 15
            reasons.append("INST buying ABOVE")
        elif above_inst_sell > above_inst_buy * 1.5 and above_inst_sell > 5000:
            score -= 15
            reasons.append("INST selling ABOVE")
        
        # 3. Time spent ABOVE (+15 if significant)
        if above_time_pct >= 30:
            score += 15
            reasons.append("Significant time ABOVE ({}%)".format(int(above_time_pct)))
        elif above_time_pct >= 15:
            score += 8
            reasons.append("Some time ABOVE ({}%)".format(int(above_time_pct)))
        elif above_time_pct < 5:
            reasons.append("Little time ABOVE ({}%)".format(int(above_time_pct)))
        
        # 4. Volume trend in ABOVE: are late segments bigger than early?
        above_segments = [s for s in segments if s.get("zone") == "ABOVE"]
        if len(above_segments) >= 4:
            mid = len(above_segments) // 2
            early_vol = sum(float(s.get("total_vol") or 0) for s in above_segments[:mid])
            late_vol = sum(float(s.get("total_vol") or 0) for s in above_segments[mid:])
            if late_vol > early_vol * 1.3:
                score += 15
                reasons.append("Volume ABOVE growing")
            elif early_vol > late_vol * 1.3:
                score -= 10
                reasons.append("Volume ABOVE fading")
        
        # 5. Green segments in ABOVE getting bigger (momentum)
        above_green = [s for s in above_segments if float(s.get("delta") or 0) > 0]
        if len(above_green) >= 3:
            vols = [float(s.get("total_vol") or 0) for s in above_green]
            if len(vols) >= 3:
                first_half = sum(vols[:len(vols)//2]) / max(1, len(vols)//2)
                second_half = sum(vols[len(vols)//2:]) / max(1, len(vols) - len(vols)//2)
                if second_half > first_half * 1.3:
                    score += 10
                    reasons.append("Buyer momentum increasing")
                elif first_half > second_half * 1.3:
                    score -= 8
                    reasons.append("Buyer momentum fading")
        
        # 6. Position migration: check if segments trend from ON → ABOVE
        zone_sequence = [s.get("zone") for s in segments[-10:]]  # Last 10 segments
        above_count_late = sum(1 for z in zone_sequence[-5:] if z == "ABOVE")
        above_count_early = sum(1 for z in zone_sequence[:5] if z == "ABOVE")
        if above_count_late > above_count_early:
            score += 10
            reasons.append("Migrating into ABOVE")
        elif above_count_early > above_count_late + 2:
            score -= 8
            reasons.append("Retreating from ABOVE")
        
        # 7. Seller exhaustion: are red/pink segments ABOVE getting smaller?
        above_red = [s for s in above_segments if float(s.get("delta") or 0) < 0]
        if len(above_red) >= 3:
            vols = [float(s.get("total_vol") or 0) for s in above_red]
            first_half = sum(vols[:len(vols)//2]) / max(1, len(vols)//2)
            second_half = sum(vols[len(vols)//2:]) / max(1, len(vols) - len(vols)//2)
            if first_half > second_half * 1.5:
                score += 10
                reasons.append("Sellers exhausting ABOVE")
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "max": max_score,
            "pct": int(score),
            "reasons": reasons,
            "grade": "HIGH" if score >= 65 else "MEDIUM" if score >= 40 else "LOW",
        }

    def _detect_distribution_pattern(
        self,
        *,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Distribution Pattern (DP): Detect "sell-the-rip" sequences.
        
        Pattern: INST_SELL ABOVE → SELL ON → SELL BELOW
        This indicates smart money distribution during rallies.
        
        If triggered, should downgrade bullish signals.
        """
        triggered = False
        trigger_count = 0
        sequences: list[dict[str, Any]] = []
        
        # Look for sequences: INST selling ABOVE followed by price dropping
        for i in range(len(segments) - 2):
            seg1 = segments[i]
            seg2 = segments[i + 1]
            seg3 = segments[i + 2]
            
            # Pattern: INST_SELL in ABOVE → negative delta ON → negative delta BELOW
            inst_sell_above = (
                seg1.get("zone") == "ABOVE" and 
                float(seg1.get("inst_sell") or 0) > float(seg1.get("inst_buy") or 0) * 1.5 and
                float(seg1.get("inst_sell") or 0) > 3000
            )
            
            sell_on = (
                seg2.get("zone") == "ON" and 
                float(seg2.get("delta") or 0) < 0
            )
            
            sell_below = (
                seg3.get("zone") == "BELOW" and 
                float(seg3.get("delta") or 0) < 0
            )
            
            if inst_sell_above and sell_on:
                triggered = True
                trigger_count += 1
                sequences.append({
                    "start_ts_ms": seg1.get("start_ts_ms"),
                    "pattern": "INST_SELL_ABOVE → SELL_ON" + (" → SELL_BELOW" if sell_below else ""),
                    "inst_sell_vol": float(seg1.get("inst_sell") or 0),
                })
            
            # Also detect: big sell ABOVE → immediate drop to ON/BELOW
            big_sell_above = (
                seg1.get("zone") == "ABOVE" and
                float(seg1.get("delta") or 0) < -5000
            )
            drop_to_on_below = seg2.get("zone") in {"ON", "BELOW"}
            
            if big_sell_above and drop_to_on_below:
                triggered = True
                trigger_count += 1
                sequences.append({
                    "start_ts_ms": seg1.get("start_ts_ms"),
                    "pattern": "BIG_SELL_ABOVE → DROP",
                    "delta": float(seg1.get("delta") or 0),
                })
        
        severity = "NONE"
        if trigger_count >= 3:
            severity = "HIGH"
        elif trigger_count >= 2:
            severity = "MEDIUM"
        elif trigger_count >= 1:
            severity = "LOW"
        
        return {
            "triggered": triggered,
            "count": trigger_count,
            "severity": severity,
            "sequences": sequences[:5],  # Limit to 5 examples
        }

    def _compute_late_momentum(
        self,
        *,
        segments: list[dict[str, Any]],
        n_segments: int = 5,
    ) -> dict[str, Any]:
        """
        Late Momentum: Who's winning in the last N segments?
        
        This determines if the session is ENDING bullish or bearish,
        which is more important for predicting the next move.
        
        Bullish signals: INST buying (blue), green dots, ABOVE zone
        Bearish signals: INST selling (pink), red dots, dropping to BELOW
        """
        if not segments or len(segments) < 2:
            return {
                "score": 0,
                "direction": "NEUTRAL",
                "signals": [],
                "last_n": 0,
            }
        
        # Get last N segments
        last_n = segments[-n_segments:] if len(segments) >= n_segments else segments
        
        score = 0
        signals: list[str] = []
        
        for seg in last_n:
            zone = seg.get("zone", "ON")
            delta = float(seg.get("delta") or 0)
            inst_buy = float(seg.get("inst_buy") or 0)
            inst_sell = float(seg.get("inst_sell") or 0)
            total_vol = float(seg.get("total_vol") or 0)
            
            # INST buying
            if inst_buy > inst_sell * 1.3 and inst_buy > 1000:
                score += 2
                if "INST buying" not in signals:
                    signals.append("INST buying")
            # INST selling
            elif inst_sell > inst_buy * 1.3 and inst_sell > 1000:
                score -= 2
                if "INST selling" not in signals:
                    signals.append("INST selling")
            
            # Delta direction
            if delta > 0 and total_vol > 0 and (delta / total_vol) > 0.1:
                score += 1
                if "Buyers winning" not in signals:
                    signals.append("Buyers winning")
            elif delta < 0 and total_vol > 0 and (abs(delta) / total_vol) > 0.1:
                score -= 1
                if "Sellers winning" not in signals:
                    signals.append("Sellers winning")
            
            # Zone position
            if zone == "ABOVE":
                score += 1
            elif zone == "BELOW":
                score -= 1
        
        # Check for strong conviction (big volume in last segments)
        if last_n:
            last_seg = last_n[-1]
            last_delta = float(last_seg.get("delta") or 0)
            last_vol = float(last_seg.get("total_vol") or 0)
            all_vols = [float(s.get("total_vol") or 0) for s in segments]
            avg_vol = sum(all_vols) / len(all_vols) if all_vols else 0
            
            # Last segment is high volume buyer
            if last_vol > avg_vol * 1.5 and last_delta > 0:
                score += 2
                signals.append("Strong buyer finish")
            # Last segment is high volume seller
            elif last_vol > avg_vol * 1.5 and last_delta < 0:
                score -= 2
                signals.append("Strong seller finish")
        
        # Determine direction
        if score >= 3:
            direction = "BULLISH"
        elif score <= -3:
            direction = "BEARISH"
        elif score > 0:
            direction = "LEAN_BULLISH"
        elif score < 0:
            direction = "LEAN_BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            "score": score,
            "direction": direction,
            "signals": signals,
            "last_n": len(last_n),
        }

    def _compute_market_state(
        self,
        *,
        defense_score: dict[str, Any],
        aggression_score: dict[str, Any],
        distribution_pattern: dict[str, Any],
        zone_analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        late_momentum: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Market State Machine:
        
        States:
        - BREAKDOWN_RISK: DS low, BELOW expanding
        - DEFENSE_ONLY: DS high, AS low (support holding but capped)
        - RECLAIM_DEVELOPING: DS rising + early AS signals
        - ACCEPTANCE: DS high + AS medium, time ABOVE increasing
        - BREAK_LIKELY: DS high + AS high, resistance failing
        
        Each state maps to an action: WAIT, SCALP, STARTER, ADD, AVOID
        
        Late momentum can upgrade/downgrade the final action.
        """
        ds = int(defense_score.get("pct") or 0)
        as_ = int(aggression_score.get("pct") or 0)
        dp_triggered = distribution_pattern.get("triggered", False)
        dp_severity = distribution_pattern.get("severity", "NONE")
        
        # Accumulation pattern detection
        accumulation = defense_score.get("accumulation_detected", False)
        accumulation_count = defense_score.get("accumulation_count", 0)
        
        # Late momentum
        late_score = int(late_momentum.get("score") or 0)
        late_direction = late_momentum.get("direction", "NEUTRAL")
        late_signals = late_momentum.get("signals", [])
        
        zones = zone_analysis.get("zones", {}) if isinstance(zone_analysis, dict) else {}
        below_zone = zones.get("BELOW", {})
        below_delta = float(below_zone.get("delta_sum") or 0)
        
        state = "UNKNOWN"
        action = "WAIT"
        reason = ""
        confidence = 0
        
        # PRIORITY CHECK: Accumulation pattern overrides low DS
        # If INST is buying dips, that's a setup even if overall DS is low
        if accumulation and accumulation_count >= 2:
            state = "ACCUMULATION_SETUP"
            action = "WATCH"
            reason = "INST accumulating on dips ({} times) despite mixed signals (DS={}, AS={})".format(
                accumulation_count, ds, as_
            )
            confidence = 60
        elif accumulation:
            state = "DEFENSE_ACCUMULATING"
            action = "WATCH"
            reason = "INST buying dips detected, watch for follow-through (DS={}, AS={})".format(ds, as_)
            confidence = 55
        
        # State determination logic (only if not accumulation)
        elif ds < 40:
            if below_delta < -5000:
                state = "BREAKDOWN_RISK"
                action = "AVOID"
                reason = "Defense weak (DS={}), selling BELOW".format(ds)
                confidence = 80
            else:
                state = "WEAK_DEFENSE"
                action = "WAIT"
                reason = "Defense not established (DS={})".format(ds)
                confidence = 60
        
        elif ds >= 65 and as_ < 35:
            # Accumulation already handled at the top - this is pure defense
            state = "DEFENSE_ONLY"
            action = "WAIT"
            reason = "Support defended (DS={}) but buyers not pressing (AS={})".format(ds, as_)
            confidence = 70
        
        elif ds >= 50 and 35 <= as_ < 60:
            if dp_triggered and dp_severity in {"MEDIUM", "HIGH"}:
                state = "DISTRIBUTION"
                action = "WAIT"
                reason = "Distribution pattern detected, AS={} may be fake".format(as_)
                confidence = 65
            else:
                state = "RECLAIM_DEVELOPING"
                action = "STARTER"
                reason = "Defense solid (DS={}), aggression building (AS={})".format(ds, as_)
                confidence = 55
        
        elif ds >= 60 and 60 <= as_ < 75:
            if dp_triggered and dp_severity == "HIGH":
                state = "ACCEPTANCE_CAUTIOUS"
                action = "STARTER"
                reason = "AS={} but distribution present - reduced size".format(as_)
                confidence = 50
            else:
                state = "ACCEPTANCE"
                action = "STARTER"
                reason = "Solid defense (DS={}), good aggression (AS={})".format(ds, as_)
                confidence = 65
        
        elif ds >= 60 and as_ >= 75:
            if dp_triggered and dp_severity == "HIGH":
                state = "BREAK_LIKELY_CAUTIOUS"
                action = "ADD"
                reason = "Strong AS={} but watch for distribution".format(as_)
                confidence = 60
            else:
                state = "BREAK_LIKELY"
                action = "ADD"
                reason = "Defense locked (DS={}), buyers pressing hard (AS={})".format(ds, as_)
                confidence = 75
        
        else:
            state = "MIXED"
            action = "WAIT"
            reason = "Unclear setup: DS={}, AS={}".format(ds, as_)
            confidence = 40
        
        # Override for high distribution severity
        if dp_severity == "HIGH" and action in {"STARTER", "ADD"}:
            action = "WAIT"
            reason += " [OVERRIDE: Heavy distribution]"
            confidence = max(30, confidence - 20)
        
        # LATE MOMENTUM UPGRADE/DOWNGRADE
        # If action is WATCH and late momentum is bullish → upgrade to STARTER
        # If late momentum is bearish → downgrade confidence
        late_upgrade = False
        late_downgrade = False
        
        if late_direction in {"BULLISH", "LEAN_BULLISH"} and late_score >= 3:
            if action == "WATCH":
                action = "STARTER"
                late_upgrade = True
                reason += " [LATE MOMENTUM: {} → upgraded]".format(", ".join(late_signals[:2]))
                confidence = min(80, confidence + 10)
            elif action == "WAIT" and state not in {"BREAKDOWN_RISK"}:
                action = "WATCH"
                late_upgrade = True
                reason += " [LATE MOMENTUM: {}]".format(", ".join(late_signals[:2]))
                confidence = min(70, confidence + 5)
        
        elif late_direction in {"BEARISH", "LEAN_BEARISH"} and late_score <= -3:
            if action in {"STARTER", "ADD"}:
                action = "WATCH"
                late_downgrade = True
                reason += " [LATE MOMENTUM BEARISH: {} → downgraded]".format(", ".join(late_signals[:2]))
                confidence = max(30, confidence - 15)
            elif action == "WATCH":
                action = "WAIT"
                late_downgrade = True
                reason += " [LATE MOMENTUM BEARISH: {}]".format(", ".join(late_signals[:2]))
                confidence = max(30, confidence - 10)
        
        return {
            "state": state,
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "ds": ds,
            "as": as_,
            "dp_triggered": dp_triggered,
            "dp_severity": dp_severity,
            "late_momentum": {
                "score": late_score,
                "direction": late_direction,
                "signals": late_signals,
                "upgrade": late_upgrade,
                "downgrade": late_downgrade,
            },
        }

    def _summarize_level_band_window(
        self,
        *,
        df: pd.DataFrame,
        start_ms: int,
        end_ms: int,
        level_price: float | None,
        level_kind: str | None,
        band_pct: float,
        interval_s: int,
        indicator_col: str | None = None,  # NEW: for dynamic indicator levels
    ) -> dict[str, Any]:
        win = self._window_df(df, start_ms, end_ms)
        out: dict[str, Any] = {
            "start_ts_ms": int(start_ms),
            "end_ts_ms": int(end_ms),
            "points": int(len(win)),
            "duration_s": float(max(0, end_ms - start_ms) / 1000.0),
        }
        if win.empty:
            return out

        # Ensure deterministic ordering for first/last calculations.
        if "timestamp" in win.columns:
            win = win.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)

        def _sum(col: str) -> float:
            if col not in win.columns:
                return 0.0
            return float(pd.to_numeric(win[col], errors="coerce").fillna(0).sum())

        def _mean(col: str) -> float | None:
            if col not in win.columns:
                return None
            s = pd.to_numeric(win[col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else None

        def _first_last_min_max(col: str) -> tuple[float | None, float | None, float | None, float | None]:
            if col not in win.columns:
                return None, None, None, None
            s = pd.to_numeric(win[col], errors="coerce").dropna()
            if s.empty:
                return None, None, None, None
            return float(s.iloc[0]), float(s.iloc[-1]), float(s.min()), float(s.max())

        # Whole-window aggregates (snapshot-level truth)
        out["trade_count_sum"] = int(_sum("trade_count"))
        out["total_volume_sum"] = float(_sum("total_volume"))
        out["buy_volume_sum"] = float(_sum("buy_volume"))
        out["sell_volume_sum"] = float(_sum("sell_volume"))
        out["unknown_volume_sum"] = float(_sum("unknown_volume"))
        out["delta_sum"] = float(_sum("delta"))

        out["relative_aggression_mean"] = _mean("relative_aggression")
        out["pct_at_ask_mean"] = _mean("pct_at_ask")
        out["pct_at_bid_mean"] = _mean("pct_at_bid")
        out["spread_pct_mean"] = _mean("spread_pct")
        out["compression_index_mean"] = _mean("compression_index")
        out["ema_confluence_score_mean"] = _mean("ema_confluence_score")
        out["atr_value_mean"] = _mean("atr_value")

        p0, p1, pmin, pmax = _first_last_min_max("last_price")
        out["last_price_first"] = p0
        out["last_price_last"] = p1
        out["last_price_min"] = pmin
        out["last_price_max"] = pmax
        if p0 is not None and p1 is not None:
            out["return_abs"] = float(p1 - p0)
            out["return_pct"] = float((p1 - p0) / p0) if p0 != 0 else None
        else:
            out["return_abs"] = None
            out["return_pct"] = None

        # Ratio-like can explode when price doesn't move; keep both raw and stable transform summaries.
        ai = None
        if "absorption_index_10s" in win.columns:
            s = pd.to_numeric(win["absorption_index_10s"], errors="coerce").dropna()
            if not s.empty:
                ai = s
        if ai is not None:
            out["absorption_index_10s_mean"] = float(ai.mean())
            out["absorption_index_10s_mean_asinh"] = float(ai.apply(lambda v: asinh(float(v))).mean())

        # Level-band focus
        out["level_kind"] = level_kind
        out["band_pct"] = float(band_pct)
        
        # For dynamic indicator levels, use real-time indicator values
        if indicator_col and indicator_col in win.columns:
            ind_vals = pd.to_numeric(win[indicator_col], errors="coerce")
            # Filter out zeros (uninitialized EMAs)
            valid_ind = ind_vals > 0
            if not valid_ind.any():
                out["level_price"] = None
                return out
            out["level_price"] = float(ind_vals[valid_ind].mean())  # Mean for reporting only
            out["indicator_col"] = indicator_col
            # Dynamic band: calculate per-row based on indicator value at that moment
            lower = ind_vals * (1.0 - float(band_pct))
            upper = ind_vals * (1.0 + float(band_pct))
            out["band_lower"] = float(lower[valid_ind].mean()) if valid_ind.any() else None
            out["band_upper"] = float(upper[valid_ind].mean()) if valid_ind.any() else None
        elif level_price is not None:
            out["level_price"] = float(level_price)
            lower = float(level_price) * (1.0 - float(band_pct))
            upper = float(level_price) * (1.0 + float(band_pct))
            out["band_lower"] = lower
            out["band_upper"] = upper
        else:
            out["level_price"] = None
            return out
        
        if "last_price" not in win.columns:
            return out

        lp = pd.to_numeric(win["last_price"], errors="coerce")
        
        # Dynamic in-band check for indicators, static for fixed price
        if indicator_col and indicator_col in win.columns:
            ind_vals = pd.to_numeric(win[indicator_col], errors="coerce")
            lower = ind_vals * (1.0 - float(band_pct))
            upper = ind_vals * (1.0 + float(band_pct))
            in_band = (lp >= lower) & (lp <= upper) & (ind_vals > 0)
        else:
            lower_static = float(level_price) * (1.0 - float(band_pct))
            upper_static = float(level_price) * (1.0 + float(band_pct))
            in_band = (lp >= lower_static) & (lp <= upper_static)
        out["time_in_band_s"] = int(in_band.fillna(False).sum()) * int(interval_s)

        band = win[in_band.fillna(False)]
        out["band_points"] = int(len(band))
        if not band.empty:
            out["band_trade_count_sum"] = int(pd.to_numeric(band.get("trade_count", 0), errors="coerce").fillna(0).sum())
            out["band_total_volume_sum"] = float(pd.to_numeric(band.get("total_volume", 0), errors="coerce").fillna(0).sum())
            out["band_buy_volume_sum"] = float(pd.to_numeric(band.get("buy_volume", 0), errors="coerce").fillna(0).sum())
            out["band_sell_volume_sum"] = float(pd.to_numeric(band.get("sell_volume", 0), errors="coerce").fillna(0).sum())
            out["band_delta_sum"] = float(pd.to_numeric(band.get("delta", 0), errors="coerce").fillna(0).sum())
            out["band_relative_aggression_mean"] = (
                float(pd.to_numeric(band["relative_aggression"], errors="coerce").dropna().mean())
                if "relative_aggression" in band.columns and not pd.to_numeric(band["relative_aggression"], errors="coerce").dropna().empty
                else None
            )
            
            # Large trade metrics within band
            out["band_large_trade_count"] = int(pd.to_numeric(band.get("large_trade_count_10s", 0), errors="coerce").fillna(0).sum())
            out["band_large_buy_volume"] = float(pd.to_numeric(band.get("large_buy_volume_10s", 0), errors="coerce").fillna(0).sum())
            out["band_large_sell_volume"] = float(pd.to_numeric(band.get("large_sell_volume_10s", 0), errors="coerce").fillna(0).sum())
            out["band_large_delta"] = float(out["band_large_buy_volume"] - out["band_large_sell_volume"])
            if out["band_large_buy_volume"] + out["band_large_sell_volume"] > 0:
                out["band_large_buy_ratio"] = float(
                    out["band_large_buy_volume"] / (out["band_large_buy_volume"] + out["band_large_sell_volume"])
                )
            else:
                out["band_large_buy_ratio"] = None
            # Large trade rate (per minute in band)
            if out.get("duration_s") and out["duration_s"] > 0:
                out["band_large_trade_rate_per_min"] = float(
                    out["band_large_trade_count"] / (out["duration_s"] / 60.0)
                )
            else:
                out["band_large_trade_rate_per_min"] = None
            # Avg large trade size
            if out["band_large_trade_count"] > 0:
                total_large_vol = out["band_large_buy_volume"] + out["band_large_sell_volume"]
                out["band_avg_large_trade_size"] = float(total_large_vol / out["band_large_trade_count"])
            else:
                out["band_avg_large_trade_size"] = None

        # Touch count: outside -> inside transitions (deterministic)
        mask = in_band.fillna(False).astype(int)
        prev = mask.shift(1).fillna(0).astype(int)
        out["touch_count"] = int(((prev == 0) & (mask == 1)).sum())

        # Cross count: sign changes around the level (uses last_price vs level/indicator)
        if indicator_col and indicator_col in win.columns:
            ind_vals = pd.to_numeric(win[indicator_col], errors="coerce")
            diff = lp - ind_vals
        else:
            diff = lp - float(level_price)
        side = diff.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
        side = side.fillna(0).astype(int)
        prev_side = side.shift(1).fillna(0).astype(int)
        out["cross_count"] = int(((side != 0) & (prev_side != 0) & (side != prev_side)).sum())

        # Reclaim attempts: after breaking beyond the band, how often does it re-enter band?
        # Also count "holds" where it stays in-band for >= 30s (~3 snapshots at 10s cadence).
        reclaim_attempts = 0
        reclaim_holds_30s = 0
        hold_n = max(1, int(round(30 / max(1, interval_s))))

        # For dynamic indicators, lower/upper are Series; for fixed price, use scalars
        if indicator_col and indicator_col in win.columns:
            ind_vals = pd.to_numeric(win[indicator_col], errors="coerce")
            lower_vals = ind_vals * (1.0 - float(band_pct))
            upper_vals = ind_vals * (1.0 + float(band_pct))
        else:
            lower_vals = float(level_price) * (1.0 - float(band_pct))
            upper_vals = float(level_price) * (1.0 + float(band_pct))
        
        broken = None
        if level_kind == "support":
            broken = lp < lower_vals
        elif level_kind == "resistance":
            broken = lp > upper_vals
        if broken is not None:
            broken = broken.fillna(False)
            # An attempt is broken==True at t-1 and in_band==True at t
            # Keep boolean dtype and avoid pandas FutureWarning about downcasting on fillna.
            broken_b = broken.fillna(False).astype(bool)
            prev_broken = broken_b.shift(1, fill_value=False)
            cur_in_band = in_band.fillna(False).astype(bool)
            attempts = (prev_broken & cur_in_band).astype(bool)
            reclaim_attempts = int(attempts.sum())
            if reclaim_attempts > 0:
                # For each attempt index, check if next hold_n snapshots are all in-band
                idxs = list(attempts[attempts].index)
                for idx in idxs:
                    sl = in_band.loc[idx : idx + hold_n - 1]
                    if len(sl) >= hold_n and bool(sl.fillna(False).all()):
                        reclaim_holds_30s += 1

        out["reclaim_attempts"] = int(reclaim_attempts)
        out["reclaim_holds_30s"] = int(reclaim_holds_30s)
        out["reclaim_hold_rate"] = (float(reclaim_holds_30s) / float(reclaim_attempts)) if reclaim_attempts > 0 else None

        return out

    def _compute_interaction_timeline(
        self,
        *,
        df: pd.DataFrame,
        level_price: float,
        level_kind: str | None,
        band_pct: float,
        interval_s: int,
        indicator_col: str | None = None,  # NEW: for dynamic indicator levels
    ) -> list[dict[str, Any]]:
        """
        Build a timeline of discrete level interactions across the entire dataframe.
        Events: touch, break, reclaim, reject.
        For indicator-based levels, uses real-time indicator values for band calculations.
        
        IMPROVED: Uses high_10s/low_10s for touch detection instead of just close.
        - For SUPPORT: price touching from above means LOW wicked into band
        - For RESISTANCE: price touching from below means HIGH wicked into band
        """
        if df.empty or "last_price" not in df.columns or "timestamp" not in df.columns:
            return []

        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        lp = pd.to_numeric(df["last_price"], errors="coerce")
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        
        # Get high/low columns if available (for wick-based touch detection)
        has_hl = "high_10s" in df.columns and "low_10s" in df.columns
        if has_hl:
            high_col = pd.to_numeric(df["high_10s"], errors="coerce")
            low_col = pd.to_numeric(df["low_10s"], errors="coerce")
        else:
            high_col = lp  # Fallback to close
            low_col = lp

        # For dynamic indicators, compute band per-row
        if indicator_col and indicator_col in df.columns:
            ind_vals = pd.to_numeric(df[indicator_col], errors="coerce")
            lower = ind_vals * (1.0 - float(band_pct))
            upper = ind_vals * (1.0 + float(band_pct))
            
            # Position based on CLOSE: -1 = below, 0 = in band, +1 = above
            positions = pd.Series(0, index=df.index)
            # Touch detection using HIGH/LOW
            touched_band = pd.Series(False, index=df.index)
            touch_from_above = pd.Series(False, index=df.index)
            touch_from_below = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                p = lp.iloc[i]
                h = high_col.iloc[i] if has_hl else p
                l_price = low_col.iloc[i] if has_hl else p
                lb = lower.iloc[i]
                ub = upper.iloc[i]
                ind_v = ind_vals.iloc[i]
                
                if pd.isna(p) or pd.isna(lb) or pd.isna(ub) or ind_v <= 0:
                    positions.iloc[i] = 0
                elif p < lb:
                    positions.iloc[i] = -1
                elif p > ub:
                    positions.iloc[i] = 1
                else:
                    positions.iloc[i] = 0
                
                # Check if the wick touched the band (even if close didn't)
                if pd.notna(h) and pd.notna(l_price) and pd.notna(lb) and pd.notna(ub):
                    # Low wicked into band from above
                    if l_price <= ub and p > ub:
                        touched_band.iloc[i] = True
                        touch_from_above.iloc[i] = True
                    # High wicked into band from below
                    if h >= lb and p < lb:
                        touched_band.iloc[i] = True
                        touch_from_below.iloc[i] = True
        else:
            lower = float(level_price) * (1.0 - float(band_pct))
            upper = float(level_price) * (1.0 + float(band_pct))
            
            # Position relative to static band: -1 = below, 0 = in band, +1 = above
            def _pos(price: float) -> int:
                if price < lower:
                    return -1
                if price > upper:
                    return 1
                return 0
            positions = lp.apply(lambda p: _pos(float(p)) if pd.notna(p) else 0)
            
            # Wick-based touch detection for static bands
            touched_band = pd.Series(False, index=df.index)
            touch_from_above = pd.Series(False, index=df.index)
            touch_from_below = pd.Series(False, index=df.index)
            
            for i in range(len(df)):
                p = lp.iloc[i] if pd.notna(lp.iloc[i]) else 0
                h = high_col.iloc[i] if (has_hl and pd.notna(high_col.iloc[i])) else p
                l_price = low_col.iloc[i] if (has_hl and pd.notna(low_col.iloc[i])) else p
                
                # Low wicked into band from above (close is above band, but low touched it)
                if l_price <= upper and p > upper:
                    touched_band.iloc[i] = True
                    touch_from_above.iloc[i] = True
                # High wicked into band from below (close is below band, but high touched it)
                if h >= lower and p < lower:
                    touched_band.iloc[i] = True
                    touch_from_below.iloc[i] = True
                    
        prev_pos = positions.shift(1, fill_value=0)

        events: list[dict[str, Any]] = []

        for i in range(1, len(df)):
            cur = int(positions.iloc[i])
            prv = int(prev_pos.iloc[i])
            t = int(ts.iloc[i])
            price = float(lp.iloc[i]) if pd.notna(lp.iloc[i]) else None
            
            # Check for wick touches (candle wicked into band but closed outside)
            if touched_band.iloc[i]:
                ev: dict[str, Any] = {"ts_ms": t, "price": price}
                ev["event"] = "touch"
                ev["wick_touch"] = True  # Flag that this was a wick touch
                if touch_from_above.iloc[i]:
                    ev["from_side"] = "above"
                elif touch_from_below.iloc[i]:
                    ev["from_side"] = "below"
                events.append(ev)
                continue  # Don't double-count with position transition logic

            if prv != cur:
                # Transition detected
                ev: dict[str, Any] = {"ts_ms": t, "price": price}

                if prv != 0 and cur == 0:
                    # Entered band from outside → touch
                    ev["event"] = "touch"
                    ev["from_side"] = "above" if prv > 0 else "below"
                elif prv == 0 and cur != 0:
                    # Left band → break or reject
                    # If support: leaving below is break, leaving above is reject (bounce)
                    # If resistance: leaving above is break, leaving below is reject
                    if level_kind == "support":
                        ev["event"] = "break" if cur < 0 else "reject"
                    elif level_kind == "resistance":
                        ev["event"] = "break" if cur > 0 else "reject"
                    else:
                        ev["event"] = "exit"
                elif prv != 0 and cur != 0 and prv != cur:
                    # Crossed THROUGH the band without dwelling (wick/fast cross)
                    # This IS a touch - price interacted with the level
                    ev["event"] = "touch"
                    ev["from_side"] = "above" if prv > 0 else "below"
                    ev["cross_through"] = True  # Flag that it passed through quickly
                    ev["to_side"] = "above" if cur > 0 else "below"
                else:
                    continue

                events.append(ev)

        # Detect reclaims: break followed by touch followed by reject on opposite side
        # Simplified: after a break, if we see a touch again, it's a reclaim attempt
        annotated: list[dict[str, Any]] = []
        last_break_side = None
        for ev in events:
            ev_copy = dict(ev)
            if ev.get("event") == "break":
                last_break_side = ev.get("to_side")
            elif ev.get("event") == "touch" and last_break_side is not None:
                ev_copy["reclaim_attempt"] = True
                ev_copy["reclaiming_from"] = last_break_side
            annotated.append(ev_copy)

        return annotated

    def _compute_touch_packets(
        self,
        *,
        df: pd.DataFrame,
        level_price: float,
        level_kind: str | None,
        band_pct: float,
        interval_s: int,
        timeline: list[dict[str, Any]],
        indicator_col: str | None = None,  # NEW: for dynamic indicator levels
    ) -> list[dict[str, Any]]:
        """
        Compute Touch Packets per Addendum K.2.
        One packet per touch event in the interaction timeline.
        For indicator-based levels, uses real-time indicator values for band calculations.
        """
        if df.empty or not timeline:
            return []

        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        ts_col = pd.to_numeric(df["timestamp"], errors="coerce")
        lp_col = pd.to_numeric(df["last_price"], errors="coerce") if "last_price" in df.columns else pd.Series([0.0]*len(df))

        # For dynamic indicators, we'll compute band per-row later
        # For static price, use fixed bands
        if indicator_col and indicator_col in df.columns:
            ind_col = pd.to_numeric(df[indicator_col], errors="coerce")
            # Use mean of valid values for reporting, but actual logic uses per-row
            valid_ind = ind_col > 0
            mean_level = float(ind_col[valid_ind].mean()) if valid_ind.any() else float(level_price)
        else:
            ind_col = None
            mean_level = float(level_price)
        
        lower = mean_level * (1.0 - float(band_pct))
        upper = mean_level * (1.0 + float(band_pct))

        # Compute session baseline for z-scores
        baseline_vol_mean = float(df["total_volume"].mean()) if "total_volume" in df.columns else 1.0
        baseline_vol_std = float(df["total_volume"].std()) if "total_volume" in df.columns else 1.0
        baseline_spread = float(df["spread_pct"].mean()) if "spread_pct" in df.columns else 0.0

        def _get_col(col: str, idx: int) -> float | None:
            if col not in df.columns or idx < 0 or idx >= len(df):
                return None
            v = df.iloc[idx][col]
            try:
                return float(v) if pd.notna(v) else None
            except Exception:
                return None

        def _window_sum(col: str, start_idx: int, n: int) -> float:
            if col not in df.columns:
                return 0.0
            end_idx = min(start_idx + n, len(df))
            if start_idx >= end_idx:
                return 0.0
            return float(pd.to_numeric(df.iloc[start_idx:end_idx][col], errors="coerce").fillna(0).sum())

        def _window_mean(col: str, start_idx: int, n: int) -> float | None:
            if col not in df.columns:
                return None
            end_idx = min(start_idx + n, len(df))
            if start_idx >= end_idx:
                return None
            s = pd.to_numeric(df.iloc[start_idx:end_idx][col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else None

        def _find_idx(ts_ms: int) -> int:
            """Find index of first row with timestamp >= ts_ms."""
            mask = ts_col >= ts_ms
            if not mask.any():
                return len(df)
            return int(mask.idxmax())

        n_30 = max(1, int(round(30 / max(1, interval_s))))
        n_60 = max(1, int(round(60 / max(1, interval_s))))

        touch_events = [e for e in timeline if e.get("event") == "touch"]
        packets: list[dict[str, Any]] = []

        for touch_num, ev in enumerate(touch_events, 1):
            t_raw = int(ev.get("ts_ms", 0))
            if t_raw <= 0:
                continue
            
            # Round down to the start of the minute candle containing this touch
            # This ensures we analyze the FULL minute, not just from the touch moment
            # e.g., touch at 12:30:30 → analyze from 12:30:00
            MINUTE_MS = 60_000
            t = (t_raw // MINUTE_MS) * MINUTE_MS  # Floor to minute start

            # idx_minute = index at minute start (for full-candle analysis)
            # idx_touch = index at actual touch time (for volume metrics)
            idx_minute = _find_idx(t)
            idx_touch = _find_idx(t_raw)
            
            if idx_touch >= len(df):
                continue
            
            # Use idx_touch for most metrics (actual touch time)
            idx = idx_touch

            # For indicator-based levels, get the indicator value at this touch moment
            if ind_col is not None and idx < len(ind_col):
                touch_level_price = float(ind_col.iloc[idx]) if ind_col.iloc[idx] > 0 else mean_level
            else:
                touch_level_price = float(level_price)
            
            pkt: dict[str, Any] = {
                "touch_id": f"touch:{touch_level_price:.4f}:{t_raw}",
                "level_price": touch_level_price,  # Real-time indicator value at touch
                "level_kind": level_kind,
                "touch_ts_ms": t_raw,  # Original touch timestamp
                "minute_start_ms": t,  # Minute-aligned start for full candle analysis
                "touch_number": touch_num,
                "from_side": ev.get("from_side"),
                "reclaim_attempt": ev.get("reclaim_attempt", False),
            }
            if indicator_col:
                pkt["indicator_col"] = indicator_col
                pkt["indicator_value_at_touch"] = touch_level_price

            # Time since last touch
            if touch_num > 1 and touch_events[touch_num - 2].get("ts_ms"):
                pkt["time_since_last_touch_s"] = float(t - int(touch_events[touch_num - 2]["ts_ms"])) / 1000.0
            else:
                pkt["time_since_last_touch_s"] = None

            # Approach (pre-touch: -60s to touch)
            pre_idx = max(0, idx - n_60)
            pkt["approach_delta_60s"] = _window_sum("delta", pre_idx, n_60)
            pkt["approach_rel_aggr_60s"] = _window_mean("relative_aggression", pre_idx, n_60)

            # Approach velocity
            atr_mean = _window_mean("atr_value", pre_idx, n_60)
            price_before = _get_col("last_price", pre_idx)
            price_at = _get_col("last_price", idx)
            if price_before is not None and price_at is not None and atr_mean and atr_mean > 0:
                ret_abs = abs(price_at - price_before)
                v = ret_abs / atr_mean / 1.0  # per minute (60s window)
                pkt["approach_velocity_atr_per_min"] = float(v)
                if v >= 1.5:
                    pkt["approach_type"] = "crash"
                elif v >= 0.75:
                    pkt["approach_type"] = "fast"
                elif v >= 0.25:
                    pkt["approach_type"] = "normal"
                else:
                    pkt["approach_type"] = "grind"
            else:
                pkt["approach_velocity_atr_per_min"] = None
                pkt["approach_type"] = None

            # Reaction flow (post-touch: 0-30s, 0-60s)
            pkt["band_delta_0_30s"] = _window_sum("delta", idx, n_30)
            pkt["band_delta_0_60s"] = _window_sum("delta", idx, n_60)
            pkt["rel_aggr_0_30s"] = _window_mean("relative_aggression", idx, n_30)
            pkt["rel_aggr_0_60s"] = _window_mean("relative_aggression", idx, n_60)

            # Delta flip flag
            pre_delta = pkt["approach_delta_60s"]
            post_delta = pkt["band_delta_0_30s"]
            if pre_delta is not None and post_delta is not None:
                pkt["delta_flip_flag"] = (pre_delta < 0 and post_delta > 0) or (pre_delta > 0 and post_delta < 0)
            else:
                pkt["delta_flip_flag"] = None

            # Dwell + Bounce
            # Find how long price stayed in-band after touch
            dwell_count = 0
            for j in range(idx, min(idx + n_60 * 2, len(df))):
                p = _get_col("last_price", j)
                if p is not None and lower <= p <= upper:
                    dwell_count += 1
                else:
                    break
            pkt["touch_dwell_s"] = float(dwell_count * interval_s)

            # Bounce return
            p_at = _get_col("last_price", idx)
            p_30 = _get_col("last_price", min(idx + n_30, len(df) - 1))
            p_60 = _get_col("last_price", min(idx + n_60, len(df) - 1))
            if p_at and p_30:
                pkt["bounce_return_30s_pct"] = float((p_30 - p_at) / p_at) if p_at != 0 else None
            else:
                pkt["bounce_return_30s_pct"] = None
            if p_at and p_60:
                pkt["bounce_return_60s_pct"] = float((p_60 - p_at) / p_at) if p_at != 0 else None
            else:
                pkt["bounce_return_60s_pct"] = None

            # Penetration / wick
            min_price_60s = None
            for j in range(idx, min(idx + n_60, len(df))):
                p = _get_col("last_price", j)
                if p is not None:
                    if min_price_60s is None or p < min_price_60s:
                        min_price_60s = p
            if min_price_60s is not None and level_kind == "support":
                penetration = lower - min_price_60s
                pkt["max_penetration_pct"] = float(penetration / level_price) if level_price != 0 else None
                # Wick recovered: penetrated below but price at +30s is back in/above band
                pkt["wick_recovered_flag"] = (penetration > 0) and (p_30 is not None and p_30 >= lower)
            else:
                pkt["max_penetration_pct"] = None
                pkt["wick_recovered_flag"] = None

            # Volume abnormality
            vol_at = _get_col("total_volume", idx)
            if vol_at is not None and baseline_vol_std > 0:
                pkt["touch_volume_z"] = float((vol_at - baseline_vol_mean) / (baseline_vol_std + 1e-9))
            else:
                pkt["touch_volume_z"] = None

            # Large trade metrics (from session_stream)
            pkt["large_trade_count_at_touch"] = _get_col("large_trade_count_10s", idx)
            pkt["large_trade_buy_ratio_at_touch"] = _get_col("large_trade_buy_ratio", idx)
            pkt["large_buy_volume_at_touch"] = _get_col("large_buy_volume_10s", idx)
            pkt["large_sell_volume_at_touch"] = _get_col("large_sell_volume_10s", idx)
            pkt["large_trade_delta_at_touch"] = _get_col("large_trade_delta", idx)
            pkt["large_trade_threshold"] = _get_col("large_trade_threshold_size", idx)
            
            # Large trade aggregates over 30s window after touch
            pkt["large_trade_count_30s"] = _window_sum("large_trade_count_10s", idx, n_30)
            pkt["large_buy_volume_30s"] = _window_sum("large_buy_volume_10s", idx, n_30)
            pkt["large_sell_volume_30s"] = _window_sum("large_sell_volume_10s", idx, n_30)
            large_buy_30 = pkt["large_buy_volume_30s"] or 0
            large_sell_30 = pkt["large_sell_volume_30s"] or 0
            if large_buy_30 + large_sell_30 > 0:
                pkt["large_trade_buy_ratio_30s"] = float(large_buy_30 / (large_buy_30 + large_sell_30))
            else:
                pkt["large_trade_buy_ratio_30s"] = None
            pkt["large_trade_delta_30s"] = float(large_buy_30 - large_sell_30)
            
            # Large trade rate (trades per minute in 30s window)
            if n_30 > 0:
                window_minutes = (n_30 * interval_s) / 60.0
                pkt["large_trade_rate_30s"] = float((pkt["large_trade_count_30s"] or 0) / max(0.001, window_minutes))
            else:
                pkt["large_trade_rate_30s"] = None
            
            # Compare to baseline (pre-touch 60s)
            baseline_large_count = _window_sum("large_trade_count_10s", pre_idx, n_60) or 0
            baseline_minutes = (n_60 * interval_s) / 60.0
            baseline_large_rate = baseline_large_count / max(0.001, baseline_minutes)
            if baseline_large_rate > 0 and pkt["large_trade_rate_30s"]:
                pkt["large_vs_baseline_ratio"] = float(pkt["large_trade_rate_30s"] / baseline_large_rate)
            else:
                pkt["large_vs_baseline_ratio"] = None
            
            # Buy vs sell trade size comparison
            buy_vol_30 = _window_sum("buy_volume", idx, n_30) or 0
            sell_vol_30 = _window_sum("sell_volume", idx, n_30) or 0
            # Approximate avg trade sizes (using total volume / trade count isn't available per side, 
            # so we use large trade volume / large trade count as proxy)
            if large_buy_30 > 0 and large_sell_30 > 0:
                # Ratio of large buy volume to large sell volume
                pkt["large_buy_vs_sell_ratio"] = float(large_buy_30 / large_sell_30)
            else:
                pkt["large_buy_vs_sell_ratio"] = None
            
            # Institutional flow flags
            lt_count_30 = pkt["large_trade_count_30s"] or 0
            lt_buy_ratio_30 = pkt["large_trade_buy_ratio_30s"]
            pkt["institutional_buying_flag"] = (
                lt_buy_ratio_30 is not None and lt_buy_ratio_30 > 0.6 and lt_count_30 >= 5
            )
            pkt["institutional_selling_flag"] = (
                lt_buy_ratio_30 is not None and lt_buy_ratio_30 < 0.4 and lt_count_30 >= 5
            )
            pkt["large_size_imbalance_flag"] = (
                pkt["large_buy_vs_sell_ratio"] is not None and pkt["large_buy_vs_sell_ratio"] > 1.5
            )

            # Absorption
            pkt["absorption_mean_0_30s"] = _window_mean("absorption_index_10s", idx, n_30)
            vol_30 = _window_sum("total_volume", idx, n_30)
            ret_30 = pkt.get("bounce_return_30s_pct")
            if ret_30 is not None and vol_30 > 0 and p_at:
                pkt["price_efficiency_0_30s"] = float(abs(ret_30 * p_at) / (vol_30 + 1e-9))
            else:
                pkt["price_efficiency_0_30s"] = None

            # Divergence
            pkt["cvd_60s_at_touch"] = _get_col("cvd_60s", idx)
            cvd_pre = _get_col("cvd_60s", pre_idx)
            cvd_at = _get_col("cvd_60s", idx)
            if cvd_pre is not None and cvd_at is not None:
                pkt["cvd_slope_into_touch"] = float(cvd_at - cvd_pre)
            else:
                pkt["cvd_slope_into_touch"] = None
            # div_flag: price new low but delta improving (simplified)
            pkt["div_flag_at_touch"] = (
                pkt.get("approach_delta_60s", 0) < 0 and
                pkt.get("band_delta_0_30s", 0) > 0
            ) if pkt.get("approach_delta_60s") is not None else None

            # Spread behavior
            pkt["spread_pct_at_touch"] = _get_col("spread_pct", idx)
            spread_pre = _window_mean("spread_pct", pre_idx, n_30)
            spread_post = _window_mean("spread_pct", idx, n_30)
            if spread_pre is not None and baseline_spread > 0:
                pkt["spread_widening_into_touch"] = spread_pre > baseline_spread * 1.1
            else:
                pkt["spread_widening_into_touch"] = None
            if spread_post is not None and pkt.get("spread_pct_at_touch") is not None:
                pkt["spread_narrowing_after_touch"] = spread_post < float(pkt["spread_pct_at_touch"]) * 0.9
            else:
                pkt["spread_narrowing_after_touch"] = None

            # Context
            pkt["compression_at_touch"] = _window_mean("compression_index", pre_idx, n_60)
            comp_early = _window_mean("compression_index", pre_idx, n_30)
            comp_late = _window_mean("compression_index", max(0, idx - n_30), n_30)
            if comp_early is not None and comp_late is not None:
                d = comp_late - comp_early
                if d > 0.05:
                    pkt["compression_trend_into_touch"] = "tightening"
                elif d < -0.05:
                    pkt["compression_trend_into_touch"] = "loosening"
                else:
                    pkt["compression_trend_into_touch"] = "flat"
            else:
                pkt["compression_trend_into_touch"] = None

            # VWAP + EMA context
            vwap = _get_col("vwap_session", idx)
            if p_at and vwap:
                if p_at > vwap * 1.001:
                    pkt["price_vs_vwap_at_touch"] = "above"
                elif p_at < vwap * 0.999:
                    pkt["price_vs_vwap_at_touch"] = "below"
                else:
                    pkt["price_vs_vwap_at_touch"] = "at"
            else:
                pkt["price_vs_vwap_at_touch"] = None

            stack = _get_col("stack_state", idx) if "stack_state" in df.columns else None
            pkt["ema_stack_at_touch"] = str(stack) if stack else None

            # Outcome: look at subsequent events for this level
            # Find next events after this touch
            subsequent = [e for e in timeline if int(e.get("ts_ms", 0)) > t]
            next_break = next((e for e in subsequent if e.get("event") == "break"), None)
            next_reject = next((e for e in subsequent if e.get("event") == "reject"), None)

            # break_confirmed_30s: exited below and stayed 30s
            if next_break:
                break_t = int(next_break.get("ts_ms", 0))
                # Check if still broken after 30s
                break_idx = _find_idx(break_t)
                still_broken = True
                for j in range(break_idx, min(break_idx + n_30, len(df))):
                    p = _get_col("last_price", j)
                    if p is not None and p >= lower:
                        still_broken = False
                        break
                pkt["break_confirmed_30s"] = still_broken and (break_t - t <= 120_000)
            else:
                pkt["break_confirmed_30s"] = False

            # reclaim_after_break + reclaim_hold_30s
            if pkt["break_confirmed_30s"]:
                # Look for reclaim touch after break
                reclaim_touch = next((e for e in subsequent if e.get("event") == "touch" and int(e.get("ts_ms", 0)) > int(next_break.get("ts_ms", 0))), None)
                pkt["reclaim_after_break"] = reclaim_touch is not None
                if reclaim_touch:
                    reclaim_t = int(reclaim_touch.get("ts_ms", 0))
                    reclaim_idx = _find_idx(reclaim_t)
                    held = True
                    for j in range(reclaim_idx, min(reclaim_idx + n_30, len(df))):
                        p = _get_col("last_price", j)
                        if p is not None and p < lower:
                            held = False
                            break
                    pkt["reclaim_hold_30s"] = held
                else:
                    pkt["reclaim_hold_30s"] = None
            else:
                pkt["reclaim_after_break"] = None
                pkt["reclaim_hold_30s"] = None

            # touch_outcome
            if not pkt.get("break_confirmed_30s"):
                pkt["touch_outcome"] = "hold"
            elif pkt.get("reclaim_after_break") and pkt.get("reclaim_hold_30s"):
                pkt["touch_outcome"] = "break_reclaim_hold"
            elif pkt.get("reclaim_after_break"):
                pkt["touch_outcome"] = "break_reclaim_fail"
            else:
                pkt["touch_outcome"] = "break"

            packets.append(pkt)

        return packets

    def _compute_reclaim_packets(
        self,
        *,
        df: pd.DataFrame,
        levels: list[dict[str, Any]],
        band_pct: float,
        interval_s: int,
    ) -> list[dict[str, Any]]:
        """
        Compute Reclaim Packets per Addendum K.3.
        Emitted when price reclaims a prior support after tagging a deeper one.
        """
        if df.empty or len(levels) < 2:
            return []

        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        ts_col = pd.to_numeric(df["timestamp"], errors="coerce")
        lp_col = pd.to_numeric(df["last_price"], errors="coerce") if "last_price" in df.columns else pd.Series([0.0]*len(df))

        n_30 = max(1, int(round(30 / max(1, interval_s))))
        n_60 = max(1, int(round(60 / max(1, interval_s))))

        def _find_idx(ts_ms: int) -> int:
            mask = ts_col >= ts_ms
            if not mask.any():
                return len(df)
            return int(mask.idxmax())

        def _window_sum(col: str, start_idx: int, n: int) -> float:
            if col not in df.columns:
                return 0.0
            end_idx = min(start_idx + n, len(df))
            if start_idx >= end_idx:
                return 0.0
            return float(pd.to_numeric(df.iloc[start_idx:end_idx][col], errors="coerce").fillna(0).sum())

        def _window_mean(col: str, start_idx: int, n: int) -> float | None:
            if col not in df.columns:
                return None
            end_idx = min(start_idx + n, len(df))
            if start_idx >= end_idx:
                return None
            s = pd.to_numeric(df.iloc[start_idx:end_idx][col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else None

        def _get_col(col: str, idx: int) -> float | None:
            if col not in df.columns or idx < 0 or idx >= len(df):
                return None
            v = df.iloc[idx][col]
            try:
                return float(v) if pd.notna(v) else None
            except Exception:
                return None

        packets: list[dict[str, Any]] = []

        # Sort levels by price (descending for supports)
        sorted_levels = sorted(levels, key=lambda x: float(x.get("level_price", 0)), reverse=True)

        for i in range(len(sorted_levels) - 1):
            prior = sorted_levels[i]
            deeper = sorted_levels[i + 1]

            prior_price = float(prior.get("level_price", 0))
            deeper_price = float(deeper.get("level_price", 0))

            if prior_price <= 0 or deeper_price <= 0:
                continue

            prior_lower = prior_price * (1.0 - band_pct)
            deeper_lower = deeper_price * (1.0 - band_pct)
            deeper_upper = deeper_price * (1.0 + band_pct)

            # Find first touch of deeper level
            t_tag_deeper = None
            for j in range(len(df)):
                p = float(lp_col.iloc[j]) if pd.notna(lp_col.iloc[j]) else None
                if p is not None and deeper_lower <= p <= deeper_upper:
                    t_tag_deeper = int(ts_col.iloc[j])
                    break

            if t_tag_deeper is None:
                continue

            # Find reclaim of prior level (price enters prior band from below after tagging deeper)
            t_reclaim = None
            was_below = False
            for j in range(_find_idx(t_tag_deeper), len(df)):
                p = float(lp_col.iloc[j]) if pd.notna(lp_col.iloc[j]) else None
                if p is None:
                    continue
                if p < prior_lower:
                    was_below = True
                elif was_below and p >= prior_lower:
                    t_reclaim = int(ts_col.iloc[j])
                    break

            if t_reclaim is None:
                continue

            reclaim_idx = _find_idx(t_reclaim)

            pkt: dict[str, Any] = {
                "reclaim_id": f"reclaim:{prior_price}:{deeper_price}:{t_reclaim}",
                "reclaimed_level_price": prior_price,
                "tagged_deeper_level_price": deeper_price,
                "t_tag_deeper_ms": t_tag_deeper,
                "t_reclaim_ms": t_reclaim,
            }

            # Speed
            pkt["time_to_reclaim_ms"] = t_reclaim - t_tag_deeper
            pkt["time_to_reclaim_s"] = float(pkt["time_to_reclaim_ms"]) / 1000.0
            pkt["snapback_flag"] = pkt["time_to_reclaim_ms"] <= 120_000

            # Quality
            pkt["reclaim_band_delta_30s"] = _window_sum("delta", reclaim_idx, n_30)
            pkt["reclaim_band_delta_60s"] = _window_sum("delta", reclaim_idx, n_60)
            pkt["reclaim_rel_aggr_30s"] = _window_mean("relative_aggression", reclaim_idx, n_30)
            pkt["reclaim_rel_aggr_60s"] = _window_mean("relative_aggression", reclaim_idx, n_60)

            # Hold check
            held_30 = True
            held_60 = True
            for j in range(reclaim_idx, min(reclaim_idx + n_30, len(df))):
                p = _get_col("last_price", j)
                if p is not None and p < prior_lower:
                    held_30 = False
                    break
            for j in range(reclaim_idx, min(reclaim_idx + n_60, len(df))):
                p = _get_col("last_price", j)
                if p is not None and p < prior_lower:
                    held_60 = False
                    break
            pkt["reclaim_hold_30s"] = held_30
            pkt["reclaim_hold_60s"] = held_60

            # Overextension depth
            pkt["drop_pct_to_deeper"] = float((deeper_price - prior_price) / prior_price) if prior_price != 0 else None
            atr_mean = _window_mean("atr_value", _find_idx(t_tag_deeper), n_60)
            if atr_mean and atr_mean > 0:
                pkt["drop_atr_to_deeper"] = abs(deeper_price - prior_price) / atr_mean
            else:
                pkt["drop_atr_to_deeper"] = None

            # Flush flag
            dur_min = max(1e-9, float(pkt["time_to_reclaim_ms"]) / 60000.0)
            if pkt.get("drop_atr_to_deeper") is not None:
                v = pkt["drop_atr_to_deeper"] / dur_min
                pkt["flush_flag"] = pkt["drop_atr_to_deeper"] >= 1.0 and v >= 0.75
            else:
                pkt["flush_flag"] = None

            packets.append(pkt)

        return packets

    def _compute_main_support_score(
        self,
        *,
        levels: list[dict[str, Any]],
        reclaim_packets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compute Main Support Score per Addendum K.5.
        Returns scores per level and identifies main_support_id + focus_level.
        """
        scores: dict[float, int] = {}

        for lv in levels:
            price = float(lv.get("level_price", 0))
            if price <= 0:
                continue

            score = 0
            tl = lv.get("timeline_summary") or {}

            # +1 if session_rejects - session_breaks >= 2
            rejects = int(tl.get("rejects", 0) or 0)
            breaks = int(tl.get("breaks", 0) or 0)
            if rejects - breaks >= 2:
                score += 1

            # -2 if role_flip == support_to_resistance
            if lv.get("role_flip") == "support_to_resistance":
                score -= 2

            # -1 per session_breaks
            score -= breaks

            # Check reclaim packets for this level
            for rp in reclaim_packets:
                if abs(float(rp.get("reclaimed_level_price", 0)) - price) < 0.0001:
                    # +2 if reclaim_hold_30s
                    if rp.get("reclaim_hold_30s"):
                        score += 2
                    # +1 if reclaim_band_delta_30s > 0 AND reclaim_rel_aggr_30s > 0.10
                    bd = rp.get("reclaim_band_delta_30s", 0) or 0
                    ra = rp.get("reclaim_rel_aggr_30s", 0) or 0
                    if bd > 0 and ra > 0.10:
                        score += 1
                    # +1 if time_to_reclaim_ms <= 120_000
                    if (rp.get("time_to_reclaim_ms") or 999999) <= 120_000:
                        score += 1

            scores[price] = score

        if not scores:
            return {"scores": {}, "main_support_id": None, "main_support_score": None, "focus_level": None}

        main_price = max(scores, key=lambda p: scores[p])
        main_score = scores[main_price]

        # Focus level: the one that got reclaimed fast after deeper tag
        focus_level = None
        for rp in reclaim_packets:
            if rp.get("snapback_flag") and rp.get("reclaim_hold_30s"):
                focus_level = rp.get("reclaimed_level_price")
                break

        return {
            "scores": {str(p): s for p, s in scores.items()},
            "main_support_id": str(main_price),
            "main_support_score": main_score,
            "focus_level": str(focus_level) if focus_level else None,
        }

    @staticmethod
    def _scenario_key(*, level_kind: str | None, level_outcome: str | None) -> str:
        """
        Deterministic scenario key for grouping comparisons:
          side ∈ {long, short, unknown} derived from level_kind (support→long, resistance→short)
          outcome ∈ {win, loss, pending}
        """
        lk = (level_kind or "").strip().lower()
        if lk == "support":
            side = "long"
        elif lk == "resistance":
            side = "short"
        else:
            side = "unknown"

        oc = (level_outcome or "").strip().lower()
        if oc not in {"win", "loss"}:
            oc = "pending"

        return f"{side}_{oc}"

    @staticmethod
    def _positive_touch_threshold(*, side: str) -> float:
        """
        Threshold used to define a "positive" touch for the purpose of cross-session comparisons.
        For long: bounce_return_30s_pct >= +0.5%
        For short: bounce_return_30s_pct <= -0.5%
        """
        if side == "short":
            return -0.005
        return 0.005

    def _compute_positive_touch_ledger(
        self,
        *,
        chain_levels: list[dict[str, Any]],
        chain_id: str,
    ) -> dict[str, Any]:
        """
        Build a deterministic "positives" ledger for this chain:
        - Extract positive touches per scenario_key (long/short × win/loss)
        - Compute per-flag counts and top combos
        """
        from collections import Counter, defaultdict

        def _flags(tp: dict[str, Any]) -> dict[str, bool]:
            return {
                "pos_aggr": isinstance(tp.get("rel_aggr_0_30s"), (int, float)) and float(tp["rel_aggr_0_30s"]) > 0,
                "pos_delta": isinstance(tp.get("band_delta_0_30s"), (int, float)) and float(tp["band_delta_0_30s"]) > 0,
                "high_compression": isinstance(tp.get("compression_at_touch"), (int, float)) and float(tp["compression_at_touch"]) >= 1.0,
                "delta_flip": tp.get("delta_flip_flag") is True,
                "div": tp.get("div_flag_at_touch") is True,
                "controlled_approach": (tp.get("approach_type") in {"grind", "normal"}),
                "from_above": (tp.get("from_side") == "above"),
            }

        scenarios: dict[str, dict[str, Any]] = {}
        all_positive: list[dict[str, Any]] = []

        for lv in chain_levels:
            lk = lv.get("level_kind")
            lo = lv.get("level_outcome")
            scenario = self._scenario_key(level_kind=lk, level_outcome=lo)
            side = scenario.split("_", 1)[0]

            thr = self._positive_touch_threshold(side=side)
            tps = lv.get("touch_packets") or []

            pos_tps: list[dict[str, Any]] = []
            for tp in tps:
                b = tp.get("bounce_return_30s_pct")
                if not isinstance(b, (int, float)):
                    continue
                # long: b >= +thr ; short: b <= thr (thr negative)
                ok = (b >= thr) if side != "short" else (b <= thr)
                if not ok:
                    continue

                fl = _flags(tp)
                entry = {
                    "scenario_key": scenario,
                    "chain_id": chain_id,
                    "level_kind": lk,
                    "level_price": lv.get("level_price"),
                    "level_outcome": lo,
                    "touch_id": tp.get("touch_id"),
                    "touch_ts_ms": tp.get("touch_ts_ms"),
                    "bounce_return_30s_pct": b,
                    "band_delta_0_30s": tp.get("band_delta_0_30s"),
                    "rel_aggr_0_30s": tp.get("rel_aggr_0_30s"),
                    "compression_at_touch": tp.get("compression_at_touch"),
                    "approach_type": tp.get("approach_type"),
                    "from_side": tp.get("from_side"),
                    "delta_flip_flag": tp.get("delta_flip_flag"),
                    "div_flag_at_touch": tp.get("div_flag_at_touch"),
                    "flags": fl,
                }
                pos_tps.append(entry)
                all_positive.append(entry)

            if scenario not in scenarios:
                scenarios[scenario] = {
                    "scenario_key": scenario,
                    "levels": [],
                    "positive_touches": [],
                }

            scenarios[scenario]["levels"].append(
                {
                    "level_price": lv.get("level_price"),
                    "level_kind": lk,
                    "level_outcome": lo,
                    "positive_touch_count": len(pos_tps),
                }
            )
            scenarios[scenario]["positive_touches"].extend(pos_tps)

        # Summaries per scenario
        for s in scenarios.values():
            pts = s.get("positive_touches") or []
            s["positive_touch_count"] = len(pts)
            flag_counts = Counter()
            combo_counts = Counter()
            for e in pts:
                fl = e.get("flags") or {}
                for k, v in fl.items():
                    if v:
                        flag_counts[k] += 1
                combo = "+".join(sorted([k for k, v in fl.items() if v]))
                if combo:
                    combo_counts[combo] += 1
            s["flag_counts"] = dict(flag_counts)
            s["combo_top"] = [
                {"combo": c, "count": int(n)} for c, n in combo_counts.most_common(10)
            ]

        return {
            "schema": "positive_touch_ledger_v1",
            "chain_id": chain_id,
            "thresholds": {"long_bounce_30s_pct": 0.005, "short_bounce_30s_pct": -0.005},
            "scenarios": list(scenarios.values()),
            "all_positive_touches": all_positive,
        }

    @staticmethod
    def _compute_level_stats(touch_packets: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute aggregated stats from touch packets for a single level.
        Returns mean, p50, rates, top bounces, and quality assessment.
        """
        import statistics
        from collections import Counter

        if not touch_packets:
            return {"touches": 0, "quality": "no_data"}

        n = len(touch_packets)

        # Outcomes
        outcomes = Counter([tp.get("touch_outcome") for tp in touch_packets])

        # Delta flip rate
        flips = sum(1 for tp in touch_packets if tp.get("delta_flip_flag") is True)
        delta_flip_rate = round(flips / n, 3) if n > 0 else 0.0

        # Compression stats
        comp_vals = [tp.get("compression_at_touch") for tp in touch_packets
                     if isinstance(tp.get("compression_at_touch"), (int, float))]
        comp_stats = {}
        if comp_vals:
            comp_stats = {
                "compression_mean": round(statistics.mean(comp_vals), 3),
                "compression_p50": round(statistics.median(comp_vals), 3),
                "compression_max": round(max(comp_vals), 3),
            }

        # Bounce 30s stats
        bounce_vals = [tp.get("bounce_return_30s_pct") for tp in touch_packets
                       if isinstance(tp.get("bounce_return_30s_pct"), (int, float))]
        bounce_stats = {}
        if bounce_vals:
            pos_count = sum(1 for b in bounce_vals if b > 0)
            bounce_stats = {
                "bounce_30s_mean": round(statistics.mean(bounce_vals), 5),
                "bounce_30s_p50": round(statistics.median(bounce_vals), 5),
                "bounce_30s_pos_rate": round(pos_count / len(bounce_vals), 3),
            }

        # Aggression stats
        aggr_vals = [tp.get("rel_aggr_0_30s") for tp in touch_packets
                     if isinstance(tp.get("rel_aggr_0_30s"), (int, float))]
        aggr_stats = {}
        if aggr_vals:
            pos_aggr = sum(1 for a in aggr_vals if a > 0)
            aggr_stats = {
                "rel_aggr_30s_mean": round(statistics.mean(aggr_vals), 4),
                "rel_aggr_30s_pos_rate": round(pos_aggr / len(aggr_vals), 3),
            }

        # Top 3 bounces
        sorted_by_bounce = sorted(
            [tp for tp in touch_packets if isinstance(tp.get("bounce_return_30s_pct"), (int, float))],
            key=lambda x: x.get("bounce_return_30s_pct", 0),
            reverse=True
        )[:3]
        top_bounces = []
        for tp in sorted_by_bounce:
            top_bounces.append({
                "ts_ms": tp.get("touch_ts_ms"),
                "bounce_pct": round(tp.get("bounce_return_30s_pct", 0), 5),
                "compression": round(tp.get("compression_at_touch", 0), 3) if tp.get("compression_at_touch") else None,
                "delta_30s": tp.get("band_delta_0_30s"),
                "rel_aggr_30s": round(tp.get("rel_aggr_0_30s", 0), 4) if tp.get("rel_aggr_0_30s") else None,
                "outcome": tp.get("touch_outcome"),
                "approach": tp.get("approach_type"),
            })

        # Quality assessment
        quality = "unknown"
        holds = outcomes.get("hold", 0)
        breaks = outcomes.get("break", 0) + outcomes.get("break_reclaim_fail", 0)
        hold_rate = holds / n if n > 0 else 0

        if n <= 3 and hold_rate >= 0.8:
            quality = "strong"
        elif n <= 7 and hold_rate >= 0.7 and bounce_stats.get("bounce_30s_pos_rate", 0) >= 0.6:
            quality = "strong"
        elif n > 10 and (breaks >= 2 or bounce_stats.get("bounce_30s_pos_rate", 0) < 0.5):
            quality = "exhausted"
        elif hold_rate >= 0.6:
            quality = "moderate"
        else:
            quality = "weak"

        return {
            "touches": n,
            "outcomes": dict(outcomes),
            "delta_flip_rate": delta_flip_rate,
            **comp_stats,
            **bounce_stats,
            **aggr_stats,
            "top_bounces": top_bounces,
            "quality": quality,
        }

    @staticmethod
    def _compute_level_flags(level_stats: dict[str, Any]) -> dict[str, Any]:
        """
        Compute green/red flags based on level stats and thresholds.
        """
        green: list[str] = []
        red: list[str] = []

        # Compression
        comp_max = level_stats.get("compression_max")
        if comp_max is not None:
            if comp_max >= 1.0:
                green.append("compression_max >= 1.0 (high energy)")
            elif comp_max < 0.5:
                red.append("compression_max < 0.5 (no setup)")

        # Bounce rate
        bounce_pos_rate = level_stats.get("bounce_30s_pos_rate")
        if bounce_pos_rate is not None:
            if bounce_pos_rate >= 0.6:
                green.append(f"bounce_30s_pos_rate = {bounce_pos_rate:.0%} (> 60%)")
            elif bounce_pos_rate < 0.5:
                red.append(f"bounce_30s_pos_rate = {bounce_pos_rate:.0%} (< 50%)")

        # Delta flip rate
        flip_rate = level_stats.get("delta_flip_rate")
        if flip_rate is not None:
            if flip_rate >= 0.5:
                green.append(f"delta_flip_rate = {flip_rate:.0%} (buyers stepping in)")

        # Aggression
        aggr_pos_rate = level_stats.get("rel_aggr_30s_pos_rate")
        if aggr_pos_rate is not None:
            if aggr_pos_rate >= 0.6:
                green.append(f"rel_aggr_30s_pos_rate = {aggr_pos_rate:.0%} (buyers in control)")
            elif aggr_pos_rate < 0.4:
                red.append(f"rel_aggr_30s_pos_rate = {aggr_pos_rate:.0%} (sellers dominating)")

        # Touch count (exhaustion)
        touches = level_stats.get("touches", 0)
        if touches > 15:
            red.append(f"touches = {touches} (> 15 = exhausted)")
        elif touches > 10:
            red.append(f"touches = {touches} (> 10 = getting tired)")

        # Outcomes
        outcomes = level_stats.get("outcomes", {})
        breaks = outcomes.get("break", 0) + outcomes.get("break_reclaim_fail", 0)
        if breaks >= 2:
            red.append(f"breaks = {breaks} (multiple failures)")

        # Quality
        quality = level_stats.get("quality")
        if quality == "strong":
            green.append("quality = strong")
        elif quality == "exhausted":
            red.append("quality = exhausted")
        elif quality == "weak":
            red.append("quality = weak")

        return {"green": green, "red": red}

    @staticmethod
    def _touch_score_palette() -> dict[int, dict[str, str]]:
        """
        Discrete 10-bucket palette for bar rendering.
        Bucket 1 = worst (dark red), bucket 10 = best (bright green).
        """
        return {
            1: {"color": "#CC0000", "label": "STRONG_SELL"},
            2: {"color": "#FF2200", "label": "SELL"},
            3: {"color": "#FF5500", "label": "SELL"},
            4: {"color": "#FF8800", "label": "CAUTION"},
            5: {"color": "#FFAA00", "label": "NEUTRAL"},
            6: {"color": "#CCCC00", "label": "NEUTRAL"},
            7: {"color": "#99CC33", "label": "BUY"},
            8: {"color": "#66CC66", "label": "BUY"},
            9: {"color": "#22DD22", "label": "STRONG_BUY"},
            10: {"color": "#00FF00", "label": "STRONG_BUY"},
        }

    @staticmethod
    def _bucket_from_score(score: float | int) -> int:
        """
        Map a score in [-20, +20] to a bucket 1..10, just like _score_touch_support_long does.
        """
        val = max(-20, min(20, float(score)))
        bucket = int(round(((val + 20) / 40) * 9 + 1))
        return max(1, min(10, bucket))

    @staticmethod
    def _score_touch_support_long(tp: dict[str, Any]) -> dict[str, Any]:
        """
        Deterministic scoring for SUPPORT-LONG only (V1).
        Produces:
          - score_v1: int (can be negative)
          - score_bucket: 1..10
          - score_color: hex
          - score_label: string
          - reasons_plus / reasons_minus: short strings
        """
        # Helper getters
        def _f(name: str) -> float | None:
            v = tp.get(name)
            return float(v) if isinstance(v, (int, float)) else None

        def _b(name: str) -> bool | None:
            v = tp.get(name)
            return bool(v) if isinstance(v, bool) else None

        score = 0
        plus: list[str] = []
        minus: list[str] = []

        # Core post-touch control (most important)
        delta30 = _f("band_delta_0_30s")
        aggr30 = _f("rel_aggr_0_30s")
        bounce30 = _f("bounce_return_30s_pct")

        # Delta points (asinh-scaled so it behaves across tickers)
        if delta30 is not None:
            # Scale: +/-50k delta maps to ~ +/-3..4 points depending on magnitude
            from math import asinh

            scaled = asinh(delta30 / 50000.0) * 4.0
            pts = int(round(max(-6.0, min(6.0, scaled))))
            if pts != 0:
                score += pts
                (plus if pts > 0 else minus).append(f"delta_30s {pts:+d}")

        # Aggression points
        if aggr30 is not None:
            if aggr30 >= 0.20:
                score += 3
                plus.append("pos_aggr_strong +3")
            elif aggr30 >= 0.05:
                score += 1
                plus.append("pos_aggr +1")
            elif aggr30 <= -0.20:
                score -= 3
                minus.append("neg_aggr_strong -3")
            elif aggr30 <= -0.05:
                score -= 1
                minus.append("neg_aggr -1")

        # Bounce points (support-long: positive bounce is bullish)
        if bounce30 is not None:
            if bounce30 >= 0.02:
                score += 6
                plus.append("bounce>=2% +6")
            elif bounce30 >= 0.01:
                score += 3
                plus.append("bounce>=1% +3")
            elif bounce30 >= 0.005:
                score += 1
                plus.append("bounce>=0.5% +1")
            elif bounce30 <= -0.01:
                score -= 6
                minus.append("bounce<=-1% -6")
            elif bounce30 <= -0.005:
                score -= 2
                minus.append("bounce<=-0.5% -2")

        # Large order flow (per your studies)
        lt_count = _f("large_trade_count_30s")
        buy_ratio = _f("large_trade_buy_ratio_30s")
        lt_delta = _f("large_trade_delta_30s")

        # Base thresholds (support-long V1): high_buy_ratio >= 55% and pos_aggr
        high_buy_ratio = (buy_ratio is not None and buy_ratio >= 0.55)
        pos_aggr = (aggr30 is not None and aggr30 > 0)

        if high_buy_ratio:
            score += 3
            plus.append("high_buy_ratio(>=55%) +3")
        if pos_aggr:
            # already scored above, but we keep a signal marker; do not double-count
            pass
        if high_buy_ratio and pos_aggr:
            score += 4
            plus.append("combo high_buy_ratio+pos_aggr +4")

        # Large delta confirmation
        if lt_delta is not None:
            if lt_delta >= 20000:
                score += 3
                plus.append("large_delta>=20k +3")
            elif lt_delta <= -20000:
                score -= 4
                minus.append("large_delta<=-20k -4")

        # Large count / activity confirmation
        if lt_count is not None:
            if lt_count >= 50:
                score += 3
                plus.append("large_count>=50 +3")
            elif lt_count >= 20:
                score += 2
                plus.append("large_count>=20 +2")
            elif lt_count < 10:
                minus.append("low_large_count(<10) cap")

        # Institutional flags (already computed from large metrics)
        if tp.get("institutional_buying_flag") is True:
            score += 4
            plus.append("INST_BUY +4")
        if tp.get("institutional_selling_flag") is True:
            score -= 6
            minus.append("INST_SELL -6")

        # Absorption: institutional selling but price still bounces
        if tp.get("institutional_selling_flag") is True and bounce30 is not None:
            if bounce30 > 0:
                score += 8
                plus.append("absorption(INST_SELL+bounce) +8")
            elif bounce30 < 0:
                score -= 3
                minus.append("confirmed_sell(INST_SELL+down) -3")

        # Break/reclaim mechanics (hard negatives / recovery positives)
        if tp.get("break_confirmed_30s") is True:
            score -= 8
            minus.append("break_confirmed -8")
            if tp.get("reclaim_after_break") and tp.get("reclaim_hold_30s") is True:
                score += 6
                plus.append("reclaim_hold_30s +6")

        # Context: compression helps, but not alone
        comp = _f("compression_at_touch")
        if comp is not None:
            if comp >= 1.0:
                score += 3
                plus.append("compression>=1.0 +3")
            elif comp >= 0.8:
                score += 2
                plus.append("compression>=0.8 +2")
            elif comp < 0.5:
                score -= 1
                minus.append("compression<0.5 -1")

        trend = tp.get("compression_trend_into_touch")
        if trend == "tightening":
            score += 1
            plus.append("compression_tightening +1")
        elif trend == "loosening":
            score -= 1
            minus.append("compression_loosening -1")

        # Clean rejection / wick recovery
        if tp.get("wick_recovered_flag") is True:
            score += 2
            plus.append("wick_recovered +2")

        dwell = _f("touch_dwell_s")
        if dwell is not None and bounce30 is not None:
            if dwell <= 20 and bounce30 > 0:
                score += 1
                plus.append("quick_touch_clean +1")

        # Low-count guard: cap ratio-based positives when conviction is low
        if lt_count is not None and lt_count < 10:
            # Remove some optimism from ratio-based signals
            # (deterministic: subtract 2 if we had any ratio-based positives)
            had_ratio = high_buy_ratio or tp.get("institutional_buying_flag") is True
            if had_ratio:
                score -= 2
                minus.append("low_count_cap -2")

        # Clamp score to keep mapping stable
        score = int(max(-20, min(20, score)))

        # Bucket mapping: [-20..+20] -> [1..10]
        # -20 => 1, 0 => 5/6, +20 => 10
        bucket = int(round(((score + 20) / 40) * 9 + 1))
        bucket = int(max(1, min(10, bucket)))

        pal = Store._touch_score_palette().get(bucket, {"color": "#CCCC00", "label": "NEUTRAL"})
        # Keep reasons short: top 6 each
        return {
            "score_v1": int(score),
            "score_bucket": int(bucket),
            "score_color": str(pal["color"]),
            "score_label": str(pal["label"]),
            "reasons_plus": plus[:6],
            "reasons_minus": minus[:6],
        }

    @staticmethod
    def _score_touch_packets_support_long(
        touch_packets: list[dict[str, Any]], *, avg_shares_per_trade: float | None = None
    ) -> dict[str, Any]:
        """
        Mutates touch_packets by adding score fields (support-long V1).
        Returns a per-level scoring summary for rendering + report.json.
        """
        if not touch_packets:
            return {
                "schema": "touch_scoring_v1",
                "mode": "support_long",
                "touches": 0,
                "verdict": "NO_DATA",
                "avg_score": None,
                "last5_avg_score": None,
                "plus": [],
                "minus": [],
                "notes": ["no touch packets"],
            }

        scores: list[int] = []
        cum_scores: list[int] = []
        running = 0
        for tp in touch_packets:
            sc = Store._score_touch_support_long(tp)
            tp.update(sc)
            scores.append(int(sc["score_v1"]))
            running += int(sc["score_v1"])
            cum_scores.append(running)
            tp["cum_score_v1"] = running

        n = len(scores)
        avg_score = float(sum(scores)) / float(n) if n > 0 else 0.0
        last5 = scores[-5:] if n >= 5 else scores
        last5_avg = float(sum(last5)) / float(len(last5)) if last5 else 0.0

        cum_min = min(cum_scores) if cum_scores else 0
        cum_max = max(cum_scores) if cum_scores else 0
        abs_max = max(abs(cum_min), abs(cum_max), 1)
        for idx, tp in enumerate(touch_packets):
            cum = cum_scores[idx]
            bucket = Store._bucket_from_score(cum)
            pal = Store._touch_score_palette().get(bucket, {"color": "#CCCC00", "label": "NEUTRAL"})
            tp["cum_score_bucket"] = bucket
            tp["cum_score_label"] = pal["label"]
            tp["cum_score_color"] = pal["color"]
            tp["cum_score_min"] = cum_min
            tp["cum_score_max"] = cum_max
            tp["cum_is_negative"] = cum < 0
            tp["cum_height_pct"] = float(abs(cum) / abs_max) * 100.0

        # Peak touch (setup formed). This is important because the last touch can be noise / post-move dump.
        peak_score = max(scores) if scores else 0
        peak_idx = int(scores.index(peak_score)) if scores else 0
        peak_touch_number = int(touch_packets[peak_idx].get("touch_number", peak_idx + 1)) if touch_packets else None
        trough_score = min(scores) if scores else 0
        trough_idx = int(scores.index(trough_score)) if scores else 0
        trough_touch_number = int(touch_packets[trough_idx].get("touch_number", trough_idx + 1)) if touch_packets else None

        # Ending matters: if last5 avg is meaningfully higher, boost verdict wording (not score)
        ending_strong = (last5_avg - avg_score) >= 2.0
        ending_weak = (avg_score - last5_avg) >= 2.0

        # Late INST_SELL (last 3 touches)
        late_inst_sell = any(tp.get("institutional_selling_flag") is True for tp in touch_packets[-3:])
        late_inst_buy = any(tp.get("institutional_buying_flag") is True for tp in touch_packets[-3:])
        late_break = any(tp.get("break_confirmed_30s") is True for tp in touch_packets[-3:])

        # Liquidity warning
        thin_liquidity = bool(avg_shares_per_trade is not None and avg_shares_per_trade < 50)

        # Verdict thresholds (support-long V1)
        # Use PEAK as the base (setup formation), and ENDING as the modifier (risk).
        verdict = "NEUTRAL"
        if peak_score >= 16:
            verdict = "STRONG_BUY"
        elif peak_score >= 10:
            verdict = "BUY"
        elif peak_score <= -16:
            verdict = "STRONG_SELL"
        elif peak_score <= -10:
            verdict = "SELL"

        # Modifier: if ending is clearly weak OR late break shows up, downgrade one notch.
        if verdict in ("STRONG_BUY", "BUY") and (late_break or (ending_weak and last5_avg <= -2)):
            verdict = "BUY" if verdict == "STRONG_BUY" else "NEUTRAL"
        if verdict in ("STRONG_SELL", "SELL") and (ending_strong and last5_avg >= 2):
            verdict = "SELL" if verdict == "STRONG_SELL" else "NEUTRAL"

        notes: list[str] = []
        if ending_strong:
            notes.append("ending_strong")
        if ending_weak:
            notes.append("ending_weak")
        if late_inst_sell:
            notes.append("late_inst_sell")
        if late_inst_buy:
            notes.append("late_inst_buy")
        if late_break:
            notes.append("late_break_confirmed")
        if thin_liquidity:
            notes.append("thin_liquidity(avg_shares/trade<50)")

        # Collect top reasons across all touches (frequency-weighted)
        from collections import Counter

        plus_counts = Counter()
        minus_counts = Counter()
        for tp in touch_packets:
            for r in (tp.get("reasons_plus") or []):
                plus_counts[str(r)] += 1
            for r in (tp.get("reasons_minus") or []):
                minus_counts[str(r)] += 1

        return {
            "schema": "touch_scoring_v1",
            "mode": "support_long",
            "touches": int(n),
            "avg_score": round(avg_score, 2),
            "last5_avg_score": round(last5_avg, 2),
            "peak_score": int(peak_score),
            "peak_touch_number": peak_touch_number,
            "trough_score": int(trough_score),
            "trough_touch_number": trough_touch_number,
            "cum_final_score": int(cum_scores[-1]) if cum_scores else 0,
            "cum_min": int(cum_min),
            "cum_max": int(cum_max),
            "cum_range": int(abs_max),
            "verdict": verdict,
            "ending_strong": bool(ending_strong),
            "ending_weak": bool(ending_weak),
            "late_inst_sell": bool(late_inst_sell),
            "late_inst_buy": bool(late_inst_buy),
            "late_break": bool(late_break),
            "thin_liquidity": bool(thin_liquidity),
            "top_plus": [{"reason": k, "count": int(v)} for k, v in plus_counts.most_common(6)],
            "top_minus": [{"reason": k, "count": int(v)} for k, v in minus_counts.most_common(6)],
            "notes": notes,
        }

    @staticmethod
    def _get_interpretation_guide() -> dict[str, Any]:
        """
        Return static interpretation guide for report display.
        """
        return {
            "post_touch_control": {
                "title": "Post-Touch Control (most important)",
                "metrics": [
                    "band_delta_0_30s > 0: buyers net-positive right after touch",
                    "rel_aggr_0_30s > 0: buyers aggressing after touch",
                    "delta_flip_flag = TRUE: flow flips at the level",
                ],
                "red_flag": "rel_aggr_0_30s < -0.3 or strongly negative band_delta_0_30s",
            },
            "compression_context": {
                "title": "Compression Context (setup quality)",
                "metrics": [
                    "compression_at_touch > 1.0: high energy, ready to move",
                    "compression_trend_into_touch = tightening: coiling into touch",
                ],
                "red_flag": "compression < 0.5 + flat trend = noise/chop",
            },
            "approach_behavior": {
                "title": "Approach Behavior (risk context)",
                "metrics": [
                    "approach_type = grind/normal: controlled, lower risk",
                    "approach_type = fast/crash: volatile, higher uncertainty",
                ],
                "red_flag": "crash approach + negative approach_delta_60s + negative rel_aggr_0_30s",
            },
            "immediate_reaction": {
                "title": "Immediate Reaction (did it bounce?)",
                "metrics": [
                    "bounce_return_30s_pct > 0: price moving up after touch",
                    "short dwell + positive bounce = clean reject",
                ],
                "red_flag": "bounce stays negative while flow metrics are negative",
            },
            "level_exhaustion": {
                "title": "Level Exhaustion (session-wide)",
                "metrics": [
                    "touches < 10: level still fresh",
                    "mostly 'hold' outcomes: level is respected",
                ],
                "red_flag": "touches > 15 + mixed breaks/reclaims = exhausted",
            },
            "cleanest_signal": "band_delta_0_30s > 0 AND rel_aggr_0_30s > 0 AND compression > 1.0 = level strengthening",
            "decision_matrix": [
                {"compression": "> 1.0", "delta_flip": "TRUE", "aggr_30s": "> 0", "signal": "STRONG (level holding)"},
                {"compression": "> 1.0", "delta_flip": "TRUE", "aggr_30s": "< 0", "signal": "WAIT (reassess in 30s)"},
                {"compression": "> 1.0", "delta_flip": "FALSE", "aggr_30s": "any", "signal": "SKIP (no flip)"},
                {"compression": "< 0.5", "delta_flip": "any", "aggr_30s": "any", "signal": "SKIP (no setup)"},
                {"compression": "any", "delta_flip": "any", "aggr_30s": "< -0.3", "signal": "SKIP (sellers winning)"},
            ],
        }

    @staticmethod
    def _flags_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
        """
        Very lightweight, human-readable flags. These are not “signals”; they’re audit hints.
        """
        flags: list[str] = []
        score = 0

        band_delta = summary.get("band_delta_sum", None)
        rel = summary.get("band_relative_aggression_mean", summary.get("relative_aggression_mean", None))
        hold = summary.get("reclaim_hold_rate", None)
        touches = summary.get("touch_count", None)

        try:
            if band_delta is not None:
                bd = float(band_delta)
                if bd > 0:
                    flags.append("green: band_delta_positive")
                    score += 1
                elif bd < 0:
                    flags.append("red: band_delta_negative")
                    score -= 1
        except Exception:
            pass

        try:
            if rel is not None:
                r = float(rel)
                if r > 0.10:
                    flags.append("green: aggression_positive")
                    score += 1
                elif r < -0.10:
                    flags.append("red: aggression_negative")
                    score -= 1
        except Exception:
            pass

        try:
            if hold is not None:
                h = float(hold)
                if h >= 0.70:
                    flags.append("green: reclaim_hold_rate_high")
                    score += 1
                elif h <= 0.30:
                    flags.append("red: reclaim_hold_rate_low")
                    score -= 1
        except Exception:
            pass

        try:
            if touches is not None:
                t = int(touches)
                if t >= 3:
                    flags.append("info: multiple_touches")
        except Exception:
            pass

        return {"score": int(score), "flags": flags}

    @staticmethod
    def _ctx_trigger_from(
        *,
        between: dict[str, Any],
        countdown: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Small, stable checklist summary.
        - Context: compression readiness (level + trend into T)
        - Trigger: reaction in the last 30s/60s before T (band_delta + aggression)
        """
        ctx: dict[str, Any] = {}
        trig: dict[str, Any] = {}

        # ---- Context: compression ----
        comp_mean = between.get("compression_index_mean", None)
        try:
            comp = float(comp_mean) if comp_mean is not None else None
        except Exception:
            comp = None

        # Trend into T: compare earliest countdown window vs latest.
        comp_early = None
        comp_late = None
        if countdown:
            def _f(x: Any) -> float | None:
                try:
                    return float(x) if x is not None else None
                except Exception:
                    return None

            comp_early = _f(countdown[0].get("compression_index_mean", None))
            comp_late = _f(countdown[-1].get("compression_index_mean", None))

        ctx["compression_mean"] = comp
        if comp is None:
            ctx["compression_level"] = None
        elif comp >= 0.75:
            ctx["compression_level"] = "high"
        elif comp <= 0.35:
            ctx["compression_level"] = "low"
        else:
            ctx["compression_level"] = "mid"

        if comp_early is not None and comp_late is not None:
            d = comp_late - comp_early
            ctx["compression_delta_into_T"] = float(d)
            if d > 0.05:
                ctx["compression_trend_into_T"] = "tightening"
            elif d < -0.05:
                ctx["compression_trend_into_T"] = "loosening"
            else:
                ctx["compression_trend_into_T"] = "flat"
        else:
            ctx["compression_delta_into_T"] = None
            ctx["compression_trend_into_T"] = None

        # ---- Trigger: last 30s/60s reaction ----
        last30 = countdown[-1] if countdown else {}
        last60 = countdown[-2] if len(countdown) >= 2 else {}

        def _g(d: dict[str, Any], k: str) -> float | None:
            try:
                v = d.get(k, None)
                return float(v) if v is not None else None
            except Exception:
                return None

        trig["band_delta_last30s"] = _g(last30, "band_delta_sum")
        trig["rel_aggr_last30s"] = _g(last30, "relative_aggression_mean")
        trig["band_delta_last60s"] = _g(last60, "band_delta_sum")
        trig["rel_aggr_last60s"] = _g(last60, "relative_aggression_mean")

        # Simple classification: reaction bullish if BOTH band_delta and aggression are positive.
        # bearish if BOTH are negative. else mixed/neutral.
        bd = trig["band_delta_last30s"]
        ra = trig["rel_aggr_last30s"]
        if bd is None or ra is None:
            trig["reaction"] = None
        elif bd > 0 and ra > 0.10:
            trig["reaction"] = "bullish"
        elif bd < 0 and ra < -0.10:
            trig["reaction"] = "bearish"
        else:
            trig["reaction"] = "mixed"

        return {"context": ctx, "trigger": trig}


    def compute_session_report(self, *, date: str, ticker: str, session_id: str) -> dict[str, Any]:
        """
        Lightweight session report for the web UI:
        - win/loss counts
        - episode metrics stats (mfe/mae/time_to_*)
        - simple feature separation ranking using baseline vs early-stress (first 60s) deltas
        - level_chains from watch-window markers (even if no episodes)
        """
        eps = self.read_episodes(date=date, ticker=ticker, session_id=session_id)
        # Note: don't return early if eps.empty; we still need to process level_chains from markers.

        # Outcome counts
        out_counts: dict[str, int] = {}
        if "outcome" in eps.columns:
            vc = eps["outcome"].astype("string").fillna("<NA>").value_counts(dropna=False)
            out_counts = {str(k): int(v) for k, v in vc.items()}
            # Make the UI friendlier: unresolved episodes are those without deterministic outcome yet.
            if "<NA>" in out_counts:
                out_counts["unresolved"] = out_counts.pop("<NA>")

        # Metrics stats
        def _stats(col: str) -> dict[str, float] | None:
            if col not in eps.columns:
                return None
            s = pd.to_numeric(eps[col], errors="coerce").dropna()
            if s.empty:
                return None
            return {
                "min": float(s.min()),
                "p50": float(s.quantile(0.5)),
                "p90": float(s.quantile(0.9)),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }

        metrics = {
            "mfe": _stats("mfe"),
            "mae": _stats("mae"),
            "time_to_mfe_ms": _stats("time_to_mfe_ms"),
            "time_to_failure_ms": _stats("time_to_failure_ms"),
        }

        # Feature separation (wins vs losses) on a small curated set
        ranked: list[dict[str, Any]] = []
        if "outcome" in eps.columns:
            e2 = eps[eps["outcome"].isin(["win", "loss"])].copy()
            if not e2.empty:
                e2["y_win"] = (e2["outcome"] == "win").astype(int)
                paths = self._glob("snapshots", date, ticker, session_id)
                snaps = _read_all_parquet_files(paths)
                if not snaps.empty:
                    snaps = snaps[snaps["episode_id"].astype(str).isin(e2["episode_id"].astype(str))]
                    snaps["timestamp"] = pd.to_numeric(snaps["timestamp"], errors="coerce")
                    snaps = snaps.dropna(subset=["episode_id", "timestamp"])

                    # windows
                    def _window(df: pd.DataFrame, which: str) -> pd.DataFrame:
                        if which == "baseline":
                            return df[df["phase"] == "baseline"].copy()
                        if which == "stress":
                            return df[df["phase"] == "stress"].copy()
                        if which == "early_stress_60s":
                            ss = df[df["phase"] == "stress"].copy()
                            ss = ss.sort_values(["episode_id", "timestamp"], kind="mergesort")
                            ss["rank"] = ss.groupby("episode_id").cumcount()
                            return ss[ss["rank"] < 6].drop(columns=["rank"])
                        raise ValueError(which)

                    candidates = [
                        "delta",
                        "cvd_60s",
                        "cvd_30s",
                        "relative_aggression",
                        "absorption_index_10s",
                        "div_score",
                        "buy_on_red",
                        "sell_on_green",
                        "spread_pct",
                        "signed_distance_to_level_atr",
                        "vol_z_60s",
                        "trade_rate_z_60s",
                        "compression_index",
                        "ema_confluence_score",
                    ]
                    candidates = [c for c in candidates if c in snaps.columns]

                    def _agg(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
                        if df.empty or not candidates:
                            return pd.DataFrame(index=pd.Index([], name="episode_id"))
                        g = df.groupby("episode_id")[candidates]
                        m = g.mean().add_suffix(f"_{suffix}_mean")
                        return m

                    b = _agg(_window(snaps, "baseline"), "baseline")
                    e = _agg(_window(snaps, "early_stress_60s"), "early")
                    joined = e2.set_index("episode_id")[["y_win"]].join([b, e], how="left")

                    # Early minus baseline deltas
                    for c in candidates:
                        cb = f"{c}_baseline_mean"
                        ce = f"{c}_early_mean"
                        if cb in joined.columns and ce in joined.columns:
                            joined[f"{c}_early_minus_base"] = joined[ce] - joined[cb]

                    # Rank by abs(mean(win) - mean(loss))
                    feats = [c for c in joined.columns if c != "y_win" and c.endswith("_early_minus_base")]
                    for f in feats:
                        x = pd.to_numeric(joined[f], errors="coerce")
                        y = joined["y_win"].astype(int)
                        win = x[y == 1].dropna()
                        loss = x[y == 0].dropna()
                        if len(win) < 2 or len(loss) < 2:
                            continue

                        mw = float(win.mean())
                        ml = float(loss.mean())
                        diff = mw - ml

                        # Scale-robust ranking: use an effect size on a transform that is stable for
                        # ratio-like features that can explode when denominators approach zero.
                        # Raw values remain in Parquet; this affects only report summarization.
                        base = f.replace("_early_minus_base", "")
                        ratio_like = base in {"absorption_index_10s", "buy_on_red", "sell_on_green", "div_score"}
                        if ratio_like:
                            win_t = win.apply(lambda v: asinh(float(v)))
                            loss_t = loss.apply(lambda v: asinh(float(v)))
                        else:
                            win_t = win
                            loss_t = loss

                        sw = float(win_t.std(ddof=0)) if len(win_t) >= 2 else 0.0
                        sl = float(loss_t.std(ddof=0)) if len(loss_t) >= 2 else 0.0
                        pooled = sqrt((sw * sw + sl * sl) / 2.0) if (sw > 0 or sl > 0) else 0.0
                        effect = (float(win_t.mean()) - float(loss_t.mean())) / (pooled + 1e-12) if pooled > 0 else 0.0

                        ranked.append(
                            {
                                "feature": f,
                                "mean_win": mw,
                                "mean_loss": ml,
                                "diff": diff,
                                "abs_diff": abs(diff),
                                "effect_size": float(effect),
                                "abs_effect_size": float(abs(effect)),
                            }
                        )
                    ranked.sort(key=lambda r: (r.get("abs_effect_size", 0.0), r.get("abs_diff", 0.0)), reverse=True)
                    ranked = ranked[:20]

        report: dict[str, Any] = {
            "date": date,
            "ticker": ticker,
            "session_id": session_id,
            "episodes_total": int(len(eps)),
            "outcome_counts": out_counts,
            "metrics": metrics,
            "feature_ranking": ranked,
            "note": (
                "Feature ranking is exploratory (small sample). It uses early-stress minus baseline deltas. "
                "Ranking uses an effect-size metric (with asinh transform for ratio-like features) to avoid "
                "penny-stock/near-zero denominators dominating by raw magnitude."
            ),
        }

        # ===== Watch-window marker analysis (pre-move microscope) =====
        # Markers come from the UI as (ts_ms + notes JSON containing optional end_ts_ms + price_tags).
        markers = self.read_markers(date=date, ticker=ticker, session_id=session_id)
        stream = self.read_session_stream(date=date, ticker=ticker, session_id=session_id)
        interval_s = self._infer_interval_s(stream)
        # Zone segments use 30-second intervals for less noisy dots
        zone_segment_interval_s = 30

        watch_windows: list[dict[str, Any]] = []
        event_markers: list[dict[str, Any]] = []
        level_chains: list[dict[str, Any]] = []

        if not markers.empty and not stream.empty and "ts_ms" in markers.columns:
            # Collect per-level markers for the new "level chain" workflow.
            # These are markers with end_ts_ms + exactly one level tag, plus optional outcome.
            chain_nodes: dict[str, list[dict[str, Any]]] = {}

            for _, row in markers.iterrows():
                try:
                    start_ms = int(row["ts_ms"])
                except Exception:
                    continue
                mtype = str(row.get("marker_type", "") or "")
                notes = self._parse_notes(row.get("notes", None))
                end_ms = notes.get("end_ts_ms", None)
                try:
                    end_ms_i = int(end_ms) if end_ms is not None and str(end_ms).strip() != "" else None
                except Exception:
                    end_ms_i = None

                level_price, level_kind = self._choose_level_tag(notes.get("price_tags", None))
                tags = self._all_level_tags(notes.get("price_tags", None))
                band_pct = float(self._DEFAULT_BAND_PCT)

                # Level-chain markers: generated by the simplified record form.
                # Accept if schema is level_chain_v1 OR if we have chain_id + watch timestamps
                schema = str(notes.get("schema", "") or "").strip()
                chain_id = str(notes.get("chain_id", "") or "").strip() or "default"
                has_watch_times = notes.get("watch_start_ts_ms") is not None and notes.get("watch_end_ts_ms") is not None
                level_type = str(notes.get("level_type", "price") or "price").strip()
                indicators = notes.get("indicators", []) or []
                is_indicator_level = level_type == "indicator" and len(indicators) > 0
                is_level_chain = schema == "level_chain_v1" or (chain_id and has_watch_times and (level_price is not None or is_indicator_level))
                
                if is_level_chain and end_ms_i is not None and end_ms_i > start_ms:
                    if is_indicator_level:
                        # Indicator-based level: create an entry for each selected indicator
                        for ind_name in indicators:
                            chain_nodes.setdefault(chain_id, []).append(
                                {
                                    "marker_type": mtype,
                                    "watch_start_ts_ms": int(start_ms),
                                    "watch_end_ts_ms": int(end_ms_i),
                                    "level_price": None,  # Dynamic, computed from stream
                                    "level_kind": level_kind,
                                    "level_index": notes.get("level_index", None),
                                    "level_outcome": notes.get("level_outcome", None),
                                    "band_pct": float(notes.get("band_pct", 0.0015)),  # 0.15% for indicators
                                    "level_type": "indicator",
                                    "indicator_name": ind_name,
                                    "direction_bias": notes.get("direction_bias", "long"),
                                }
                            )
                    elif level_price is not None:
                        # Price-based level: existing behavior
                        chain_nodes.setdefault(chain_id, []).append(
                            {
                                "marker_type": mtype,
                                "watch_start_ts_ms": int(start_ms),
                                "watch_end_ts_ms": int(end_ms_i),
                                "level_price": float(level_price),
                                "level_kind": level_kind,
                                "level_index": notes.get("level_index", None),
                                "level_outcome": notes.get("level_outcome", None),
                                "band_pct": band_pct,
                                "level_type": "price",
                                "direction_bias": notes.get("direction_bias", "long"),
                            }
                        )

                if end_ms_i is not None and end_ms_i > start_ms:
                    # Per-level analysis (macro + micro tags)
                    if not tags and level_price is not None:
                        tags = [{"price": float(level_price), "kind": level_kind}]

                    countdown_specs = [(240, 180), (180, 120), (120, 60), (60, 30), (30, 0)]
                    T = int(end_ms_i)
                    per_levels: list[dict[str, Any]] = []
                    for tag in tags:
                        lp = float(tag["price"])
                        lk = tag.get("kind", None)
                        between = self._summarize_level_band_window(
                            df=stream,
                            start_ms=start_ms,
                            end_ms=end_ms_i,
                            level_price=lp,
                            level_kind=lk,
                            band_pct=band_pct,
                            interval_s=interval_s,
                        )
                        between["flags"] = self._flags_from_summary(between)

                        countdown: list[dict[str, Any]] = []
                        for a, b in countdown_specs:
                            w_end = T - (b * 1000)
                            w_start = T - (a * 1000)
                            ws = max(int(start_ms), int(w_start))
                            we = min(int(end_ms_i), int(w_end))
                            if we <= ws:
                                continue
                            summary = self._summarize_level_band_window(
                                df=stream,
                                start_ms=ws,
                                end_ms=we,
                                level_price=lp,
                                level_kind=lk,
                                band_pct=band_pct,
                                interval_s=interval_s,
                            )
                            summary["label"] = f"T-{a}s->T-{b}s"
                            summary["flags"] = self._flags_from_summary(summary)
                            countdown.append(summary)

                        # Checklist summary for fast reading
                        checklist = self._ctx_trigger_from(between=between, countdown=countdown)

                        per_levels.append(
                            {
                                "level_price": lp,
                                "level_kind": lk,
                                "between": between,
                                "countdown": countdown,
                                "checklist": checklist,
                            }
                        )

                    watch_windows.append(
                        {
                            "marker_type": mtype,
                            "watch_start_ts_ms": int(start_ms),
                            "watch_end_ts_ms": int(end_ms_i),
                            "band_pct": band_pct,
                            "levels": per_levels,
                            "note": "Watch window uses marker start→end. End is the move anchor T. Band stats are computed within ±band_pct of level_price.",
                        }
                    )
                else:
                    # Single-moment marker ("move"): compute pre (last 60s) + during/post
                    T = int(start_ms)
                    pre = self._summarize_level_band_window(
                        df=stream,
                        start_ms=T - 60_000,
                        end_ms=T,
                        level_price=level_price,
                        level_kind=level_kind,
                        band_pct=band_pct,
                        interval_s=interval_s,
                    )
                    during = self._summarize_level_band_window(
                        df=stream,
                        start_ms=T,
                        end_ms=T + 30_000,
                        level_price=level_price,
                        level_kind=level_kind,
                        band_pct=band_pct,
                        interval_s=interval_s,
                    )
                    post = self._summarize_level_band_window(
                        df=stream,
                        start_ms=T + 30_000,
                        end_ms=T + 120_000,
                        level_price=level_price,
                        level_kind=level_kind,
                        band_pct=band_pct,
                        interval_s=interval_s,
                    )
                    event_markers.append(
                        {
                            "marker_type": mtype,
                            "ts_ms": int(T),
                            "band_pct": band_pct,
                            "level_price": float(level_price) if level_price is not None else None,
                            "level_kind": level_kind,
                            "pre_60s": pre,
                            "during_30s": during,
                            "post_90s": post,
                            "note": "Event marker uses T=marker.ts_ms with pre(60s), during(30s), post(90s) windows anchored at T.",
                        }
                    )

            # Build level-chain report blocks (ordered levels + auto-computed move gaps)
            for cid, nodes in chain_nodes.items():
                if not nodes:
                    continue
                # Order by watch start (handle None level_price for indicator levels)
                nodes.sort(key=lambda x: (int(x.get("watch_start_ts_ms", 0)), float(x.get("level_price") or 0.0)))
                chain_levels: list[dict[str, Any]] = []
                prev = None

                for n in nodes:
                    ws = int(n["watch_start_ts_ms"])
                    we = int(n["watch_end_ts_ms"])
                    lk = n.get("level_kind", None)
                    level_type_n = n.get("level_type", "price")
                    indicator_name = n.get("indicator_name", None)
                    
                    # Compute level price: either fixed or from indicator column
                    ind_col_name: str | None = None  # Column name for dynamic indicators
                    
                    if level_type_n == "indicator" and indicator_name:
                        # Get indicator column for dynamic level, respecting timeframe
                        indicator_tf = n.get("indicator_timeframe", "1m")  # Default to 1m
                        
                        # Build column name based on timeframe
                        # 1m EMAs use base names (ema9, ema20, etc.)
                        # 2m/3m use suffixed names (ema9_2m, ema20_2m, etc.)
                        if indicator_tf == "1m" or indicator_tf == "30s":
                            ind_col_map = {
                                "vwap": "vwap_session",
                                "ema9": "ema9",
                                "ema20": "ema20",
                                "ema30": "ema30",
                                "ema200": "ema200",
                            }
                        elif indicator_tf == "2m":
                            ind_col_map = {
                                "vwap": "vwap_session",  # VWAP is session-based, no timeframe
                                "ema9": "ema9_2m",
                                "ema20": "ema20_2m",
                                "ema30": "ema30_2m",
                                "ema200": "ema200_2m",
                            }
                        elif indicator_tf == "3m":
                            ind_col_map = {
                                "vwap": "vwap_session",  # VWAP is session-based, no timeframe
                                "ema9": "ema9_3m",
                                "ema20": "ema20_3m",
                                "ema30": "ema30_3m",
                                "ema200": "ema200_3m",
                            }
                        else:
                            # Fallback to 1m
                            ind_col_map = {
                                "vwap": "vwap_session",
                                "ema9": "ema9",
                                "ema20": "ema20",
                                "ema30": "ema30",
                                "ema200": "ema200",
                            }
                        ind_col_name = ind_col_map.get(indicator_name)
                        if ind_col_name and ind_col_name in stream.columns:
                            window_df = stream[(stream["timestamp"] >= ws) & (stream["timestamp"] <= we)]
                            if not window_df.empty and ind_col_name in window_df.columns:
                                # Calculate mean for display only (real-time values used in analysis)
                                ind_values = window_df[ind_col_name].dropna()
                                valid_values = ind_values[ind_values > 0]  # Filter uninitialized zeros
                                if len(valid_values) > 0:
                                    lp = float(valid_values.mean())  # Display mean only
                                else:
                                    continue  # Skip if no valid indicator data
                            else:
                                continue  # Skip if no data in window
                        else:
                            continue  # Skip if indicator column not found
                    else:
                        # Fixed price level
                        if n.get("level_price") is None:
                            continue
                        lp = float(n["level_price"])

                    # Pass indicator_col for dynamic band calculations
                    between = self._summarize_level_band_window(
                        df=stream,
                        start_ms=ws,
                        end_ms=we,
                        level_price=lp,
                        level_kind=lk,
                        band_pct=float(n.get("band_pct", band_pct)),
                        interval_s=interval_s,
                        indicator_col=ind_col_name,  # NEW: pass indicator column
                    )
                    between["flags"] = self._flags_from_summary(between)

                    countdown_specs = [(240, 180), (180, 120), (120, 60), (60, 30), (30, 0)]
                    countdown: list[dict[str, Any]] = []
                    for a, b in countdown_specs:
                        w_end = we - (b * 1000)
                        w_start = we - (a * 1000)
                        ws2 = max(int(ws), int(w_start))
                        we2 = min(int(we), int(w_end))
                        if we2 <= ws2:
                            continue
                        summary = self._summarize_level_band_window(
                            df=stream,
                            start_ms=ws2,
                            end_ms=we2,
                            level_price=lp,
                            level_kind=lk,
                            band_pct=float(n.get("band_pct", band_pct)),
                            interval_s=interval_s,
                            indicator_col=ind_col_name,  # Pass indicator column
                        )
                        summary["label"] = f"T-{a}s->T-{b}s"
                        summary["flags"] = self._flags_from_summary(summary)
                        countdown.append(summary)

                    checklist = self._ctx_trigger_from(between=between, countdown=countdown)

                    move_from_prev = None
                    if prev is not None:
                        ms = int(prev["watch_end_ts_ms"])
                        me = int(ws)
                        if me > ms:
                            mv = self._summarize_level_band_window(
                                df=stream,
                                start_ms=ms,
                                end_ms=me,
                                level_price=None,
                                level_kind=None,
                                band_pct=float(n.get("band_pct", band_pct)),
                                interval_s=interval_s,
                            )
                            # Velocity bucket (ATR-normalized if available, else pct/min)
                            dur_min = max(1e-9, float(mv.get("duration_s", 0.0)) / 60.0)
                            ret_abs = mv.get("return_abs", None)
                            atr_m = mv.get("atr_value_mean", None)
                            bucket = None
                            try:
                                if ret_abs is not None and atr_m is not None and float(atr_m) > 0:
                                    v = abs(float(ret_abs)) / float(atr_m) / dur_min  # ATR per minute
                                    if v >= 1.5:
                                        bucket = "crash"
                                    elif v >= 0.75:
                                        bucket = "fast"
                                    elif v >= 0.25:
                                        bucket = "normal"
                                    else:
                                        bucket = "grind"
                                    mv["abs_return_atr"] = abs(float(ret_abs)) / float(atr_m)
                                    mv["abs_return_atr_per_min"] = float(v)
                                else:
                                    rp = mv.get("return_pct", None)
                                    if rp is not None:
                                        v = abs(float(rp)) / dur_min  # pct per minute (fraction)
                                        if v >= 0.007:
                                            bucket = "crash"
                                        elif v >= 0.003:
                                            bucket = "fast"
                                        elif v >= 0.001:
                                            bucket = "normal"
                                        else:
                                            bucket = "grind"
                                        mv["abs_return_pct"] = abs(float(rp))
                                        mv["abs_return_pct_per_min"] = float(v)
                            except Exception:
                                bucket = None
                            mv["velocity_bucket"] = bucket

                            # ---- Move verdict (human-friendly, deterministic) ----
                            # Direction from price change (fallback: neutral)
                            direction = None
                            try:
                                if ret_abs is not None:
                                    ra = float(ret_abs)
                                    if ra > 0:
                                        direction = "up"
                                    elif ra < 0:
                                        direction = "down"
                                    else:
                                        direction = "flat"
                            except Exception:
                                direction = None
                            mv["direction"] = direction

                            # Flow dominance from delta/volume imbalance
                            flow_side = None
                            flow_strength = None
                            imbalance = None
                            try:
                                dsum = float(mv.get("delta_sum", 0.0) or 0.0)
                                tv = float(mv.get("total_volume_sum", 0.0) or 0.0)
                                imbalance = abs(dsum) / (tv + 1e-9)
                                if dsum > 0:
                                    flow_side = "buy"
                                elif dsum < 0:
                                    flow_side = "sell"
                                else:
                                    flow_side = "flat"

                                # Thresholds tuned to be stable across symbols (since it's ratio)
                                if imbalance >= 0.35:
                                    flow_strength = "strong"
                                elif imbalance >= 0.15:
                                    flow_strength = "moderate"
                                else:
                                    flow_strength = "balanced"
                            except Exception:
                                flow_side = None
                                flow_strength = None
                                imbalance = None

                            mv["flow_side"] = flow_side
                            mv["flow_strength"] = flow_strength
                            mv["flow_imbalance_ratio"] = float(imbalance) if imbalance is not None else None

                            # Compact verdict string for UI/CSV
                            parts = []
                            if bucket:
                                parts.append(str(bucket))
                            if direction:
                                parts.append(str(direction))
                            if flow_side and flow_strength:
                                parts.append(f"{flow_side}_{flow_strength}")
                            mv["verdict"] = " / ".join(parts) if parts else None
                            move_from_prev = mv

                    # Session-wide continuous tracking for this level (full session, not just window)
                    session_start_ms = int(stream["timestamp"].min()) if not stream.empty else ws
                    session_end_ms = int(stream["timestamp"].max()) if not stream.empty else we
                    session_wide = self._summarize_level_band_window(
                        df=stream,
                        start_ms=session_start_ms,
                        end_ms=session_end_ms,
                        level_price=lp,
                        level_kind=lk,
                        band_pct=float(n.get("band_pct", band_pct)),
                        interval_s=interval_s,
                        indicator_col=ind_col_name,  # Pass indicator column
                    )
                    session_wide["flags"] = self._flags_from_summary(session_wide)

                    # Zone story (watch window) + last minutes bias
                    zone_analysis = self._compute_zone_metrics(
                        df=stream,
                        level_price=float(lp),
                        start_ms=ws,
                        end_ms=we,
                        interval_s=interval_s,
                        above_pct=self._DEFAULT_ZONE_ABOVE_PCT,
                        below_pct=self._DEFAULT_ZONE_BELOW_PCT,
                        on_pct=self._DEFAULT_ZONE_ON_PCT,
                        on_pct_down=self._DEFAULT_ZONE_ON_PCT_DOWN,
                    )
                    zone_analysis["last_minutes"] = {
                        "last_3m": self._compute_last_minutes_bias(
                            df=stream, end_ms=we, minutes=3
                        ),
                        "last_5m": self._compute_last_minutes_bias(
                            df=stream, end_ms=we, minutes=5
                        ),
                    }
                    zone_segments = self._compute_zone_segments(
                        df=stream,
                        level_price=float(lp),
                        start_ms=ws,
                        end_ms=we,
                        interval_s=zone_segment_interval_s,  # 30s segments for less noisy dots
                        above_pct=self._DEFAULT_ZONE_ABOVE_PCT,
                        below_pct=self._DEFAULT_ZONE_BELOW_PCT,
                        on_pct=self._DEFAULT_ZONE_ON_PCT,
                        on_pct_down=self._DEFAULT_ZONE_ON_PCT_DOWN,
                    )
                    zone_segments = self._annotate_zone_segments(
                        segments=zone_segments, zone_analysis=zone_analysis
                    )
                    zone_signals = self._compute_additional_signals(
                        segments=zone_segments, zone_analysis=zone_analysis
                    )
                    risk_reward = self._compute_risk_reward(
                        level_price=float(lp),
                        zone_analysis=zone_analysis,
                        segments=zone_segments,
                    )
                    zone_verdict = self._compute_final_verdict(
                        zone_analysis=zone_analysis,
                        additional=zone_signals,
                        segments=zone_segments,
                        risk_reward=risk_reward,
                    )
                    trade_trigger = self._compute_trade_trigger(
                        level_price=float(lp),
                        risk_reward=risk_reward,
                        zone_analysis=zone_analysis,
                    )
                    role_behavior = self._compute_role_behavior(
                        zone_analysis=zone_analysis,
                        segments=zone_segments,
                        level_kind=lk,
                    )

                    # DS/AS/DP State Machine scoring
                    defense_score = self._compute_defense_score(
                        segments=zone_segments,
                        zone_analysis=zone_analysis,
                    )
                    aggression_score = self._compute_aggression_score(
                        segments=zone_segments,
                        zone_analysis=zone_analysis,
                    )
                    distribution_pattern = self._detect_distribution_pattern(
                        segments=zone_segments,
                    )
                    late_momentum = self._compute_late_momentum(
                        segments=zone_segments,
                        n_segments=5,
                    )
                    market_state = self._compute_market_state(
                        defense_score=defense_score,
                        aggression_score=aggression_score,
                        distribution_pattern=distribution_pattern,
                        zone_analysis=zone_analysis,
                        segments=zone_segments,
                        late_momentum=late_momentum,
                    )

                    # Interaction timeline: every touch/break/reclaim/reject across entire session
                    interaction_timeline = self._compute_interaction_timeline(
                        df=stream,
                        level_price=lp,
                        level_kind=lk,
                        band_pct=float(n.get("band_pct", band_pct)),
                        interval_s=interval_s,
                        indicator_col=ind_col_name,  # Pass indicator column
                    )

                    # Summarize timeline events
                    timeline_summary = {
                        "total_events": len(interaction_timeline),
                        "touches": sum(1 for e in interaction_timeline if e.get("event") == "touch"),
                        "breaks": sum(1 for e in interaction_timeline if e.get("event") == "break"),
                        "rejects": sum(1 for e in interaction_timeline if e.get("event") == "reject"),
                        "reclaim_attempts": sum(1 for e in interaction_timeline if e.get("reclaim_attempt")),
                        "cross_throughs": sum(1 for e in interaction_timeline if e.get("event") == "cross_through"),
                    }

                    # Detect role flip: if level_kind is support but we see rejects from below after breaks, it may have flipped to resistance
                    role_flip = None
                    if lk == "support":
                        # Check if there are rejects from above AFTER a break below
                        saw_break_below = False
                        saw_reject_above_after = False
                        for e in interaction_timeline:
                            if e.get("event") == "break" and e.get("to_side") == "below":
                                saw_break_below = True
                            if saw_break_below and e.get("event") == "reject" and e.get("to_side") == "above":
                                saw_reject_above_after = True
                        if saw_break_below and saw_reject_above_after:
                            role_flip = "support_to_resistance"
                    elif lk == "resistance":
                        saw_break_above = False
                        saw_reject_below_after = False
                        for e in interaction_timeline:
                            if e.get("event") == "break" and e.get("to_side") == "above":
                                saw_break_above = True
                            if saw_break_above and e.get("event") == "reject" and e.get("to_side") == "below":
                                saw_reject_below_after = True
                        if saw_break_above and saw_reject_below_after:
                            role_flip = "resistance_to_support"

                    level_entry = {
                        "chain_id": cid,
                        "level_index": n.get("level_index", None),
                        "level_kind": lk,
                        "level_price": lp,
                        "level_outcome": n.get("level_outcome", None),
                        "watch_start_ts_ms": ws,
                        "watch_end_ts_ms": we,
                        "band_pct": float(n.get("band_pct", band_pct)),
                        "between": between,
                        "countdown": countdown,
                        "checklist": checklist,
                        "move_from_prev": move_from_prev,
                        "session_wide": session_wide,
                        "zone_analysis": zone_analysis,
                        "zone_segments": zone_segments,
                        "zone_signals": zone_signals,
                        "risk_reward": risk_reward,
                        "zone_verdict": zone_verdict,
                        "trade_trigger": trade_trigger,
                        "role_behavior": role_behavior,
                        "interaction_timeline": interaction_timeline,
                        "timeline_summary": timeline_summary,
                        "role_flip": role_flip,
                        "level_type": level_type_n,
                        "direction_bias": n.get("direction_bias", "long"),
                        "indicator_col": ind_col_name,  # NEW: for dynamic levels
                        # DS/AS/DP State Machine
                        "defense_score": defense_score,
                        "aggression_score": aggression_score,
                        "distribution_pattern": distribution_pattern,
                        "market_state": market_state,
                        "late_momentum": late_momentum,
                    }
                    # Add indicator info if applicable
                    if level_type_n == "indicator" and indicator_name:
                        level_entry["indicator_name"] = indicator_name
                    
                    chain_levels.append(level_entry)
                    prev = n

                # Compute Touch Packets for each level in chain
                all_touch_packets: list[dict[str, Any]] = []
                for lv in chain_levels:
                    tl = lv.get("interaction_timeline") or []
                    pkts = self._compute_touch_packets(
                        df=stream,
                        level_price=float(lv.get("level_price", 0)),
                        level_kind=lv.get("level_kind"),
                        band_pct=float(lv.get("band_pct", band_pct)),
                        interval_s=interval_s,
                        timeline=tl,
                        indicator_col=lv.get("indicator_col"),  # Pass indicator column
                    )
                    lv["touch_packets"] = pkts
                    all_touch_packets.extend(pkts)

                    # Touch scoring (support-long V1 for now)
                    # Uses avg_shares_per_trade from session_wide band stats as a liquidity guard.
                    sw = lv.get("session_wide") or {}
                    band_vol = float(sw.get("band_total_volume_sum") or 0)
                    band_trades = float(sw.get("band_trade_count_sum") or 0)
                    avg_shares_per_trade = (band_vol / band_trades) if band_trades > 0 else None
                    if str(lv.get("level_kind") or "").lower() == "support":
                        lv["touch_scoring"] = self._score_touch_packets_support_long(
                            pkts, avg_shares_per_trade=avg_shares_per_trade
                        )
                    else:
                        lv["touch_scoring"] = {
                            "schema": "touch_scoring_v1",
                            "mode": "unsupported",
                            "touches": len(pkts),
                            "verdict": "N/A",
                            "notes": ["only support-long scoring implemented"],
                        }

                    # Compute level stats and flags from touch packets
                    lv["level_stats"] = self._compute_level_stats(pkts)
                    lv["level_flags"] = self._compute_level_flags(lv["level_stats"])

                # Compute Reclaim Packets for the chain
                reclaim_packets = self._compute_reclaim_packets(
                    df=stream,
                    levels=chain_levels,
                    band_pct=band_pct,
                    interval_s=interval_s,
                )

                # Compute Main Support Score
                main_support = self._compute_main_support_score(
                    levels=chain_levels,
                    reclaim_packets=reclaim_packets,
                )

                # Add gap_to_next_support_atr and support_cluster_flag
                sorted_prices = sorted([float(lv.get("level_price", 0)) for lv in chain_levels], reverse=True)
                for lv in chain_levels:
                    lp = float(lv.get("level_price", 0))
                    idx = sorted_prices.index(lp) if lp in sorted_prices else -1
                    if idx >= 0 and idx < len(sorted_prices) - 1:
                        next_price = sorted_prices[idx + 1]
                        atr_mean = (lv.get("session_wide") or {}).get("atr_value_mean")
                        if atr_mean and atr_mean > 0:
                            gap = abs(lp - next_price) / atr_mean
                            lv["gap_to_next_support_atr"] = float(gap)
                            lv["support_cluster_flag"] = gap < 0.5
                        else:
                            lv["gap_to_next_support_atr"] = None
                            lv["support_cluster_flag"] = None
                    else:
                        lv["gap_to_next_support_atr"] = None
                        lv["support_cluster_flag"] = None

                    # Add historical hold rate
                    tl_sum = lv.get("timeline_summary") or {}
                    touches = int(tl_sum.get("touches", 0) or 0)
                    rejects = int(tl_sum.get("rejects", 0) or 0)
                    if touches > 0:
                        lv["historical_hold_rate"] = float(rejects) / float(touches)
                    else:
                        lv["historical_hold_rate"] = None

                    # Add focus_returns_to_this flag
                    lv["focus_returns_to_this"] = str(lp) == main_support.get("focus_level")

                level_chains.append(
                    {
                        "schema": "level_chain_v3",
                        "chain_id": cid,
                        "levels": chain_levels,
                        "touch_packets": all_touch_packets,
                        "reclaim_packets": reclaim_packets,
                        "main_support": main_support,
                        "positive_touch_ledger": self._compute_positive_touch_ledger(
                            chain_levels=chain_levels, chain_id=cid
                        ),
                        "interpretation_guide": self._get_interpretation_guide(),
                        "note": "Move segments computed automatically. Touch/Reclaim packets + level_stats per Addendum K.",
                    }
                )

        report["watch_windows"] = watch_windows
        report["event_markers"] = event_markers
        report["level_chains"] = level_chains
        report["watch_note"] = "To use watch windows: set marker end time to the move-start time; tag a support/resistance price; band_pct defaults to 1.5%."

        # Add unified analysis to each level chain
        for chain in level_chains:
            for lv in chain.get("levels", []):
                tps = lv.get("touch_packets", [])
                if tps:
                    unified = self._compute_unified_analysis(tps, lv.get("session_wide", {}), stream)
                    lv["unified_analysis"] = unified

        return report

    def _compute_unified_analysis(self, tps: list, session_wide: dict, stream: pd.DataFrame = None) -> dict:
        """Compute unified analysis with phases, trend scoring, and verdict."""
        if not tps:
            return {"error": "No touch data"}
        
        # Overall balance
        total_inst_buy = sum(1 for tp in tps if tp.get("institutional_buying_flag"))
        total_inst_sell = sum(1 for tp in tps if tp.get("institutional_selling_flag"))
        inst_buy_vol = sum(tp.get("large_buy_volume_30s", 0) or 0 for tp in tps if tp.get("institutional_buying_flag"))
        inst_sell_vol = sum(tp.get("large_sell_volume_30s", 0) or 0 for tp in tps if tp.get("institutional_selling_flag"))
        
        # Absorption rate
        absorbed = sum(1 for tp in tps if tp.get("institutional_selling_flag") and (tp.get("bounce_return_30s_pct", 0) or 0) > 0)
        confirmed = sum(1 for tp in tps if tp.get("institutional_selling_flag") and (tp.get("bounce_return_30s_pct", 0) or 0) <= 0)
        absorption_rate = absorbed / total_inst_sell if total_inst_sell > 0 else 0
        
        # Minute-by-minute breakdown - use FULL MINUTE data from session_stream
        from collections import defaultdict
        from datetime import datetime, timezone
        import pytz
        
        def format_ts_chicago(ts_ms: int) -> str:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            chicago = pytz.timezone("America/Chicago")
            return dt.astimezone(chicago).strftime("%H:%M")
        
        # First, count touches and INST flags per minute from touch packets
        minute_data = defaultdict(lambda: {"touches": 0, "buy_vol": 0, "sell_vol": 0, "inst_buy": 0, "inst_sell": 0})
        
        for tp in tps:
            ts = tp.get("touch_ts_ms") or tp.get("ts_ms", 0)
            if ts:
                minute_key = format_ts_chicago(ts)
                minute_data[minute_key]["touches"] += 1
                if tp.get("institutional_buying_flag"):
                    minute_data[minute_key]["inst_buy"] += 1
                if tp.get("institutional_selling_flag"):
                    minute_data[minute_key]["inst_sell"] += 1
        
        # Now get FULL MINUTE buy/sell volumes from session_stream
        if stream is not None and not stream.empty and "timestamp" in stream.columns:
            chicago = pytz.timezone("America/Chicago")
            stream_copy = stream.copy()
            stream_copy["minute"] = pd.to_datetime(stream_copy["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/Chicago").dt.strftime("%H:%M")
            
            # Sum full minute data for minutes that have touches
            for minute_key in list(minute_data.keys()):
                minute_rows = stream_copy[stream_copy["minute"] == minute_key]
                if not minute_rows.empty:
                    # Use ALL buy/sell volume, not just large (institutional)
                    minute_data[minute_key]["buy_vol"] = float(minute_rows["buy_volume"].fillna(0).sum())
                    minute_data[minute_key]["sell_vol"] = float(minute_rows["sell_volume"].fillna(0).sum())
        
        minute_breakdown = []
        minute_progression = []
        for minute in sorted(minute_data.keys()):
            m = minute_data[minute]
            delta = m["buy_vol"] - m["sell_vol"]
            
            if m["buy_vol"] > m["sell_vol"] * 1.1:
                winner = "BUYERS"
                minute_progression.append("+")
            elif m["sell_vol"] > m["buy_vol"] * 1.1:
                winner = "SELLERS"
                minute_progression.append("-")
            else:
                winner = "EVEN"
                minute_progression.append("=")
            
            minute_breakdown.append({
                "minute": minute,
                "touches": m["touches"],
                "buy_vol": m["buy_vol"],
                "sell_vol": m["sell_vol"],
                "delta": delta,
                "inst_buy": m["inst_buy"],
                "inst_sell": m["inst_sell"],
                "winner": winner,
                "winner_symbol": "+" if winner == "BUYERS" else "-" if winner == "SELLERS" else "="
            })
        
        # === WASH ENGINE (Consolidated Absorption + Sequence Analysis) ===
        # Detects dump events and evaluates recovery quality with scoring
        
        sorted_minutes = sorted(minute_breakdown, key=lambda x: x["minute"])
        
        wash_engine = {
            "dumps": [],
            "signals": [],
            "score": 0,
            "hard_stop": None,
            "verdict_modifier": None
        }
        
        # Helper functions
        def _is_dump(m: dict) -> bool:
            return m["sell_vol"] > m["buy_vol"] * 2 and m["sell_vol"] > 20000
        
        def _is_seller_dominant(m: dict) -> bool:
            return m["sell_vol"] > m["buy_vol"] * 1.2
        
        def _is_buyer_dominant(m: dict) -> bool:
            return m["buy_vol"] > m["sell_vol"] * 1.2
        
        # Scan for dumps and evaluate recovery
        for i, m in enumerate(sorted_minutes):
            if not _is_dump(m):
                continue
            
            dump = {
                "minute": m["minute"],
                "sell_vol": m["sell_vol"],
                "buy_vol": m["buy_vol"],
                "inst_sell": m["inst_sell"],
                "recovery_minutes": [],
                "recovery_buy": 0,
                "recovery_sell": 0,
                "signals": [],
                "score": 0,
                "hard_stop": False,
                "absorbed": False
            }
            
            # === HARD STOP A: Breakdown continues ===
            # If dump AND next minute is also seller-dominant = no bounce
            if i + 1 < len(sorted_minutes):
                next_m = sorted_minutes[i + 1]
                if _is_seller_dominant(next_m):
                    dump["hard_stop"] = True
                    dump["signals"].append("HARD_STOP_A: breakdown continues (next minute seller-dominant)")
                    wash_engine["hard_stop"] = "BREAKDOWN_CONTINUES"
            
            # Look at recovery in next 5 minutes
            recovery_mins = []
            prev_buy = m["buy_vol"]
            accel_count = 0
            
            for j in range(i + 1, min(i + 6, len(sorted_minutes))):
                rm = sorted_minutes[j]
                recovery_mins.append(rm)
                dump["recovery_buy"] += rm["buy_vol"]
                dump["recovery_sell"] += rm["sell_vol"]
                dump["recovery_minutes"].append(rm["minute"])
                
                # Track acceleration (buy increasing)
                if rm["buy_vol"] > prev_buy * 1.1:
                    accel_count += 1
                prev_buy = rm["buy_vol"]
            
            # === HARD STOP B: No buyers at all ===
            # If next 3 min combined buy < 30% of dump
            if dump["recovery_buy"] < dump["sell_vol"] * 0.30:
                dump["hard_stop"] = True
                dump["signals"].append("HARD_STOP_B: no buyers (recovery < 30% of dump)")
                wash_engine["hard_stop"] = "NO_BUYERS"
            
            # === SOFT SIGNALS (only if no hard stop) ===
            if not dump["hard_stop"]:
                dump_vol = dump["sell_vol"]
                rec_buy = dump["recovery_buy"]
                rec_ratio = (rec_buy / dump_vol) if dump_vol > 0 else 0
                
                # S1: Recovery ratio
                if rec_ratio >= 0.75:
                    dump["score"] += 2
                    dump["signals"].append(f"S1_STRONG: recovery {rec_ratio:.0%} >= 75%")
                elif rec_ratio >= 0.50:
                    dump["score"] += 1
                    dump["signals"].append(f"S1_DECENT: recovery {rec_ratio:.0%} >= 50%")
                
                # S2: Buyer acceleration
                if accel_count >= 2:
                    dump["score"] += 2
                    dump["signals"].append(f"S2_STRONG: {accel_count}x acceleration")
                elif accel_count >= 1:
                    dump["score"] += 1
                    dump["signals"].append(f"S2_WEAK: {accel_count}x acceleration")
                
                # S3: INST confirmation in recovery
                late_inst_buy = sum(rm["inst_buy"] for rm in recovery_mins)
                late_inst_sell = sum(rm["inst_sell"] for rm in recovery_mins)
                
                if late_inst_buy >= 2:
                    dump["score"] += 2
                    dump["signals"].append(f"S3_STRONG: {late_inst_buy} INST_BUY in recovery")
                elif late_inst_buy >= 1:
                    dump["score"] += 1
                    dump["signals"].append(f"S3_WEAK: {late_inst_buy} INST_BUY in recovery")
                
                # S4: INST Shift (INST_SELL in dump → INST_BUY in recovery)
                if m["inst_sell"] >= 1 and late_inst_buy >= 1 and late_inst_sell == 0:
                    dump["score"] += 2
                    dump["signals"].append("S4_INST_SHIFT: INST flipped from sell to buy")
                
                # Mark as absorbed if score >= 2
                dump["absorbed"] = dump["score"] >= 2
            
            wash_engine["dumps"].append(dump)
            wash_engine["score"] += dump["score"]
        
        # === LAST MINUTE BIAS (S5) ===
        if sorted_minutes:
            last_m = sorted_minutes[-1]
            if _is_buyer_dominant(last_m):
                wash_engine["score"] += 1
                wash_engine["signals"].append(f"S5_LAST_MIN: final minute buyers won (+{last_m['buy_vol'] - last_m['sell_vol']:,.0f})")
            if last_m["inst_buy"] >= 1:
                wash_engine["score"] += 1
                wash_engine["signals"].append(f"S5_LAST_INST: INST_BUY in final minute")
        
        # === BID/ASK IMBALANCE (S6) - from session_wide if available ===
        # Check if we have bid/ask data in touch packets
        late_tps = tps[-3:] if len(tps) >= 3 else tps
        bid_sum = sum(tp.get("bid", 0) or 0 for tp in late_tps)
        ask_sum = sum(tp.get("ask", 0) or 0 for tp in late_tps)
        if bid_sum > 0 and ask_sum > 0:
            # We have quote data - check spread behavior
            avg_spread_pct = sum(tp.get("spread_pct", 0) or 0 for tp in late_tps) / len(late_tps)
            if avg_spread_pct < 0.005:  # Tight spread < 0.5%
                wash_engine["score"] += 1
                wash_engine["signals"].append("S6_TIGHT_SPREAD: spread tightening in late minutes")
        
        # === COMPUTE WASH VERDICT ===
        absorbed_count = sum(1 for d in wash_engine["dumps"] if d.get("absorbed"))
        failed_dumps = sum(1 for d in wash_engine["dumps"] if not d.get("absorbed") and not d.get("hard_stop"))
        hard_stopped = any(d.get("hard_stop") for d in wash_engine["dumps"])
        
        if hard_stopped:
            wash_engine["verdict_modifier"] = "HARD_STOP"
        elif wash_engine["score"] >= 5:
            wash_engine["verdict_modifier"] = "STRONG_WASH"  # → TAKE
        elif wash_engine["score"] >= 3:
            wash_engine["verdict_modifier"] = "DECENT_WASH"  # → LEAN
        elif wash_engine["score"] >= 1:
            wash_engine["verdict_modifier"] = "WEAK_WASH"    # → WAIT
        else:
            wash_engine["verdict_modifier"] = "NO_WASH"
        
        wash_engine["absorbed_count"] = absorbed_count
        wash_engine["failed_dumps"] = failed_dumps
        
        # Phase analysis - split touches into 4 phases by touch order
        n = len(tps)
        phase_size = max(1, n // 4)
        phases = []
        for i in range(0, n, phase_size):
            end_idx = min(i + phase_size, n)
            phase_tps = tps[i:end_idx]
            
            buy_vol = sum(tp.get("large_buy_volume_30s", 0) or 0 for tp in phase_tps)
            sell_vol = sum(tp.get("large_sell_volume_30s", 0) or 0 for tp in phase_tps)
            delta = buy_vol - sell_vol
            inst_buy = sum(1 for tp in phase_tps if tp.get("institutional_buying_flag"))
            inst_sell = sum(1 for tp in phase_tps if tp.get("institutional_selling_flag"))
            
            winner = "BUYERS" if delta > 0 else "SELLERS" if delta < 0 else "CONTESTED"
            
            first_touch = phase_tps[0].get("touch_number", i + 1) if phase_tps else i + 1
            last_touch = phase_tps[-1].get("touch_number", end_idx) if phase_tps else end_idx
            
            phases.append({
                "phase_num": len(phases) + 1,
                "first_touch": first_touch,
                "last_touch": last_touch,
                "touch_count": len(phase_tps),
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "delta": delta,
                "inst_buy": inst_buy,
                "inst_sell": inst_sell,
                "winner": winner,
                "winner_symbol": "+" if winner == "BUYERS" else "-" if winner == "SELLERS" else "="
            })
        
        # === PATTERN LIBRARY (~50 patterns) ===
        # Each pattern is a simple yes/no check with a clear +/- score
        
        pattern_score = 0
        matched_patterns = []
        hard_stop = None
        
        # Helper: count phase winners
        n_phases = len(phases)
        buyers_won = [p for p in phases if p["winner"] == "BUYERS"]
        sellers_won = [p for p in phases if p["winner"] == "SELLERS"]
        last_3 = phases[-3:] if n_phases >= 3 else phases
        last_4 = phases[-4:] if n_phases >= 4 else phases
        first_phase = phases[0] if phases else None
        last_phase = phases[-1] if phases else None
        
        # === HARD STOPS (instant AVOID) ===
        # Check wash engine for hard stops
        if wash_engine.get("hard_stop"):
            hard_stop = wash_engine["hard_stop"]
        
        # INST_SELL_DOMINATES: INST_SELL vol > 3x INST_BUY vol
        if inst_sell_vol > inst_buy_vol * 3 and inst_sell_vol > 5000:
            hard_stop = "INST_SELL_DOMINATES"
        
        # === PHASE WINNERS ===
        # Buyers won 2 of last 3 phases
        if len([p for p in last_3 if p["winner"] == "BUYERS"]) >= 2:
            pattern_score += 1
            matched_patterns.append("+1 Buyers won 2 of last 3 phases")
        
        # Buyers won 3 of last 4 phases
        if len([p for p in last_4 if p["winner"] == "BUYERS"]) >= 3:
            pattern_score += 2
            matched_patterns.append("+2 Buyers won 3 of last 4 phases")
        
        # Buyers won ALL phases
        if n_phases >= 2 and len(buyers_won) == n_phases:
            pattern_score += 3
            matched_patterns.append("+3 Buyers won ALL phases")
        
        # Sellers won 2 of last 3 phases
        if len([p for p in last_3 if p["winner"] == "SELLERS"]) >= 2:
            pattern_score -= 1
            matched_patterns.append("-1 Sellers won 2 of last 3 phases")
        
        # Sellers won 3 of last 4 phases
        if len([p for p in last_4 if p["winner"] == "SELLERS"]) >= 3:
            pattern_score -= 2
            matched_patterns.append("-2 Sellers won 3 of last 4 phases")
        
        # Last phase = BUYERS
        if last_phase and last_phase["winner"] == "BUYERS":
            pattern_score += 1
            matched_patterns.append("+1 Last phase = BUYERS")
        
        # Last phase = SELLERS
        if last_phase and last_phase["winner"] == "SELLERS":
            pattern_score -= 1
            matched_patterns.append("-1 Last phase = SELLERS")
        
        # Started SELLERS, ended BUYERS (Turnaround)
        if first_phase and last_phase and first_phase["winner"] == "SELLERS" and last_phase["winner"] == "BUYERS":
            pattern_score += 2
            matched_patterns.append("+2 Turnaround: started SELLERS, ended BUYERS")
        
        # Started BUYERS, ended SELLERS (Breakdown)
        if first_phase and last_phase and first_phase["winner"] == "BUYERS" and last_phase["winner"] == "SELLERS":
            pattern_score -= 2
            matched_patterns.append("-2 Breakdown: started BUYERS, ended SELLERS")
        
        # === INST ACTIVITY ===
        # INST_BUY vol > INST_SELL vol
        if inst_buy_vol > inst_sell_vol:
            pattern_score += 1
            matched_patterns.append(f"+1 INST_BUY vol > INST_SELL vol ({inst_buy_vol:,.0f} vs {inst_sell_vol:,.0f})")
        
        # INST_BUY vol > 2x INST_SELL vol
        if inst_buy_vol > inst_sell_vol * 2:
            pattern_score += 2
            matched_patterns.append("+2 INST_BUY vol > 2x INST_SELL vol")
        
        # INST_SELL vol > INST_BUY vol
        if inst_sell_vol > inst_buy_vol:
            pattern_score -= 1
            matched_patterns.append(f"-1 INST_SELL vol > INST_BUY vol ({inst_sell_vol:,.0f} vs {inst_buy_vol:,.0f})")
        
        # INST_SELL vol > 2x INST_BUY vol
        if inst_sell_vol > inst_buy_vol * 2:
            pattern_score -= 2
            matched_patterns.append("-2 INST_SELL vol > 2x INST_BUY vol")
        
        # INST_BUY in last 2 phases
        late_2 = phases[-2:] if n_phases >= 2 else phases
        late_inst_buy = sum(p["inst_buy"] for p in late_2)
        late_inst_sell = sum(p["inst_sell"] for p in late_2)
        early_2 = phases[:2] if n_phases >= 2 else phases
        early_inst_buy = sum(p["inst_buy"] for p in early_2)
        early_inst_sell = sum(p["inst_sell"] for p in early_2)
        
        if late_inst_buy >= 1:
            pattern_score += 1
            matched_patterns.append(f"+1 INST_BUY in last 2 phases ({late_inst_buy})")
        
        if late_inst_sell >= 1:
            pattern_score -= 1
            matched_patterns.append(f"-1 INST_SELL in last 2 phases ({late_inst_sell})")
        
        # INST shift: SELL→BUY (early INST_SELL, late INST_BUY, no late INST_SELL)
        if early_inst_sell >= 1 and late_inst_buy >= 1 and late_inst_sell == 0:
            pattern_score += 2
            matched_patterns.append("+2 INST shift: SELL→BUY (flip bullish)")
        
        # INST shift: BUY→SELL
        if early_inst_buy >= 1 and late_inst_sell >= 1 and late_inst_buy == 0:
            pattern_score -= 2
            matched_patterns.append("-2 INST shift: BUY→SELL (flip bearish)")
        
        # Avg buy size > avg sell size
        avg_inst_buy_size = (inst_buy_vol / total_inst_buy) if total_inst_buy > 0 else 0
        avg_inst_sell_size = (inst_sell_vol / total_inst_sell) if total_inst_sell > 0 else 0
        
        if avg_inst_buy_size > avg_inst_sell_size * 1.5 and total_inst_buy >= 1:
            pattern_score += 1
            matched_patterns.append(f"+1 Bigger buyers (avg {avg_inst_buy_size:,.0f} vs {avg_inst_sell_size:,.0f})")
        
        if avg_inst_sell_size > avg_inst_buy_size * 1.5 and total_inst_sell >= 1:
            pattern_score -= 1
            matched_patterns.append(f"-1 Bigger sellers (avg {avg_inst_sell_size:,.0f} vs {avg_inst_buy_size:,.0f})")
        
        # 2+ INST_BUY back-to-back
        for i in range(1, len(sorted_minutes)):
            if sorted_minutes[i]["inst_buy"] >= 1 and sorted_minutes[i-1]["inst_buy"] >= 1:
                pattern_score += 2
                matched_patterns.append("+2 Back-to-back INST_BUY (stacking conviction)")
                break
        
        # 2+ INST_SELL back-to-back
        for i in range(1, len(sorted_minutes)):
            if sorted_minutes[i]["inst_sell"] >= 1 and sorted_minutes[i-1]["inst_sell"] >= 1:
                pattern_score -= 2
                matched_patterns.append("-2 Back-to-back INST_SELL (stacking distribution)")
                break
        
        # === MINUTE FLOW ===
        if sorted_minutes:
            last_min = sorted_minutes[-1]
            last_2_mins = sorted_minutes[-2:] if len(sorted_minutes) >= 2 else sorted_minutes
            
            # Last minute = BUYERS
            if last_min["buy_vol"] > last_min["sell_vol"] * 1.2:
                pattern_score += 1
                matched_patterns.append("+1 Last minute = BUYERS")
            
            # Last 2 minutes = BUYERS
            last_2_buy = sum(m["buy_vol"] for m in last_2_mins)
            last_2_sell = sum(m["sell_vol"] for m in last_2_mins)
            if last_2_buy > last_2_sell * 1.2 and len(last_2_mins) >= 2:
                pattern_score += 2
                matched_patterns.append("+2 Last 2 minutes = BUYERS (momentum close)")
            
            # Last minute = SELLERS
            if last_min["sell_vol"] > last_min["buy_vol"] * 1.2:
                pattern_score -= 1
                matched_patterns.append("-1 Last minute = SELLERS")
            
            # Ended with INST_BUY
            if last_min["inst_buy"] >= 1:
                pattern_score += 1
                matched_patterns.append("+1 Ended with INST_BUY")
            
            # Ended with INST_SELL
            if last_min["inst_sell"] >= 1:
                pattern_score -= 1
                matched_patterns.append("-1 Ended with INST_SELL")
            
            # Buy volume accelerating
            if len(sorted_minutes) >= 3:
                buys = [m["buy_vol"] for m in sorted_minutes[-3:]]
                if buys[1] > buys[0] and buys[2] > buys[1]:
                    pattern_score += 1
                    matched_patterns.append("+1 Buy volume accelerating")
            
            # Sell volume accelerating
            if len(sorted_minutes) >= 3:
                sells = [m["sell_vol"] for m in sorted_minutes[-3:]]
                if sells[1] > sells[0] and sells[2] > sells[1]:
                    pattern_score -= 1
                    matched_patterns.append("-1 Sell volume accelerating")
        
        # === VOLUME BALANCE ===
        total_buy_vol = sum(p["buy_vol"] for p in phases)
        total_sell_vol = sum(p["sell_vol"] for p in phases)
        
        if total_buy_vol > total_sell_vol:
            pattern_score += 1
            matched_patterns.append(f"+1 Total buy > sell ({total_buy_vol:,.0f} vs {total_sell_vol:,.0f})")
        
        if total_sell_vol > total_buy_vol:
            pattern_score -= 1
            matched_patterns.append(f"-1 Total sell > buy ({total_sell_vol:,.0f} vs {total_buy_vol:,.0f})")
        
        # === DELTA FLIP ===
        if n_phases >= 2:
            early_delta = sum(p["delta"] for p in phases[:n_phases//2])
            late_delta = sum(p["delta"] for p in phases[n_phases//2:])
            
            # Delta flipped - to + (Recovery)
            if early_delta < 0 and late_delta > 0:
                pattern_score += 2
                matched_patterns.append("+2 Delta flipped - to + (Recovery)")
            
            # Delta flipped + to - (Breakdown)
            if early_delta > 0 and late_delta < 0:
                pattern_score -= 2
                matched_patterns.append("-2 Delta flipped + to - (Breakdown)")
        
        # === PROGRESSION SHAPE ===
        if n_phases >= 2:
            progression = "".join(p["winner_symbol"] for p in phases)
            
            # Ends [+][+]
            if progression.endswith("++"):
                pattern_score += 2
                matched_patterns.append("+2 Ends [+][+] (momentum up)")
            
            # Ends [-][-]
            if progression.endswith("--"):
                pattern_score -= 2
                matched_patterns.append("-2 Ends [-][-] (momentum down)")
            
            # Pattern [-] then [+][+] (reversal)
            if "-++" in progression:
                pattern_score += 2
                matched_patterns.append("+2 Pattern [-]→[+][+] (reversal)")
            
            # Pattern [+] then [-][-] (breakdown)
            if "+--" in progression:
                pattern_score -= 2
                matched_patterns.append("-2 Pattern [+]→[-][-] (breakdown)")
        
        # === ABSORPTION PATTERNS (from wash engine) ===
        if wash_engine.get("absorbed_count", 0) >= 1:
            pattern_score += 2
            matched_patterns.append(f"+2 Dump absorbed ({wash_engine['absorbed_count']}x)")
        
        if wash_engine.get("absorbed_count", 0) >= 2:
            pattern_score += 3
            matched_patterns.append("+3 Double absorption (very strong)")
        
        if wash_engine.get("failed_dumps", 0) >= 1:
            pattern_score -= 2
            matched_patterns.append(f"-2 Failed dump ({wash_engine['failed_dumps']}x)")
        
        # === VERDICT FROM PATTERN SCORE ===
        if hard_stop:
            direction = "SHORT"
            confidence = "AVOID"
            matched_patterns.insert(0, f"⛔ HARD STOP: {hard_stop}")
        elif pattern_score >= 8:
            direction = "LONG"
            confidence = "TAKE"
        elif pattern_score >= 4:
            direction = "LONG"
            confidence = "LEAN"
        elif pattern_score >= 1:
            direction = "LONG"
            confidence = "WAIT"
        elif pattern_score >= -3:
            direction = "NEUTRAL"
            confidence = "WAIT"
        elif pattern_score >= -7:
            direction = "SHORT"
            confidence = "LEAN"
        else:
            direction = "SHORT"
            confidence = "AVOID"
        
        # Legacy compatibility
        trend_score = pattern_score
        trend_signals = matched_patterns
        caution_reasons = []
        early_score = 0.0
        mid_score = 0.0
        late_score = float(pattern_score)
        
        # Note: Hard stops and confidence already set by pattern library above
        
        # Progression string
        progression = "".join(f"[{p['winner_symbol']}]" for p in phases)
        
        return {
            "overall": {
                "inst_buy_trades": total_inst_buy,
                "inst_sell_trades": total_inst_sell,
                "inst_buy_volume": inst_buy_vol,
                "inst_sell_volume": inst_sell_vol,
                "absorption_rate": absorption_rate,
                "absorbed_count": absorbed,
                "confirmed_count": confirmed
            },
            "minute_breakdown": minute_breakdown,
            "minute_progression": "".join(f"[{s}]" for s in minute_progression),
            "wash_engine": wash_engine,
            "phases": phases,
            "progression": progression,
            "trend_score": trend_score,
            "trend_signals": trend_signals,
            "weighted_scores": {
                "early": early_score,
                "mid": mid_score,
                "late": late_score,
                "total": early_score + mid_score + late_score
            },
            "verdict": {
                "direction": direction,
                "confidence": confidence,
                "cautions": caution_reasons,
                "pattern_score": pattern_score,
                "hard_stop": hard_stop
            }
        }

    # ============ LIVE ANALYSIS ============

    def compute_live_analysis(
        self,
        *,
        session_df: pd.DataFrame,
        level_price: float,
        level_kind: str,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Compute analysis for live dashboard from a session stream DataFrame.
        Returns a level_data dict compatible with the live.html template.
        """
        if session_df.empty:
            return {}

        # Use the full time range of the session data
        ts_col = "timestamp" if "timestamp" in session_df.columns else "ts_ms"
        start_ms = int(session_df[ts_col].min())
        end_ms = int(session_df[ts_col].max())

        # Compute zone analysis
        zone_analysis = self._compute_zone_metrics(
            df=session_df,
            level_price=level_price,
            start_ms=start_ms,
            end_ms=end_ms,
            above_pct=self._DEFAULT_ZONE_ABOVE_PCT,
            below_pct=self._DEFAULT_ZONE_BELOW_PCT,
            on_pct=self._DEFAULT_ZONE_ON_PCT,
            on_pct_down=self._DEFAULT_ZONE_ON_PCT_DOWN,
            interval_s=self._DEFAULT_SNAPSHOT_INTERVAL_S,
        )

        # Compute zone segments
        zone_segments = self._compute_zone_segments(
            df=session_df,
            level_price=level_price,
            start_ms=start_ms,
            end_ms=end_ms,
            above_pct=self._DEFAULT_ZONE_ABOVE_PCT,
            below_pct=self._DEFAULT_ZONE_BELOW_PCT,
            on_pct=self._DEFAULT_ZONE_ON_PCT,
            on_pct_down=self._DEFAULT_ZONE_ON_PCT_DOWN,
            interval_s=self._DEFAULT_SNAPSHOT_INTERVAL_S,
        )
        zone_segments = self._annotate_zone_segments(
            segments=zone_segments, zone_analysis=zone_analysis
        )

        # Defense/Aggression scoring
        defense_score = self._compute_defense_score(
            segments=zone_segments,
            zone_analysis=zone_analysis,
        )
        aggression_score = self._compute_aggression_score(
            segments=zone_segments,
            zone_analysis=zone_analysis,
        )
        distribution_pattern = self._detect_distribution_pattern(
            segments=zone_segments,
        )
        late_momentum = self._compute_late_momentum(
            segments=zone_segments,
            n_segments=5,
        )
        market_state = self._compute_market_state(
            defense_score=defense_score,
            aggression_score=aggression_score,
            distribution_pattern=distribution_pattern,
            zone_analysis=zone_analysis,
            segments=zone_segments,
            late_momentum=late_momentum,
        )

        return {
            "level_price": level_price,
            "level_kind": level_kind,
            "ticker": ticker,
            "zone_analysis": zone_analysis,
            "zone_segments": zone_segments,
            "defense_score": defense_score,
            "aggression_score": aggression_score,
            "distribution_pattern": distribution_pattern,
            "late_momentum": late_momentum,
            "market_state": market_state,
        }
