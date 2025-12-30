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


@dataclass(frozen=True, slots=True)
class Store:
    root_dir: Path

    _DEFAULT_BAND_PCT: float = 0.0015  # 0.15% band for touch detection
    _DEFAULT_SNAPSHOT_INTERVAL_S: int = 10

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
                        "interaction_timeline": interaction_timeline,
                        "timeline_summary": timeline_summary,
                        "role_flip": role_flip,
                        "level_type": level_type_n,
                        "direction_bias": n.get("direction_bias", "long"),
                        "indicator_col": ind_col_name,  # NEW: for dynamic levels
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


