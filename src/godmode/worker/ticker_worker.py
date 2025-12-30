from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.config import AppConfig
from godmode.core.enums import ATRSeedSource, ATRStatus, DirectionBias, EpisodePhase
from godmode.core.models import Episode, Level, Quote, Snapshot, Trade
from godmode.engine.atr import ATRCalculator
from godmode.engine.bar_builder import BarBuilder
from godmode.engine.ema import EMACalculator
from godmode.engine.ring_buffer import TimeRingBuffer
from godmode.engine.vwap import VWAPSessionCalculator
from godmode.features.ema_structure import EMAStructureFeatureEngine
from godmode.features.level_spine import LevelSpineFeatureEngine
from godmode.features.microstructure import MicrostructureFeatureEngine
from godmode.features.orderflow import OrderFlowFeatureEngine
from godmode.features.smart_money import SmartMoneyFeatureEngine
from godmode.features.volatility import VolatilityFeatureEngine
from godmode.episode.state_machine import EpisodeStateMachine
from godmode.episode.snapshot_builder import LeeReadyClassifier
from godmode.zone.zone_gate import ZoneGate, ZoneSnapshot
from godmode.worker.session_writer import SessionWriter
from godmode.core.markers import Marker, MarkerType
from godmode.episode.resolution import evaluate_resolution
from godmode.episode.metrics import apply_outcome_and_metrics


@dataclass(slots=True)
class _EpisodeCtx:
    episode: Episode
    level: Level
    snapshots: list[Snapshot]
    spine: LevelSpineFeatureEngine
    tf_rows: list[dict]
    tf_history_by_tf: dict[str, list[dict]]


class TickerWorker:
    """
    Minimal per-ticker worker (Day 14 wiring) that supports replay end-to-end:
    - Maintains trade/quote ring buffers (event-time)
    - Produces deterministic 10s ticker snapshots
    - Runs ZoneGate + EpisodeStateMachine
    - Builds per-episode snapshots by merging ticker snapshot + per-level zone snapshot,
      then applies LevelSpine features (level-specific)
    - Persists episodes/snapshots via SessionWriter
    """

    def __init__(
        self,
        *,
        ticker: str,
        session_id: str,
        config: AppConfig,
        levels: list[Level],
        direction_bias: DirectionBias,
        record_session_stream: bool = True,
    ) -> None:
        self._ticker = ticker
        self._session_id = session_id
        self._cfg = config
        self._levels: list[Level] = list(levels)
        self._direction_bias = direction_bias
        self._record_session_stream = bool(record_session_stream)

        self._interval_ms = int(config.snapshot.interval_seconds) * 1000

        # Buffers (5 min)
        max_age_ms = 5 * 60 * 1000
        self._trades = TimeRingBuffer[Trade](max_age_ms=max_age_ms)
        self._quotes = TimeRingBuffer[Quote](max_age_ms=max_age_ms)
        self._ticker_snaps = TimeRingBuffer[Snapshot](max_age_ms=max_age_ms)

        # Zone snapshots per level id (for baseline backfill)
        self._zone_snaps: dict[str, TimeRingBuffer[ZoneSnapshot]] = {
            lvl.level_id: TimeRingBuffer[ZoneSnapshot](max_age_ms=max_age_ms) for lvl in levels
        }

        # Indicators
        # Bar builders: 1m is used for ATR; 30s/2m/3m for multi-timeframe analysis.
        self._bars_1m = BarBuilder(symbol=ticker, bucket_seconds=60)
        self._bars_30s = BarBuilder(symbol=ticker, bucket_seconds=30)
        self._bars_2m = BarBuilder(symbol=ticker, bucket_seconds=120)
        self._bars_3m = BarBuilder(symbol=ticker, bucket_seconds=180)
        self._atr = ATRCalculator(period_bars=config.atr.period_bars, atr_blend_alpha=config.atr.atr_blend_alpha)
        # EMA calculators for 1m timeframe
        self._ema9_1m = EMACalculator(period=9)
        self._ema20_1m = EMACalculator(period=20)
        self._ema30_1m = EMACalculator(period=30)
        self._ema200_1m = EMACalculator(period=200)

        # EMA calculators for 30s timeframe
        self._ema9_30s = EMACalculator(period=9)
        self._ema20_30s = EMACalculator(period=20)
        self._ema30_30s = EMACalculator(period=30)
        self._ema200_30s = EMACalculator(period=200)
        
        # EMA calculators for 2m timeframe
        self._ema9_2m = EMACalculator(period=9)
        self._ema20_2m = EMACalculator(period=20)
        self._ema30_2m = EMACalculator(period=30)
        self._ema200_2m = EMACalculator(period=200)
        
        # EMA calculators for 3m timeframe
        self._ema9_3m = EMACalculator(period=9)
        self._ema20_3m = EMACalculator(period=20)
        self._ema30_3m = EMACalculator(period=30)
        self._ema200_3m = EMACalculator(period=200)
        self._vwap = VWAPSessionCalculator(
            exchange_timezone=config.session.exchange_timezone,
            rth_open_time=config.session.rth_open_time,
            include_premarket_in_vwap=config.session.include_premarket_in_vwap,
        )

        # Zone + episodes
        self._zone = ZoneGate(ticker=ticker)
        self._episodes = EpisodeStateMachine(ticker=ticker, session_id=session_id, config=config)

        # Feature engines (ticker-level)
        self._micro = MicrostructureFeatureEngine()
        self._orderflow = OrderFlowFeatureEngine()
        self._vol = VolatilityFeatureEngine()
        self._ema_struct = EMAStructureFeatureEngine()
        self._smart = SmartMoneyFeatureEngine(config=config, snapshot_interval_seconds=config.snapshot.interval_seconds)

        # Per-level episode contexts
        self._active: dict[str, _EpisodeCtx] = {}
        
        # Lee-Ready classifier for accurate buy/sell classification
        self._classifier = LeeReadyClassifier()

        # Writer
        self._writer = SessionWriter(config=config, session_id=session_id, ticker=ticker)
        self._session_stream_out: list[Snapshot] = []
        self._session_stream_all: list[Snapshot] = []
        self._markers: list[Marker] = []

        # Ticker snapshot sequencing
        self._seq = 0
        self._next_tick_end_ms: Optional[int] = None
        self._last_price: float = 0.0

        # Seed ATR deterministically (fixed default) for this MVP wiring; provider seeds can be layered later.
        self._atr.reset_session(
            atr_seed=float(config.atr.fixed_default_atr),
            seed_source=ATRSeedSource.FIXED_DEFAULT,
            atr_blend_alpha=config.atr.atr_blend_alpha,
        )
        self._atr_state_value = float(config.atr.fixed_default_atr)
        self._atr_state_status = ATRStatus.SEEDED
        self._atr_state_source = ATRSeedSource.FIXED_DEFAULT
        self._atr_is_warm = False

    def _floor_end_ms(self, ts_ms: int) -> int:
        start = (int(ts_ms) // self._interval_ms) * self._interval_ms
        return start + self._interval_ms - 1

    def on_trade(self, trade: Trade) -> None:
        self._trades.append(trade)
        self._last_price = float(trade.price)

        # VWAP uses trades directly
        self._vwap.update(trade)

        # 1m bars + ATR/EMA updates
        finished_1m = self._bars_1m.update(trade)
        if finished_1m is not None:
            atr_snap = self._atr.update(finished_1m)
            self._atr_state_value = float(atr_snap.value)
            self._atr_state_status = atr_snap.status
            self._atr_state_source = atr_snap.seed_source or self._atr_state_source
            self._atr_is_warm = atr_snap.is_warm

            # Update EMAs on bar close
            self._ema9_1m.update(finished_1m)
            self._ema20_1m.update(finished_1m)
            self._ema30_1m.update(finished_1m)
            self._ema200_1m.update(finished_1m)

        finished_30s = self._bars_30s.update(trade)
        if finished_30s is not None:
            self._ema9_30s.update(finished_30s)
            self._ema20_30s.update(finished_30s)
            self._ema30_30s.update(finished_30s)
            self._ema200_30s.update(finished_30s)

        finished_2m = self._bars_2m.update(trade)
        if finished_2m is not None:
            self._ema9_2m.update(finished_2m)
            self._ema20_2m.update(finished_2m)
            self._ema30_2m.update(finished_2m)
            self._ema200_2m.update(finished_2m)

        finished_3m = self._bars_3m.update(trade)
        if finished_3m is not None:
            self._ema9_3m.update(finished_3m)
            self._ema20_3m.update(finished_3m)
            self._ema30_3m.update(finished_3m)
            self._ema200_3m.update(finished_3m)

    def on_quote(self, quote: Quote) -> None:
        self._quotes.append(quote)

    def _build_ticker_snapshot(self, end_ms: int) -> Snapshot:
        s = Snapshot(
            episode_id="__ticker__",
            sequence_id=self._seq,
            timestamp=end_ms,
            phase=EpisodePhase.STRESS,
        )
        self._seq += 1
        s.atr_value = float(self._atr_state_value)

        # last_price from last trade <= end_ms
        if self._last_price > 0:
            s.last_price = self._last_price

        # latest quote in window [start..end]
        start_ms = end_ms - (self._interval_ms - 1)
        latest_q: Optional[Quote] = None
        for q in self._quotes.window(start_ts_ms=start_ms, end_ts_ms=end_ms):
            latest_q = q
        if latest_q is not None:
            s.bid = float(latest_q.bid)
            s.ask = float(latest_q.ask)
            s.mid_price = (s.bid + s.ask) / 2.0 if (s.bid > 0 and s.ask > 0) else 0.0
            s.spread_abs = (s.ask - s.bid) if (s.bid > 0 and s.ask > 0) else 0.0
            s.spread_pct = (s.spread_abs / s.mid_price) if s.mid_price > 0 else 0.0
            s.quote_age_ms = int(end_ms - int(latest_q.ts_ms))

        # VWAP context
        vwap_val = self._vwap.current_vwap
        if vwap_val is not None and s.last_price > 0:
            s.vwap_session = float(vwap_val)
            s.price_minus_vwap = float(s.last_price - s.vwap_session)
            s.price_minus_vwap_atr = float(s.price_minus_vwap / self._atr_state_value) if self._atr_state_value > 0 else 0.0

        # EMA values - 1m (default)
        s.ema9 = float(self._ema9_1m.value or 0.0)
        s.ema20 = float(self._ema20_1m.value or 0.0)
        s.ema30 = float(self._ema30_1m.value or 0.0)
        s.ema200 = float(self._ema200_1m.value or 0.0)
        
        # EMA values - 2m
        s.ema9_2m = float(self._ema9_2m.value or 0.0)
        s.ema20_2m = float(self._ema20_2m.value or 0.0)
        s.ema30_2m = float(self._ema30_2m.value or 0.0)
        s.ema200_2m = float(self._ema200_2m.value or 0.0)
        
        # EMA values - 3m
        s.ema9_3m = float(self._ema9_3m.value or 0.0)
        s.ema20_3m = float(self._ema20_3m.value or 0.0)
        s.ema30_3m = float(self._ema30_3m.value or 0.0)
        s.ema200_3m = float(self._ema200_3m.value or 0.0)

        # Order flow base fields for this 10s window:
        window_trades = self._trades.window(start_ts_ms=start_ms, end_ts_ms=end_ms)
        s.trade_count = len(window_trades)
        s.total_volume = sum(float(t.size) for t in window_trades)

        # Track high/low for the window (for wick-based touch detection)
        # Filter out non-regular prints that can distort candle highs/lows in charts (e.g. condition "W").
        if window_trades:
            prices = [
                float(t.price)
                for t in window_trades
                if getattr(t, "conditions", None) is None or ("W" not in tuple(getattr(t, "conditions", ()) or ()))
            ]
            s.high_10s = max(prices) if prices else s.last_price
            s.low_10s = min(prices) if prices else s.last_price
        else:
            # No trades in window - use last_price as both high/low
            s.high_10s = s.last_price
            s.low_10s = s.last_price

        # Lee-Ready classification for accurate buy/sell assignment
        # Uses quote rule first, then tick rule for in-spread trades
        # Track classifications for reuse in large trade analysis
        buy = sell = unk = 0.0
        trade_classifications: dict[int, str] = {}  # ts_ms -> side
        for t in window_trades:
            # latest quote at-or-before trade time
            q_at: Optional[Quote] = None
            for q in self._quotes:
                if q.ts_ms <= t.ts_ms:
                    q_at = q
                else:
                    break
            if q_at is None:
                unk += float(t.size)
                trade_classifications[t.ts_ms] = "unknown"
                continue
            # Use Lee-Ready classifier (quote rule + tick rule)
            side = self._classifier.classify(
                trade_price=float(t.price),
                bid=float(q_at.bid),
                ask=float(q_at.ask)
            )
            trade_classifications[t.ts_ms] = side
            if side == "buy":
                buy += float(t.size)
            elif side == "sell":
                sell += float(t.size)
            else:
                unk += float(t.size)

        s.buy_volume = buy
        s.sell_volume = sell
        s.unknown_volume = unk
        s.delta = buy - sell
        s.pct_at_ask = (buy / s.total_volume) if s.total_volume > 0 else 0.0
        s.pct_at_bid = (sell / s.total_volume) if s.total_volume > 0 else 0.0
        s.relative_aggression = ((buy - sell) / (s.total_volume + 1e-9)) if s.total_volume > 0 else 0.0

        # Derived ticker-level features from history:
        hist = list(self._ticker_snaps)
        self._micro.apply_to_snapshot(s, self._micro.compute(current=s, history=hist))
        self._orderflow.apply_to_snapshot(
            s,
            self._orderflow.compute(episode_id="__ticker__", current=s, history=hist, window_trades=window_trades),
        )
        self._vol.apply_to_snapshot(s, self._vol.compute(current=s, history=hist))

        # EMA structure features: need confluence_ref (store per episode; here use config)
        dummy_ep = Episode(
            episode_id="__ticker__",
            session_id=self._session_id,
            ticker=self._ticker,
            level_id="",
            level_price=0.0,
            level_type="",
            level_source="",
            level_width_atr=0.25,
            level_entry_side="",
            start_time=0,
            zone_entry_time=0,
            ema_confluence_ref=self._cfg.ema.confluence_ref,
            atr_value=self._atr_state_value,
            atr_status=self._atr_state_status,
            atr_seed_source=self._atr_state_source,
            atr_is_warm=self._atr_is_warm,
            direction_bias=self._direction_bias,
        )
        self._ema_struct.apply_to_snapshot(
            s,
            self._ema_struct.compute(episode=dummy_ep, current=s, history=hist, atr_14_1m=self._atr_state_value),
        )

        # Large-trade imbalance (Addendum I4): rolling p95 trade size over 5m, then large buckets in this 10s window.
        five_min_start = end_ms - 300_000
        sizes_5m = [float(t.size) for t in self._trades.window(start_ts_ms=five_min_start, end_ts_ms=end_ms) if float(t.size) > 0.0]
        if sizes_5m:
            sizes_5m.sort()
            n = len(sizes_5m)
            idx = int(((0.95 * n) + 0.999999) - 1)  # ceil(0.95*n)-1
            idx = max(0, min(n - 1, idx))
            thr = float(sizes_5m[idx])
        else:
            thr = 0.0
        s.large_trade_threshold_size = thr

        large_count = 0
        lb = ls = lu = 0.0
        if thr > 0.0:
            for t in window_trades:
                if float(t.size) < thr:
                    continue
                large_count += 1
                # Use cached Lee-Ready classification from first pass
                side = trade_classifications.get(t.ts_ms, "unknown")
                if side == "buy":
                    lb += float(t.size)
                elif side == "sell":
                    ls += float(t.size)
                else:
                    lu += float(t.size)

        s.large_trade_count_10s = int(large_count)
        s.large_buy_volume_10s = float(lb)
        s.large_sell_volume_10s = float(ls)
        s.large_unknown_volume_10s = float(lu)
        s.large_trade_delta = float(lb - ls)
        s.large_trade_buy_ratio = float(lb / (lb + ls + self._cfg.smart_money.epsilon)) if (lb + ls) > 0 else 0.0
        s.large_trade_share_of_total_vol_10s = float((lb + ls + lu) / (s.total_volume + self._cfg.smart_money.epsilon)) if s.total_volume > 0 else 0.0

        return s

    def _tf_row(
        self,
        *,
        episode: Episode,
        base_snapshot: Snapshot,
        timeframe: str,
        ema9: float,
        ema20: float,
        ema30: float,
        ema200: float,
        tf_history: list[dict],
    ) -> dict:
        ts = int(base_snapshot.timestamp)
        target = ts - 60_000

        prev = None
        for r in tf_history:
            if int(r["timestamp"]) <= target:
                prev = r
            else:
                break

        def slope(cur: float, prev_val: Optional[float]) -> float:
            if prev_val is None:
                return 0.0
            return (cur - prev_val) / 60.0

        prev_ema9 = float(prev["ema9"]) if prev is not None else None
        prev_ema20 = float(prev["ema20"]) if prev is not None else None
        prev_ema30 = float(prev["ema30"]) if prev is not None else None

        spread_9_20 = float(ema9 - ema20)
        spread_20_30 = float(ema20 - ema30)

        atr = float(base_snapshot.atr_value) if float(base_snapshot.atr_value) > 0.0 else float(episode.atr_value)
        compression = (abs(spread_9_20) + abs(spread_20_30)) / atr if atr > 0 else 0.0
        confluence = max(0.0, min(1.0, 1.0 - (compression / float(episode.ema_confluence_ref))))

        if ema9 > ema20 > ema30:
            stack = "bull"
        elif ema9 < ema20 < ema30:
            stack = "bear"
        else:
            stack = "mixed"

        price = float(base_snapshot.last_price)
        bit0 = 1 if price > ema9 and ema9 != 0.0 else 0
        bit1 = 1 if price > ema20 and ema20 != 0.0 else 0
        bit2 = 1 if price > ema30 and ema30 != 0.0 else 0
        bit3 = 1 if price > ema200 and ema200 != 0.0 else 0
        mask = bit0 + 2 * bit1 + 4 * bit2 + 8 * bit3

        stretch_200 = (price - ema200) / atr if (atr > 0 and ema200 != 0.0 and price != 0.0) else 0.0

        return {
            "episode_id": episode.episode_id,
            "timestamp": ts,
            "timeframe": timeframe,
            "ema9": float(ema9),
            "ema20": float(ema20),
            "ema30": float(ema30),
            "ema200": float(ema200),
            "slope_ema9_60s": float(slope(ema9, prev_ema9)),
            "slope_ema20_60s": float(slope(ema20, prev_ema20)),
            "slope_ema30_60s": float(slope(ema30, prev_ema30)),
            "ema_spread_9_20": float(spread_9_20),
            "ema_spread_20_30": float(spread_20_30),
            "compression_index": float(compression),
            "ema_confluence_score": float(confluence),
            "stack_state": stack,
            "price_vs_emas": int(mask),
            "stretch_200_atr": float(stretch_200),
        }

    def add_marker(
        self,
        *,
        ts_ms: int,
        marker_type: MarkerType,
        marker_id: str | None = None,
        direction_bias: DirectionBias | None = None,
        notes: str | None = None,
    ) -> Marker:
        """
        Record a marker event (Addendum H).
        Marker episode extraction is performed at flush() using the recorded session stream.
        """
        ts_ms = int(ts_ms)
        if marker_id is None:
            marker_id = f"marker:{self._ticker}:{ts_ms}:{marker_type.value}"
        m = Marker(
            marker_id=str(marker_id),
            session_id=self._session_id,
            ticker=self._ticker,
            ts_ms=ts_ms,
            marker_type=marker_type,
            direction_bias=direction_bias,
            notes=notes,
        )
        self._markers.append(m)
        return m

    def _infer_marker_bias(self, marker_type: MarkerType) -> Optional[DirectionBias]:
        # Deterministic default mapping from SPEC Addendum H6.
        if marker_type == MarkerType.SUPPORT_BOUNCE:
            return DirectionBias.LONG
        if marker_type == MarkerType.DOWNTREND_BREAK:
            return DirectionBias.LONG
        if marker_type == MarkerType.BREAKDOWN:
            return DirectionBias.SHORT
        return None

    def _extract_marker_episodes(self) -> None:
        if not self._markers:
            return
        # Use the full session stream recorded in-memory (replay/integration).
        stream = self._session_stream_all
        if not stream:
            return

        # Deterministic de-dupe: identical (ts_ms, marker_type) keep first.
        seen: set[tuple[int, str]] = set()
        markers = []
        for m in sorted(self._markers, key=lambda x: (int(x.ts_ms), x.marker_id)):
            key = (int(m.ts_ms), m.marker_type.value)
            if key in seen:
                continue
            seen.add(key)
            markers.append(m)

        for m in markers:
            M = int(m.ts_ms)
            baseline_start = M - 300_000
            stress_start = M - 120_000
            stress_end = M + 120_000
            resolution_end = M + 600_000

            # Find anchor snapshot (latest <= M)
            anchor = None
            for s in stream:
                if int(s.timestamp) <= M:
                    anchor = s
                else:
                    break
            # Deterministic fallback: if marker occurs before the first snapshot tick,
            # use the first available snapshot as anchor rather than dropping the marker.
            if anchor is None and stream and int(stream[0].timestamp) > M:
                anchor = stream[0]
            if anchor is None or float(anchor.last_price) <= 0.0:
                continue

            bias = m.direction_bias or self._infer_marker_bias(m.marker_type)
            if bias is None:
                # Requires explicit direction bias; skip if missing.
                continue

            # Episode record (modeled as pseudo level)
            ep = Episode(
                episode_id=f"{m.marker_id}",
                session_id=self._session_id,
                ticker=self._ticker,
                episode_source="marker_extract",
                marker_type=m.marker_type.value,
                marker_ts_ms=M,
                marker_id=m.marker_id,
                level_id=m.marker_id,
                level_price=float(anchor.last_price),
                level_type=m.marker_type.value,
                level_source="marker",
                level_width_atr=0.0,
                level_entry_side="",
                start_time=baseline_start,
                zone_entry_time=stress_start,
                zone_exit_time=stress_end,
                resolution_time=None,
                end_time=resolution_end,
                direction_bias=bias,
                atr_value=float(anchor.atr_value) if float(anchor.atr_value) > 0 else float(self._atr_state_value),
                atr_status=self._atr_state_status,
                atr_seed_source=self._atr_state_source,
                atr_blend_alpha=self._cfg.atr.atr_blend_alpha,
                atr_is_warm=self._atr_is_warm,
                success_threshold_atr=self._cfg.resolution.success_threshold_atr,
                failure_threshold_atr=self._cfg.resolution.failure_threshold_atr,
                timeout_seconds=int((resolution_end - stress_end) / 1000),
                ema_confluence_ref=self._cfg.ema.confluence_ref,
            )

            # Extract snapshots from stream and map phases by windows
            snaps = []
            for s in stream:
                t = int(s.timestamp)
                if t < baseline_start:
                    continue
                if t > resolution_end:
                    break
                out = Snapshot(
                    episode_id=ep.episode_id,
                    sequence_id=int(s.sequence_id),
                    timestamp=int(s.timestamp),
                    phase=EpisodePhase.BASELINE,
                    atr_value=float(s.atr_value),
                )
                for k in Snapshot.__slots__:
                    if k in {"episode_id", "phase"}:
                        continue
                    setattr(out, k, getattr(s, k))
                if t < stress_start:
                    out.phase = EpisodePhase.BASELINE
                elif t <= stress_end:
                    out.phase = EpisodePhase.STRESS
                else:
                    out.phase = EpisodePhase.RESOLUTION
                snaps.append(out)

            # Determine deterministic resolution trigger/time by scanning snapshots after stress_end
            resolved = None
            for s in snaps:
                t = int(s.timestamp)
                if t < stress_end:
                    continue
                d = evaluate_resolution(
                    ts_ms=t,
                    last_price=float(s.last_price),
                    level_price=float(ep.level_price),
                    atr_value=float(ep.atr_value),
                    direction_bias=ep.direction_bias,
                    success_threshold_atr=ep.success_threshold_atr,
                    failure_threshold_atr=ep.failure_threshold_atr,
                    zone_exit_time=ep.zone_exit_time,
                    timeout_seconds=ep.timeout_seconds,
                )
                if d is not None:
                    resolved = d
                    break
            if resolved is not None:
                ep.resolution_trigger = resolved.trigger
                ep.resolution_time = resolved.resolution_time
                ep.end_time = resolved.resolution_time

            # Persist extracted episode/snapshots (marker persistence is handled in flush()).
            apply_outcome_and_metrics(ep=ep, snapshots=snaps)
            self._writer.write_snapshots(snaps)
            self._writer.write_episodes([ep])

    def add_level(
        self,
        *,
        created_ts_ms: int,
        level_price: float,
        level_type: str,
        level_width_atr: float = 0.25,
        level_source: str = "manual",
        level_id: str | None = None,
        notes: str | None = None,
    ) -> Level:
        """
        Manual level override (Addendum G).
        Becomes active immediately for ZoneGate/Episode creation.
        """
        created_ts_ms = int(created_ts_ms)
        if level_id is None:
            level_id = f"manual:{self._ticker}:{created_ts_ms}:{float(level_price)}"
        lvl = Level(
            level_id=str(level_id),
            level_price=float(level_price),
            level_type=str(level_type),
            level_source=str(level_source),
            level_width_atr=float(level_width_atr),
            created_ts_ms=created_ts_ms,
            notes=notes,
        )
        self._levels.append(lvl)
        # Create per-level zone snapshot buffer for baseline backfill.
        if lvl.level_id not in self._zone_snaps:
            max_age_ms = 5 * 60 * 1000
            self._zone_snaps[lvl.level_id] = TimeRingBuffer[ZoneSnapshot](max_age_ms=max_age_ms)
        return lvl

    # Orchestrator command helpers (avoid orchestrator imports to prevent cycles)
    def on_add_level(self, cmd: object) -> None:
        self.add_level(
            created_ts_ms=int(getattr(cmd, "ts_ms")),
            level_price=float(getattr(cmd, "level_price")),
            level_type=str(getattr(cmd, "level_type")),
            level_width_atr=float(getattr(cmd, "level_width_atr", 0.25)),
            level_id=getattr(cmd, "level_id", None),
            notes=getattr(cmd, "notes", None),
        )

    def on_add_marker(self, cmd: object) -> None:
        self.add_marker(
            ts_ms=int(getattr(cmd, "ts_ms")),
            marker_type=getattr(cmd, "marker_type"),
            marker_id=getattr(cmd, "marker_id", None),
            direction_bias=getattr(cmd, "direction_bias", None),
            notes=getattr(cmd, "notes", None),
        )

    def _merge_episode_snapshot(self, *, ep: Episode, base: Snapshot, zone: ZoneSnapshot) -> Snapshot:
        # Clone base snapshot and inject episode linkage + level spine base fields.
        s = Snapshot(
            episode_id=ep.episode_id,
            sequence_id=base.sequence_id,
            timestamp=base.timestamp,
            phase=EpisodePhase.BASELINE,  # will be overwritten below
        )
        # Copy all Snapshot slot fields deterministically (Snapshot uses slots, so no __dict__).
        for k in Snapshot.__slots__:
            if k in {"episode_id", "phase"}:
                continue
            setattr(s, k, getattr(base, k))

        if int(base.timestamp) < int(ep.zone_entry_time):
            s.phase = EpisodePhase.BASELINE
        elif ep.zone_exit_time is None or int(base.timestamp) <= int(ep.zone_exit_time):
            s.phase = EpisodePhase.STRESS
        else:
            s.phase = EpisodePhase.RESOLUTION

        s.signed_distance_to_level = zone.signed_distance_to_level
        s.signed_distance_to_level_atr = zone.signed_distance_to_level_atr
        s.abs_distance_to_level_atr = zone.abs_distance_to_level_atr
        s.touch_count = zone.touch_count

        return s

    def on_event_time(self, ts_ms: int) -> None:
        """
        Advance snapshot ticks up to ts_ms, producing ticker snapshots and updating episodes.
        Call this after ingesting all events with event-time <= ts_ms.
        """
        if self._next_tick_end_ms is None:
            self._next_tick_end_ms = self._floor_end_ms(ts_ms)

        while self._next_tick_end_ms is not None and self._next_tick_end_ms <= ts_ms:
            end_ms = self._next_tick_end_ms

            # Build ticker snapshot and store in ring buffer
            base = self._build_ticker_snapshot(end_ms)
            self._ticker_snaps.append(base)
            if self._record_session_stream:
                self._session_stream_out.append(base)
                self._session_stream_all.append(base)

            # Zone gate for all levels at this tick (needs last_price and ATR)
            zone_snaps, zone_events = self._zone.update(
                ts_ms=end_ms,
                last_price=float(base.last_price),
                atr_value=float(self._atr_state_value),
                levels=self._levels,
            )
            for zs in zone_snaps:
                self._zone_snaps[zs.level_id].append(zs)

            # Handle enter/exit events to manage episodes
            for ev in zone_events:
                started = self._episodes.on_zone_event(
                    event=ev,
                    direction_bias=self._direction_bias,
                    atr_value=float(self._atr_state_value),
                    atr_status=self._atr_state_status,
                    atr_seed_source=self._atr_state_source,
                    atr_is_warm=self._atr_is_warm,
                )
                if started is not None:
                    # Start episode context and backfill baseline snapshots from ticker ring buffer
                    lvl = next(x for x in self._levels if x.level_id == started.level_id)
                    ctx = _EpisodeCtx(
                        episode=started,
                        level=lvl,
                        snapshots=[],
                        spine=LevelSpineFeatureEngine(snapshot_interval_ms=self._interval_ms),
                        tf_rows=[],
                        tf_history_by_tf={"30s": [], "1m": []},
                    )

                    # Backfill baseline window: use stored ticker snapshots + stored zone snapshots for this level.
                    base_hist = self._ticker_snaps.window(start_ts_ms=started.start_time, end_ts_ms=started.zone_entry_time - 1)
                    zone_hist = self._zone_snaps[started.level_id].window(start_ts_ms=started.start_time, end_ts_ms=started.zone_entry_time - 1)
                    zone_by_ts = {int(z.ts_ms): z for z in zone_hist}

                    for b in base_hist:
                        z = zone_by_ts.get(int(b.timestamp))
                        if z is None:
                            continue
                        merged = self._merge_episode_snapshot(ep=started, base=b, zone=z)
                        # Apply level spine features using episode history
                        feats = ctx.spine.compute(
                            episode=started,
                            current=merged,
                            history=ctx.snapshots,
                            in_zone=bool(z.in_zone),
                            touch_count=int(z.touch_count),
                        )
                        ctx.spine.apply_to_snapshot(merged, feats)
                        sm = self._smart.compute(episode=started, current=merged, history=ctx.snapshots)
                        self._smart.apply_to_snapshot(merged, sm)
                        ctx.snapshots.append(merged)

                        # Multi-timeframe structure rows (30s, 1m)
                        r30 = self._tf_row(
                            episode=started,
                            base_snapshot=merged,
                            timeframe="30s",
                            ema9=float(self._ema9_30s.value or 0.0),
                            ema20=float(self._ema20_30s.value or 0.0),
                            ema30=float(self._ema30_30s.value or 0.0),
                            ema200=float(self._ema200_30s.value or 0.0),
                            tf_history=ctx.tf_history_by_tf["30s"],
                        )
                        ctx.tf_history_by_tf["30s"].append(r30)
                        ctx.tf_rows.append(r30)

                        r1 = self._tf_row(
                            episode=started,
                            base_snapshot=merged,
                            timeframe="1m",
                            ema9=float(self._ema9_1m.value or 0.0),
                            ema20=float(self._ema20_1m.value or 0.0),
                            ema30=float(self._ema30_1m.value or 0.0),
                            ema200=float(self._ema200_1m.value or 0.0),
                            tf_history=ctx.tf_history_by_tf["1m"],
                        )
                        ctx.tf_history_by_tf["1m"].append(r1)
                        ctx.tf_rows.append(r1)

                    self._active[started.level_id] = ctx

            # For each active episode, append current merged snapshot and check resolution
            resolved_eps: list[Episode] = []
            for level_id, ctx in list(self._active.items()):
                # current zone snapshot for this level at this tick
                z = next((zs for zs in zone_snaps if zs.level_id == level_id), None)
                if z is None:
                    continue

                merged = self._merge_episode_snapshot(ep=ctx.episode, base=base, zone=z)
                feats = ctx.spine.compute(
                    episode=ctx.episode,
                    current=merged,
                    history=ctx.snapshots,
                    in_zone=bool(z.in_zone),
                    touch_count=int(z.touch_count),
                )
                ctx.spine.apply_to_snapshot(merged, feats)
                # Smart money proxies (episode-level; uses baseline stats)
                sm = self._smart.compute(episode=ctx.episode, current=merged, history=ctx.snapshots)
                self._smart.apply_to_snapshot(merged, sm)
                ctx.snapshots.append(merged)

                r30 = self._tf_row(
                    episode=ctx.episode,
                    base_snapshot=merged,
                    timeframe="30s",
                    ema9=float(self._ema9_30s.value or 0.0),
                    ema20=float(self._ema20_30s.value or 0.0),
                    ema30=float(self._ema30_30s.value or 0.0),
                    ema200=float(self._ema200_30s.value or 0.0),
                    tf_history=ctx.tf_history_by_tf["30s"],
                )
                ctx.tf_history_by_tf["30s"].append(r30)
                ctx.tf_rows.append(r30)

                r1 = self._tf_row(
                    episode=ctx.episode,
                    base_snapshot=merged,
                    timeframe="1m",
                    ema9=float(self._ema9_1m.value or 0.0),
                    ema20=float(self._ema20_1m.value or 0.0),
                    ema30=float(self._ema30_1m.value or 0.0),
                    ema200=float(self._ema200_1m.value or 0.0),
                    tf_history=ctx.tf_history_by_tf["1m"],
                )
                ctx.tf_history_by_tf["1m"].append(r1)
                ctx.tf_rows.append(r1)

                # If exited, evaluate resolution at this tick
                finalized = self._episodes.update_resolution(
                    level_id=level_id,
                    ts_ms=end_ms,
                    last_price=float(base.last_price),
                    atr_value=float(self._atr_state_value),
                )
                if finalized is not None:
                    # Persist this episode's snapshots + episode record
                    apply_outcome_and_metrics(ep=finalized, snapshots=ctx.snapshots)
                    self._writer.write_snapshots(ctx.snapshots)
                    self._writer.write_episodes([finalized])
                    self._writer.write_tf_indicators(ctx.tf_rows)
                    resolved_eps.append(finalized)
                    del self._active[level_id]

            self._next_tick_end_ms = end_ms + self._interval_ms

    def flush(self) -> None:
        """Finalize any in-progress bar (does not force-resolve episodes)."""
        self._bars_1m.flush()
        self._bars_30s.flush()
        # Persist markers at end-of-session for replay (deterministic), even if episode extraction skips.
        if self._markers:
            # Deterministic de-dupe: identical (ts_ms, marker_type) keep first.
            seen: set[tuple[int, str]] = set()
            markers: list[Marker] = []
            for m in sorted(self._markers, key=lambda x: (int(x.ts_ms), x.marker_id)):
                key = (int(m.ts_ms), m.marker_type.value)
                if key in seen:
                    continue
                seen.add(key)
                markers.append(m)
            self._writer.write_markers(markers)

        # Marker episode extraction occurs at end-of-session for replay (deterministic).
        self._extract_marker_episodes()
        if self._record_session_stream and self._session_stream_out:
            self._writer.write_session_stream(self._session_stream_out)
            self._session_stream_out.clear()


