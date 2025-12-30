from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from godmode.core.config import AppConfig
from godmode.core.enums import EpisodePhase, VWAPState
from godmode.core.models import Episode, Quote, Snapshot, Trade
from godmode.engine.ring_buffer import TimeRingBuffer
from godmode.engine.vwap import VWAPSessionCalculator
from godmode.zone.zone_gate import ZoneSnapshot


def _floor_to_interval_ms(ts_ms: int, interval_ms: int) -> int:
    return (int(ts_ms) // interval_ms) * interval_ms


def _phase_for_episode(ep: Episode, ts_ms: int) -> EpisodePhase:
    if ts_ms < ep.zone_entry_time:
        return EpisodePhase.BASELINE
    if ep.zone_exit_time is None or ts_ms <= ep.zone_exit_time:
        return EpisodePhase.STRESS
    return EpisodePhase.RESOLUTION


def _latest_quote_in_window(quotes: TimeRingBuffer[Quote], start_ms: int, end_ms: int) -> Optional[Quote]:
    # Addendum D: "latest NBBO quote observed during the 10-second window"
    latest: Optional[Quote] = None
    for q in quotes.window(start_ts_ms=start_ms, end_ts_ms=end_ms):
        latest = q
    return latest


def _latest_quote_at_or_before(quotes: TimeRingBuffer[Quote], ts_ms: int) -> Optional[Quote]:
    # Deterministic tie-breakers are inherited from the ring buffer ordering.
    latest: Optional[Quote] = None
    for q in quotes:
        if q.ts_ms <= ts_ms:
            latest = q
        else:
            break
    return latest


def classify_trade(*, trade_price: float, bid: float, ask: float) -> str:
    """Simple quote rule classification (legacy, no state)."""
    # Spec quote test:
    # trade >= ask → buy; trade <= bid → sell; else → unknown
    if trade_price >= ask:
        return "buy"
    if trade_price <= bid:
        return "sell"
    return "unknown"


class LeeReadyClassifier:
    """
    Lee-Ready trade classification algorithm.
    
    1. Quote rule first: price >= ask → buy, price <= bid → sell
    2. If in-spread, use tick rule: compare to previous trade price
       - Price up → buy (buyer lifted)
       - Price down → sell (seller hit)
       - Same price → use previous classification
    """
    
    def __init__(self):
        self._last_price: Optional[float] = None
        self._last_side: str = "unknown"
    
    def classify(self, *, trade_price: float, bid: float, ask: float) -> str:
        """Classify a trade using Lee-Ready algorithm."""
        side: str
        
        # Step 1: Quote rule (always try first)
        if trade_price >= ask:
            side = "buy"
        elif trade_price <= bid:
            side = "sell"
        else:
            # Step 2: In-spread - use tick rule
            if self._last_price is not None:
                if trade_price > self._last_price:
                    side = "buy"  # Price went up, buyer was aggressive
                elif trade_price < self._last_price:
                    side = "sell"  # Price went down, seller was aggressive
                else:
                    # Same price - use previous classification
                    side = self._last_side if self._last_side in ("buy", "sell") else "unknown"
            else:
                side = "unknown"
        
        # Update state
        self._last_price = trade_price
        if side in ("buy", "sell"):
            self._last_side = side
        
        return side
    
    def reset(self):
        """Reset state (e.g., at start of new session)."""
        self._last_price = None
        self._last_side = "unknown"


@dataclass(slots=True)
class SnapshotBuilder:
    """
    Builds 10-second snapshots (event-time) linked to an Episode.

    Determinism:
    - Snapshot timestamps are aligned to fixed interval boundaries in event-time.
    - Quote sampling uses the latest quote observed during the snapshot window (Addendum D).
    - Buy/sell classification uses the latest quote at-or-before each trade's event-time.
    """

    config: AppConfig
    interval_ms: int = 10_000
    epsilon: float = 1e-9
    _seq: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_trade_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _classifier: LeeReadyClassifier = field(default_factory=LeeReadyClassifier, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be > 0")
        # ensure config snapshot interval and this builder align unless intentionally overridden
        cfg_ms = int(self.config.snapshot.interval_seconds) * 1000
        if cfg_ms != self.interval_ms:
            # keep deterministic but allow explicit override
            pass

        # internal state stored in dataclass fields for slots compatibility

    def build(
        self,
        *,
        ts_ms: int,
        episode: Episode,
        zone: ZoneSnapshot,
        atr_value: float,
        trades: TimeRingBuffer[Trade],
        quotes: TimeRingBuffer[Quote],
        vwap: VWAPSessionCalculator,
        ema9: Optional[float] = None,
        ema20: Optional[float] = None,
        ema30: Optional[float] = None,
        ema200: Optional[float] = None,
    ) -> Snapshot:
        snap_ts = _floor_to_interval_ms(ts_ms, self.interval_ms)
        start_ms = snap_ts
        end_ms = snap_ts + self.interval_ms - 1

        if end_ms < episode.start_time:
            raise ValueError("snapshot timestamp is before episode start_time")

        seq = self._seq.get(episode.episode_id, 0)
        self._seq[episode.episode_id] = seq + 1

        # last_price: last trade observed at-or-before snapshot end (deterministic).
        last_price = self._last_trade_price.get(episode.ticker, 0.0)
        for t in trades.window(start_ts_ms=episode.start_time, end_ts_ms=end_ms):
            last_price = float(t.price)
        if last_price != 0.0:
            self._last_trade_price[episode.ticker] = last_price

        # Quotes: latest during the 10s window (Addendum D)
        q_latest = _latest_quote_in_window(quotes, start_ms, end_ms)
        bid = float(q_latest.bid) if q_latest else 0.0
        ask = float(q_latest.ask) if q_latest else 0.0
        mid = (bid + ask) / 2.0 if (bid > 0.0 and ask > 0.0) else 0.0
        spread_abs = (ask - bid) if (bid > 0.0 and ask > 0.0) else 0.0
        spread_pct = (spread_abs / mid) if mid > 0.0 else 0.0
        quote_age_ms = (end_ms - int(q_latest.ts_ms)) if q_latest else None

        # Trades inside the snapshot window
        window_trades = trades.window(start_ts_ms=start_ms, end_ts_ms=end_ms)
        trade_count = len(window_trades)
        total_volume = sum(float(t.size) for t in window_trades)

        # Track high/low for the window (for touch detection)
        high_10s = last_price
        low_10s = last_price
        if window_trades:
            # Filter out non-regular prints that can distort candle highs/lows in charts.
            # In practice, Webull (and many chart feeds) ignore certain "special" trades.
            # Alpaca can include these in getTrades() with condition codes like "W".
            prices = [
                float(t.price)
                for t in window_trades
                if getattr(t, "conditions", None) is None or ("W" not in tuple(getattr(t, "conditions", ()) or ()))
            ]
            high_10s = max(prices) if prices else last_price
            low_10s = min(prices) if prices else last_price
        # If no trades in window, use last_price as both high/low
        if high_10s == 0.0:
            high_10s = last_price
        if low_10s == 0.0:
            low_10s = last_price

        buy_vol = 0.0
        sell_vol = 0.0
        unk_vol = 0.0

        # Classify each trade using Lee-Ready algorithm (quote rule + tick rule)
        for t in window_trades:
            q_at = _latest_quote_at_or_before(quotes, int(t.ts_ms))
            if q_at is None:
                unk_vol += float(t.size)
                continue
            # Use Lee-Ready classifier for accurate buy/sell assignment
            side = self._classifier.classify(
                trade_price=float(t.price),
                bid=float(q_at.bid),
                ask=float(q_at.ask)
            )
            if side == "buy":
                buy_vol += float(t.size)
            elif side == "sell":
                sell_vol += float(t.size)
            else:
                unk_vol += float(t.size)

        delta = buy_vol - sell_vol
        pct_at_ask = (buy_vol / total_volume) if total_volume > 0 else 0.0
        pct_at_bid = (sell_vol / total_volume) if total_volume > 0 else 0.0
        relative_aggr = ((buy_vol - sell_vol) / (total_volume + self.epsilon)) if total_volume > 0 else 0.0

        # VWAP context (session)
        vwap_val = vwap.current_vwap
        price_minus_vwap = (last_price - vwap_val) if (vwap_val is not None and last_price != 0.0) else 0.0
        price_minus_vwap_atr = (price_minus_vwap / atr_value) if (atr_value > 0) else 0.0
        if vwap_val is None or last_price == 0.0:
            vwap_state = VWAPState.AT
        else:
            if abs(price_minus_vwap) < 1e-12:
                vwap_state = VWAPState.AT
            elif price_minus_vwap > 0:
                vwap_state = VWAPState.ABOVE
            else:
                vwap_state = VWAPState.BELOW

        # EMA defaults
        ema9 = float(ema9) if ema9 is not None else 0.0
        ema20 = float(ema20) if ema20 is not None else 0.0
        ema30 = float(ema30) if ema30 is not None else 0.0
        ema200 = float(ema200) if ema200 is not None else 0.0

        # Level distances: prefer the ZoneSnapshot spine values (already ATR-normalized deterministically)
        signed = float(zone.signed_distance_to_level)
        signed_atr = float(zone.signed_distance_to_level_atr)
        abs_atr = float(zone.abs_distance_to_level_atr)

        phase = _phase_for_episode(episode, end_ms)

        return Snapshot(
            episode_id=episode.episode_id,
            sequence_id=seq,
            timestamp=end_ms,  # snapshot end-time (event-time)
            phase=phase,
            signed_distance_to_level=signed,
            signed_distance_to_level_atr=signed_atr,
            abs_distance_to_level_atr=abs_atr,
            # max_penetration/cross/oscillation/time-in-zone computed in FeatureEngine (Days 10-12)
            touch_count=int(zone.touch_count),
            last_price=float(last_price),
            high_10s=float(high_10s),
            low_10s=float(low_10s),
            bid=bid,
            ask=ask,
            mid_price=mid,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
            quote_age_ms=quote_age_ms,
            vwap_session=float(vwap_val) if vwap_val is not None else 0.0,
            price_minus_vwap=float(price_minus_vwap),
            price_minus_vwap_atr=float(price_minus_vwap_atr),
            vwap_state=vwap_state,
            ema9=ema9,
            ema20=ema20,
            ema30=ema30,
            ema200=ema200,
            trade_count=int(trade_count),
            total_volume=float(total_volume),
            buy_volume=float(buy_vol),
            sell_volume=float(sell_vol),
            unknown_volume=float(unk_vol),
            delta=float(delta),
            pct_at_ask=float(pct_at_ask),
            pct_at_bid=float(pct_at_bid),
            relative_aggression=float(relative_aggr),
        )


