from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .enums import (
    ATRSeedSource,
    ATRStatus,
    ApproachVelocity,
    DirectionBias,
    EpisodePhase,
    Outcome,
    ResolutionTrigger,
    ResolutionType,
    VWAPState,
)


@dataclass(frozen=True, slots=True)
class Trade:
    """Single trade event (event-time)."""

    ts_ms: int
    symbol: str
    price: float
    size: float
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Quote:
    """NBBO quote event (event-time)."""

    ts_ms: int
    symbol: str
    bid: float
    bid_size: float
    ask: float
    ask_size: float


@dataclass(frozen=True, slots=True)
class Bar:
    """Internal 1-minute OHLCV bar built from trades (Addendum A1)."""

    ts_ms: int  # bar open timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int


@dataclass(frozen=True, slots=True)
class Level:
    level_id: str
    level_price: float
    level_type: str
    level_source: str
    level_width_atr: float = 0.25
    # For manual runtime levels (Addendum G). Keep 0 if unknown/not applicable.
    created_ts_ms: int = 0
    notes: Optional[str] = None


@dataclass(slots=True)
class Episode:
    # Identity
    episode_id: str
    session_id: str
    ticker: str

    # Level reference (denormalized for auditability)
    level_id: str
    level_price: float
    level_type: str
    level_source: str
    level_width_atr: float
    level_entry_side: str  # above | below at zone entry

    # Deterministic timing (ms, event-time)
    start_time: int
    zone_entry_time: int
    zone_exit_time: Optional[int] = None
    resolution_time: Optional[int] = None
    end_time: Optional[int] = None

    # Rules & triggers
    zone_rule: str = "abs(signed_distance_to_level_atr) <= 0.25"
    resolution_trigger: Optional[ResolutionTrigger] = None
    direction_bias: DirectionBias = DirectionBias.LONG  # set explicitly by episode creator

    # Thresholds (auditability; Addendum B)
    success_threshold_atr: float = 0.50
    failure_threshold_atr: float = 0.35
    timeout_seconds: int = 300

    # EMA structure auditability (Day 12 determinism patch)
    ema_confluence_ref: float = 0.25

    # ATR state at episode start (Implementation Decisions #2 + Bulletproofing B)
    atr_value: float = 0.0
    atr_status: ATRStatus = ATRStatus.SEEDED
    atr_seed_source: Optional[ATRSeedSource] = None
    atr_blend_alpha: float = 0.7
    atr_is_warm: bool = False

    # Outcomes (objective)
    outcome: Optional[Outcome] = None
    resolution_type: Optional[ResolutionType] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    r_multiple: Optional[float] = None
    time_to_mfe_ms: Optional[int] = None
    time_to_failure_ms: Optional[int] = None

    # Context (recommended; Addendum E)
    time_of_day_bucket: str = "mid"  # open | mid | close
    gap_pct: Optional[float] = None
    spy_return_5m: Optional[float] = None
    spy_bucket: Optional[str] = None

    # Provenance (Addendum H)
    episode_source: str = "level_gate"  # level_gate | marker_extract
    marker_id: Optional[str] = None
    marker_type: Optional[str] = None
    marker_ts_ms: Optional[int] = None


@dataclass(slots=True)
class Snapshot:
    # Linkage & control
    episode_id: str
    sequence_id: int
    timestamp: int  # event-time (ms)
    phase: EpisodePhase
    atr_value: float = 0.0  # optional context (helps audit and marker extraction)

    # === A) LEVEL INTERACTION SPINE ===
    signed_distance_to_level: float = 0.0
    signed_distance_to_level_atr: float = 0.0
    abs_distance_to_level_atr: float = 0.0
    max_penetration_atr: float = 0.0

    cross_count_60s: int = 0
    cross_density: float = 0.0
    oscillation_amplitude_atr_60s: float = 0.0

    time_in_zone_rolling: float = 0.0
    total_time_in_zone_episode: float = 0.0
    touch_count: int = 0
    avg_time_per_touch: float = 0.0

    # === B) PRICE & SPREAD MICROSTRUCTURE ===
    last_price: float = 0.0
    high_10s: float = 0.0  # Highest trade price in 10s window (for touch detection)
    low_10s: float = 0.0   # Lowest trade price in 10s window (for touch detection)
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0
    spread_abs: float = 0.0
    spread_pct: float = 0.0
    spread_volatility_60s: float = 0.0
    spread_zscore: Optional[float] = None  # optional
    quote_age_ms: Optional[int] = None  # optional

    # === C) VWAP (Context) ===
    vwap_session: float = 0.0
    price_minus_vwap: float = 0.0
    price_minus_vwap_atr: float = 0.0
    vwap_state: VWAPState = VWAPState.AT

    # === D) EMA STRUCTURE (No crosses) ===
    # 1-minute EMAs (default)
    ema9: float = 0.0
    ema20: float = 0.0
    ema30: float = 0.0
    ema200: float = 0.0
    
    # 2-minute EMAs
    ema9_2m: float = 0.0
    ema20_2m: float = 0.0
    ema30_2m: float = 0.0
    ema200_2m: float = 0.0
    
    # 3-minute EMAs
    ema9_3m: float = 0.0
    ema20_3m: float = 0.0
    ema30_3m: float = 0.0
    ema200_3m: float = 0.0

    slope_ema9_60s: float = 0.0
    slope_ema20_60s: float = 0.0
    slope_ema30_60s: float = 0.0

    ema_spread_9_20: float = 0.0
    ema_spread_20_30: float = 0.0

    compression_index: float = 0.0
    ema_confluence_score: float = 0.0
    stack_state: str = ""
    price_vs_emas: int = 0
    stretch_200_atr: float = 0.0

    # === E) TIME & SALES / ORDER FLOW ===
    trade_count: int = 0
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    unknown_volume: float = 0.0
    delta: float = 0.0

    # Spec storage columns are `%_at_ask` / `%_at_bid`; internal safe names here.
    pct_at_ask: float = 0.0
    pct_at_bid: float = 0.0

    relative_aggression: float = 0.0
    relative_aggression_zscore_60s: float = 0.0

    delta_velocity: float = 0.0
    delta_acceleration: float = 0.0

    absorption_index_10s: float = 0.0

    avg_trade_size: float = 0.0
    trade_size_std: float = 0.0
    trade_size_cv: float = 0.0
    top_decile_volume_share: float = 0.0

    # === F) VOLATILITY & APPROACH STATE ===
    realized_volatility_60s: float = 0.0
    approach_return_60s: float = 0.0
    approach_volatility_60s: float = 0.0
    approach_velocity_bucket: ApproachVelocity = ApproachVelocity.GRIND

    # === Addendum I: Smart Money Proxies ===
    cvd_10s: float = 0.0
    cvd_30s: float = 0.0
    cvd_60s: float = 0.0
    cvd_30s_slope: float = 0.0
    cvd_60s_slope: float = 0.0

    return_10s: float = 0.0
    return_60s: float = 0.0
    return_norm: float = 0.0
    delta_norm: float = 0.0
    delta_sign: int = 0
    return_sign: int = 0
    divergence_flag: int = 0
    div_score: float = 0.0

    buy_on_red: float = 0.0
    sell_on_green: float = 0.0
    return_sign_10s: int = 0

    vol_60s: float = 0.0
    trade_count_60s: int = 0
    baseline_vol_60s_mean: float = 0.0
    baseline_vol_60s_std: float = 0.0
    vol_z_60s: float = 0.0
    baseline_trade_count_60s_mean: float = 0.0
    baseline_trade_count_60s_std: float = 0.0
    trade_rate_z_60s: float = 0.0

    large_trade_threshold_size: float = 0.0
    large_trade_count_10s: int = 0
    large_buy_volume_10s: float = 0.0
    large_sell_volume_10s: float = 0.0
    large_unknown_volume_10s: float = 0.0
    large_trade_buy_ratio: float = 0.0
    large_trade_delta: float = 0.0
    large_trade_share_of_total_vol_10s: float = 0.0

    @property
    def ts_ms(self) -> int:
        """
        Event-time alias for internal buffering/ordering.
        Snapshot schema uses `timestamp`; internal engines may require `ts_ms`.
        """
        return int(self.timestamp)



