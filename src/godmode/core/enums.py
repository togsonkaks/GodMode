from __future__ import annotations

from enum import Enum


class EpisodePhase(str, Enum):
    BASELINE = "baseline"
    STRESS = "stress"
    RESOLUTION = "resolution"
    COMPLETE = "complete"


class ResolutionTrigger(str, Enum):
    THRESHOLD_HIT = "threshold_hit"
    INVALIDATION = "invalidation"
    TIMEOUT = "timeout"


class ATRStatus(str, Enum):
    SEEDED = "seeded"
    BLENDING = "blending"
    LIVE = "live"


class ATRSeedSource(str, Enum):
    PRIOR_SESSION = "prior_session"
    DAILY_FALLBACK = "daily_fallback"
    FIXED_DEFAULT = "fixed_default"


class LevelType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    VWAP = "vwap"
    TRENDLINE = "trendline"


class LevelSource(str, Enum):
    MANUAL = "manual"
    DERIVED = "derived"
    PRIOR_LOW = "prior_low"
    VWAP = "vwap"
    TRENDLINE = "trendline"


class DirectionBias(str, Enum):
    LONG = "long"
    SHORT = "short"


class VWAPState(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    AT = "at"


class ApproachVelocity(str, Enum):
    CRASH = "crash"
    GRIND = "grind"


class Outcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    SCRATCH = "scratch"
    NO_TRADE = "no-trade"


class ResolutionType(str, Enum):
    REVERSAL_SUCCESS = "reversal_success"
    FAKE_BREAK = "fake_break"
    CONTINUATION = "continuation"
    CHOP = "chop"



