from __future__ import annotations

import math

import pytest

from godmode.core.enums import ATRSeedSource, ATRStatus
from godmode.core.models import Bar
from godmode.engine.atr import ATRCalculator, daily_atr_fallback_seed, true_range


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float = 1.0, symbol: str = "AAPL") -> Bar:
    return Bar(ts_ms=ts_ms, symbol=symbol, open=o, high=h, low=l, close=c, volume=v, trade_count=1)


def test_true_range_matches_spec() -> None:
    # No prev close -> TR = high-low
    assert true_range(high=12, low=10, prev_close=None) == 2

    # With prev close -> max of three terms
    tr = true_range(high=12, low=10, prev_close=20)
    assert tr == max(2, abs(12 - 20), abs(10 - 20))


def test_requires_seed_before_update() -> None:
    atr = ATRCalculator(period_bars=14, atr_blend_alpha=0.7)
    with pytest.raises(RuntimeError):
        atr.update(_bar(0, 10, 11, 9, 10))


def test_seeded_then_blending_then_live_status() -> None:
    atr = ATRCalculator(period_bars=3, atr_blend_alpha=0.7)
    atr.reset_session(atr_seed=10.0, seed_source=ATRSeedSource.PRIOR_SESSION)

    # 1st bar: partial exists immediately, so we are blending (not seeded)
    s1 = atr.update(_bar(0, 10, 12, 10, 11))
    assert s1.status == ATRStatus.BLENDING
    assert s1.is_warm is False

    # 2nd bar: still not warm (needs 3)
    s2 = atr.update(_bar(60_000, 11, 13, 11, 12))
    assert s2.status == ATRStatus.BLENDING
    assert s2.is_warm is False

    # 3rd bar: warm -> live
    s3 = atr.update(_bar(120_000, 12, 14, 12, 13))
    assert s3.status == ATRStatus.LIVE
    assert s3.is_warm is True
    assert s3.bars_today == 3


def test_blending_formula_uses_alpha_and_partial() -> None:
    atr = ATRCalculator(period_bars=14, atr_blend_alpha=0.7)
    atr.reset_session(atr_seed=10.0, seed_source=ATRSeedSource.PRIOR_SESSION)

    # First bar: prev_close None, TR = high-low = 2, partial=2
    s1 = atr.update(_bar(0, 10, 12, 10, 11))
    expected = 0.7 * 10.0 + 0.3 * 2.0
    assert math.isclose(s1.value, expected, rel_tol=1e-12)


def test_daily_atr_fallback_seed() -> None:
    seed = daily_atr_fallback_seed(daily_atr=3.9)
    assert math.isclose(seed, 3.9 / math.sqrt(390), rel_tol=1e-12)


