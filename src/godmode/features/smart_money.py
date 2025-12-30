from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from godmode.core.config import AppConfig
from godmode.core.enums import EpisodePhase
from godmode.core.models import Episode, Snapshot


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _pop_mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _pop_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _pop_mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return sqrt(v)


@dataclass(frozen=True, slots=True)
class SmartMoneyFeatures:
    cvd_10s: float
    cvd_30s: float
    cvd_60s: float
    cvd_30s_slope: float
    cvd_60s_slope: float

    return_10s: float
    return_60s: float
    return_norm: float
    delta_norm: float
    delta_sign: int
    return_sign: int
    divergence_flag: int
    div_score: float

    buy_on_red: float
    sell_on_green: float
    return_sign_10s: int

    vol_60s: float
    trade_count_60s: int
    baseline_vol_60s_mean: float
    baseline_vol_60s_std: float
    vol_z_60s: float
    baseline_trade_count_60s_mean: float
    baseline_trade_count_60s_std: float
    trade_rate_z_60s: float


class SmartMoneyFeatureEngine:
    """
    Addendum I: Smart Money Proxies.

    Computes episode-level flow/response features using the episode snapshot stream:
    - CVD windows (10s/30s/60s) from delta
    - return_10s/return_60s (price diffs), normalization by ATR and baseline volume
    - divergence flag + magnitude score
    - buy_on_red / sell_on_green (normalized with return floor)
    - vol/trade-rate zscores using baseline window stats
    """

    def __init__(self, *, config: AppConfig, snapshot_interval_seconds: int = 10) -> None:
        self._cfg = config
        self._dt = int(snapshot_interval_seconds)
        self._eps = float(config.smart_money.epsilon)
        self._floor_atr = float(config.smart_money.return_floor_atr)

    def compute(
        self,
        *,
        episode: Episode,
        current: Snapshot,
        history: list[Snapshot],
    ) -> SmartMoneyFeatures:
        # Use the last N snapshots (inclusive) deterministically by list order.
        # Expect history in increasing timestamp order for the episode.
        def last_n(snaps: list[Snapshot], n: int) -> list[Snapshot]:
            return snaps[-n:] if len(snaps) >= n else snaps[:]

        all_snaps = history + [current]

        # CVD windows from delta
        cvd_10 = float(current.delta)
        cvd_30 = sum(float(s.delta) for s in last_n(all_snaps, 3))
        cvd_60 = sum(float(s.delta) for s in last_n(all_snaps, 6))

        # Slopes: difference vs previous window value
        if len(history) >= 1:
            prev_all = history
            prev_cvd_30 = sum(float(s.delta) for s in last_n(prev_all, 3))
            prev_cvd_60 = sum(float(s.delta) for s in last_n(prev_all, 6))
            cvd_30_slope = cvd_30 - prev_cvd_30
            cvd_60_slope = cvd_60 - prev_cvd_60
        else:
            cvd_30_slope = 0.0
            cvd_60_slope = 0.0

        # Returns (simple diffs)
        if len(history) >= 1:
            prev_price = float(history[-1].last_price)
            ret_10 = float(current.last_price) - prev_price
        else:
            ret_10 = 0.0

        if len(all_snaps) >= 7:
            price_60_ago = float(all_snaps[-7].last_price)  # t-6 snapshots (60s)
            ret_60 = float(current.last_price) - price_60_ago
        else:
            ret_60 = 0.0

        atr = float(current.atr_value) if float(current.atr_value) > 0 else float(episode.atr_value)
        return_norm = ret_60 / (atr + self._eps) if atr > 0 else 0.0

        # Volume/trade_count 60s rolling sums
        vol_60s = sum(float(s.total_volume) for s in last_n(all_snaps, 6))
        trade_count_60s = int(sum(int(s.trade_count) for s in last_n(all_snaps, 6)))

        # Baseline stats computed from baseline window samples (using vol_60s/trade_count_60s samples).
        baseline_snaps = [s for s in all_snaps if s.phase == EpisodePhase.BASELINE]
        # Build sample series from baseline snaps using their own rolling sums up to that point.
        baseline_vol_samples: list[float] = []
        baseline_trade_samples: list[float] = []
        for i, s in enumerate(baseline_snaps):
            # Use baseline subset up to i as deterministic history for that baseline point
            subset = baseline_snaps[: i + 1]
            baseline_vol_samples.append(sum(float(x.total_volume) for x in subset[-6:]))
            baseline_trade_samples.append(float(sum(int(x.trade_count) for x in subset[-6:])))

        b_vol_mean = _pop_mean(baseline_vol_samples)
        b_vol_std = _pop_std(baseline_vol_samples)
        b_tr_mean = _pop_mean(baseline_trade_samples)
        b_tr_std = _pop_std(baseline_trade_samples)

        vol_z = (vol_60s - b_vol_mean) / (b_vol_std + self._eps) if b_vol_std > 0 else 0.0
        trade_z = (float(trade_count_60s) - b_tr_mean) / (b_tr_std + self._eps) if b_tr_std > 0 else 0.0

        # Divergence normalization
        delta_norm = cvd_60 / (b_vol_mean + self._eps) if b_vol_mean > 0 else 0.0
        ds = _sign(delta_norm)
        rs = _sign(return_norm)
        divergence_flag = 1 if ds != rs else 0
        div_score = abs(delta_norm) / (abs(return_norm) + self._eps) if (abs(return_norm) > 0 or abs(delta_norm) > 0) else 0.0

        # Buy-on-red / sell-on-green
        abs_return = abs(ret_10)
        return_floor = self._floor_atr * atr
        abs_return_adj = max(abs_return, return_floor)
        buy_on_red = float(current.buy_volume) / (abs_return_adj + self._eps) if ret_10 < 0 else 0.0
        sell_on_green = float(current.sell_volume) / (abs_return_adj + self._eps) if ret_10 > 0 else 0.0
        return_sign_10s = _sign(ret_10)

        return SmartMoneyFeatures(
            cvd_10s=float(cvd_10),
            cvd_30s=float(cvd_30),
            cvd_60s=float(cvd_60),
            cvd_30s_slope=float(cvd_30_slope),
            cvd_60s_slope=float(cvd_60_slope),
            return_10s=float(ret_10),
            return_60s=float(ret_60),
            return_norm=float(return_norm),
            delta_norm=float(delta_norm),
            delta_sign=int(ds),
            return_sign=int(rs),
            divergence_flag=int(divergence_flag),
            div_score=float(div_score),
            buy_on_red=float(buy_on_red),
            sell_on_green=float(sell_on_green),
            return_sign_10s=int(return_sign_10s),
            vol_60s=float(vol_60s),
            trade_count_60s=int(trade_count_60s),
            baseline_vol_60s_mean=float(b_vol_mean),
            baseline_vol_60s_std=float(b_vol_std),
            vol_z_60s=float(vol_z),
            baseline_trade_count_60s_mean=float(b_tr_mean),
            baseline_trade_count_60s_std=float(b_tr_std),
            trade_rate_z_60s=float(trade_z),
        )

    def apply_to_snapshot(self, snapshot: Snapshot, feats: SmartMoneyFeatures) -> None:
        snapshot.cvd_10s = feats.cvd_10s
        snapshot.cvd_30s = feats.cvd_30s
        snapshot.cvd_60s = feats.cvd_60s
        snapshot.cvd_30s_slope = feats.cvd_30s_slope
        snapshot.cvd_60s_slope = feats.cvd_60s_slope

        snapshot.return_10s = feats.return_10s
        snapshot.return_60s = feats.return_60s
        snapshot.return_norm = feats.return_norm
        snapshot.delta_norm = feats.delta_norm
        snapshot.delta_sign = feats.delta_sign
        snapshot.return_sign = feats.return_sign
        snapshot.divergence_flag = feats.divergence_flag
        snapshot.div_score = feats.div_score

        snapshot.buy_on_red = feats.buy_on_red
        snapshot.sell_on_green = feats.sell_on_green
        snapshot.return_sign_10s = feats.return_sign_10s

        snapshot.vol_60s = feats.vol_60s
        snapshot.trade_count_60s = feats.trade_count_60s
        snapshot.baseline_vol_60s_mean = feats.baseline_vol_60s_mean
        snapshot.baseline_vol_60s_std = feats.baseline_vol_60s_std
        snapshot.vol_z_60s = feats.vol_z_60s
        snapshot.baseline_trade_count_60s_mean = feats.baseline_trade_count_60s_mean
        snapshot.baseline_trade_count_60s_std = feats.baseline_trade_count_60s_std
        snapshot.trade_rate_z_60s = feats.trade_rate_z_60s


