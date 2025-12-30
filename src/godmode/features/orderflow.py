from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from godmode.core.models import Snapshot, Trade


def _pop_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return sqrt(v)


@dataclass(frozen=True, slots=True)
class OrderFlowFeatures:
    relative_aggression_zscore_60s: float
    delta_velocity: float
    delta_acceleration: float
    absorption_index_10s: float
    avg_trade_size: float
    trade_size_std: float
    trade_size_cv: float
    top_decile_volume_share: float


class OrderFlowFeatureEngine:
    """
    Order flow feature additions for Day 11.

    Deterministic definitions:
    - relative_aggression_zscore_60s: population z-score over last 60s of snapshots (incl current).
    - absorption_index_10s: max(0, -delta) / (abs(price_return_10s) + epsilon)
    - trade size structure computed from trades in the current 10s snapshot window.
    """

    def __init__(self, *, rolling_window_seconds: int = 60, epsilon: float = 1e-9) -> None:
        self._win_ms = int(rolling_window_seconds * 1000)
        self._eps = float(epsilon)

        # Per-episode state for delta derivatives and last price.
        self._prev_delta: dict[str, float] = {}
        self._prev_delta_velocity: dict[str, float] = {}
        self._prev_last_price: dict[str, float] = {}

    def compute(
        self,
        *,
        episode_id: str,
        current: Snapshot,
        history: list[Snapshot],
        window_trades: list[Trade],
    ) -> OrderFlowFeatures:
        end_ms = int(current.timestamp)
        start_ms = end_ms - self._win_ms
        window_snaps = [s for s in history if start_ms <= int(s.timestamp) <= end_ms] + [current]

        # relative_aggression_zscore_60s
        xs = [float(s.relative_aggression) for s in window_snaps]
        std = _pop_std(xs)
        if len(xs) < 2 or std == 0.0:
            z = 0.0
        else:
            m = sum(xs) / len(xs)
            z = (float(current.relative_aggression) - m) / (std + self._eps)

        # delta derivatives
        prev_delta = self._prev_delta.get(episode_id)
        if prev_delta is None:
            dv = 0.0
        else:
            dv = float(current.delta) - prev_delta

        prev_dv = self._prev_delta_velocity.get(episode_id)
        if prev_dv is None:
            da = 0.0
        else:
            da = dv - prev_dv

        self._prev_delta[episode_id] = float(current.delta)
        self._prev_delta_velocity[episode_id] = float(dv)

        # price_return_10s (deterministic, per episode)
        prev_price = self._prev_last_price.get(episode_id)
        if prev_price is None or prev_price == 0.0 or float(current.last_price) == 0.0:
            price_ret = 0.0
        else:
            price_ret = float(current.last_price) - prev_price
        self._prev_last_price[episode_id] = float(current.last_price)

        absorption = max(0.0, -float(current.delta)) / (abs(price_ret) + self._eps)

        # trade size stats from the current 10s window trades
        sizes = [float(t.size) for t in window_trades if float(t.size) > 0.0]
        if not sizes:
            avg = std_sz = cv = top_dec_share = 0.0
        else:
            avg = sum(sizes) / len(sizes)
            std_sz = _pop_std(sizes)
            cv = (std_sz / avg) if avg > 0 else 0.0

            total_vol = sum(sizes)
            if total_vol <= 0:
                top_dec_share = 0.0
            else:
                # top decile by trade size: take the largest ceil(10%) trades by size and compute their volume share.
                k = max(1, int((len(sizes) * 0.1) + 0.999999))
                top = sorted(sizes, reverse=True)[:k]
                top_dec_share = sum(top) / total_vol

        return OrderFlowFeatures(
            relative_aggression_zscore_60s=float(z),
            delta_velocity=float(dv),
            delta_acceleration=float(da),
            absorption_index_10s=float(absorption),
            avg_trade_size=float(avg),
            trade_size_std=float(std_sz),
            trade_size_cv=float(cv),
            top_decile_volume_share=float(top_dec_share),
        )

    def apply_to_snapshot(self, snapshot: Snapshot, feats: OrderFlowFeatures) -> None:
        snapshot.relative_aggression_zscore_60s = feats.relative_aggression_zscore_60s
        snapshot.delta_velocity = feats.delta_velocity
        snapshot.delta_acceleration = feats.delta_acceleration
        snapshot.absorption_index_10s = feats.absorption_index_10s
        snapshot.avg_trade_size = feats.avg_trade_size
        snapshot.trade_size_std = feats.trade_size_std
        snapshot.trade_size_cv = feats.trade_size_cv
        snapshot.top_decile_volume_share = feats.top_decile_volume_share


