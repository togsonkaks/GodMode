from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from godmode.core.enums import DirectionBias
from godmode.core.models import Episode, Snapshot


def _sign(x: float, *, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


@dataclass(frozen=True, slots=True)
class LevelSpineFeatures:
    max_penetration_atr: float
    cross_count_60s: int
    cross_density: float
    oscillation_amplitude_atr_60s: float
    time_in_zone_rolling: float
    total_time_in_zone_episode: float
    avg_time_per_touch: float


@dataclass(slots=True)
class _EpisodeSpineState:
    max_penetration_atr: float = 0.0
    total_time_in_zone_episode: float = 0.0


class LevelSpineFeatureEngine:
    """
    Computes the Level Interaction Spine features (SPEC §5A) deterministically.

    Definitions (deterministic, implementation-level):
    - cross_count_60s: number of sign changes of signed_distance_to_level_atr across the last 60 seconds of snapshots.
      - sign(x) uses eps=1e-12; zeros do not create crossings and are ignored in sign-change counting.
    - cross_density: cross_count_60s / 60.0 (crosses per second).
    - oscillation_amplitude_atr_60s: max(signed_distance_to_level_atr) - min(signed_distance_to_level_atr) over last 60s.
    - time_in_zone_rolling: seconds spent in-zone over last 60 seconds (snapshots cadence-based).
    - total_time_in_zone_episode: cumulative seconds in-zone across the episode so far (cadence-based).
    - avg_time_per_touch: total_time_in_zone_episode / touch_count (0 if touch_count==0).
    - max_penetration_atr: maximum penetration to the *opposite side* of the entry side, in ATR units.
      - If entry side is 'above': penetration is max(0, -signed_distance_to_level_atr).
      - If entry side is 'below': penetration is max(0, +signed_distance_to_level_atr).

    Note: 60s windows align with other 60s features in the spec (e.g., cross_count_60s).
    """

    def __init__(self, *, snapshot_interval_ms: int = 10_000, rolling_window_seconds: int = 60) -> None:
        if snapshot_interval_ms <= 0:
            raise ValueError("snapshot_interval_ms must be > 0")
        if rolling_window_seconds <= 0:
            raise ValueError("rolling_window_seconds must be > 0")
        self._dt_s = snapshot_interval_ms / 1000.0
        self._win_ms = int(rolling_window_seconds * 1000)

        self._state: dict[str, _EpisodeSpineState] = {}

    def compute(
        self,
        *,
        episode: Episode,
        current: Snapshot,
        history: list[Snapshot],
        in_zone: bool,
        touch_count: int,
    ) -> LevelSpineFeatures:
        """
        Compute spine features for `current` snapshot.

        `history` should include prior snapshots for the same episode in deterministic order.
        """
        st = self._state.setdefault(episode.episode_id, _EpisodeSpineState())

        # === max_penetration_atr ===
        entry_side = (episode.level_entry_side or "").lower()
        signed_atr = float(current.signed_distance_to_level_atr)
        if entry_side == "above":
            penetration = max(0.0, -signed_atr)
        elif entry_side == "below":
            penetration = max(0.0, signed_atr)
        else:
            # If not known, treat penetration as abs distance beyond 0 in either direction.
            penetration = abs(signed_atr)
        st.max_penetration_atr = max(st.max_penetration_atr, penetration)

        # === rolling window selection ===
        end_ms = int(current.timestamp)
        start_ms = end_ms - self._win_ms
        window_snaps = [s for s in history if start_ms <= int(s.timestamp) <= end_ms] + [current]

        # === cross_count_60s ===
        # Count sign changes ignoring zeros (eps-based).
        signs: list[int] = []
        for s in window_snaps:
            sig = _sign(float(s.signed_distance_to_level_atr))
            if sig != 0:
                signs.append(sig)
        cross_count = 0
        for a, b in zip(signs, signs[1:]):
            if a != b:
                cross_count += 1

        cross_density = cross_count / 60.0

        # === oscillation amplitude ===
        vals = [float(s.signed_distance_to_level_atr) for s in window_snaps]
        osc_amp = (max(vals) - min(vals)) if vals else 0.0

        # === time in zone ===
        # cadence-based: each snapshot represents dt_s seconds.
        time_in_zone_rolling = sum(self._dt_s for s in window_snaps if abs(float(s.signed_distance_to_level_atr)) <= episode.level_width_atr)
        if in_zone:
            st.total_time_in_zone_episode += self._dt_s

        avg_time_per_touch = (st.total_time_in_zone_episode / float(touch_count)) if touch_count > 0 else 0.0

        return LevelSpineFeatures(
            max_penetration_atr=float(st.max_penetration_atr),
            cross_count_60s=int(cross_count),
            cross_density=float(cross_density),
            oscillation_amplitude_atr_60s=float(osc_amp),
            time_in_zone_rolling=float(time_in_zone_rolling),
            total_time_in_zone_episode=float(st.total_time_in_zone_episode),
            avg_time_per_touch=float(avg_time_per_touch),
        )

    def apply_to_snapshot(self, snapshot: Snapshot, features: LevelSpineFeatures) -> None:
        snapshot.max_penetration_atr = features.max_penetration_atr
        snapshot.cross_count_60s = features.cross_count_60s
        snapshot.cross_density = features.cross_density
        snapshot.oscillation_amplitude_atr_60s = features.oscillation_amplitude_atr_60s
        snapshot.time_in_zone_rolling = features.time_in_zone_rolling
        snapshot.total_time_in_zone_episode = features.total_time_in_zone_episode
        snapshot.avg_time_per_touch = features.avg_time_per_touch


