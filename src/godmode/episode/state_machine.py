from __future__ import annotations

from typing import Optional
from uuid import uuid4

from godmode.core.config import AppConfig
from godmode.core.enums import (
    ATRSeedSource,
    ATRStatus,
    DirectionBias,
)
from godmode.core.models import Episode
from godmode.episode.resolution import evaluate_resolution
from godmode.zone.zone_gate import ZoneEvent, ZoneEventType


class EpisodeStateMachine:
    """
    Deterministic episode lifecycle driven by ZoneGate events.

    Phases:
    - Baseline: [zone_entry_time - baseline_window_ms, zone_entry_time)
    - Stress:   [zone_entry_time, zone_exit_time] (if exit occurs)
    - Resolution: after exit until trigger (threshold_hit|invalidation|timeout)

    Notes:
    - This does NOT compute snapshots; SnapshotBuilder (Day 9) consumes episode timing.
    - direction_bias is required at episode start (do not guess long/short).
    """

    def __init__(
        self,
        *,
        ticker: str,
        session_id: str,
        config: AppConfig,
        baseline_window_seconds: int = 300,
    ) -> None:
        self._ticker = ticker
        self._session_id = session_id
        self._cfg = config
        self._baseline_ms = int(baseline_window_seconds) * 1000

        # One active episode per level_id.
        self._active: dict[str, Episode] = {}

    def on_zone_event(
        self,
        *,
        event: ZoneEvent,
        direction_bias: DirectionBias,
        atr_value: float,
        atr_status: ATRStatus,
        atr_seed_source: Optional[ATRSeedSource],
        atr_is_warm: bool,
    ) -> Optional[Episode]:
        """
        Process a ZoneGate event and start/update an episode.
        Returns a newly started episode on ENTER, otherwise None.
        """
        if event.ticker != self._ticker:
            raise ValueError("ticker mismatch")

        if event.type == ZoneEventType.ENTER:
            episode_id = str(uuid4())
            start_time = int(event.ts_ms - self._baseline_ms)

            ep = Episode(
                episode_id=episode_id,
                session_id=self._session_id,
                ticker=self._ticker,
                episode_source="level_gate",
                level_id=event.level_id,
                level_price=float(event.level_price),
                level_type=str(event.level_type),
                level_source=str(event.level_source),
                level_width_atr=float(event.level_width_atr),
                level_entry_side=str(event.level_entry_side or ""),
                start_time=start_time,
                zone_entry_time=int(event.ts_ms),
                zone_exit_time=None,
                resolution_time=None,
                end_time=None,
                zone_rule="abs(signed_distance_to_level_atr) <= 0.25",
                resolution_trigger=None,
                direction_bias=direction_bias,
                success_threshold_atr=self._cfg.resolution.success_threshold_atr,
                failure_threshold_atr=self._cfg.resolution.failure_threshold_atr,
                timeout_seconds=self._cfg.resolution.timeout_seconds,
                ema_confluence_ref=self._cfg.ema.confluence_ref,
                atr_value=float(atr_value),
                atr_status=atr_status,
                atr_seed_source=atr_seed_source,
                atr_blend_alpha=self._cfg.atr.atr_blend_alpha,
                atr_is_warm=bool(atr_is_warm),
            )

            self._active[event.level_id] = ep
            return ep

        if event.type == ZoneEventType.EXIT:
            ep = self._active.get(event.level_id)
            if ep is None:
                return None
            if ep.zone_exit_time is None:
                ep.zone_exit_time = int(event.ts_ms)
            return None

        return None

    def update_resolution(
        self,
        *,
        level_id: str,
        ts_ms: int,
        last_price: float,
        atr_value: float,
    ) -> Optional[Episode]:
        """
        Call on each snapshot tick once an episode has exited the zone.
        When resolution occurs, returns the finalized episode and removes it from active.
        """
        ep = self._active.get(level_id)
        if ep is None:
            return None

        # Only resolve after zone_exit_time exists.
        decision = evaluate_resolution(
            ts_ms=int(ts_ms),
            last_price=float(last_price),
            level_price=float(ep.level_price),
            atr_value=float(atr_value),
            direction_bias=ep.direction_bias,
            success_threshold_atr=ep.success_threshold_atr,
            failure_threshold_atr=ep.failure_threshold_atr,
            zone_exit_time=ep.zone_exit_time,
            timeout_seconds=ep.timeout_seconds,
        )
        if decision is None:
            return None

        ep.resolution_time = int(decision.resolution_time)
        ep.resolution_trigger = decision.trigger
        ep.end_time = int(decision.resolution_time)

        # Episode is finalized; remove from active.
        del self._active[level_id]
        return ep

    def get_active(self, level_id: str) -> Optional[Episode]:
        return self._active.get(level_id)


