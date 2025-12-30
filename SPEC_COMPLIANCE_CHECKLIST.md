# Spec Compliance Checklist (v1.1.1)

This checklist is **normative**. “Pass” means the implementation matches `SPEC.md` exactly and does not introduce new behavior.

## A) Determinism & Time Alignment (Live == Replay)

- [ ] **Event schema**: every trade/quote event includes **`ts_ms`** and **`symbol`**.
- [ ] **Event-time only**: all aggregation and snapshot timestamps use **`ts_ms`**, not receipt-time.
- [ ] **Ordering**: streams are processed in non-decreasing `ts_ms`; ties use a **stable tie-breaker** (e.g., sequence counter).
- [ ] **Snapshot cadence**: snapshots are built on deterministic **10-second boundaries** in event-time.
- [ ] **Quote sampling**: for each 10s snapshot, use the **latest NBBO quote observed in that 10s window** (Addendum D).
- [ ] **quote_age_ms**: if implemented, is derived from snapshot end-time minus latest quote `ts_ms` (event-time).

## B) Bars, ATR, and Cold Start (Deterministic + Auditable)

- [ ] **1-minute bars** are built internally from trades (Addendum A1): open=first, high=max, low=min, close=last, volume=sum.
- [ ] **Session rules**: RTH-only by default; extended hours controlled by `include_extended_hours`.
- [ ] **ATR_14_1m** is computed from internal 1-minute bars (Addendum A2).
- [ ] **ATR seed definition**: `ATR_seed = prior_session_RTH_ATR_14_1m` built from the same bar builder (Bulletproofing B).
- [ ] **Seed fallback**: if seed missing, use `daily_ATR / sqrt(390)` or fixed default; store `atr_seed_source`.
- [ ] **No 15-min wait**: zone gating does not block on warm ATR.
- [ ] **Blending policy** (Implementation decision #2):
  - [ ] If `bars_today >= 14`: ATR is **live**.
  - [ ] Else: `ATR = alpha * ATR_seed + (1 - alpha) * ATR_partial`.
  - [ ] `atr_blend_alpha` default **0.7**.
- [ ] **Stored audit fields** per episode: `atr_status`, `atr_seed_source`, `atr_blend_alpha`, `atr_is_warm`.

## C) Zone Gating & Touch Count

- [ ] **Zone rule** matches spec: `abs(signed_distance_to_level_atr) <= 0.25` (or configurable but default must be 0.25).
- [ ] **Touch count** matches Addendum C:
  - [ ] touch increments on outside→inside (entry) AND inside→outside for **≥ 1 full snapshot** (exit).

## D) Episode State Machine & Resolution

- [ ] **Phases** exist and match: Baseline (3–5m pre-entry), Stress (entry→interaction), Resolution (post exit up to window).
- [ ] **Deterministic timestamps stored**: `zone_entry_time`, `zone_exit_time`, `resolution_time`, `resolution_trigger`.
- [ ] **Resolution triggers** only: `threshold_hit | invalidation | timeout`.
- [ ] **Default thresholds stored per episode** (Addendum B):
  - [ ] `success_threshold_atr = 0.50`
  - [ ] `failure_threshold_atr = 0.35`
  - [ ] `timeout_seconds = 300`

## E) Order Flow Classification (Quote Test)

- [ ] Buy/sell classification exactly matches spec:
  - [ ] `trade >= ask → buy`
  - [ ] `trade <= bid → sell`
  - [ ] else `unknown`
- [ ] Store all buckets: buy, sell, unknown volumes.

## F) Features & Schema Fidelity

- [ ] All required snapshot fields exist per `SPEC.md` §5 (A–G).
- [ ] All distances/volatility are normalized in **ATR units or %** as specified.
- [ ] `%_at_ask` and `%_at_bid` are represented in storage schema exactly; Python-safe internal naming uses explicit serialization aliasing.

## G) Storage (Parquet) & Atomicity

- [ ] Parquet output uses partitioning exactly:
  - [ ] `snapshots/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`
  - [ ] `episodes/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`
  - [ ] `labels/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`
- [ ] **Atomic write**: write temp then rename (`.tmp.parquet → .parquet`).
- [ ] Compression: **zstd preferred** (snappy acceptable if needed).

## H) Replay/Backtesting Parity

- [ ] Replay is treated as another provider; **same engine**, different feed.
- [ ] Live and replay share ZoneGating, Episode engine, SnapshotBuilder, Feature engine.



