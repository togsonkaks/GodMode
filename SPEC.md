# NUMERIC MARKET-STATE RECORDER — Complete Specification

> **Version**: 1.1.1 (Final Frozen Spec + Implementation Decisions)  
> **Project Codename**: GodMode  
> **Created**: December 22, 2025  
> **Status**: LOCKED — All decisions finalized

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Strategic Philosophy](#2-core-strategic-philosophy)
3. [System Architecture](#3-system-architecture)
4. [Data Model](#4-data-model)
5. [Feature Set](#5-feature-set)
6. [Buy/Sell Classification Rule](#6-buysell-classification-rule)
7. [Excluded By Design](#7-excluded-by-design)
8. [Data Collection Rules](#8-data-collection-rules)
9. [AI Analysis Objective](#9-ai-analysis-objective)
10. [Execution Roadmap](#10-execution-roadmap)
11. [Anti-Overfitting Rules](#11-anti-overfitting-rules)
12. [Final Positioning](#12-final-positioning)
13. [Addendum A — Indicator Source Definitions](#addendum-a--indicator-source-definitions)
14. [Addendum B — Resolution Threshold Defaults](#addendum-b--resolution-threshold-defaults)
15. [Addendum C — Touch Count Definition](#addendum-c--touch-count-definition)
16. [Addendum D — Quote/Spread Sampling Policy](#addendum-d--quotespread-sampling-policy)
17. [Addendum E — Episode Context Fields](#addendum-e--episode-context-fields)
18. [Implementation Decisions](#implementation-decisions)
19. [Bulletproofing Micro-Edits](#bulletproofing-micro-edits)
20. [Project Structure](#project-structure)
21. [Data Models (Code)](#data-models-code)
22. [Tech Stack](#tech-stack)

---

## 1) INTRODUCTION

### Purpose & Problem Statement

Build a numeric market-state recording system to solve a specific trading failure mode:

**High-quality reversals are missed because confirmation arrives too late.**

Traditional candle-based analysis captures outcomes, not causes. Candles summarize what already happened; they do not explain why price changed when it did.

This system records **pressure, structure, velocity, and interaction behavior** *before price resolves*, enabling human + AI analysis to identify early reversal fingerprints—especially around:

- Downtrend breaks
- Reclaim attempts
- Key horizontal levels
- VWAP zones

**This is not a trading bot.**  

This is a **market research engine** whose outputs can later inform discretionary trading, alerting systems, or automated decision layers.

---

## 2) CORE STRATEGIC PHILOSOPHY

- **Candles are outputs, not inputs**
- **Reversals are regime shifts, not confirmations**
- **Contrast is king** (baseline vs stressed state)
- **Velocity > statics** (derivatives matter)
- **Interaction > touch**
- **Fewer features, higher signal**

---

## 3) SYSTEM ARCHITECTURE (THE ENGINE)

### 3.1 Watchlist Monitoring

- Maintain a **lightweight ring buffer in RAM per ticker**.
- Buffer holds the last ~5 minutes of **computed snapshots** (optionally raw trades/quotes).
- Buffer is always-on but **not persisted**.

### 3.2 Zone Gating (Critical)

Persist data **only** when price enters an ATR-normalized zone around a user-defined level.

Example zone rule:

- `abs(signed_distance_to_level_atr) <= 0.25`

### 3.3 Episode State Machine (Deterministic)

Each interaction with a level is captured as one Episode with phases:

- **Baseline Window**: 3–5 minutes pre zone-entry
- **Stress Window**: zone-entry through interaction
- **Resolution Window**: up to 5 minutes after first of threshold/invalidation/timeout

Deterministic timestamps:

- `zone_entry_time`
- `zone_exit_time`
- `resolution_time`
- `resolution_trigger`: `threshold_hit | invalidation | timeout`

---

## 4) DATA MODEL (AI-READY)

### 4.1 Level Object (Required)

| Field | Description |
|-------|-------------|
| `level_id` | Unique identifier |
| `level_price` | Price of the level |
| `level_type` | support \| resistance \| vwap \| trendline |
| `level_source` | manual \| derived \| prior_low \| vwap \| trendline |
| `level_width_atr` | Zone width in ATR units (default 0.25) |
| `level_entry_side` | above \| below (at zone entry) |

### 4.2 Episodes

**Metadata:**

| Field | Description |
|-------|-------------|
| `episode_id` | Unique identifier |
| `session_id` | Session identifier |
| `ticker` | Symbol |
| `level_id` | Reference to level |
| `level_price` | Price of the level |
| `level_type` | Type of level |
| `level_source` | Source of level |
| `level_width_atr` | Zone width |
| `level_entry_side` | above \| below (at zone entry) |
| `direction_bias` | long \| short |
| `zone_rule` | Zone entry rule used |
| `start_time` | Episode start (baseline begin) |
| `zone_entry_time` | When price entered zone |
| `zone_exit_time` | When price exited zone |
| `resolution_time` | When resolution occurred |
| `end_time` | Episode end |
| `resolution_trigger` | threshold_hit \| invalidation \| timeout |

**Outcomes (objective):**

| Field | Description |
|-------|-------------|
| `outcome` | win \| loss \| scratch \| no-trade |
| `resolution_type` | reversal_success \| fake_break \| continuation \| chop |
| `R_multiple` | Risk multiple (if trade-based) |
| `MFE` | Maximum Favorable Excursion |
| `MAE` | Maximum Adverse Excursion |
| `time_to_MFE` | Time to MFE (milliseconds) |
| `time_to_failure` | Time to failure (milliseconds) |

**Manual labels stored separately.**

**Context fields (recommended):**

| Field | Description |
|-------|-------------|
| `time_of_day_bucket` | open \| mid \| close |
| `gap_pct` | Gap vs prior close |
| `spy_return_5m` | Optional: SPY 5-min return |
| `spy_bucket` | Optional: SPY state bucket |

**Threshold fields (auditability):**

| Field | Default |
|-------|---------|
| `success_threshold_atr` | 0.50 |
| `failure_threshold_atr` | 0.35 |
| `timeout_seconds` | 300 |

### 4.3 Snapshots

- Every **10 seconds**
- Linked by `episode_id`
- Includes baseline + stress + resolution phases
- Must include: `timestamp`, `sequence_id`

---

## 5) FEATURE SET (FINAL — V1.1.1)

### Global Normalization Rule

Distances/volatility must be in **ATR units** or **%**.

---

### A) LEVEL INTERACTION SPINE

| Feature | Description |
|---------|-------------|
| `signed_distance_to_level` | Raw distance to level |
| `signed_distance_to_level_atr` | Distance in ATR units |
| `abs_distance_to_level_atr` | Absolute distance in ATR |
| `max_penetration_atr` | Max penetration through level |

**Interaction/oscillation:**

| Feature | Description |
|---------|-------------|
| `cross_count_60s` | Level crosses in last 60s |
| `cross_density` | Cross frequency |
| `oscillation_amplitude_atr_60s` | Price oscillation range |

**Time in zone:**

| Feature | Description |
|---------|-------------|
| `time_in_zone_rolling` | Rolling time in zone |
| `total_time_in_zone_episode` | Total episode time in zone |
| `touch_count` | Number of zone touches |
| `avg_time_per_touch` | Average duration per touch |

---

### B) PRICE & SPREAD MICROSTRUCTURE

| Feature | Description |
|---------|-------------|
| `last_price` | Last trade price |
| `bid` | Best bid |
| `ask` | Best ask |
| `mid_price` | (bid + ask) / 2 |
| `spread_abs` | ask - bid |
| `spread_pct` | spread / mid_price |
| `spread_volatility_60s` | Spread volatility |
| `spread_zscore` | Optional: Z-score of spread |
| `quote_age_ms` | Optional: Age of quote in ms |

**Deterministic definitions (rolling microstructure):**

- Window: last **60 seconds** of snapshots (inclusive of current), aligned to snapshot cadence.
- Only include samples where `bid > 0` and `ask > 0` (i.e., valid NBBO observed).
- `spread_volatility_60s`: population standard deviation of `spread_pct` over the window (ddof = 0). If <2 valid samples, use 0.
- `spread_zscore` (optional): \((spread_pct - mean(spread_pct_{60s})) / (spread_volatility_60s + epsilon)\), epsilon = 1e-9. If `spread_volatility_60s == 0`, use 0.

---

### C) VWAP (Context)

| Feature | Description |
|---------|-------------|
| `vwap_session` | Session VWAP |
| `price_minus_vwap` | Distance from VWAP |
| `price_minus_vwap_atr` | Distance in ATR units |
| `vwap_state` | above \| below \| at |

---

### D) EMA STRUCTURE (No Crosses)

| Feature | Description |
|---------|-------------|
| `ema9` | 9-period EMA |
| `ema20` | 20-period EMA |
| `ema30` | 30-period EMA |
| `ema200` | 200-period EMA |
| `slope_ema9_60s` | EMA9 slope over 60s |
| `slope_ema20_60s` | EMA20 slope over 60s |
| `slope_ema30_60s` | EMA30 slope over 60s |
| `ema_spread_9_20` | EMA9 - EMA20 |
| `ema_spread_20_30` | EMA20 - EMA30 |
| `compression_index` | EMA compression measure |
| `ema_confluence_score` | EMA confluence score |
| `stack_state` | EMA stack configuration |
| `price_vs_emas` | Price position vs EMAs |
| `stretch_200_atr` | `(last_price - ema200) / ATR_14_1m` |

**Deterministic definitions (EMA structure):**

- `slope_emaX_60s` uses **per-second slope** over 60 seconds:
  - `slope_emaX_60s = (emaX_t - emaX_{t-60s}) / 60`

- `compression_index` is EMA gap sum normalized by ATR:
  - `compression_index = (abs(ema9-ema20) + abs(ema20-ema30)) / ATR_14_1m`

- `ema_confluence_score` is a bounded 0–1 squeeze score (lower compression ⇒ higher score):
  - `ema_confluence_score = clamp01(1 - compression_index / confluence_ref)`
  - `confluence_ref = 0.25` (config constant; **store per episode for auditability**)

- `stack_state` rules (use 9/20/30 only; 200 is separate via stretch):
  - `bull` if `ema9 > ema20 > ema30`
  - `bear` if `ema9 < ema20 < ema30`
  - else `mixed`

- `price_vs_emas` is an integer **bitmask** in fixed order `[9,20,30,200]`:
  - `bit0 = 1 if price > ema9 else 0`
  - `bit1 = 1 if price > ema20 else 0`
  - `bit2 = 1 if price > ema30 else 0`
  - `bit3 = 1 if price > ema200 else 0`
  - `price_vs_emas = bit0 + 2*bit1 + 4*bit2 + 8*bit3`

---

### E) TIME & SALES / ORDER FLOW

**Volume/flow:**

| Feature | Description |
|---------|-------------|
| `trade_count` | Number of trades |
| `total_volume` | Total volume |
| `buy_volume` | Volume at ask |
| `sell_volume` | Volume at bid |
| `unknown_volume` | Unclassified volume |
| `delta` | buy_volume - sell_volume |

**Aggression:**

| Feature | Description |
|---------|-------------|
| `%_at_ask` | Percent volume at ask |
| `%_at_bid` | Percent volume at bid |
| `relative_aggression` | Aggression ratio |
| `relative_aggression_zscore_60s` | Z-score over 60s |

**Serialization note (determinism + practicality):** the official storage column names remain `%_at_ask` and `%_at_bid`. In Python code, use safe identifiers (e.g. `pct_at_ask`, `pct_at_bid`) and serialize with explicit column aliasing so Parquet output matches the official schema exactly.

**Velocity/acceleration:**

| Feature | Description |
|---------|-------------|
| `delta_velocity` | `delta[t] - delta[t-1]` |
| `delta_acceleration` | `delta_velocity[t] - delta_velocity[t-1]` |

**Deterministic definitions (rolling order flow):**

- Window: last **60 seconds** of snapshots (inclusive of current), aligned to snapshot cadence.
- `relative_aggression_zscore_60s`: z-score of `relative_aggression` over the window using population std dev (ddof = 0) and epsilon = 1e-9. If <2 samples or std==0, use 0.

---

## ADDENDUM I — Smart Money Proxies (Recommended)

These are **behavioral proxies** for accumulation/distribution using only trades/quotes (no L2). They do **not** identify “smart money” directly; they quantify **flow vs price response** in deterministic, AI-friendly form.

### I1) CVD windows (10s / 30s / 60s)

Let `delta_10s` be the snapshot’s `delta` value.

- `cvd_10s = delta_10s`
- `cvd_30s = sum(delta_10s over last 3 snapshots)` (inclusive)
- `cvd_60s = sum(delta_10s over last 6 snapshots)` (inclusive)

Slopes (first differences):

- `cvd_30s_slope = cvd_30s[t] - cvd_30s[t-1]`
- `cvd_60s_slope = cvd_60s[t] - cvd_60s[t-1]`

### I2) Divergence score (normalized) + raw flags

Compute:

- `return_10s = last_price[t] - last_price[t-1]`
- `return_60s = last_price[t] - last_price[t-6]`
- `return_norm = return_60s / (ATR_14_1m + eps)`

Normalize delta by baseline activity:

- `vol_60s = sum(total_volume over last 6 snapshots)`
- `baseline_vol_60s_mean` and `baseline_vol_60s_std` are computed over the **Baseline window** using `vol_60s` samples (population mean/std, ddof=0).
- `delta_norm = cvd_60s / (baseline_vol_60s_mean + eps)`

Then:

- `delta_sign = sign(delta_norm)` in {-1,0,+1}
- `return_sign = sign(return_norm)` in {-1,0,+1}
- `divergence_flag = 1 if delta_sign != return_sign else 0`
- `div_score = abs(delta_norm) / (abs(return_norm) + eps)`

Store: `delta_norm`, `return_norm`, `delta_sign`, `return_sign`, `divergence_flag`, `div_score`.

### I3) Buy-on-red / Sell-on-green (normalized)

Prevent blowups on flat returns:

- `abs_return = abs(return_10s)`
- `return_floor = floor_atr * ATR_14_1m` where `floor_atr` default = `0.02` (config constant)
- `abs_return_adj = max(abs_return, return_floor)`

Then:

- `buy_on_red = buy_volume / (abs_return_adj + eps) if return_10s < 0 else 0`
- `sell_on_green = sell_volume / (abs_return_adj + eps) if return_10s > 0 else 0`

Store also: `return_sign_10s = sign(return_10s)`.

### I4) Large-trade imbalance (rolling p95 over 5m)

Compute a per-ticker large trade threshold:

- `large_trade_threshold_size = rolling_p95_trade_size_5m`

Where `rolling_p95_trade_size_5m` is computed from all trade sizes in the last **5 minutes** of trades (event-time), using deterministic nearest-rank:

- sort sizes ascending
- index = ceil(0.95*N) - 1

Then, within the current 10s snapshot window, for trades with `size >= large_trade_threshold_size`, compute:

- `large_trade_count_10s`
- `large_buy_volume_10s`, `large_sell_volume_10s`, `large_unknown_volume_10s`
- `large_trade_buy_ratio = large_buy_volume / (large_buy_volume + large_sell_volume + eps)`
- `large_trade_delta = large_buy_volume - large_sell_volume`
- `large_trade_share_of_total_vol_10s = (large_buy + large_sell + large_unknown) / (total_volume + eps)`

### I5) Volume abnormality z-score (baseline = episode baseline)

Using the Baseline window (population stats, ddof=0):

- `vol_z_60s = (vol_60s - baseline_vol_60s_mean) / (baseline_vol_60s_std + eps)`
- `trade_count_60s = sum(trade_count over last 6 snapshots)`
- `baseline_trade_count_60s_mean`, `baseline_trade_count_60s_std`
- `trade_rate_z_60s = (trade_count_60s - baseline_trade_count_60s_mean) / (baseline_trade_count_60s_std + eps)`


**Absorption:**

| Feature | Formula |
|---------|---------|
| `absorption_index_10s` | `max(0, -delta) / (abs(price_return_10s) + epsilon)` |

**Trade size structure:**

| Feature | Description |
|---------|-------------|
| `avg_trade_size` | Average trade size |
| `trade_size_std` | Trade size std dev |
| `trade_size_cv` | Coefficient of variation |
| `top_decile_volume_share` | Top 10% trade share |

---

### F) VOLATILITY & APPROACH STATE

| Feature | Description |
|---------|-------------|
| `realized_volatility_60s` | 60s realized volatility |
| `approach_return_60s` | Return over approach |
| `approach_volatility_60s` | Approach volatility |
| `approach_velocity_bucket` | crash \| grind |

**Deterministic definitions (volatility/approach):**

- Window: last **60 seconds** of snapshots (inclusive), aligned to snapshot cadence.
- Use **10-second log returns** from `last_price`:
  - `r_t = ln(p_t / p_{t-1})` (only if both prices are > 0)
  - `realized_volatility_60s = pop_std(r over window)` with ddof = 0; if <2 returns, use 0
- `approach_return_60s = ln(p_t / p_{t-60s})` (if `p_{t-60s}` exists and both prices > 0; else 0)
- `approach_volatility_60s = realized_volatility_60s`

---

### G) SEQUENCE CONTROL

| Feature | Description |
|---------|-------------|
| `sequence_id` | Snapshot sequence number |
| `timestamp` | Event timestamp (ms) |

---

## 6) BUY/SELL CLASSIFICATION RULE

**Quote test (MUST BE DEFINED):**

| Condition | Classification |
|-----------|----------------|
| `trade >= ask` | BUY |
| `trade <= bid` | SELL |
| else | UNKNOWN |

**Store all buckets.** Never discard unknown volume.

---

## 7) EXCLUDED BY DESIGN

The following are **intentionally excluded**:

- ❌ Candle patterns
- ❌ MACD
- ❌ RSI
- ❌ Full Level 2 depth
- ❌ Visual confirmation logic

---

## 8) DATA COLLECTION RULES

Persist data only when:

1. **In-zone** (abs_distance_to_level_atr <= 0.25), OR
2. **Setup detector flags candidate** (optional)

Always include: **baseline + stress + resolution** phases.

**Target:** 50–200 high-quality episodes

---

## 9) AI ANALYSIS OBJECTIVE

AI compares wins vs losses to:

- Find early-shift features
- Discover reversal fingerprints
- Output probabilities (not rigid triggers)

The goal is **pattern discovery**, not rule creation.

---

## 10) EXECUTION ROADMAP

### Phase 1 (Days 1–14): Core Infrastructure

| Day | Deliverable |
|-----|-------------|
| 1 | Project scaffold, pyproject.toml, core models (Trade, Quote, Level, Episode, Snapshot), enums, config |
| 2 | RingBuffer with time-based eviction, unit tests |
| 3 | BarBuilder (1-minute OHLCV from trades), unit tests |
| 4 | ATRCalculator with seeding/blending logic, status tracking, unit tests |
| 5 | VWAPCalculator, EMACalculator, unit tests |
| 6 | Provider abstraction (DataProvider base), replay adapter |
| 7 | ZoneGate (entry/exit detection), LevelManager, unit tests |
| 8 | EpisodeStateMachine (phase transitions), deterministic timestamps |
| 9 | SnapshotBuilder (10-second aggregation, event-time based) |
| 10 | FeatureEngine — Level spine features |
| 11 | FeatureEngine — Microstructure + Order flow features |
| 12 | FeatureEngine — Volatility + EMA structure features |
| 13 | StorageWriter (atomic parquet, partitioning), TickerWorker integration |
| 14 | Orchestrator, CLI, integration tests, end-to-end flow |

### Phase 2 (Days 15–45): Data Collection

- 1 ticker, 1 setup type
- ~50 labeled episodes
- Iterate on labeling workflow

### Phase 3 (Day 45+): AI Discovery

- AI discovery → playbook fingerprints
- Out-of-sample validation
- Fingerprint documentation

---

## 11) ANTI-OVERFITTING RULES

All fingerprints must pass:

1. **Market Physics Test**: Does this make sense mechanically?
2. **Session/Context Sanity Test**: Does it hold across different market conditions?
3. **Out-of-Sample Ticker Test**: Does it generalize to unseen symbols?

---

## 12) FINAL POSITIONING

A numeric market-state recorder that converts intuition into structured, learnable evidence—enabling earlier recognition of regime change than candles allow.

> **The edge is not prediction.**  
> **The edge is recognizing the shift before it becomes obvious.**

---

## ADDENDUM A — Indicator Source Definitions

### A1) 1-Minute Bars (Internal Construction)

| Property | Definition |
|----------|------------|
| **Source** | Last-trade prices (prints) |
| **Open** | First trade in the minute |
| **High** | Max trade price in the minute |
| **Low** | Min trade price in the minute |
| **Close** | Last trade in the minute |
| **Volume** | Sum trade sizes in the minute |
| **Session** | RTH by default |
| **Extended Hours** | Controlled by `include_extended_hours: true\|false` |

### A2) ATR Definition

- **ATR_14_1m** computed on the internal 1-minute OHLC bars.
- Default: **RTH-only** unless `include_extended_hours=true`.

### ATR Calculation Method (Deterministic)

- Use **Wilder’s ATR (RMA)** with period `n = 14` on internal 1-minute OHLC bars (TR computed per bar).

- True Range:

  - `TR_t = max(high_t - low_t, abs(high_t - close_{t-1}), abs(low_t - close_{t-1}))`

- Wilder update:

  - `ATR_t = (ATR_{t-1} * (n - 1) + TR_t) / n`

Initialization:

- The initial `ATR` for a session is seeded per the ATR cold-start policy (`ATR_seed`), then blended until `atr_status = live`.

### A3) VWAP Definition (Session Boundary)

- **VWAP resets daily at RTH open by default.**
- VWAP uses trades: `price * size / total_size`
- Optional config: `include_premarket_in_vwap: true|false`

---

## ADDENDUM B — Resolution Threshold Defaults

### B1) Default Thresholds

| Threshold | Value | Description |
|-----------|-------|-------------|
| **Success** | +0.50 ATR | From level in thesis direction |
| **Failure** | -0.35 ATR | Against thesis before success |
| **Timeout** | 300 seconds | After `zone_exit_time` or max cap |

### B2) Store Per Episode (Auditability)

- `success_threshold_atr` (default 0.50)
- `failure_threshold_atr` (default 0.35)
- `timeout_seconds` (default 300)

---

## ADDENDUM C — Touch Count Definition

A touch increments when:

1. **Entry**: Outside zone → inside zone
2. **Exit**: Inside zone → outside zone for **≥1 full snapshot** (10s)

Both conditions must be met sequentially for a touch to count.

---

## ADDENDUM D — Quote/Spread Sampling Policy

For each 10-second snapshot:

- Use the **latest NBBO quote observed during the 10-second window**
- Store `quote_age_ms` to detect stale quotes

---

## ADDENDUM E — Episode Context Fields

Add to Episode metadata:

| Field | Description |
|-------|-------------|
| `time_of_day_bucket` | open \| mid \| close |
| `gap_pct` | Gap percentage vs prior close |
| `spy_return_5m` | Optional: SPY 5-min return |
| `spy_bucket` | Optional: SPY state category |

---

## ADDENDUM F — Multi-Timeframe Structure Layer (Recommended)

Goal: keep **10-second snapshots** as the micro “pressure” layer, while computing the **EMA structure block** on higher structure timeframes so later analysis can express:

> “30s looks ready, 1m close, 2m not ready”

…without candle pattern logic.

### F1) Timeframes (V1.x)

- Required in V1.x: `30s`, `1m`
- Optional later: `2m`

### F2) Bar construction (30s / 1m / 2m)

Bars are constructed from **trades** using the same rules as Addendum A1, with bucket size \(B\) in seconds:

- `open`: first trade in the bucket
- `high`: max trade price in the bucket
- `low`: min trade price in the bucket
- `close`: last trade in the bucket
- `volume`: sum trade sizes in the bucket
- `trade_count`: number of trades in the bucket

Bucket start is deterministic in event-time:

- `bucket_start_ts_ms = floor(ts_ms / (B*1000)) * (B*1000)`

### F3) EMA structure per timeframe (no crosses)

Compute EMAs on each timeframe’s bar closes:

- `ema9_tf`, `ema20_tf`, `ema30_tf`, `ema200_tf`

The **EMA structure derived fields** per timeframe use the exact same deterministic definitions as §5D:

- `slope_emaX_60s_tf = (emaX_t - emaX_{t-60s}) / 60`
- `ema_spread_9_20_tf = ema9_tf - ema20_tf`
- `ema_spread_20_30_tf = ema20_tf - ema30_tf`
- `compression_index_tf = (abs(ema9_tf-ema20_tf) + abs(ema20_tf-ema30_tf)) / ATR_14_1m`
- `ema_confluence_score_tf = clamp01(1 - compression_index_tf / confluence_ref)`
- `stack_state_tf` uses 9/20/30:
  - bull if `ema9_tf > ema20_tf > ema30_tf`
  - bear if `ema9_tf < ema20_tf < ema30_tf`
  - else mixed
- `price_vs_emas_tf` bitmask uses `[9,20,30,200]` against **last_price** at the snapshot timestamp.

**Normalization rule:** EMA compression is normalized by **ATR_14_1m** (not ATR of that timeframe) unless explicitly changed in a future spec version.

### F4) Alignment to 10-second snapshots (deterministic)

Higher timeframe EMA values are sampled for each 10-second snapshot timestamp `T` using:

- the **latest fully completed timeframe bar** with `bar_end_ts_ms <= T`

This produces a deterministic “structure block” aligned to each 10-second snapshot without candles.

### F5) Storage (recommended)

Write a separate Parquet dataset for multi-timeframe indicators:

- `tf_indicators/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`

Each row is keyed by:

- `episode_id`
- `timestamp` (10-second snapshot timestamp, event-time)
- `timeframe` (`30s` | `1m` | `2m`)

And contains:

- `ema9`, `ema20`, `ema30`, `ema200`
- `slope_ema9_60s`, `slope_ema20_60s`, `slope_ema30_60s`
- `ema_spread_9_20`, `ema_spread_20_30`
- `compression_index`, `ema_confluence_score`
- `stack_state`, `price_vs_emas`
- `stretch_200_atr = (last_price - ema200) / ATR_14_1m`

---

## ADDENDUM G — Manual Level Override (Must-Have)

Even if no level is predefined, the operator must be able to add levels mid-session to create an anchor immediately (e.g., “TSLA downtrend break is about to trigger”).

### G1) Command/API

Support a runtime operation:

- `add_level(ticker, level_price, level_type, level_width_atr=0.25, level_source=manual, level_id optional)`

### G2) Deterministic `level_id` rule

- If the user supplies `level_id`, use it.
- Else generate deterministically:

`level_id = "manual:{ticker}:{created_ts_ms}:{level_price}"`

### G3) Required stored fields (auditability)

Store on the Level record:

- `level_id`
- `level_price`
- `level_type`
- `level_source = manual`
- `level_width_atr`
- `created_ts_ms` (event-time when added; snapshot/end-of-window time is acceptable)

Optional:

- `notes`

### G4) Behavior

- Manual levels become active **immediately** for ZoneGate and Episode creation.
- Episodes created from manual levels must store the same level metadata (`level_id`, `level_type`, `level_source`, `level_width_atr`) as any other level.

---

## ADDENDUM H — Session Recording + Markers → Deterministic Episode Extraction (Recommended)

This addendum adds a low-friction workflow without compromising dataset quality:

- **Layer 1 (Session recording)**: continuously record numeric time-series per ticker during a watch session.
- **Layer 2 (Episode extraction)**: markers do **not** become the dataset directly; markers trigger deterministic extraction of standardized episodes.

### H1) Layer 1 — Session recording (continuous numeric stream)

When the operator clicks “Record” on a ticker:

- Log the numeric stream continuously (raw + computed as available) per ticker.
- Store as time-series data partitioned by date/ticker/session.
- This stream is *not* the training dataset by itself; it is the source used to extract episodes.

Recommended storage:

- `session_stream/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`

Each row corresponds to the existing 10-second snapshot schema (same fields as `Snapshot`), keyed by:

- `session_id`, `ticker`, `timestamp`, `sequence_id`

### H2) Layer 2 — Markers (small fixed vocabulary)

Markers represent **human intent** and must use a small fixed vocabulary (no freeform primary labels).

Marker fields:

- `marker_id` (deterministic if not provided)
- `session_id`, `ticker`
- `ts_ms` (event-time, milliseconds)
- `marker_type` (enum)
- optional: `direction_bias` (long|short) if not inferable from marker_type
- optional: `notes` (freeform, not used as primary label)

Recommended marker storage:

- `markers/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-*.parquet`

### H3) Deterministic episode extraction from markers (standard windows)

Markers trigger deterministic extraction rules. When a marker is placed at time `M`:

- **Baseline window**: `[M - 5m, M - 2m]`
- **Stress window**: `[M - 2m, M + 2m]`
- **Resolution window**: `[M + 2m, M + 10m]` **OR** until deterministic resolution triggers fire (threshold/invalidation/timeout), whichever comes first.

The extracted episode must attach:

- `marker_type` (as the label class)
- `episode_source = marker_extract`

### H4) Anchor price for marker episodes (deterministic)

Marker episodes use an anchor price `anchor_price`:

- `anchor_price = last_price` from the session stream snapshot whose timestamp is the **latest <= M**

The marker episode is modeled as an Episode around a pseudo “level”:

- `level_id = marker_id`
- `level_price = anchor_price`
- `level_source = marker`
- `level_type = marker_type`

Resolution uses the standard deterministic thresholds (Addendum B) relative to `level_price` and `direction_bias`.

### H5) Overlap policy (deterministic)

To prevent dataset mess:

- Allow overlapping markers, but **do not merge episodes**.
- If two markers have identical `ts_ms` and `marker_type`, keep the first and ignore duplicates (stable tie-breaker by insertion order).

### H6) Marker vocabulary (V1 recommendation)

Start with:

- `consolidation`
- `support_bounce`
- `breakdown`
- `downtrend_break`

`direction_bias` inference (deterministic default mapping):

- `support_bounce` → long
- `downtrend_break` → long
- `breakdown` → short
- `consolidation` → requires explicit `direction_bias` (or store as no-trade intent)

---

## ADDENDUM J — Web UI Orchestration (Recommended)

This addendum covers the **web-based control panel** for triggering recordings and visualizing results. The web UI is a **thin orchestration layer**—all business logic runs in the existing backend engine.

### J1) Record Form (multi-ticker support)

The `/record` page provides a form with:

| Field | Type | Description |
|-------|------|-------------|
| `tickers` | string | Comma-separated ticker list (e.g., `TSLA, NVDA, AAPL`) |
| `start` | datetime | Start of recording window (local time, converted to UTC internally) |
| `end` | datetime | End of recording window |
| `direction_bias` | enum | `long \| short` (applies to all tickers) |
| `timezone` | string | Timezone name (default `America/Chicago`); form inputs are converted to UTC before fetching |
| `markers[]` | list (optional) | One or more markers, each with start time, optional end time, types, and optional price tags |

### J2) Multi-Ticker Parallel Jobs

When the user submits the form:

1. Backend creates one **Job** per ticker.
2. Each job runs: `polygon_export(ticker, start, end)` → `replay_run(ticker, session_id, ...)`.
3. Jobs execute in parallel (async or thread pool).
4. Job state is tracked: `pending → running → done | error`.

### J3) Job Model (deterministic)

```python
@dataclass
class Job:
    job_id: str                     # UUID or deterministic hash
    ticker: str
    session_id: str
    status: str                     # pending | running | done | error
    created_at: int                 # ms since epoch
    started_at: Optional[int]
    finished_at: Optional[int]
    error_message: Optional[str]
```

### J4) API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/record` | Submit record request; returns list of `job_id`s |
| `GET` | `/api/jobs` | List all jobs (optionally filter by status) |
| `GET` | `/api/jobs/{job_id}` | Get single job status |

### J5) Frontend Behavior

1. After form submission, redirect to `/jobs` page.
2. `/jobs` page polls `GET /api/jobs` every 2 seconds.
3. When all jobs are `done`, show links to session results.
4. On `error`, show error message.

### J6) Marker from Form (optional)

If markers are provided in the form:

- Each marker is emitted as one or more `add_marker` commands at `marker_start_ts_ms`.
- If a marker includes multiple types (e.g., `downtrend_break` + `double_bottom`), emit **one marker command per type** (no composite enums).
- Marker `end_ts_ms` is stored in marker `notes` JSON for auditability.

This triggers deterministic episode extraction per Addendum H.

### J6.1) Marker Price Tags → Runtime Levels (support/resistance)

Markers may include optional **price tags** (support/resistance levels) to analyze behavior at specific prices.

For each marker price tag:
- Emit an `add_level` command at the marker start timestamp (`marker_start_ts_ms`)
- `level_price = tag.price`
- `level_type = support|resistance|manual` (manual if not specified)
- `level_source = manual` (via existing `add_level` behavior)

This enables level-spine + orderflow/smart-money features to be computed relative to those tagged prices.

### J7) In-Memory Job Queue (V1)

For V1, jobs are stored in-memory (Python dict). This is sufficient for single-user local usage. Future versions may use Redis/Celery for persistence and multi-worker support.

---

## IMPLEMENTATION DECISIONS

### 1) Data Provider

- Implement a **provider abstraction layer from day 1**
- Build one adapter for V1 (primary provider), keep interface stable

**Interface:**

```python
class DataProvider(ABC):
    async def getTrades(symbol: str, start: int, end: int) -> list[Trade]
    async def getQuotes(symbol: str, start: int, end: int) -> list[Quote]
    async def subscribeTrades(symbols: list[str]) -> AsyncIterator[Trade]
    async def subscribeQuotes(symbols: list[str]) -> AsyncIterator[Quote]
```

### 2) ATR Cold Start (Deterministic, Configurable)

- Zone gating **must NOT wait 15 minutes**
- Use deterministic seeding and expose blend config

**Fields to store:**

| Field | Values |
|-------|--------|
| `atr_status` | seeded \| blending \| live |
| `atr_blend_alpha` | Config, default 0.7 |
| `atr_is_warm` | Boolean, true when ≥14 1m bars |
| `atr_seed_source` | prior_session \| daily_fallback \| fixed_default |

**Policy:**

```
IF bars_today >= 14:
    ATR = ATR_14_1m (live)
    atr_status = live
ELSE:
    ATR = atr_blend_alpha * ATR_seed + (1 - atr_blend_alpha) * ATR_partial
    atr_status = blending (or seeded if no partial data)
```

### 3) Labeling Interface

V1 uses a simple labels file (CSV/Parquet) merged into episodes:

| Field | Required |
|-------|----------|
| `episode_id` | Yes |
| `manual_outcome` | Yes |
| `manual_notes` | No |
| `setup_type` | No |
| `confidence` | No |

Web UI is optional for later phases.

### 4) Replay / Backtesting (Feed-Agnostic)

- The pipeline treats replay as **"just another data source"**
- Same Episode engine, SnapshotBuilder, and ZoneGating
- Different feed implementations:
  - Live: WebSocket stream
  - Replay: File iterator over recorded trades/quotes

### 5) Multi-Ticker Orchestration

- Build **multi-ticker capable architecture from day 1**
- Run 1 ticker in Phase 2

**Pattern:**

```
Orchestrator (1)
    ├── TickerWorker (per symbol)
    │   ├── RingBuffer (trades)
    │   ├── RingBuffer (quotes)
    │   ├── BarBuilder
    │   ├── ATRCalculator
    │   ├── ZoneGate
    │   ├── EpisodeStateMachine
    │   ├── SnapshotBuilder
    │   └── FeatureEngine
    │
    └── StorageWriter (shared)
```

### 6) Storage Format (Parquet)

**Partitioning:**

```
snapshots/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
episodes/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
labels/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
tf_indicators/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
session_stream/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
markers/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
```

**Settings:**

- Append via rolling part files (per episode or per chunk size/time)
- Compression: **zstd** preferred; snappy acceptable if needed

---

## BULLETPROOFING MICRO-EDITS

### A) Timestamp Contract (Live == Replay)

All trade/quote events **MUST** include:

| Field | Description |
|-------|-------------|
| `ts_ms` | Exchange or SIP event timestamp in milliseconds |
| `symbol` | Ticker symbol |

**Rules:**

- SnapshotBuilder uses **event-time (`ts_ms`)**, not receipt-time
- Replay feeds and live feeds must preserve ordering by `ts_ms`
- Use stable tie-breakers if needed (sequence number)

> This prevents "works in replay, breaks live" drift.

### B) ATR Seed Source (Deterministic + Auditable)

**Define `ATR_seed` precisely:**

```
ATR_seed = prior_session_RTH_ATR_14_1m
```

Computed from the **same internal 1-minute bar builder** used today.

**If seed missing (first run), fallback:**

```
ATR_seed = daily_ATR / sqrt(390)  # Approx per-minute scaling
```

Or use a fixed safe default.

**Store:**

| Field | Values |
|-------|--------|
| `atr_seed_source` | prior_session \| daily_fallback \| fixed_default |
| `atr_status` | seeded \| blending \| live |
| `atr_blend_alpha` | Config value (default 0.7) |

### C) Storage: File Naming + Atomic Writes

**Parquet writing must be atomic:**

```
1. Write to: part-000.tmp.parquet
2. Rename to: part-000.parquet
```

**Partitioning path includes `session_id`:**

```
snapshots/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
episodes/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
labels/date=YYYY-MM-DD/ticker=XYZ/session=<session_id>/part-000.parquet
```

> This prevents corrupted partial files and makes sessions easy to delete/re-run.

---

## PROJECT STRUCTURE

```
godmode/
├── pyproject.toml                 # Project metadata, dependencies
├── README.md
├── SPEC.md                        # This file
├── config/
│   └── default.yaml               # Default configuration
│
├── src/
│   └── godmode/
│       ├── __init__.py
│       │
│       ├── core/                  # Core domain models
│       │   ├── __init__.py
│       │   ├── models.py          # Level, Episode, Snapshot, Trade, Quote
│       │   ├── enums.py           # EpisodePhase, ResolutionTrigger, etc.
│       │   └── config.py          # Pydantic config models
│       │
│       ├── providers/             # Data provider abstraction
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract DataProvider interface
│       │   ├── polygon.py         # Polygon.io adapter (V1)
│       │   └── replay.py          # File replay adapter
│       │
│       ├── engine/                # Core processing engine
│       │   ├── __init__.py
│       │   ├── ring_buffer.py     # Time-bounded ring buffer
│       │   ├── bar_builder.py     # 1-minute OHLCV aggregation
│       │   ├── atr.py             # ATR calculator with seeding/blending
│       │   ├── vwap.py            # Session VWAP calculator
│       │   ├── ema.py             # EMA calculations
│       │   └── indicators.py      # Composite indicator state
│       │
│       ├── zone/                  # Zone gating logic
│       │   ├── __init__.py
│       │   ├── zone_gate.py       # Zone entry/exit detection
│       │   └── level_manager.py   # Level loading and management
│       │
│       ├── episode/               # Episode state machine
│       │   ├── __init__.py
│       │   ├── state_machine.py   # Baseline → Stress → Resolution
│       │   ├── snapshot_builder.py# 10-second snapshot construction
│       │   └── resolution.py      # Threshold/invalidation/timeout logic
│       │
│       ├── features/              # Feature computation
│       │   ├── __init__.py
│       │   ├── level_spine.py     # Distance, penetration, crosses
│       │   ├── microstructure.py  # Spread, quote analysis
│       │   ├── orderflow.py       # Delta, aggression, absorption
│       │   ├── volatility.py      # Realized vol, approach state
│       │   └── compute.py         # Master feature engine
│       │
│       ├── storage/               # Persistence layer
│       │   ├── __init__.py
│       │   ├── writer.py          # Atomic parquet writer
│       │   ├── reader.py          # Parquet reader utilities
│       │   └── partitioning.py    # Path generation logic
│       │
│       ├── worker/                # Per-ticker worker
│       │   ├── __init__.py
│       │   └── ticker_worker.py   # Main worker loop
│       │
│       ├── orchestrator/          # Multi-ticker orchestration
│       │   ├── __init__.py
│       │   └── session.py         # Session lifecycle management
│       │
│       └── cli/                   # Command-line interface
│           ├── __init__.py
│           └── main.py            # Entry points (run, replay, export)
│
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── unit/
│   │   ├── test_ring_buffer.py
│   │   ├── test_atr.py
│   │   ├── test_zone_gate.py
│   │   ├── test_state_machine.py
│   │   ├── test_features.py
│   │   └── test_storage.py
│   └── integration/
│       └── test_episode_flow.py
│
└── data/                          # Local data directory (gitignored)
    ├── levels/                    # Level definitions per ticker
    │   └── AAPL.yaml
    ├── seeds/                     # ATR seed cache
    │   └── AAPL_atr_seed.json
    └── output/                    # Parquet output
        ├── snapshots/
        ├── episodes/
        └── labels/
```

---

## DATA MODELS (CODE)

### Enums

```python
from enum import Enum

class EpisodePhase(Enum):
    BASELINE = "baseline"
    STRESS = "stress"
    RESOLUTION = "resolution"
    COMPLETE = "complete"

class ResolutionTrigger(Enum):
    THRESHOLD_HIT = "threshold_hit"
    INVALIDATION = "invalidation"
    TIMEOUT = "timeout"

class ATRStatus(Enum):
    SEEDED = "seeded"
    BLENDING = "blending"
    LIVE = "live"

class ATRSeedSource(Enum):
    PRIOR_SESSION = "prior_session"
    DAILY_FALLBACK = "daily_fallback"
    FIXED_DEFAULT = "fixed_default"

class LevelType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    VWAP = "vwap"
    TRENDLINE = "trendline"

class LevelSource(Enum):
    MANUAL = "manual"
    DERIVED = "derived"
    PRIOR_LOW = "prior_low"
    VWAP = "vwap"
    TRENDLINE = "trendline"

class DirectionBias(Enum):
    LONG = "long"
    SHORT = "short"

class VWAPState(Enum):
    ABOVE = "above"
    BELOW = "below"
    AT = "at"

class ApproachVelocity(Enum):
    CRASH = "crash"
    GRIND = "grind"

class Outcome(Enum):
    WIN = "win"
    LOSS = "loss"
    SCRATCH = "scratch"
    NO_TRADE = "no-trade"

class ResolutionType(Enum):
    REVERSAL_SUCCESS = "reversal_success"
    FAKE_BREAK = "fake_break"
    CONTINUATION = "continuation"
    CHOP = "chop"
```

### Core Models

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class Trade:
    """Single trade event."""
    ts_ms: int              # Exchange/SIP timestamp in milliseconds
    symbol: str
    price: float
    size: float
    conditions: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Quote:
    """NBBO quote event."""
    ts_ms: int
    symbol: str
    bid: float
    bid_size: float
    ask: float
    ask_size: float

@dataclass(frozen=True, slots=True)
class Bar:
    """1-minute OHLCV bar."""
    ts_ms: int              # Bar open timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int

@dataclass
class Level:
    """User-defined price level for zone gating."""
    level_id: str
    level_price: float
    level_type: str         # support | resistance | vwap | trendline
    level_source: str       # manual | derived | prior_low | vwap
    level_width_atr: float = 0.25  # Zone width in ATR units

@dataclass
class Episode:
    """Complete episode record."""
    episode_id: str
    session_id: str
    ticker: str
    
    # Level reference
    level_id: str
    level_price: float
    level_type: str
    level_source: str
    level_width_atr: float
    level_entry_side: str   # above | below
    
    # Timing (all in ms)
    start_time: int
    zone_entry_time: int
    zone_exit_time: Optional[int] = None
    resolution_time: Optional[int] = None
    end_time: Optional[int] = None
    
    # Resolution
    resolution_trigger: Optional[str] = None
    direction_bias: str = "long"    # long | short
    
    # Thresholds (for auditability)
    success_threshold_atr: float = 0.50
    failure_threshold_atr: float = 0.35
    timeout_seconds: int = 300
    
    # ATR state at episode start
    atr_value: float = 0.0
    atr_status: str = "seeded"
    atr_seed_source: Optional[str] = None
    atr_blend_alpha: float = 0.7
    atr_is_warm: bool = False
    
    # Outcomes (filled at resolution)
    outcome: Optional[str] = None
    resolution_type: Optional[str] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    r_multiple: Optional[float] = None
    time_to_mfe_ms: Optional[int] = None
    time_to_failure_ms: Optional[int] = None
    
    # Context
    time_of_day_bucket: str = "mid"
    gap_pct: Optional[float] = None
    spy_return_5m: Optional[float] = None

@dataclass
class Snapshot:
    """10-second market state snapshot."""
    episode_id: str
    sequence_id: int
    timestamp: int          # Event-time (ms)
    phase: str              # baseline | stress | resolution
    
    # === LEVEL SPINE ===
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
    
    # === PRICE & SPREAD ===
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0
    spread_abs: float = 0.0
    spread_pct: float = 0.0
    spread_volatility_60s: float = 0.0
    quote_age_ms: Optional[int] = None
    
    # === VWAP ===
    vwap_session: float = 0.0
    price_minus_vwap: float = 0.0
    price_minus_vwap_atr: float = 0.0
    vwap_state: str = "at"
    
    # === EMA STRUCTURE ===
    ema9: float = 0.0
    ema20: float = 0.0
    ema30: float = 0.0
    ema200: float = 0.0
    slope_ema9_60s: float = 0.0
    slope_ema20_60s: float = 0.0
    slope_ema30_60s: float = 0.0
    ema_spread_9_20: float = 0.0
    ema_spread_20_30: float = 0.0
    compression_index: float = 0.0
    ema_confluence_score: float = 0.0
    stack_state: str = ""
    price_vs_emas: str = ""
    stretch_200_atr: float = 0.0
    
    # === ORDER FLOW ===
    trade_count: int = 0
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    unknown_volume: float = 0.0
    delta: float = 0.0
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
    
    # === VOLATILITY & APPROACH ===
    realized_volatility_60s: float = 0.0
    approach_return_60s: float = 0.0
    approach_volatility_60s: float = 0.0
    approach_velocity_bucket: str = "grind"
```

---

## TECH STACK

| Package | Purpose | Version |
|---------|---------|---------|
| Python | Runtime | 3.11+ |
| asyncio | Real-time event loop | stdlib |
| numpy | Feature math | latest |
| pandas | DataFrame ops | latest |
| pyarrow | Parquet I/O | latest |
| websockets | Streaming data | latest |
| pydantic | Data validation / config | 2.x |
| pytest | Testing | latest |
| pytest-asyncio | Async test support | latest |
| polars | Optional: faster analytics | latest |
| pyyaml | Config file loading | latest |
| click | CLI framework | latest |

---

## Addendum K: Support Chain Analysis (Level Relationships)

### K.1 Philosophy

Supports don't exist in isolation. A **chain** of supports (S1 → S2 → S3) tells a story:
- S1 fails → price flushes to S2 → S2 holds → price reclaims S1 → "main support" is S2

This addendum defines the **deterministic metrics** to capture that relationship.

**Thresholding rule**: Store raw magnitudes (e.g., `drop_atr = 1.23`) AND simple buckets (e.g., `flush_flag = true`). Tune thresholds later from labeled data. Don't hard-code subjective cutoffs prematurely.

**Event naming rule**: Keep events mechanical (`touch`, `break`, `reclaim`, `hold`). Don't add subjective pattern names (e.g., "failed breakdown"). Let AI discover patterns from boring events.

---

### K.2 Touch Packet (emitted for every touch of a support)

| Field | Type | Definition |
|-------|------|------------|
| `touch_id` | str | Unique ID for this touch event |
| `touch_ts_ms` | int | Timestamp of the touch |
| `touch_number` | int | 1st, 2nd, 3rd, etc. touch of this level in session |
| `time_since_last_touch_s` | float | Seconds since previous touch of same level |
| **Approach (pre-touch)** | | |
| `approach_velocity_atr_per_min` | float | \|return_60s\| / atr / 1min |
| `approach_type` | str | crash (≥1.5) / fast (≥0.75) / normal (≥0.25) / grind (<0.25) |
| `approach_delta_60s` | float | sum(delta) in 60s leading into touch |
| `approach_rel_aggr_60s` | float | mean(rel_aggr) in 60s leading into touch |
| **Dwell + Bounce (post-touch)** | | |
| `touch_dwell_s` | float | Seconds stayed in-band after touch |
| `bounce_return_30s_pct` | float | Return from touch to touch+30s |
| `bounce_return_60s_pct` | float | Return from touch to touch+60s |
| **Reaction flow** | | |
| `band_delta_0_30s` | float | sum(delta) in [touch, touch+30s] while in-band |
| `band_delta_0_60s` | float | sum(delta) in [touch, touch+60s] while in-band |
| `rel_aggr_0_30s` | float | mean(rel_aggr) in [touch, touch+30s] |
| `rel_aggr_0_60s` | float | mean(rel_aggr) in [touch, touch+60s] |
| `delta_flip_flag` | bool | sign(delta before) ≠ sign(delta after) |
| **Penetration / wick** | | |
| `max_penetration_pct` | float | How far price stabbed below band (% of level price) |
| `wick_recovered_flag` | bool | Penetrated but closed back inside band within 30s |
| **Volume abnormality** | | |
| `touch_volume_z` | float | z-score of volume at touch vs session baseline |
| `large_trade_count_at_touch` | int | Count of trades > p95 size in [touch-10s, touch+10s] |
| `large_trade_buy_ratio` | float | large_buy_vol / (large_buy + large_sell) |
| **Absorption** | | |
| `absorption_mean_0_30s` | float | mean(absorption_index_10s) in [touch, touch+30s] |
| `price_efficiency_0_30s` | float | \|return\| / volume (low = absorption) |
| **Divergence** | | |
| `cvd_60s_at_touch` | float | CVD_60s value at touch time |
| `cvd_slope_into_touch` | float | CVD trend leading into touch |
| `div_flag_at_touch` | bool | Price new low but delta improving |
| **Spread behavior** | | |
| `spread_pct_at_touch` | float | Spread at touch moment |
| `spread_widening_into_touch` | bool | Spread in [touch-30s] > baseline |
| `spread_narrowing_after_touch` | bool | Spread in [touch+30s] < spread at touch |
| **Context** | | |
| `compression_at_touch` | float | mean compression_index in [touch-60s, touch] |
| `compression_trend_into_touch` | str | tightening / loosening / flat |
| `price_vs_vwap_at_touch` | str | above / below / at |
| `ema_stack_at_touch` | str | bull / bear / mixed |
| **Outcome** | | |
| `break_confirmed_30s` | bool | Exited below band and stayed out for ≥30s |
| `reclaim_after_break` | bool | Re-entered band within 60s after break |
| `reclaim_hold_30s` | bool | Stayed in/above band for 30s after reclaim |
| `touch_outcome` | str | hold / break / break_reclaim_hold / break_reclaim_fail |

---

### K.3 Reclaim Packet (emitted when price reclaims a prior support after tagging deeper)

| Field | Type | Definition |
|-------|------|------------|
| `reclaim_id` | str | Unique ID |
| `reclaimed_level_price` | float | The prior support being reclaimed (e.g., S2) |
| `tagged_deeper_level_price` | float | The deeper support that was touched (e.g., S3) |
| `t_tag_deeper_ms` | int | Timestamp when deeper support was touched |
| `t_reclaim_ms` | int | Timestamp when prior support was reclaimed |
| **Speed** | | |
| `time_to_reclaim_ms` | int | t_reclaim - t_tag_deeper |
| `time_to_reclaim_s` | float | Same in seconds |
| `snapback_flag` | bool | time_to_reclaim ≤ 120s |
| **Quality (post-reclaim)** | | |
| `reclaim_band_delta_30s` | float | band_delta in [reclaim, reclaim+30s] |
| `reclaim_band_delta_60s` | float | band_delta in [reclaim, reclaim+60s] |
| `reclaim_rel_aggr_30s` | float | mean(rel_aggr) in [reclaim, reclaim+30s] |
| `reclaim_rel_aggr_60s` | float | mean(rel_aggr) in [reclaim, reclaim+60s] |
| `reclaim_hold_30s` | bool | Stayed in/above band for 30s after reclaim |
| `reclaim_hold_60s` | bool | Stayed in/above band for 60s after reclaim |
| **Overextension depth** | | |
| `drop_pct_to_deeper` | float | (deeper_price - prior_price) / prior_price |
| `drop_atr_to_deeper` | float | \|price diff\| / atr_mean |
| `flush_flag` | bool | drop_atr ≥ 1.0 AND velocity ≥ 0.75 ATR/min |

---

### K.4 Chain Relationship Metrics (per support in chain)

| Field | Type | Definition |
|-------|------|------------|
| `level_index` | int | Position in chain (1, 2, 3, ...) |
| `level_price` | float | Support price |
| `level_kind` | str | support / resistance |
| `level_outcome` | str | win / loss / pending |
| **Gap to next** | | |
| `gap_to_next_support_atr` | float | Distance to next support in ATR units |
| `support_cluster_flag` | bool | Gap < 0.5 ATR (levels too close) |
| **Session-wide history** | | |
| `session_touches` | int | Total touches of this level in session |
| `session_breaks` | int | Total breaks |
| `session_rejects` | int | Total rejects |
| `session_reclaim_attempts` | int | Reclaim attempts after breaks |
| `session_reclaim_hold_rate` | float | Holds / attempts |
| `historical_hold_rate` | float | rejects / touches |
| `role_flip` | str | support_to_resistance / resistance_to_support / None |
| **Focus detection** | | |
| `focus_returns_to_this` | bool | Deeper support tagged → this level reclaimed fast + held |

---

### K.5 Main Support Scoring (computed per chain)

| Field | Type | Definition |
|-------|------|------------|
| `main_support_score` | int | Composite score per level |
| `main_support_id` | str | Level with highest score |
| `focus_level` | str | Level where "attention returns" after flush |

**Score formula:**

```
main_support_score =
  + 2 * (reclaim_hold_30s after deeper tag)
  + 1 * (reclaim_band_delta_30s > 0 AND reclaim_rel_aggr_30s > 0.10)
  + 1 * (time_to_reclaim_ms <= 120_000)
  + 1 * (session_rejects - session_breaks >= 2)
  - 2 * (role_flip == support_to_resistance)
  - 1 * (session_breaks)
```

---

### K.6 Report + CSV Output

All of the above will be available in:
- **`report.json`** per session (structured JSON)
- **`/api/level_chains.csv`** export (flat CSV with all metrics)
- **`/api/touch_packets.csv`** export (one row per touch event)
- **`/api/reclaim_packets.csv`** export (one row per reclaim event)

---

## CHANGELOG

### v1.1.2 (Addendum K)

- Added Support Chain Analysis (Addendum K)
- Touch Packet, Reclaim Packet, Chain Relationship Metrics
- Main Support Scoring formula

### v1.1.1 (Final Frozen)

- Added determinism patches (Addendum A-E)
- Locked all threshold defaults
- Defined touch count formally
- Added quote sampling policy
- Added episode context fields

### v1.1.0

- Added implementation decisions
- Added bulletproofing micro-edits
- Finalized project structure

### v1.0.0

- Initial specification

