# Zone Findings (Support/Resistance Microstructure)

This file is a living log of **what worked vs what failed** around a user-selected level (support/resistance) using GodMode `/zones` analytics (30s segments, ON-band = ±1.5%).

Goal: collect enough examples to extract repeatable “tells” without overfitting.

---

## Definitions (Current)

- **Level (L):** user-selected support/resistance price.
- **ON band:** `L ± 1.5%` (primary “what’s happening on the line” range).
- **ABOVE / BELOW:** used for context only unless explicitly noted.
- **Stages:** split the zone window into early → confirm buckets (e.g. 1/2/3). Stage 3 is “confirmation” by intent, but short windows may be noisy.

Key ON-band metrics (per zone and per stage):
- `ON segs`: number of 30s segments in ON.
- `ON Δ`: `ON buy - ON sell` (aggressive flow proxy).
- `ON buyers%`: `ON buy / (ON buy + ON sell)`.
- `ON INST B/S` and `ON INST Δ`: large-print bias at/near the level.

---

## Pattern Candidates (Draft)

### A) Reclaim Reversal (sell → fail → reclaim)

Signature:
- Early stages ON are **negative** (selling into the level).
- **No meaningful BELOW acceptance** (little time/segments below).
- Stage 3 flips to **positive ON Δ** and/or strong **ABOVE Δ** soon after.

What it means:
- Sellers tried to break the level but couldn’t hold below.
- Buyers step in late and reclaim.

### B) Breakdown Acceptance (sell → hold below)

Signature:
- ON turns negative and then you see **sustained BELOW** (time/segments) with continued negative Δ.
- Any pops back ON fail quickly and roll back below.

### C) Pop Without Follow-Through (reclaim impulse only)

Signature:
- Stage 2 strong positive (often ABOVE) but Stage 3 goes flat/negative ON and price drifts back into ON.
- Often still tradable, but lower expectation for continuation unless it keeps printing ABOVE positives.

---

## Logged Examples

### Example 1 — Reversal (Sell pressure failed, reclaim)

Window: `2026-01-16 12:05 → 12:09` @ `L=4.0523`

- Price: `+0.73%`
- ON band (total): `segs=8`, `buy=32,626`, `sell=35,855`, `Δ=-3,230`
- Stage ON deltas:
  - Stage 1 ON: `Δ=-4,964` (selling into level)
  - Stage 2 ON: `Δ=-6,195` (continued pressure)
  - Stage 3 ON: `Δ=+7,930` (flip / reclaim)
- Context: Stage 3 also showed strong buyers% and price ended green.

Interpretation (draft): **Sell → fail → reclaim** (reversal signature shows in Stage 3 ON flip).

---

### Example 2 — Reclaim + Push (weak/short confirmation)

Window: `2026-01-16 12:19 → 12:27` @ `L=4.4293`

- Price: `+2.70%`
- ON band (total): `segs=9`, `Δ=+4,321`, `INST 5,000B/0S`
- ABOVE (context): `Δ=+22,311`, `INST 10,037B/0S`
- BELOW (context): `segs=2`, `Δ=-5,960` (brief dip/pressure)
- Stage summary:
  - Stage 1: `px=-1.30%`, `Δ=-6,942` (dip/pressure)
  - Stage 2: `px=+3.42%`, `Δ=+27,614`, `INST Δ=+15,037` (reclaim impulse)
  - Stage 3: `px=+0.66%`, `Δ≈0` (short window; confirmation not huge)

Interpretation (draft): **Reclaim and push**; short window so “confirmation” can be small but the pattern is present.

---

### Example 3 — Absorption / “Bullish outcome with bearish tape”

Window: `2026-01-16 12:47 → 12:57` @ `L=4.5533`

High-level:
- Price: `+1.77%` (ended green)
- ON band (total): `segs=14`, `buy=104,000`, `sell=125,156`, `Δ=-21,155` (seller-dominant *while ON*)
- BELOW (context): `segs=7`, `Δ=-29,681` (brief breakdown attempt)
- ABOVE (context): `segs=1`, `Δ=+4,079` (reclaim pop)

Tape sequence (selected 30s segments):
- Early shove + immediate sell response:
  - `12:46:30 ON Δ +6,192` then `12:47:30 ON Δ -5,152`
- Sustained ON selling (seller pressure while still “at level”):
  - `12:48:00 ON Δ -3,456`
  - `12:48:30 ON Δ -5,507`
  - `12:49:00 ON Δ -6,087`
- Breakdown attempt (BELOW) with heavy sell delta:
  - `12:50:01 BELOW Δ -13,043` (largest sell hit in the window)
  - Followed by continued BELOW pressure into `12:51:30`
- Stabilize + reclaim:
  - `12:52:00 BELOW Δ +812` (first clear buyer push while still below)
  - `12:52:31 ON Δ +193` (reclaim to ON)
  - `12:55:01 ABOVE Δ +4,079` (pop through)
- Even after reclaim, selling returns but price doesn’t collapse:
  - `12:56:00 ON Δ -9,677` while price still prints ~`4.59`

Interpretation (draft):
- This is a classic **absorption / ineffective selling** case: the tape shows persistent seller dominance ON and even a strong BELOW sell hit, yet price recovers and finishes green.
- It’s a good reminder that **negative ON Δ does not automatically mean “fail”**; you need to watch whether selling actually produces **acceptance below** vs getting absorbed and reclaimed.

---

### Example 4 — Volatile Pop, Then Fade (distribution risk)

Window: `2026-01-16 08:09 → 09:12` @ `L=5.5762`

High-level:
- Price: `+2.88%` (ended slightly up), but with a very large intrazone range (`~21%` of start close).
- Overall tape: `buyers 45.4%`, `Delta=-95,333` (seller-dominant).
- ON band (total): `segs=5`, `Δ=-43,731`, `INST 0B/5,887S` (selling bias *at the level*).

Stage read:
- Stage 1: `px +0.28%` with heavy selling (`buyers 24.6%`, `Δ=-56,103`) → early pressure.
- Stage 2: large push up (`px +6.53%`) while delta is still negative (`Δ=-32,493`, `INST Δ=+1,070`) → **absorption / lift despite selling** (the “pump” leg).
- Stage 3: pullback (`px -2.45%`) with `INST Δ=-5,887` (inst selling) → **fade / distribution risk** into the end of the window.

Interpretation (draft):
- This is not a clean reclaim reversal; it’s closer to **“pop without confirmation”**: a strong up leg occurs even with negative delta, then the “confirm” leg is negative and shows **institutional selling on/near the level**.
- Useful tell: when Stage 2 rallies on negative delta, you want Stage 3 to **hold above / flip ON positive**; here Stage 3 does the opposite.

---

## What We Still Need To Log

- Cases where ON is positive but price still fails (false positives).
- Cases where ON is negative but price still rips (absorption scenarios).
- “Whipsaw ON” cases: alternating ON Δ signs before a decisive move.
- Compare outcomes by:
  - ON time/segments density (how long price stayed “at the level”)
  - Size composition (few big vs many small)
  - Presence of INST in ON specifically (not just overall)

---

## Notes / Rules (Non-final)

- Avoid “single-number” conclusions; sequence matters (late-stage flips can dominate outcome).
- Treat Stage 3 as the most important *when it exists*, but short windows can compress signals.
- **ON band is not a “buy signal” by itself.** ON just means “battle at the level.” The signal comes from:
  - ON delta *changing* late (e.g., Stage 3 flips negative → positive)
  - Whether price gets **acceptance BELOW** (time/segments) vs quick reclaim back ON/ABOVE
  - “Trap-door then reclaim” sequences (big BELOW/ON sell → fast ON/ABOVE positive)
