# Safe Execution Pattern (Opus implements → GPT‑5.2 audits)

This repo follows a strict **author/reviewer** workflow:

- **Author (Implementation / Plumbing)**: Opus 4.5
- **Auditor (Core logic review / determinism)**: GPT‑5.2
- **One-off helpers**: Codex mini
- **Documentation**: Opus 4.5

## Hard rule (do not break)

**Opus may not define new behavior.**  
Opus may only implement behavior already specified in `SPEC.md`.

If anything feels underspecified:

1. Pause
2. Clarify/update `SPEC.md`
3. Then implement

## Step 1 — Give Opus an exact task

**Prompt style (required):**

“Implement `SnapshotBuilder` exactly per `SPEC.md` §5 and Addendum D.  
Do not invent logic. If anything is missing or ambiguous, ask questions before coding.”

**Inputs Opus must be given:**

- The exact `SPEC.md` sections (copy/paste)
- The target files/modules to create/modify
- The acceptance criteria (tests, schema, invariants)

## Step 2 — Run GPT‑5.2 as a code auditor

Give GPT‑5.2:

- The relevant `SPEC.md` sections
- The produced code (diff or full files)
- This instruction:

“Check for any violations of determinism, time alignment, ATR logic, delta classification, or replay/live parity.  
List concrete violations and required fixes. Do not propose new behavior.”

## What “audit pass” means in this repo

An audit passes only if:

- Implementation matches `SPEC.md` **exactly**
- Determinism requirements hold (event-time, ordering, stable tie-breakers)
- Replay and live paths share the same engine logic
- Storage format/partitioning/atomic writes match the spec
- Feature names and serialization match the official schema

## Checklist

Use `SPEC_COMPLIANCE_CHECKLIST.md` for every module PR/change.



