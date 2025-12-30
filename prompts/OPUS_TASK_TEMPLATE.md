# Opus Implementation Task Template (STRICT)

## Role
You are implementing code **only**. You may not define new behavior. The spec is authoritative.

## Hard rules
- Do **not** invent logic, defaults, fields, or thresholds.
- If anything is ambiguous or missing, **stop and ask questions** before coding.
- Keep naming consistent with `SPEC.md` schema (storage columns) and use explicit aliasing when Python identifiers differ.

## Task
Implement: `<MODULE_OR_COMPONENT_NAME>`

## Spec excerpts (paste exact sections)
Paste relevant sections from `SPEC.md` here:

```
<PASTE_SPEC_SECTIONS_HERE>
```

## Inputs / Constraints
- Target files to create/modify:
  - `<file1>`
  - `<file2>`
- Must support: live + replay parity (same engine, different feed)
- Must respect determinism: event-time (`ts_ms`), stable ordering/tie-breakers

## Acceptance criteria
- Unit tests to add/update:
  - `<test1>`: `<what it proves>`
  - `<test2>`: `<what it proves>`
- Determinism invariants:
  - `<invariant1>`
  - `<invariant2>`
- Storage schema expectations:
  - `<schema note>`

## Output format
- Provide a focused diff or full file contents.
- Call out any unanswered questions **before** writing code.



