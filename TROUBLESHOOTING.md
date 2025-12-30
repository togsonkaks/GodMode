# GodMode Troubleshooting (avoid repeated fixes)

This project spans **Windows + Python tooling + market data providers + Parquet**. This doc captures the common failure modes we hit and the permanent fixes / prevention steps.

## Environment / install

### `python` opens Microsoft Store / wrong Python
- **Symptom**: `python` launches Store or `python` not found.
- **Fix**: install Python 3.11+ and run it via full path, e.g.:
  - `C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`
- **Prevention**: disable Windows “App Execution Aliases” for Python if needed.

### Missing packages at runtime/tests
- **Symptom**: `ModuleNotFoundError` (e.g. `httpx`) or FastAPI form error.
- **Fix**: install project dependencies (from `pyproject.toml`).
- **Prevention**:
  - FastAPI HTML forms require **`python-multipart`**
  - Providers may require **`httpx`**

## Provider / API keys

### Missing API keys
- **Symptom**: config validation errors like “API key required”
- **Fix**: set environment variables and restart the server:
  - `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- **Prevention**: keep keys out of git; rotate if ever exposed.

### 403 / Not authorized (Polygon)
- **Cause**: plan does not include tick trades/quotes.
- **Fix**: upgrade plan or use Alpaca.

## Parquet / schema drift

### `pyarrow.lib.ArrowTypeError: Unable to merge ... string vs dictionary`
- **Cause**: inconsistent dtype inference between writes (string vs dictionary-encoded categorical).
- **Fix**: ensure consistent types before writing OR write via stable schema.
- **Prevention**: avoid pandas categorical columns; keep `ticker` as plain string.

### “Unsupported cast from string to null”
- **Cause**: schema drift across parquet fragments (some parts wrote a column as `null` type because it was empty).
- **Fix**: read fragment-by-fragment in analysis tools.
- **Prevention**: prefer writing non-empty columns consistently; avoid emitting columns sometimes missing entirely.

## Dataclasses / `slots=True` gotchas

### `AttributeError` on `_seq` / `_writer` in `@dataclass(slots=True)`
- **Cause**: you cannot add new attributes dynamically.
- **Fix**: declare them as dataclass fields with `init=False`.

### `TypeError: non-default argument ... follows default argument`
- **Cause**: dataclass ordering rules.
- **Fix**: reorder fields so required fields come before defaults.

### Duplicated dataclass fields silently overwrite
- **Cause**: accidental duplicate field names in `models.py`.
- **Fix**: removed duplicates; tests cover basic import smoke.

## Analysis / “false alarms”

### “duplicate timestamps” in `snapshots/`
- **Cause**: `snapshots/` is **per-episode**; many episodes share the same timestamps.
- **Fix**: analyze cadence **per `episode_id`**, not globally.

## Reruns (don’t refetch)

If you already have replay files under `data/output/replay/<TICKER>_<SESSION_ID>/`, you can rerun pipeline without calling Alpaca again:

- `trades.parquet`
- `quotes.parquet`
- optional `commands.csv`

Use:
- `python -m godmode.cli.main replay-run ...`


