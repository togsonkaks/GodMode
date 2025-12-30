# GodMode — Numeric Market-State Recorder

This repository implements the **Numeric Market-State Recorder** per `SPEC.md` (v1.1.1).

- **Not a trading bot**: it records market microstructure + context around user-defined levels.
- **Deterministic** by design: event-time (`ts_ms`), replay/live parity, atomic parquet writes.

## Source of truth

Read: `SPEC.md`  
Workflow: `WORKFLOW.md`  
Audit checklist: `SPEC_COMPLIANCE_CHECKLIST.md`

## Quickstart (dev)

Create a virtual environment and install:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run CLI (stub for now):

```bash
godmode --help
```

## Web UI

The web app lets you:
- **Record**: fetch data from Alpaca, run the pipeline, save results
- **Browse**: view sessions, episodes, snapshots, markers

### Setup (Alpaca)

Set your API keys:

```powershell
setx ALPACA_API_KEY "YOUR_API_KEY_ID"
setx ALPACA_SECRET_KEY "YOUR_SECRET_KEY"
```

Restart your shell after setting them.

### Run

```powershell
$env:ALPACA_API_KEY = "YOUR_API_KEY_ID"; $env:ALPACA_SECRET_KEY = "YOUR_SECRET_KEY"; godmode web --host 127.0.0.1 --port 8000
```

Or if env vars are already set:

```bash
godmode web --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

### Workflow

1. Click **+ Record**
2. Enter tickers (e.g., `TSLA, NVDA`), start/end times, direction bias
3. Optionally add a marker time + type (e.g., "8:40 AM, downtrend_break")
4. Click **Fetch & Analyze**
5. Watch progress on the Jobs page (auto-refreshes)
6. When done, click **View →** to see results

## Data Providers

### Alpaca (recommended)

Alpaca Algo Trader Plus ($99/mo) provides:
- Full tick-level trades
- Full NBBO quotes (Level 1)
- 7+ years historical data
- Real-time streaming

This is what GodMode uses by default.

### Polygon (alternative)

Polygon Developer ($79/mo) has trades but NO quotes.
Polygon Advanced ($199/mo) has both trades and quotes.

To use Polygon instead of Alpaca, set `POLYGON_API_KEY` and modify `jobs.py` to use `export_polygon_to_replay`.




