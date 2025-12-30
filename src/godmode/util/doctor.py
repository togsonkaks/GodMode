from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal


Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    status: Status
    message: str


def run_doctor(*, project_root: Path | None = None) -> List[Check]:
    root = project_root or Path(".")
    checks: list[Check] = []

    # Load local .env if present (best-effort), so doctor reflects what the app will see.
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=root / ".env", override=False)
    except Exception:
        pass

    checks.append(Check("ok", f"python={sys.executable}"))
    checks.append(Check("ok", f"version={sys.version.split()[0]}"))

    if sys.version_info < (3, 11):
        checks.append(Check("fail", "Python 3.11+ required"))
        return checks

    # Core deps
    try:
        import pyarrow  # noqa: F401
        import pandas  # noqa: F401
        import pydantic  # noqa: F401

        checks.append(Check("ok", "Core deps import (pyarrow/pandas/pydantic)"))
    except Exception as e:
        checks.append(Check("fail", f"Core dependency import failed: {type(e).__name__}: {e}"))
        return checks

    # Web deps (optional)
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import jinja2  # noqa: F401
        import multipart  # noqa: F401  # python-multipart

        checks.append(Check("ok", "Web deps import (fastapi/uvicorn/jinja2/python-multipart)"))
    except Exception as e:
        checks.append(Check("warn", f"Web deps import incomplete: {type(e).__name__}: {e}"))

    # Config file
    cfg = root / "config" / "default.yaml"
    checks.append(Check("ok" if cfg.exists() else "warn", "config/default.yaml present" if cfg.exists() else "config/default.yaml missing"))

    # Provider env vars
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        checks.append(Check("ok", "Alpaca env vars present (ALPACA_API_KEY/ALPACA_SECRET_KEY)"))
    else:
        checks.append(Check("warn", "Alpaca env vars missing (needed for live/export via Alpaca)"))

    # Output roots (informational)
    for d in [root / "data" / "output", root / "data" / "output_clean"]:
        if d.exists():
            checks.append(Check("ok", f"found {d.as_posix()}"))

    return checks


def print_doctor(checks: List[Check]) -> int:
    exit_code = 0
    for c in checks:
        tag = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[c.status]
        print(f"{tag} {c.message}")
        if c.status == "fail":
            exit_code = 2
    return exit_code


