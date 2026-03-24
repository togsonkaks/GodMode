from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from godmode.scan.downtrend_break_scanner import DowntrendBreakScanner, ScannerConfig
from godmode.scan.expo_push import ExpoPushMessage, send_expo_push
from godmode.scan.scanner_state import ScannerState, load_state, save_state


def _ct_day_key() -> str:
    from datetime import datetime, timezone
    import pytz

    dt = datetime.now(tz=timezone.utc)
    chicago = pytz.timezone("America/Chicago")
    return dt.astimezone(chicago).strftime("%Y-%m-%d")


def _ct_now_str() -> str:
    from datetime import datetime, timezone
    import pytz

    dt = datetime.now(tz=timezone.utc)
    chicago = pytz.timezone("America/Chicago")
    return dt.astimezone(chicago).strftime("%Y-%m-%d %H:%M:%S CT")


def _ct_hm() -> float:
    from datetime import datetime, timezone
    import pytz

    dt = datetime.now(tz=timezone.utc)
    chicago = pytz.timezone("America/Chicago")
    local = dt.astimezone(chicago)
    return float(local.hour) + float(local.minute) / 60.0


async def _alpaca_top_gainers(*, top_n: int) -> list[str]:
    import httpx

    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        return []
    url = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"top": int(top_n)}
    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    movers = data.get("movers", []) if isinstance(data, dict) else []
    syms: list[str] = []
    for m in movers:
        if not isinstance(m, dict) or not m.get("symbol"):
            continue
        # Prefer gainers only if change field exists.
        ch = m.get("percent_change", m.get("change", None))
        try:
            if ch is not None and float(ch) <= 0:
                continue
        except Exception:
            pass
        syms.append(str(m["symbol"]).upper())
    return syms[: int(top_n)]


def _parse_tokens() -> list[str]:
    raw = (os.environ.get("EXPO_PUSH_TOKENS") or "").strip()
    if not raw:
        return []
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    return toks


async def run_scanner_once(*, state_path: Path, cfg: ScannerConfig) -> dict[str, Any]:
    # Only run 3am–3pm CT, unless forced (for testing).
    force = (os.environ.get("SCANNER_FORCE") or "").strip() == "1"
    hm = _ct_hm()
    if (not force) and (hm < 3.0 or hm >= 15.0):
        return {"ok": True, "skipped": True, "reason": "outside_session", "now": _ct_now_str()}

    day = _ct_day_key()
    state = load_state(state_path, day=day)
    scanner = DowntrendBreakScanner(cfg=cfg)
    try:
        tickers = await _alpaca_top_gainers(top_n=cfg.top_n)
        now_ms = int(time.time() * 1000)
        results = await scanner.scan(tickers, now_ms=now_ms)
    finally:
        await scanner.aclose()

    tf_order = ["30s", "1m", "2m", "3m", "5m"]
    new_alerts: list[dict[str, Any]] = []

    for r in results:
        t = r.ticker
        st = state.tickers.setdefault(
            t,
            {"cycle": 0, "fired": [], "noted_align": False, "armed": False, "low_at_cycle": None},
        )
        fired = set(st.get("fired") or [])

        if st.get("low_at_cycle") is not None and r.session_low is not None:
            prev_low = float(st["low_at_cycle"])
            cur_low = float(r.session_low)
            if cur_low < prev_low * (1.0 - float(cfg.new_low_realert_min_drop_pct)):
                st["armed"] = True

        if st.get("armed"):
            any_positive = any(s.ema_ok and s.macd_ok for s in (r.signals or []))
            if any_positive:
                st["cycle"] = int(st.get("cycle") or 0) + 1
                fired = set()
                st["noted_align"] = False
                st["armed"] = False
                st["low_at_cycle"] = r.session_low

        if st.get("low_at_cycle") is None:
            st["low_at_cycle"] = r.session_low

        sig_by_tf = {s.timeframe: s for s in (r.signals or [])}
        for tf in tf_order:
            s = sig_by_tf.get(tf)
            if s is None or not (s.ema_ok and s.macd_ok):
                continue
            key = f"{tf}:cycle{st['cycle']}"
            if key in fired:
                continue
            fired.add(key)
            new_alerts.append(
                {"ticker": t, "kind": f"{tf} EMA+MACD", "when": _ct_now_str(), "detail": f"MACD={s.macd_tier}"}
            )

        for s in (r.signals or []):
            if not s.ema200_touch_regain:
                continue
            key = f"ema200:{s.timeframe}:cycle{st['cycle']}"
            if key in fired:
                continue
            fired.add(key)
            new_alerts.append(
                {"ticker": t, "kind": f"{s.timeframe} EMA200 regain", "when": _ct_now_str(), "detail": s.ema200_reason}
            )

        if not st.get("noted_align"):
            need = {f"{tf}:cycle{st['cycle']}" for tf in ("30s", "1m", "2m", "3m")}
            if need.issubset(fired):
                st["noted_align"] = True
                new_alerts.append(
                    {"ticker": t, "kind": "ALIGN 30s+1m+2m+3m", "when": _ct_now_str(), "detail": "multi-timeframe alignment"}
                )

        st["fired"] = sorted(fired)

    if new_alerts:
        state.alerts = (new_alerts + state.alerts)[:500]

    save_state(state_path, state)

    tokens = _parse_tokens()
    push_result = None
    if tokens and new_alerts:
        messages: list[ExpoPushMessage] = []
        for a in new_alerts[:25]:
            title = f"{a['ticker']} {a['kind']}"
            body = a.get("detail") or ""
            data = {
                "type": "alert",
                "symbol": a.get("ticker"),
                "kind": a.get("kind"),
                "detail": a.get("detail"),
                "when": a.get("when"),
            }
            for tok in tokens:
                messages.append(ExpoPushMessage(to=tok, title=title, body=body, data=data))
        push_result = await send_expo_push(messages=messages)

    return {
        "ok": True,
        "skipped": False,
        "now": _ct_now_str(),
        "tickers": tickers,
        "new_alerts": new_alerts,
        "push": push_result,
        "results": [asdict(r) | {"signals": [asdict(s) for s in (r.signals or [])]} for r in results],
    }


def main() -> None:
    state_path = Path(os.environ.get("SCANNER_STATE_PATH") or "data/scanner/scanner_state.json")
    cfg = ScannerConfig(top_n=int(os.environ.get("SCANNER_TOP_N") or 12), refresh_seconds=5)
    res = asyncio.run(run_scanner_once(state_path=state_path, cfg=cfg))
    print(res)


if __name__ == "__main__":
    main()
