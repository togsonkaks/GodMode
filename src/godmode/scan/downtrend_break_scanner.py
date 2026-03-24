from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import logging
from typing import Any, Literal

import pandas as pd

from godmode.engine.bar_builder import BarBuilder
from godmode.providers.alpaca import AlpacaProvider

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


Timeframe = Literal["30s", "1m", "2m", "3m", "5m"]


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    top_n: int = 12
    refresh_seconds: int = 5

    # Session window (America/Chicago): 3:00am–3:00pm CT
    session_start_ct: time = time(3, 0, 0)
    session_end_ct: time = time(15, 0, 0)

    min_volume: float = 3_000_000
    min_up_pct: float = 0.50
    min_off_hod_pct: float = 0.15

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # EMA
    ema_fast: int = 9
    ema_mid: int = 20
    ema_slow: int = 30
    ema_long: int = 200

    # Alert logic
    new_low_realert_min_drop_pct: float = 0.0  # allow any new low by default
    ema200_regain_confirm_s: int = 60

    # How much recent data to pull for signals
    recent_minutes_for_signals: int = 90


def _chicago_tz() -> Any:
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available (Python 3.9+) required")
    return ZoneInfo("America/Chicago")


def _now_ct() -> datetime:
    return datetime.now(tz=_chicago_tz())


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def session_window_ms(*, day: date, cfg: ScannerConfig) -> tuple[int, int]:
    tz = _chicago_tz()
    start = datetime.combine(day, cfg.session_start_ct, tzinfo=tz)
    end = datetime.combine(day, cfg.session_end_ct, tzinfo=tz)
    return _ms(start), _ms(end)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()


def compute_macd(close: pd.Series, *, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _macd_ok(hist: pd.Series, macd_line: pd.Series) -> tuple[bool, str]:
    """
    Returns (ok, tier) where tier is one of: flip, expanding, none.
    """
    if len(hist) < 3:
        return False, "none"
    h2, h1, h0 = float(hist.iloc[-3]), float(hist.iloc[-2]), float(hist.iloc[-1])
    if h1 <= 0 and h0 > 0:
        return True, "flip"
    if h0 > 0 and h0 > h1 > h2:
        return True, "expanding"
    return False, "none"


def _ema_ok(ema9: pd.Series, ema20: pd.Series, ema30: pd.Series) -> bool:
    if len(ema9) < 2 or len(ema20) < 2 or len(ema30) < 2:
        return False
    e9 = float(ema9.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e30 = float(ema30.iloc[-1])
    if not (e9 > e20):
        return False
    if e9 > e30:
        return True
    # Crossover over EMA30 in the last bar
    prev9 = float(ema9.iloc[-2])
    prev30 = float(ema30.iloc[-2])
    return prev9 <= prev30 and e9 > e30


def _ema200_touch_regain(
    *,
    bars: pd.DataFrame,
    ema200: pd.Series,
    confirm_seconds: int,
) -> tuple[bool, str]:
    """
    Touch: last closed bar range intersects EMA200.
    Regain: within ~confirm_seconds, a later bar closes above EMA200.
    This uses bar-level logic; for timeframes > 1m, regain can be slower.
    """
    if bars.empty or len(bars) < 2:
        return False, "no_bars"
    if any(c not in bars.columns for c in ("ts_ms", "high", "low", "close")):
        return False, "missing_cols"
    e = pd.to_numeric(ema200, errors="coerce").dropna()
    if e.empty:
        return False, "no_ema200"

    last_idx = len(bars) - 2  # last closed bar (ignore possibly in-progress)
    low = float(bars["low"].iloc[last_idx])
    high = float(bars["high"].iloc[last_idx])
    e200 = float(ema200.iloc[last_idx])
    touched = low <= e200 <= high
    if not touched:
        return False, "no_touch"

    touch_ts = int(bars["ts_ms"].iloc[last_idx])
    # Find first subsequent bar within confirm window that closes above EMA200
    for j in range(last_idx + 1, len(bars)):
        ts = int(bars["ts_ms"].iloc[j])
        if ts - touch_ts > int(confirm_seconds * 1000):
            break
        close = float(bars["close"].iloc[j])
        e200j = float(ema200.iloc[j])
        if close > e200j:
            return True, "touch_regain"
    return False, "touch_failed"


def _to_bar_df_from_trades(trades: list[Any], *, symbol: str, bucket_seconds: int) -> pd.DataFrame:
    bb = BarBuilder(symbol=symbol, bucket_seconds=bucket_seconds, allow_out_of_order=False)
    rows = []
    for t in trades:
        finished = bb.update(t)
        if finished is None:
            continue
        rows.append(
            {
                "ts_ms": int(finished.ts_ms),
                "open": float(finished.open),
                "high": float(finished.high),
                "low": float(finished.low),
                "close": float(finished.close),
                "volume": float(finished.volume),
                "trade_count": int(finished.trade_count),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)


def _resample_bars(bars_1m: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
    if bars_1m.empty:
        return bars_1m
    b = bars_1m.copy()
    b["dt"] = pd.to_datetime(b["ts_ms"], unit="ms", utc=True)
    b = b.set_index("dt")
    rule = f"{int(minutes)}min"
    o = b["open"].resample(rule).first()
    h = b["high"].resample(rule).max()
    l = b["low"].resample(rule).min()
    c = b["close"].resample(rule).last()
    v = b["volume"].resample(rule).sum()
    n = b["trade_count"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v, n], axis=1).dropna()
    out = out.rename(
        columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "trade_count": "trade_count"}
    )
    out["ts_ms"] = (out.index.view("int64") // 1_000_000).astype("int64")
    out = out.reset_index(drop=True)
    return out[["ts_ms", "open", "high", "low", "close", "volume", "trade_count"]]


@dataclass(slots=True)
class TimeframeSignal:
    timeframe: Timeframe
    ema_ok: bool
    macd_ok: bool
    macd_tier: str
    ema200_touch_regain: bool
    ema200_reason: str


@dataclass(slots=True)
class TickerScanResult:
    ticker: str
    session_open: float | None
    session_low: float | None
    hod: float | None
    last: float | None
    pct_up: float | None
    pct_off_hod: float | None
    volume: float | None
    passes_day_filters: bool
    signals: list[TimeframeSignal]


class DowntrendBreakScanner:
    def __init__(self, *, cfg: ScannerConfig) -> None:
        self._cfg = cfg
        self._provider = AlpacaProvider.from_env()
        self._log = logging.getLogger("godmode.scan.downtrend_break_scanner")

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def scan(self, tickers: list[str], *, now_ms: int) -> list[TickerScanResult]:
        """
        Scan a list of tickers. This is intentionally stateless; alerting is handled elsewhere.
        """
        out: list[TickerScanResult] = []

        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        now_ct = now_dt.astimezone(_chicago_tz())
        day = now_ct.date()
        start_ms, end_ms = session_window_ms(day=day, cfg=self._cfg)
        end_ms = min(end_ms, now_ms)
        if end_ms <= start_ms:
            return out

        recent_start_ms = max(start_ms, end_ms - int(self._cfg.recent_minutes_for_signals * 60_000))

        for t in tickers:
            t = t.upper().strip()
            try:
                bars_rows = await self._provider.getBars(t, start_ms, end_ms, timeframe="1Min")
            except Exception as e:
                self._log.warning("getBars failed ticker=%s err=%r", t, e)
                out.append(
                    TickerScanResult(
                        ticker=t,
                        session_open=None,
                        session_low=None,
                        hod=None,
                        last=None,
                        pct_up=None,
                        pct_off_hod=None,
                        volume=None,
                        passes_day_filters=False,
                        signals=[],
                    )
                )
                continue
            bars_1m_day = pd.DataFrame(bars_rows) if bars_rows else pd.DataFrame()
            stats = None
            if not bars_1m_day.empty:
                bars_1m_day = bars_1m_day.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)
                stats = {
                    "open": float(bars_1m_day["open"].iloc[0]),
                    "low": float(pd.to_numeric(bars_1m_day["low"], errors="coerce").min()),
                    "hod": float(pd.to_numeric(bars_1m_day["high"], errors="coerce").max()),
                    "last": float(bars_1m_day["close"].iloc[-1]),
                    "volume": float(pd.to_numeric(bars_1m_day["volume"], errors="coerce").fillna(0).sum()),
                }

            if stats is None:
                out.append(
                    TickerScanResult(
                        ticker=t,
                        session_open=None,
                        session_low=None,
                        hod=None,
                        last=None,
                        pct_up=None,
                        pct_off_hod=None,
                        volume=None,
                        passes_day_filters=False,
                        signals=[],
                    )
                )
                continue

            open_px = float(stats["open"])
            low_px = float(stats["low"])
            hod = float(stats["hod"])
            last = float(stats["last"])
            vol = float(stats["volume"])

            pct_up = (hod - open_px) / open_px if open_px > 0 else None
            pct_off = (hod - last) / hod if hod > 0 else None

            passes_day = (
                pct_up is not None
                and pct_off is not None
                and pct_up >= self._cfg.min_up_pct
                and pct_off >= self._cfg.min_off_hod_pct
                and vol >= self._cfg.min_volume
            )

            tf_signals: list[TimeframeSignal] = []
            if passes_day:
                # Use bars for 1m and resample for higher TFs. Use trades only for 30s.
                try:
                    bars_recent_rows = await self._provider.getBars(t, recent_start_ms, end_ms, timeframe="1Min")
                except Exception as e:
                    self._log.warning("getBars(recent) failed ticker=%s err=%r", t, e)
                    bars_recent_rows = []
                bars_1m = pd.DataFrame(bars_recent_rows) if bars_recent_rows else pd.DataFrame()
                if not bars_1m.empty:
                    bars_1m = bars_1m.sort_values(["ts_ms"], kind="mergesort").reset_index(drop=True)

                try:
                    trades_recent = await self._provider.getTrades(t, recent_start_ms, end_ms)
                except Exception as e:
                    self._log.warning("getTrades failed ticker=%s err=%r", t, e)
                    trades_recent = []
                bars_30s = _to_bar_df_from_trades(trades_recent, symbol=t, bucket_seconds=30)
                bars_2m = _resample_bars(bars_1m, minutes=2) if not bars_1m.empty else pd.DataFrame()
                bars_3m = _resample_bars(bars_1m, minutes=3) if not bars_1m.empty else pd.DataFrame()
                bars_5m = _resample_bars(bars_1m, minutes=5) if not bars_1m.empty else pd.DataFrame()

                tf_map: dict[Timeframe, pd.DataFrame] = {
                    "30s": bars_30s,
                    "1m": bars_1m,
                    "2m": bars_2m,
                    "3m": bars_3m,
                    "5m": bars_5m,
                }

                for tf, b in tf_map.items():
                    if b.empty or len(b) < 35:
                        continue
                    close = pd.to_numeric(b["close"], errors="coerce").dropna()
                    if close.empty:
                        continue
                    ema9 = _ema(close, self._cfg.ema_fast)
                    ema20 = _ema(close, self._cfg.ema_mid)
                    ema30 = _ema(close, self._cfg.ema_slow)
                    ema200 = _ema(close, self._cfg.ema_long)

                    ema_ok = _ema_ok(ema9, ema20, ema30)
                    macd_line, signal_line, hist = compute_macd(
                        close, fast=self._cfg.macd_fast, slow=self._cfg.macd_slow, signal=self._cfg.macd_signal
                    )
                    macd_ok, tier = _macd_ok(hist, macd_line)

                    # EMA200 touch+regain: use same timeframe bars (best-effort)
                    touch_regain, touch_reason = _ema200_touch_regain(
                        bars=b, ema200=ema200, confirm_seconds=self._cfg.ema200_regain_confirm_s
                    )

                    tf_signals.append(
                        TimeframeSignal(
                            timeframe=tf,
                            ema_ok=bool(ema_ok),
                            macd_ok=bool(macd_ok),
                            macd_tier=tier,
                            ema200_touch_regain=bool(touch_regain),
                            ema200_reason=touch_reason,
                        )
                    )

            out.append(
                TickerScanResult(
                    ticker=t,
                    session_open=open_px,
                    session_low=low_px,
                    hod=hod,
                    last=last,
                    pct_up=float(pct_up) if pct_up is not None else None,
                    pct_off_hod=float(pct_off) if pct_off is not None else None,
                    volume=vol,
                    passes_day_filters=bool(passes_day),
                    signals=tf_signals,
                )
            )

        return out
