"""Price-bar fallback for when the TradingView scanner keeps failing.

TradingView's scanner sometimes answers every request with an empty body for
minutes at a time. The resilience layer in ``screener_provider`` retries and
serves its stale cache; when both fail, the single-symbol tools
(coin_analysis, multi_timeframe_analysis, multi_agent_analysis, ...) come here.

This module gets OHLCV bars from Yahoo Finance first, then from Tiingo (US
stocks, ``TIINGO_API_KEY``), then from Alpha Vantage (1D/1W/1M stocks and 1D
crypto, ``ALPHAVANTAGE_API_KEY``). It calculates the TradingView indicator
columns from the bars and builds the same ``tradingview_ta`` ``Analysis``
object, so the tools need no second code path. Each such analysis has a
``data_source`` attribute ("yahoo:BTC-USD", "tiingo:AAPL", "alphavantage:IBM")
so a tool can tell its caller where the numbers came from.

Limits:
- Only symbols with a known mapping: crypto pairs, US stocks, BIST, EGX,
  TADAWUL. Other venues return None, and the caller keeps its TradingView error.
- Crypto volume is Yahoo's aggregate, not the volume of the named exchange.
- Values are calculated locally and differ a little from TradingView's. The
  pivot columns and three of the per-indicator ``Rec.*`` votes stay None, so
  the ``Recommend.*`` ratings use only the indicators calculated here.
- Tiingo intraday bars are IEX bars: IEX trades and IEX volume only. Its 1h
  and 4h bars are grouped from 15min bars from 9:30 New York time, the US
  session open. Free keys allow 50 requests per hour and 1000 per day.
- Alpha Vantage free keys allow 25 requests per day and no intraday data.
- The caps below count this process's requests in rolling windows, which is
  stricter than any fixed hour or day the API resets on.

Tunables (env vars):
  TIINGO_API_KEY              unset = Tiingo is not used
  TIINGO_HOURLY_LIMIT         default 50 (requests per rolling hour)
  TIINGO_DAILY_LIMIT          default 1000 (requests per rolling 24 hours)
  TIINGO_CACHE_TTL            default 300 (seconds a Tiingo answer is reused)
  ALPHAVANTAGE_API_KEY        unset = Alpha Vantage is not used
  ALPHAVANTAGE_DAILY_LIMIT    default 25 (requests per rolling 24 hours)
  ALPHAVANTAGE_CACHE_TTL      default 3600 (seconds an Alpha Vantage answer is reused)
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .indicators import AVERAGE_VOLUME, RELATIVE_VOLUME
from .indicators_calc import (
    calc_atr,
    calc_bollinger,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_sma,
)

# Single-symbol tools only. A scan of a whole exchange cannot be served from
# per-symbol price requests.
MAX_SYMBOLS = 3

# Fewer bars than this and most columns would be None (AO needs 34).
_MIN_BARS = 35

# Pair quote -> the Yahoo / Alpha Vantage market it trades against.
_CRYPTO_QUOTES = (
    ("FDUSD", "USD"), ("USDT", "USD"), ("USDC", "USD"), ("BUSD", "USD"),
    ("USD", "USD"), ("EUR", "EUR"),
)
_YAHOO_SUFFIX = {"BIST": ".IS", "EGX": ".CA", "TADAWUL": ".SR"}

# timeframe -> (Yahoo interval, Yahoo range). 4h is grouped from 1h.
_YAHOO_INTERVALS = {
    "5m": ("5m", "5d"),
    "15m": ("15m", "1mo"),
    "1h": ("1h", "3mo"),
    "4h": ("1h", "1y"),
    "1D": ("1d", "2y"),
    "1W": ("1wk", "10y"),
    "1M": ("1mo", "max"),
}

# Bars merged into one output bar share this key of their UTC start time:
# 4h blocks 00-04, 04-08, ... like TradingView's crypto 4h bars, ISO weeks
# (Monday start) and calendar months.
_GROUP_KEYS = {
    "4h": lambda ts: ts // (3600 * 4),
    "1W": lambda ts: datetime.fromtimestamp(ts, timezone.utc).isocalendar()[:2],
    "1M": lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m"),
}

# US stock 1h and 4h bars start at the 9:30 New York session open
# (9:30-10:30 ... 15:30-16:00, and 9:30-13:30, 13:30-16:00), not on clock hours.
_NEW_YORK = ZoneInfo("America/New_York")


def _session_key(hours: int):
    def key(ts: int) -> tuple:
        t = datetime.fromtimestamp(ts, _NEW_YORK)
        return t.date(), (t.hour * 60 + t.minute - 570) // (60 * hours)
    return key


_TIINGO_GROUP_KEYS = {**_GROUP_KEYS, "1h": _session_key(1), "4h": _session_key(4)}

_INTRADAY_SECONDS = {"5m": 300, "15m": 900, "1h": 3600}
_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
_UA = "Mozilla/5.0 (compatible; tradingview-mcp)"

_AV_URL = "https://www.alphavantage.co/query"
_AV_FUNCTIONS = {
    ("stock", "1D"): "TIME_SERIES_DAILY",
    ("stock", "1W"): "TIME_SERIES_WEEKLY",
    ("stock", "1M"): "TIME_SERIES_MONTHLY",
    ("crypto", "1D"): "DIGITAL_CURRENCY_DAILY",
}

_TIINGO_URL = "https://api.tiingo.com"
# timeframe -> (Tiingo path, IEX resampleFreq, days of history). Measured on
# 2026-11-07 with a free key: 5 years of end-of-day rows, 365 days of IEX
# 1hour rows, 150 days of 15min rows (2808 rows in one answer). 1h, 4h, 1W
# and 1M are grouped here (see _TIINGO_GROUP_KEYS).
_TIINGO_INTERVALS = {
    "5m": ("iex", "5min", 7),
    "15m": ("iex", "15min", 30),
    "1h": ("iex", "15min", 90),
    "4h": ("iex", "15min", 150),
    "1D": ("tiingo/daily", None, 730),
    "1W": ("tiingo/daily", None, 1825),
    "1M": ("tiingo/daily", None, 1825),
}

DATA_NOTE = (
    "TradingView did not answer, so these values come from {source} price bars "
    "with indicators calculated locally. They can differ a little from "
    "TradingView's values."
)


def _log(msg: str) -> None:
    with contextlib.suppress(Exception):  # a closed stderr must not break a tool
        print(f"[tradingview_mcp] {msg}", file=sys.stderr)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class _Quota:
    """Requests this process may still send to one keyed API: a cap per
    rolling window, a stop after the API itself reports its limit, and a
    cache of the answers."""

    def __init__(self, windows: dict[int, tuple[str, float]], ttl: tuple[str, float]):
        self.windows = windows  # seconds -> (env var, default cap)
        self.ttl = ttl  # (env var, default seconds)
        self.lock = threading.Lock()
        self.cache: dict[Any, tuple[float, list[dict]]] = {}
        self.sent: list[float] = []
        self.blocked_until = 0.0

    def cached(self, key: Any) -> list[dict] | None:
        with self.lock:
            hit = self.cache.get(key)
        if hit and time.time() - hit[0] < _env_float(*self.ttl):
            return hit[1]
        return None

    def store(self, key: Any, candles: list[dict]) -> None:
        with self.lock:
            self.cache[key] = (time.time(), candles)

    def take(self) -> bool:
        """Count one request, or False when a cap or a block says no."""
        now = time.time()
        with self.lock:
            if now < self.blocked_until:
                return False
            self.sent = [t for t in self.sent if now - t < max(self.windows)]
            for seconds, (var, default) in self.windows.items():
                if sum(now - t < seconds for t in self.sent) >= _env_float(var, default):
                    return False
            self.sent.append(now)
            return True

    def block(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until = time.time() + seconds

    def reset(self) -> None:
        with self.lock:
            self.cache.clear()
            self.sent.clear()
            self.blocked_until = 0.0


_TIINGO = _Quota({3600: ("TIINGO_HOURLY_LIMIT", 50), 86_400: ("TIINGO_DAILY_LIMIT", 1000)},
                 ("TIINGO_CACHE_TTL", 300))
_AV = _Quota({86_400: ("ALPHAVANTAGE_DAILY_LIMIT", 25)}, ("ALPHAVANTAGE_CACHE_TTL", 3600))


# ── Symbol mapping ────────────────────────────────────────────────────────────

def provider_symbols(screener: str, tv_symbol: str) -> tuple[str, dict | None] | None:
    """``(yahoo_symbol, alpha_vantage_params or None)``, or None when we do
    not know the symbol on Yahoo (better no data than another instrument's)."""
    exchange, _, ticker = (tv_symbol or "").strip().upper().partition(":")
    if not ticker:
        return None
    if screener == "crypto":
        for quote, market in _CRYPTO_QUOTES:
            if ticker.endswith(quote) and len(ticker) > len(quote):
                base = ticker[: -len(quote)]
                return f"{base}-{market}", {"kind": "crypto", "symbol": base, "market": market}
        return None
    if screener == "america":
        return ticker.replace(".", "-"), {"kind": "stock", "symbol": ticker}
    suffix = _YAHOO_SUFFIX.get(exchange)
    if suffix:
        return ticker + suffix, None
    return None


# ── Bar sources ───────────────────────────────────────────────────────────────

def _get_json(url: str, proxy_retry: bool = False, headers: dict | None = None) -> Any:
    """GET ``url`` as JSON. With ``proxy_retry``, a failed direct request is
    sent again through the configured proxy, as backtest_service does."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        if not proxy_retry:
            raise
    from .proxy_manager import build_opener_with_proxy
    with build_opener_with_proxy(_UA).open(url, timeout=18) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_yahoo(yahoo_symbol: str, timeframe: str, us_session: bool = False) -> list[dict]:
    """Bars from Yahoo's chart API, oldest first. Prices are not rounded
    (backtest_service._fetch_ohlcv rounds to 4 decimals, which zeroes
    small-price coins), and not dividend-adjusted, like TradingView's.
    4h bars are grouped from 1h bars: with ``us_session`` from the 9:30 New
    York open (Yahoo's US stock 1h bars start at 9:30, 10:30, ... measured
    2026-11-07), else in UTC blocks."""
    interval, period = _YAHOO_INTERVALS[timeframe]
    url = f"{_YAHOO_URL}/{urllib.parse.quote(yahoo_symbol)}?interval={interval}&range={period}"
    data = _get_json(url, proxy_retry=True)
    result = data["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    candles = []
    for i, ts in enumerate(result.get("timestamp") or []):
        bar = [q[k][i] for k in ("open", "high", "low", "close")]
        if None in bar:
            continue
        candles.append({"ts": ts, "open": bar[0], "high": bar[1], "low": bar[2],
                        "close": bar[3], "volume": q["volume"][i] or 0})
    # Intraday answers end with a last-price tick as an extra bar (volume 0,
    # e.g. BTC-USD 15m at 08:36:54 after the 08:30 bar). It is not a bar.
    step = _INTRADAY_SECONDS.get(interval)
    if step and len(candles) > 1 and candles[-1]["ts"] - candles[-2]["ts"] < step:
        candles.pop()
    if timeframe != "4h":
        return candles
    return _group(candles, _session_key(4) if us_session else _GROUP_KEYS["4h"])


def _group(candles: list[dict], key) -> list[dict]:
    """Consecutive bars with the same ``key(ts)`` merged into one bar."""
    out: list[dict] = []
    last = None
    for c in candles:
        k = key(c["ts"])
        if out and k == last:
            bar = out[-1]
            bar["high"] = max(bar["high"], c["high"])
            bar["low"] = min(bar["low"], c["low"])
            bar["close"] = c["close"]
            bar["volume"] += c["volume"]
        else:
            out.append(dict(c))
            last = k
    return out


def _fetch_tiingo(ticker: str, timeframe: str) -> list[dict] | None:
    """US stock bars from Tiingo, oldest first: IEX bars intraday, end-of-day
    bars from 1D up. Split-adjusted, not dividend-adjusted, like TradingView's."""
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        return None
    path, freq, days = _TIINGO_INTERVALS[timeframe]
    symbol = ticker.replace(".", "-").lower()  # BRK.B is brk-b (measured)
    cached = _TIINGO.cached((symbol, timeframe))
    if cached is not None:
        return cached
    if not _TIINGO.take():
        return None
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    query = {"startDate": start}
    if freq:
        # IEX sends no volume unless the columns are named.
        query.update(resampleFreq=freq, columns="open,high,low,close,volume")
    url = f"{_TIINGO_URL}/{path}/{urllib.parse.quote(symbol)}/prices?{urllib.parse.urlencode(query)}"
    try:
        # The key goes in the header, never in the URL.
        rows = _get_json(url, headers={"Authorization": f"Token {key}"})
    except urllib.error.HTTPError as e:
        # 429 is the usual HTTP rate-limit status. Tiingo's documentation does
        # not name one (read 2026-11-07), so this is not verified.
        if e.code == 429:
            _TIINGO.block(3600)
        _log(f"Tiingo {path} {symbol} failed: HTTP {e.code}")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"Tiingo {path} {symbol} failed: {type(e).__name__}")
        return None
    if not isinstance(rows, list):
        _log(f"Tiingo {path} {symbol} gave no data: {str(rows)[:200]}")
        return None

    # End-of-day prices are as traded. splitFactor on the split day (NVDA
    # 2024-06-10: 10.0) divides the prices of all earlier rows.
    candles: list[dict] = []
    factor = 1.0
    for r in reversed(rows):
        try:
            ts = int(datetime.fromisoformat(r["date"].replace("Z", "+00:00")).timestamp())
            candles.append({
                "ts": ts,
                "open": float(r["open"]) / factor,
                "high": float(r["high"]) / factor,
                "low": float(r["low"]) / factor,
                "close": float(r["close"]) / factor,
                "volume": float(r.get("volume") or 0) * factor,
            })
            factor *= float(r.get("splitFactor") or 1.0)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    candles.reverse()
    group = _TIINGO_GROUP_KEYS.get(timeframe)
    if group:
        candles = _group(candles, group)
    _TIINGO.store((symbol, timeframe), candles)
    return candles


def _fetch_alpha_vantage(av: dict, timeframe: str) -> list[dict] | None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    function = _AV_FUNCTIONS.get((av["kind"], timeframe))
    if not key or not function:
        return None
    params = {"function": function, "symbol": av["symbol"]}
    if av["kind"] == "crypto":
        params["market"] = av["market"]
    cache_key = tuple(sorted(params.items()))
    cached = _AV.cached(cache_key)
    if cached is not None:
        return cached
    if not _AV.take():
        return None

    url = f"{_AV_URL}?{urllib.parse.urlencode({**params, 'apikey': key})}"
    try:
        body = _get_json(url)
    except Exception as e:  # noqa: BLE001 - never log the URL, it holds the key
        _log(f"Alpha Vantage {function} {av['symbol']} failed: {type(e).__name__}")
        return None
    if not isinstance(body, dict):
        return None

    series = next((v for k, v in body.items() if k != "Meta Data" and isinstance(v, dict)), None)
    if not series:
        # "Information" / "Note" is the daily or per-minute limit: stop for today.
        # "Error Message" is an unknown symbol: only this request fails.
        if "Information" in body or "Note" in body:
            _AV.block(86_400 - time.time() % 86_400)  # until the next UTC day
        _log(f"Alpha Vantage {function} {av['symbol']} gave no data: {str(body)[:200]}")
        return None

    candles = []
    for date in sorted(series):  # Alpha Vantage sends newest first
        bar = series[date]
        try:
            candles.append({
                "date": date,
                "open": float(bar["1. open"]),
                "high": float(bar["2. high"]),
                "low": float(bar["3. low"]),
                "close": float(bar["4. close"]),
                "volume": float(bar["5. volume"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    _AV.store(cache_key, candles)
    return candles


# ── Indicators ────────────────────────────────────────────────────────────────

def _rolling(values: Sequence[float | None], n: int, fn) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1 : i + 1]
        if None not in window:
            out[i] = fn(window)
    return out


def _mean(w: list[float]) -> float:
    return sum(w) / len(w)


def _wma(values: Sequence[float | None], n: int) -> list[float | None]:
    weights = range(1, n + 1)
    return _rolling(values, n, lambda w: sum(x * k for x, k in zip(w, weights)) / sum(weights))


def _stoch(src: Sequence[float | None], hi: Sequence[float | None],
           lo: Sequence[float | None], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(src)
    for i in range(n - 1, len(src)):
        x = src[i]
        h = [y for y in hi[i - n + 1 : i + 1] if y is not None]
        l = [y for y in lo[i - n + 1 : i + 1] if y is not None]
        if x is None or len(h) < n or len(l) < n or max(h) == min(l):
            continue
        out[i] = 100 * (x - min(l)) / (max(h) - min(l))
    return out


def _dmi(h: list[float], l: list[float], c: list[float], n: int = 14):
    """Wilder's ADX, +DI and -DI."""
    size = len(c)
    adx: list[float | None] = [None] * size
    pdi: list[float | None] = [None] * size
    mdi: list[float | None] = [None] * size
    if size < 2 * n + 1:
        return adx, pdi, mdi
    tr, pdm, mdm = [0.0], [0.0], [0.0]
    for i in range(1, size):
        up, down = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    s_tr, s_p, s_m = sum(tr[1 : n + 1]), sum(pdm[1 : n + 1]), sum(mdm[1 : n + 1])
    dx: list[float] = []
    for i in range(n, size):
        if i > n:
            s_tr = s_tr - s_tr / n + tr[i]
            s_p = s_p - s_p / n + pdm[i]
            s_m = s_m - s_m / n + mdm[i]
        p, m = (100 * s_p / s_tr, 100 * s_m / s_tr) if s_tr else (0.0, 0.0)
        pdi[i], mdi[i] = p, m
        dx.append(100 * abs(p - m) / (p + m) if p + m else 0.0)
        if len(dx) == n:
            adx[i] = _mean(dx)
        elif len(dx) > n:
            prev = adx[i - 1]
            adx[i] = None if prev is None else (prev * (n - 1) + dx[-1]) / n
    return adx, pdi, mdi


def _psar(h: list[float], l: list[float], step: float = 0.02, cap: float = 0.2) -> float | None:
    """Parabolic SAR value for the newest bar."""
    if len(h) < 3:
        return None
    up = h[1] >= h[0]
    sar, ep, af = (l[0], h[1], step) if up else (h[0], l[1], step)
    for i in range(2, len(h)):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, cap)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, cap)
    return sar


def _ultimate(h: list[float], l: list[float], c: list[float]) -> float | None:
    if len(c) < 29:
        return None
    bp = [c[i] - min(l[i], c[i - 1]) for i in range(1, len(c))]
    tr = [max(h[i], c[i - 1]) - min(l[i], c[i - 1]) for i in range(1, len(c))]
    avg = {}
    for n in (7, 14, 28):
        t = sum(tr[-n:])
        if not t:
            return None
        avg[n] = sum(bp[-n:]) / t
    return 100 * (4 * avg[7] + 2 * avg[14] + avg[28]) / 7


def calculate_indicators(candles: list[dict]) -> dict[str, Any]:
    """TradingView scanner columns (bare names) for the newest bar."""
    o = [float(b["open"]) for b in candles]
    h = [float(b["high"]) for b in candles]
    l = [float(b["low"]) for b in candles]
    c = [float(b["close"]) for b in candles]
    v = [float(b.get("volume") or 0) for b in candles]

    def at(series: Sequence[float | None], back: int = 0) -> float | None:
        i = len(series) - 1 - back
        x = series[i] if i >= 0 else None
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    rsi = calc_rsi(c, 14)
    stoch_k = _rolling(_stoch(c, h, l, 14), 3, _mean)
    stoch_d = _rolling(stoch_k, 3, _mean)
    tp = [(a + b + d) / 3 for a, b, d in zip(h, l, c)]
    tp_sma = calc_sma(tp, 20)
    cci: list[float | None] = [None] * len(c)
    for i in range(19, len(c)):
        mid = tp_sma[i]
        if mid is None:
            continue
        dev = _mean([abs(x - mid) for x in tp[i - 19 : i + 1]])
        cci[i] = (tp[i] - mid) / (0.015 * dev) if dev else 0.0
    adx, pdi, mdi = _dmi(h, l, c)
    median = [(a + b) / 2 for a, b in zip(h, l)]
    ao = [a - b if a is not None and b is not None else None
          for a, b in zip(calc_sma(median, 5), calc_sma(median, 34))]
    mom = [c[i] - c[i - 10] if i >= 10 else None for i in range(len(c))]
    macd = calc_macd(c)
    stoch_rsi_k = _rolling(_stoch(rsi, rsi, rsi, 14), 3, _mean)
    hh14, ll14 = _rolling(h, 14, max), _rolling(l, 14, min)
    ema13 = calc_ema(c, 13)
    bb = calc_bollinger(c, 20, 2.0)
    hull = _wma([2 * a - b if a is not None and b is not None else None
                 for a, b in zip(_wma(c, 4), _wma(c, 9))], 3)
    hh26, ll26 = _rolling(h, 26, max), _rolling(l, 26, min)
    vol20 = sum(v[-20:])
    hi14, lo14, ema13_now = at(hh14), at(ll14), at(ema13)
    hi26, lo26 = at(hh26), at(ll26)

    ind: dict[str, Any] = {
        "RSI": at(rsi), "RSI[1]": at(rsi, 1),
        "Stoch.K": at(stoch_k), "Stoch.D": at(stoch_d),
        "Stoch.K[1]": at(stoch_k, 1), "Stoch.D[1]": at(stoch_d, 1),
        "CCI20": at(cci), "CCI20[1]": at(cci, 1),
        "ADX": at(adx), "ADX+DI": at(pdi), "ADX-DI": at(mdi),
        "ADX+DI[1]": at(pdi, 1), "ADX-DI[1]": at(mdi, 1),
        "AO": at(ao), "AO[1]": at(ao, 1), "AO[2]": at(ao, 2),
        "Mom": at(mom), "Mom[1]": at(mom, 1),
        "MACD.macd": at(macd["macd"]), "MACD.signal": at(macd["signal"]),
        "Stoch.RSI.K": at(stoch_rsi_k),
        "W.R": (-100 * (hi14 - c[-1]) / (hi14 - lo14)
                if hi14 is not None and lo14 is not None and hi14 != lo14 else None),
        "BBPower": (h[-1] + l[-1] - 2 * ema13_now) if ema13_now is not None else None,
        "UO": _ultimate(h, l, c),
        "close": c[-1], "open": o[-1], "high": h[-1], "low": l[-1], "volume": v[-1],
        "change": (c[-1] / c[-2] - 1) * 100 if len(c) > 1 and c[-2] else None,
        "Ichimoku.BLine": (hi26 + lo26) / 2 if hi26 is not None and lo26 is not None else None,
        "VWMA": sum(a * b for a, b in zip(c[-20:], v[-20:])) / vol20 if len(c) >= 20 and vol20 else None,
        "HullMA9": at(hull),
        "P.SAR": _psar(h, l),
        "BB.lower": at(bb["lower"]), "BB.upper": at(bb["upper"]),
        "ATR": at(calc_atr(h, l, c, 14)),
        AVERAGE_VOLUME: _mean(v[-10:]) if len(v) >= 10 else None,
        RELATIVE_VOLUME: (v[-1] / _mean(v[-11:-1])) if len(v) >= 11 and _mean(v[-11:-1]) else None,
    }
    for n in (5, 10, 20, 30, 50, 100, 200):
        ind[f"EMA{n}"] = at(calc_ema(c, n))
        ind[f"SMA{n}"] = at(calc_sma(c, n))

    # TradingView's per-indicator votes, where its rule is known. Measured on
    # 2026-11-07 against the scanner (300 US + 230 crypto rows): Rec.VWMA and
    # Rec.HullMA9 = sign(close - MA) and Rec.UO = +1 above 70, -1 below 30
    # agreed on 526-530 of 530 rows, and Rec.Ichimoku was 0 on all 530. A
    # threshold rule for Rec.WR / Rec.Stoch.RSI agreed on only 67-85%, so
    # those (and Rec.BBPower) stay None and do not vote.
    def above(ma: float | None) -> int | None:
        return None if ma is None else (c[-1] > ma) - (c[-1] < ma)

    uo = ind["UO"]
    ind["Rec.VWMA"] = above(ind["VWMA"])
    ind["Rec.HullMA9"] = above(ind["HullMA9"])
    ind["Rec.UO"] = None if uo is None else (1 if uo > 70 else -1 if uo < 30 else 0)
    ind["Rec.Ichimoku"] = 0
    return ind


def _recommend(counter: dict[str, int]) -> float:
    total = counter["BUY"] + counter["SELL"] + counter["NEUTRAL"]
    return (counter["BUY"] - counter["SELL"]) / total if total else 0.0


def build_analysis(candles: list[dict], screener: str, tv_symbol: str, timeframe: str):
    """A tradingview_ta ``Analysis`` from bars, built like the scanner's."""
    from tradingview_ta.main import TradingView, calculate

    from .screener_provider import TA_EXTRA_COLUMNS

    computed = calculate_indicators(candles)
    keys = TradingView.indicators.copy() + list(TA_EXTRA_COLUMNS) + ["ATR"]
    values = {k: computed.get(k) for k in keys}
    exchange, _, ticker = tv_symbol.partition(":")

    def run():
        return calculate(indicators=values, indicators_key=keys, screener=screener,
                         symbol=ticker, exchange=exchange, interval=timeframe)

    # calculate() needs the three Recommend.* ratings, which TradingView
    # computes server-side. Run it once to get the BUY/SELL/NEUTRAL votes of
    # the indicators we have, then rate from those votes like TradingView does.
    values.update({"Recommend.Other": 0.0, "Recommend.All": 0.0, "Recommend.MA": 0.0})
    first = run()
    other = _recommend(first.oscillators)
    ma = _recommend(first.moving_averages)
    values.update({"Recommend.Other": other, "Recommend.MA": ma, "Recommend.All": (other + ma) / 2})
    return run()


def fallback_analysis(screener: str, tv_symbol: str, timeframe: str):
    """``Analysis`` from Yahoo, Tiingo or Alpha Vantage bars (first with
    enough bars wins), or None when none has usable bars."""
    mapped = provider_symbols(screener, tv_symbol)
    if mapped is None or timeframe not in _YAHOO_INTERVALS:
        return None
    yahoo_symbol, av = mapped
    sources = [(f"yahoo:{yahoo_symbol}", lambda: _fetch_yahoo(yahoo_symbol, timeframe, us_session=screener == "america"))]
    if av and av["kind"] == "stock":
        sources.append((f"tiingo:{av['symbol']}", lambda: _fetch_tiingo(av["symbol"], timeframe)))
    if av:
        sources.append((f"alphavantage:{av['symbol']}", lambda: _fetch_alpha_vantage(av, timeframe)))
    # A fallback must never replace the TradingView error with its own, so
    # every failure here ends as None.
    for source, fetch in sources:
        try:
            candles = fetch()
        except Exception as e:  # noqa: BLE001 - any failure moves on to the next source
            _log(f"fallback {source} {timeframe} failed: {e!r}")
            continue
        if candles and len(candles) >= _MIN_BARS:
            break
    else:
        return None
    try:
        analysis = build_analysis(candles, screener, tv_symbol, timeframe)
    except Exception as e:  # noqa: BLE001
        _log(f"fallback analysis for {tv_symbol} {timeframe} failed: {e!r}")
        return None
    analysis.data_source = source
    _log(f"served {tv_symbol} {timeframe} from {source} ({len(candles)} bars)")
    return analysis


def fallback_analyses(screener: str, symbols: list[str], timeframe: str) -> dict[str, Any] | None:
    """``{symbol: Analysis or None}`` for up to MAX_SYMBOLS symbols, or None
    when there are too many symbols or no symbol could be served."""
    if not symbols or len(symbols) > MAX_SYMBOLS:
        return None
    out = {s: fallback_analysis(screener, s, timeframe) for s in symbols}
    return out if any(a is not None for a in out.values()) else None


def _source_of(item: Any) -> str | None:
    if isinstance(item, dict):  # a resilient_scan_columns row
        return item.get("data_source")
    return getattr(item, "data_source", None)


def source_fields(*items) -> dict[str, str]:
    """``data_source`` / ``data_note`` for a tool result when any of the
    analyses (or scan rows) came from this fallback; an empty dict otherwise."""
    sources = sorted({s for item in items if (s := _source_of(item))})
    if not sources:
        return {}
    joined = ", ".join(sources)
    return {"data_source": joined, "data_note": DATA_NOTE.format(source=joined)}
