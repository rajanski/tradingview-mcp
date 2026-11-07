"""Yahoo / Tiingo / Alpha Vantage fallback for when the TradingView scanner keeps failing.

On the self-hosted server (2026-10-29 .. 2026-11-07) every request that got an
empty body also failed its two retries, 50 of 50, so the tools returned errors.
The single-symbol tools now get bars from Yahoo (then Tiingo for US stocks,
then Alpha Vantage for 1D/1W/1M), calculate the indicators locally and say so
in ``data_source``.
"""
from __future__ import annotations

import json
import math
import urllib.error
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest
import tradingview_ta
from tradingview_screener import query as tv_query

from tradingview_mcp.core.services import ohlcv_fallback as fb
from tradingview_mcp.core.services import screener_provider as sp
from tradingview_mcp.core.services import screener_service


@pytest.fixture(autouse=True)
def _fast_resilience(monkeypatch):
    with sp._SCREENER_CACHE_LOCK:
        sp._SCREENER_CACHE.clear()
    with sp._TA_FAILURE_LOCK:
        sp._LAST_TA_FAILURE_TS = 0.0
    monkeypatch.setenv("TRADINGVIEW_MCP_RETRY_DELAYS", "0.01")
    monkeypatch.setenv("TRADINGVIEW_MCP_RETRY_JITTER", "0")
    monkeypatch.setenv("TRADINGVIEW_MCP_FAILURE_COOLDOWN_S", "0")
    monkeypatch.setenv("TRADINGVIEW_MCP_MIN_INTERVAL_S", "0")
    monkeypatch.setattr(screener_service, "_TA_AVAILABLE", True)
    yield
    with sp._SCREENER_CACHE_LOCK:
        sp._SCREENER_CACHE.clear()


# ── fakes ──────────────────────────────────────────────────────────────────

def _price(i: int) -> float:
    return 100 + 10 * math.sin(i / 7) + i * 0.1


def _yahoo_payload(n: int = 300, step: int = 86_400, start: int = 1_699_920_000,
                   last_tick: bool = False) -> dict:
    ts = [start + i * step for i in range(n)]
    close = [_price(i) for i in range(n)]
    quote = {
        "open": [c - 0.5 for c in close],
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [1_000 + i for i in range(n)],
    }
    if last_tick:  # Yahoo's last-price tick after the newest intraday bar
        ts.append(ts[-1] + step // 3)
        for k in ("open", "high", "low", "close"):
            quote[k].append(close[-1])
        quote["volume"].append(0)
    return {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [quote]}}]}}


def _av_payload(n: int = 100) -> dict:
    series = {}
    for i in range(n):
        p = _price(i)
        series[f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"] = {
            "1. open": str(p - 0.5), "2. high": str(p + 1), "3. low": str(p - 1),
            "4. close": str(p), "5. volume": str(1_000 + i),
        }
    newest_first = dict(sorted(series.items(), reverse=True))
    return {"Meta Data": {}, "Time Series (Daily)": newest_first}


def _tiingo_rows(n: int = 300, step: int = 86_400, start: int = 1_699_920_000) -> list[dict]:
    rows = []
    for i in range(n):
        p = _price(i)
        date = datetime.fromtimestamp(start + i * step, timezone.utc)
        rows.append({"date": date.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "open": p - 0.5,
                     "high": p + 1, "low": p - 1, "close": p, "volume": 1_000 + i,
                     "splitFactor": 1.0})
    return rows


class _Net:
    """Stands in for ``ohlcv_fallback._get_json``; records every URL and header."""

    def __init__(self, yahoo: Callable[[str], Any] | None = None,
                 av: Callable[[str], Any] | None = None,
                 tiingo: Callable[[str], Any] | None = None):
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self._yahoo, self._av, self._tiingo = yahoo, av, tiingo

    def __call__(self, url: str, proxy_retry: bool = False, headers: dict | None = None) -> Any:
        self.urls.append(url)
        self.headers.append(headers or {})
        if "alphavantage" in url:
            handler = self._av
        elif "tiingo" in url:
            handler = self._tiingo
        else:
            handler = self._yahoo
        if handler is None:
            raise ConnectionError("down")
        return handler(url)

    def count(self, host: str) -> int:
        return sum(host in u for u in self.urls)


def _tradingview_down(monkeypatch) -> None:
    def empty_body(*args, **kwargs):
        raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(tradingview_ta, "get_multiple_analysis", empty_body)
    monkeypatch.setattr(tv_query.Query, "get_scanner_data_raw", empty_body)


# ── symbol mapping ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("screener, symbol, expected", [
    ("crypto", "KUCOIN:BTCUSDT", ("BTC-USD", {"kind": "crypto", "symbol": "BTC", "market": "USD"})),
    ("crypto", "BINANCE:ETHFDUSD", ("ETH-USD", {"kind": "crypto", "symbol": "ETH", "market": "USD"})),
    ("crypto", "KRAKEN:BTCEUR", ("BTC-EUR", {"kind": "crypto", "symbol": "BTC", "market": "EUR"})),
    ("america", "NYSE:BRK.B", ("BRK-B", {"kind": "stock", "symbol": "BRK.B"})),
    ("turkey", "BIST:THYAO", ("THYAO.IS", None)),
    ("egypt", "EGX:COMI", ("COMI.CA", None)),
])
def test_known_symbols_map_to_yahoo_and_alpha_vantage(screener, symbol, expected):
    assert fb.provider_symbols(screener, symbol) == expected


@pytest.mark.parametrize("screener, symbol", [
    ("uk", "LSE:BP"),                 # plain "BP" on Yahoo is the US listing
    ("crypto", "BINANCE:BTCUSDT.P"),  # perpetual future, not the spot pair
    ("cfd", "TVC:GOLD"),
    ("america", "AAPL"),              # no exchange prefix
])
def test_unknown_symbols_get_no_fallback(screener, symbol):
    assert fb.provider_symbols(screener, symbol) is None


# ── indicators ─────────────────────────────────────────────────────────────

def _candles(n: int = 300) -> list[dict]:
    q = _yahoo_payload(n)["chart"]["result"][0]["indicators"]["quote"][0]
    return [{k: q[k][i] for k in q} for i in range(n)]


def test_indicators_follow_their_definitions():
    candles = _candles()
    closes = [c["close"] for c in candles]
    vols = [c["volume"] for c in candles]
    ind = fb.calculate_indicators(candles)

    assert ind["close"] == closes[-1]
    assert ind["SMA20"] == pytest.approx(sum(closes[-20:]) / 20)
    assert ind["change"] == pytest.approx((closes[-1] / closes[-2] - 1) * 100)
    assert ind["Mom"] == pytest.approx(closes[-1] - closes[-11])
    assert ind[fb.AVERAGE_VOLUME] == pytest.approx(sum(vols[-10:]) / 10)
    assert ind[fb.RELATIVE_VOLUME] == pytest.approx(vols[-1] / (sum(vols[-11:-1]) / 10))
    assert ind["BB.lower"] < ind["SMA20"] < ind["BB.upper"]
    assert -100 <= ind["W.R"] <= 0
    assert 0 <= ind["Stoch.K"] <= 100 and 0 <= ind["RSI"] <= 100
    assert 0 <= ind["ADX"] <= 100 and ind["ATR"] > 0
    assert ind["Rec.VWMA"] == (1 if closes[-1] > ind["VWMA"] else -1)


def test_analysis_has_ratings_and_all_tradingview_columns():
    a = fb.build_analysis(_candles(), "america", "NASDAQ:AAPL", "1D")
    assert a.summary["RECOMMENDATION"] in {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}
    assert set(tradingview_ta.main.TradingView.indicators) <= set(a.indicators)
    assert -1 <= a.indicators["Recommend.All"] <= 1
    assert (a.exchange, a.symbol, a.interval) == ("NASDAQ", "AAPL", "1D")


def test_too_few_bars_give_no_analysis(monkeypatch):
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload(n=20)))
    assert fb.fallback_analysis("america", "NASDAQ:AAPL", "1D") is None


# ── Yahoo bars ─────────────────────────────────────────────────────────────

def test_yahoo_last_price_tick_is_not_a_bar(monkeypatch):
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload(n=50, step=900, last_tick=True)))
    candles = fb._fetch_yahoo("BTC-USD", "15m")
    assert len(candles) == 50
    assert candles[-1]["volume"] == 1_049


def test_4h_bars_are_built_from_1h_bars_in_utc_blocks(monkeypatch):
    start = 1_699_920_000  # 2023-11-14 00:00 UTC, a 4h block border
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload(n=8, step=3600, start=start)))
    bars = fb._fetch_yahoo("BTC-USD", "4h")
    assert len(bars) == 2
    assert bars[0]["open"] == _price(0) - 0.5
    assert bars[0]["close"] == _price(3)
    assert bars[0]["high"] == max(_price(i) for i in range(4)) + 1
    assert bars[0]["volume"] == 1_000 + 1_001 + 1_002 + 1_003


# ── Tiingo ─────────────────────────────────────────────────────────────────

def test_tiingo_serves_stocks_before_alpha_vantage(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    net = _Net(tiingo=lambda u: _tiingo_rows(), av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)

    a = fb.fallback_analysis("america", "NYSE:BRK.B", "1D")
    assert a.data_source == "tiingo:BRK.B"
    assert a.indicators["close"] == pytest.approx(_price(299))
    assert "/tiingo/daily/brk-b/prices?startDate=" in net.urls[-1]
    assert net.headers[-1] == {"Authorization": "Token TIINGOKEY7"}
    assert not any("TIINGOKEY7" in u for u in net.urls)
    assert net.count("alphavantage") == 0

    fb.fallback_analysis("america", "NYSE:BRK.B", "1D")  # answer is reused
    assert net.count("tiingo") == 1


def test_tiingo_intraday_asks_iex_for_volume(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    net = _Net(tiingo=lambda u: _tiingo_rows(step=900))
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NASDAQ:AAPL", "15m").data_source == "tiingo:AAPL"
    assert "/iex/aapl/prices?" in net.urls[-1]
    assert "resampleFreq=15min" in net.urls[-1]
    assert "columns=open%2Chigh%2Clow%2Cclose%2Cvolume" in net.urls[-1]


def test_tiingo_prices_are_split_adjusted(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    rows = _tiingo_rows(n=4)
    for r in rows[:2]:  # as traded before a 10:1 split on rows[2]
        for k in ("open", "high", "low", "close"):
            r[k] *= 10
    rows[2]["splitFactor"] = 10.0
    monkeypatch.setattr(fb, "_get_json", _Net(tiingo=lambda u: rows))

    bars = fb._fetch_tiingo("NVDA", "1D")
    assert [b["close"] for b in bars] == pytest.approx([_price(i) for i in range(4)])
    assert [b["volume"] for b in bars] == [10_000, 10_010, 1_002, 1_003]


@pytest.mark.parametrize("open_utc", [
    datetime(2026, 11, 6, 14, 30, tzinfo=timezone.utc),  # 9:30 EST
    datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),   # 9:30 EDT
])
@pytest.mark.parametrize("timeframe, rows_per_bar", [
    ("1h", [4, 4, 4, 4, 4, 4, 2]),  # 9:30-10:30 ... 15:30-16:00
    ("4h", [16, 10]),               # 9:30-13:30, 13:30-16:00
])
def test_tiingo_1h_and_4h_bars_start_at_the_session_open(monkeypatch, open_utc, timeframe, rows_per_bar):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    net = _Net(tiingo=lambda u: _tiingo_rows(n=26, step=900, start=int(open_utc.timestamp())))
    monkeypatch.setattr(fb, "_get_json", net)

    bars = fb._fetch_tiingo("AAPL", timeframe)
    assert "resampleFreq=15min" in net.urls[-1]
    assert len(bars) == len(rows_per_bar)
    first = 0
    for bar, n in zip(bars, rows_per_bar):
        assert bar["open"] == _price(first) - 0.5
        assert bar["close"] == _price(first + n - 1)
        assert bar["volume"] == sum(1_000 + i for i in range(first, first + n))
        first += n


def test_tiingo_weekly_bars_are_grouped_from_daily_rows(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    monday = 1_699_833_600  # 2023-11-13 00:00 UTC
    monkeypatch.setattr(fb, "_get_json", _Net(tiingo=lambda u: _tiingo_rows(n=10, start=monday)))
    bars = fb._fetch_tiingo("AAPL", "1W")
    assert len(bars) == 2  # Mon 13 .. Sun 19, Mon 20 .. Wed 22
    assert bars[0]["open"] == _price(0) - 0.5
    assert bars[0]["close"] == _price(6)
    assert bars[0]["volume"] == sum(1_000 + i for i in range(7))


def test_tiingo_hourly_limit_is_kept(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    monkeypatch.setenv("TIINGO_HOURLY_LIMIT", "1")
    net = _Net(tiingo=lambda u: _tiingo_rows())
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is not None
    assert fb.fallback_analysis("america", "NASDAQ:MSFT", "1D") is None
    assert net.count("tiingo") == 1


def test_tiingo_rate_limit_status_stops_requests_for_an_hour(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")

    def limited(url):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    net = _Net(tiingo=limited)
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is None
    assert fb.fallback_analysis("america", "NASDAQ:MSFT", "1D") is None
    assert net.count("tiingo") == 1


def test_tiingo_failure_moves_on_to_alpha_vantage(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    net = _Net(tiingo=lambda u: {"detail": "Error: Ticker 'IBM' not found"},
               av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D").data_source == "alphavantage:IBM"
    assert net.count("tiingo") == 1


def test_tiingo_is_not_asked_for_crypto(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "TIINGOKEY7")
    net = _Net(tiingo=lambda u: _tiingo_rows())
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("crypto", "KUCOIN:BTCUSDT", "1D") is None
    assert net.count("tiingo") == 0


# ── Alpha Vantage ──────────────────────────────────────────────────────────

def test_alpha_vantage_serves_daily_when_yahoo_fails(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    net = _Net(av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)

    a = fb.fallback_analysis("america", "NYSE:IBM", "1D")
    assert a.data_source == "alphavantage:IBM"
    assert a.indicators["close"] == pytest.approx(_price(99))  # oldest-first order
    assert "function=TIME_SERIES_DAILY" in net.urls[-1]

    # No intraday data on the free plan: no request is spent on 15m.
    assert fb.fallback_analysis("america", "NYSE:IBM", "15m") is None
    assert net.count("alphavantage") == 1


def test_alpha_vantage_answers_are_reused(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    net = _Net(av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    fb.fallback_analysis("america", "NYSE:IBM", "1D")
    fb.fallback_analysis("america", "NYSE:IBM", "1D")
    assert net.count("alphavantage") == 1


def test_alpha_vantage_daily_limit_is_kept(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    monkeypatch.setenv("ALPHAVANTAGE_DAILY_LIMIT", "1")
    net = _Net(av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is not None
    assert fb.fallback_analysis("america", "NASDAQ:MSFT", "1D") is None
    assert net.count("alphavantage") == 1


def test_alpha_vantage_limit_message_stops_requests_for_the_day(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "KEY123")
    net = _Net(av=lambda u: {"Information": "standard API rate limit is 25 requests per day"})
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is None
    assert fb.fallback_analysis("america", "NASDAQ:MSFT", "1D") is None
    assert net.count("alphavantage") == 1


def test_alpha_vantage_key_never_reaches_the_log(monkeypatch, capsys):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "SECRETKEY42")

    def fail_with_url(url):
        raise ConnectionError(f"could not open {url}")

    monkeypatch.setattr(fb, "_get_json", _Net(av=fail_with_url))
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is None
    assert "SECRETKEY42" not in capsys.readouterr().err


def test_no_key_no_alpha_vantage(monkeypatch):
    net = _Net(av=lambda u: _av_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    assert fb.fallback_analysis("america", "NYSE:IBM", "1D") is None
    assert net.count("alphavantage") == 0


# ── the TradingView wrappers ───────────────────────────────────────────────

def test_ta_outage_is_served_from_yahoo(monkeypatch):
    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    result = sp.resilient_get_multiple_analysis("america", "1D", ["NASDAQ:AAPL"])
    assert result["NASDAQ:AAPL"].data_source == "yahoo:AAPL"


def test_ta_outage_on_a_scan_still_raises(monkeypatch):
    _tradingview_down(monkeypatch)
    net = _Net(yahoo=lambda u: _yahoo_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    symbols = [f"NASDAQ:S{i}" for i in range(fb.MAX_SYMBOLS + 1)]
    with pytest.raises(RuntimeError, match="transient errors"):
        sp.resilient_get_multiple_analysis("america", "1D", symbols)
    assert net.urls == []


def test_ta_outage_without_fallback_data_keeps_the_tradingview_error(monkeypatch):
    _tradingview_down(monkeypatch)  # and Yahoo is blocked by conftest
    with pytest.raises(RuntimeError, match="transient errors"):
        sp.resilient_get_multiple_analysis("america", "1D", ["NASDAQ:AAPL"])


def test_multi_timeframe_outage_is_served_per_timeframe(monkeypatch):
    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    out = sp.resilient_multi_timeframe_analysis("crypto", "KUCOIN:BTCUSDT", ["1W", "1D", "4h"])
    assert set(out) == {"1W", "1D", "4h"}
    assert all(a.data_source == "yahoo:BTC-USD" for a in out.values())


def test_scan_columns_outage_is_served_for_one_symbol(monkeypatch):
    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    out = sp.resilient_scan_columns("america", "1D", ["NASDAQ:AAPL"], ["close", "RSI"])
    assert set(out["NASDAQ:AAPL"]) == {"close", "RSI", "data_source"}


def test_scan_columns_real_error_is_not_served(monkeypatch):
    def bad_request(*args, **kwargs):
        raise RuntimeError("Unknown field 'XYZ'")

    monkeypatch.setattr(tv_query.Query, "get_scanner_data_raw", bad_request)
    net = _Net(yahoo=lambda u: _yahoo_payload())
    monkeypatch.setattr(fb, "_get_json", net)
    with pytest.raises(RuntimeError, match="Unknown field"):
        sp.resilient_scan_columns("america", "1D", ["NASDAQ:AAPL"], ["XYZ"])
    assert net.urls == []


# ── the tools, end to end ──────────────────────────────────────────────────

def _assert_marked(result: dict, source: str) -> None:
    assert "error" not in result, result
    assert result["data_source"] == source
    assert source in result["data_note"]


def test_coin_analysis_runs_on_fallback_data(monkeypatch):
    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    result = screener_service.analyze_coin("AAPL", "NASDAQ", "1D")
    _assert_marked(result, "yahoo:AAPL")
    assert result["price_data"]["current_price"] == pytest.approx(_price(299))


def test_multi_timeframe_analysis_runs_on_fallback_data(monkeypatch):
    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    _assert_marked(screener_service.run_multi_timeframe_analysis("KUCOIN:BTCUSDT", "KUCOIN"), "yahoo:BTC-USD")


def test_volume_confirmation_runs_on_fallback_data(monkeypatch):
    from tradingview_mcp.core.services.scanner_service import (
        volume_confirmation_analyze,
    )

    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    _assert_marked(volume_confirmation_analyze("AAPL", "NASDAQ", "1D"), "yahoo:AAPL")


def test_multi_agent_runs_on_fallback_data(monkeypatch):
    from tradingview_mcp.core.services.multi_agent_service import (
        run_multi_agent_analysis,
    )

    _tradingview_down(monkeypatch)
    monkeypatch.setattr(fb, "_get_json", _Net(yahoo=lambda u: _yahoo_payload()))
    _assert_marked(run_multi_agent_analysis("NASDAQ:AAPL", "nasdaq", "1D"), "yahoo:AAPL")
