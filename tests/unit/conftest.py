"""Unit tests stay off the network, also on the Yahoo / Alpha Vantage
fallback path that a simulated TradingView outage now reaches."""
from __future__ import annotations

import pytest

from tradingview_mcp.core.services import ohlcv_fallback


@pytest.fixture(autouse=True)
def _no_fallback_network(monkeypatch):
    def blocked(url, proxy_retry=False):
        raise ConnectionError("network is blocked in unit tests")

    monkeypatch.setattr(ohlcv_fallback, "_get_json", blocked)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with ohlcv_fallback._AV_LOCK:
        ohlcv_fallback._AV_CACHE.clear()
        ohlcv_fallback._AV_STATE.update(day=None, count=0, blocked=False)
