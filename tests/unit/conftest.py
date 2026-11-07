"""Unit tests stay off the network, also on the Yahoo / Tiingo / Alpha Vantage
fallback path that a simulated TradingView outage now reaches."""
from __future__ import annotations

import pytest

from tradingview_mcp.core.services import ohlcv_fallback


@pytest.fixture(autouse=True)
def _no_fallback_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise ConnectionError("network is blocked in unit tests")

    monkeypatch.setattr(ohlcv_fallback, "_get_json", blocked)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    ohlcv_fallback._AV.reset()
    ohlcv_fallback._TIINGO.reset()
