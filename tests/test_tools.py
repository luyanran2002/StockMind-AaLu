import json

from app.tools import build_tools
from app.tools.providers import MockMarketDataProvider


def _tool_map():
    provider = MockMarketDataProvider(seed=1)
    return {tool.name: tool for tool in build_tools(provider)}


def test_get_stock_price():
    tool = _tool_map()["get_stock_price"]
    payload = json.loads(tool.invoke({"ticker": "NVDA"}))
    assert payload["ticker"] == "NVDA"
    assert payload["price"] > 0
    assert payload["source"] == "mock"
    assert payload["timestamp"]
    assert "Simulated" in payload["note"]


def test_get_historical_prices():
    tool = _tool_map()["get_historical_prices"]
    payload = json.loads(tool.invoke({"ticker": "NVDA", "period": "1mo", "interval": "1d"}))
    assert payload["ticker"] == "NVDA"
    assert len(payload["bars"]) > 0
    assert {"date", "open", "high", "low", "close", "volume"} <= set(payload["bars"][0])


def test_search_company_news():
    tool = _tool_map()["search_company_news"]
    payload = json.loads(tool.invoke({"ticker": "NVDA", "query": "recent", "limit": 5}))
    assert payload["ticker"] == "NVDA"
    assert len(payload["items"]) == 5
    assert payload["items"][0]["title"].startswith("[MOCK]")


def test_mock_provider_is_deterministic():
    p1 = MockMarketDataProvider(seed=7)
    p2 = MockMarketDataProvider(seed=7)
    assert p1.get_stock_price("NVDA").price == p2.get_stock_price("NVDA").price

