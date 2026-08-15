import json

from app.tools import build_tools
from app.tools.providers import MockMarketDataProvider


def _tools():
    return {tool.name: tool for tool in build_tools(MockMarketDataProvider(seed=1))}


def _invoke(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


def test_financial_tools_return_provenance():
    tools = _tools()
    income = _invoke(tools["get_income_statement"], ticker="NVDA")
    assert income["ticker"] == "NVDA"
    assert income["revenue"] > 0
    assert income["source"] == "mock"
    assert income["timestamp"]

    balance = _invoke(tools["get_balance_sheet"], ticker="NVDA")
    assert balance["total_assets"] > 0
    assert balance["cash_and_equivalents"] > 0

    cashflow = _invoke(tools["get_cash_flow"], ticker="NVDA")
    assert cashflow["free_cash_flow"] > 0


def test_eps_and_revenue():
    tools = _tools()
    eps = _invoke(tools["get_eps"], ticker="NVDA")
    assert eps["eps_ttm"] > 0
    revenue = _invoke(tools["get_revenue"], ticker="NVDA")
    assert revenue["revenue"] > 0


def test_financials_are_internally_consistent():
    provider = MockMarketDataProvider(seed=1)
    income = provider.get_income_statement("NVDA")
    stats = provider.get_key_stats("NVDA")
    assert income.eps_diluted == stats.eps_ttm


def test_valuation_tools():
    tools = _tools()
    pe = _invoke(tools["calculate_pe"], ticker="NVDA")
    assert pe["metric"] == "pe"
    assert pe["value"] > 0

    ev_ebitda = _invoke(tools["calculate_ev_ebitda"], ticker="NVDA")
    assert ev_ebitda["value"] > 0

    fcf_yield = _invoke(tools["calculate_fcf_yield"], ticker="NVDA")
    assert 0 < fcf_yield["value"] < 1

    dcf = _invoke(
        tools["calculate_dcf"],
        ticker="NVDA",
        growth_rate=0.08,
        discount_rate=0.10,
        terminal_growth=0.03,
        years=5,
    )
    assert dcf["value"]["intrinsic_value_per_share"] > 0
    assert "assumptions" in dcf["value"]


def test_technical_tools():
    tools = _tools()
    ma = _invoke(tools["calculate_ma"], ticker="NVDA", period="3mo", window=20)
    assert ma["metric"] == "moving_average"
    assert ma["value"] > 0

    rsi = _invoke(tools["calculate_rsi"], ticker="NVDA", period="6mo", window=14)
    assert 0 <= rsi["value"] <= 100

    macd = _invoke(tools["calculate_macd"], ticker="NVDA", period="6mo")
    assert set(macd["value"]) == {"macd", "signal", "histogram"}

    vol = _invoke(tools["calculate_volatility"], ticker="NVDA", period="1y")
    assert vol["value"] >= 0

    dd = _invoke(tools["calculate_drawdown"], ticker="NVDA", period="1y")
    assert dd["value"]["max_drawdown"] >= 0


def test_search_market_news():
    tools = _tools()
    news = _invoke(tools["search_market_news"], query="rates", limit=5)
    assert news["ticker"] == "MARKET"
    assert len(news["items"]) == 5
    assert news["items"][0]["title"].startswith("[MOCK]")

