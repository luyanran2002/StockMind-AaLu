from langchain_core.tools import BaseTool

from app.tools.financial import build_financial_tools
from app.tools.market import build_market_tools
from app.tools.news import build_news_tools
from app.tools.providers import (
    MarketDataProvider,
    MockMarketDataProvider,
    YFinanceMarketDataProvider,
    get_data_provider,
)
from app.tools.technical import build_technical_tools
from app.tools.valuation import build_valuation_tools


def build_tools(provider: MarketDataProvider) -> list[BaseTool]:
    """Build the full Phase 2 tool set for the given data provider."""
    return (
        build_market_tools(provider)
        + build_news_tools(provider)
        + build_financial_tools(provider)
        + build_technical_tools(provider)
        + build_valuation_tools(provider)
    )


__all__ = [
    "MarketDataProvider",
    "MockMarketDataProvider",
    "YFinanceMarketDataProvider",
    "get_data_provider",
    "build_tools",
]
