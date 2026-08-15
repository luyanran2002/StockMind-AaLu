"""News tools for the ReAct agent (Phase 1)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools.providers import MarketDataProvider


class SearchCompanyNewsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    query: str = Field(default="", description="Optional keyword to focus the search")
    limit: int = Field(default=5, description="Maximum number of news items to return", ge=1, le=20)


class SearchMarketNewsInput(BaseModel):
    query: str = Field(default="", description="Optional keyword to focus the search")
    limit: int = Field(default=5, description="Maximum number of news items to return", ge=1, le=20)


def build_news_tools(provider: MarketDataProvider) -> list[BaseTool]:
    """Build the news tools bound to ``provider``."""

    def search_company_news(ticker: str, query: str = "", limit: int = 5) -> str:
        """Search for recent company and market news for a ticker."""
        return provider.search_company_news(ticker, query, limit).model_dump_json()

    def search_market_news(query: str = "", limit: int = 5) -> str:
        """Search for recent market-wide news (indices, macro, sectors)."""
        return provider.search_market_news(query, limit).model_dump_json()

    return [
        StructuredTool.from_function(
            func=search_company_news,
            name="search_company_news",
            description="Search recent company and market news for a stock ticker.",
            args_schema=SearchCompanyNewsInput,
        ),
        StructuredTool.from_function(
            func=search_market_news,
            name="search_market_news",
            description="Search recent market-wide news (indices, macro, sectors) not tied to one ticker.",
            args_schema=SearchMarketNewsInput,
        ),
    ]
