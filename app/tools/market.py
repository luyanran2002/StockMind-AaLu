"""Market-data tools for the ReAct agent (Phase 1)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools.providers import MarketDataProvider


class GetStockPriceInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")


class GetHistoricalPricesInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    period: str = Field(
        default="1mo",
        description="Lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y",
    )
    interval: str = Field(
        default="1d",
        description="Bar interval: 1m, 5m, 15m, 1d, 1wk",
    )


def build_market_tools(provider: MarketDataProvider) -> list[BaseTool]:
    """Build the market-data tools bound to ``provider``."""

    def get_stock_price(ticker: str) -> str:
        """Fetch the latest price for a ticker, with source and timestamp."""
        return provider.get_stock_price(ticker).model_dump_json()

    def get_historical_prices(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
        """Fetch recent historical OHLCV bars for a ticker."""
        return provider.get_historical_prices(ticker, period, interval).model_dump_json()

    return [
        StructuredTool.from_function(
            func=get_stock_price,
            name="get_stock_price",
            description="Get the latest available price for a stock ticker.",
            args_schema=GetStockPriceInput,
        ),
        StructuredTool.from_function(
            func=get_historical_prices,
            name="get_historical_prices",
            description="Get recent historical price bars (open/high/low/close/volume) for a stock ticker.",
            args_schema=GetHistoricalPricesInput,
        ),
    ]

