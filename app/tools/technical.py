"""Technical-analysis tools (deterministic computation over market data, Phase 2)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools import analysis
from app.tools.base import metric_json
from app.tools.providers import HistoricalPricesResult, MarketDataProvider


class TickerPeriodInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    period: str = Field(
        default="1y",
        description="Lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y",
    )


class TickerPeriodWindowInput(TickerPeriodInput):
    window: int = Field(default=14, description="Rolling window size in bars")


def _closes(result: HistoricalPricesResult) -> list[float]:
    return [bar.close for bar in result.bars]


def build_technical_tools(provider: MarketDataProvider) -> list[BaseTool]:
    def calculate_ma(ticker: str, period: str = "3mo", window: int = 20) -> str:
        """Calculate the latest simple moving average of closing prices."""
        history = provider.get_historical_prices(ticker, period, "1d")
        value = analysis.compute_ma(_closes(history), window)
        return metric_json(
            "moving_average",
            value,
            {"ticker": ticker.upper(), "period": period, "window": window},
            history.source,
            history.note,
        )

    def calculate_rsi(ticker: str, period: str = "6mo", window: int = 14) -> str:
        """Calculate the latest Relative Strength Index (RSI)."""
        history = provider.get_historical_prices(ticker, period, "1d")
        value = analysis.compute_rsi(_closes(history), window)
        return metric_json(
            "rsi",
            value,
            {"ticker": ticker.upper(), "period": period, "window": window},
            history.source,
            history.note,
        )

    def calculate_macd(ticker: str, period: str = "6mo") -> str:
        """Calculate the latest MACD line, signal line and histogram."""
        history = provider.get_historical_prices(ticker, period, "1d")
        value = analysis.compute_macd(_closes(history))
        return metric_json(
            "macd",
            value,
            {"ticker": ticker.upper(), "period": period},
            history.source,
            history.note,
        )

    def calculate_volatility(ticker: str, period: str = "1y") -> str:
        """Calculate annualized price volatility from daily returns."""
        history = provider.get_historical_prices(ticker, period, "1d")
        value = analysis.compute_annualized_volatility(_closes(history))
        return metric_json(
            "annualized_volatility",
            value,
            {"ticker": ticker.upper(), "period": period},
            history.source,
            history.note,
        )

    def calculate_drawdown(ticker: str, period: str = "1y") -> str:
        """Calculate max and current drawdown from peak price."""
        history = provider.get_historical_prices(ticker, period, "1d")
        value = analysis.compute_drawdown(_closes(history))
        return metric_json(
            "drawdown",
            value,
            {"ticker": ticker.upper(), "period": period},
            history.source,
            history.note,
        )

    return [
        StructuredTool.from_function(
            func=calculate_ma,
            name="calculate_ma",
            description="Calculate the latest simple moving average of a stock's closing prices.",
            args_schema=TickerPeriodWindowInput,
        ),
        StructuredTool.from_function(
            func=calculate_rsi,
            name="calculate_rsi",
            description="Calculate the latest Relative Strength Index (RSI, 0-100) of a stock.",
            args_schema=TickerPeriodWindowInput,
        ),
        StructuredTool.from_function(
            func=calculate_macd,
            name="calculate_macd",
            description="Calculate the latest MACD line, signal line and histogram of a stock.",
            args_schema=TickerPeriodInput,
        ),
        StructuredTool.from_function(
            func=calculate_volatility,
            name="calculate_volatility",
            description="Calculate annualized price volatility of a stock from daily returns.",
            args_schema=TickerPeriodInput,
        ),
        StructuredTool.from_function(
            func=calculate_drawdown,
            name="calculate_drawdown",
            description="Calculate max and current drawdown of a stock from its price history.",
            args_schema=TickerPeriodInput,
        ),
    ]
