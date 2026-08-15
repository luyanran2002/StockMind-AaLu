"""Financial-statement tools (data retrieval, Phase 2)."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools.providers import MarketDataProvider


class TickerInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")


def build_financial_tools(provider: MarketDataProvider) -> list[BaseTool]:
    def get_income_statement(ticker: str) -> str:
        """Get the latest income statement (revenue, EBITDA, net income, EPS)."""
        return provider.get_income_statement(ticker).model_dump_json()

    def get_balance_sheet(ticker: str) -> str:
        """Get the latest balance sheet (assets, liabilities, debt, cash, equity)."""
        return provider.get_balance_sheet(ticker).model_dump_json()

    def get_cash_flow(ticker: str) -> str:
        """Get the latest cash flow statement (operating, investing, financing, FCF)."""
        return provider.get_cash_flow(ticker).model_dump_json()

    def get_eps(ticker: str) -> str:
        """Get trailing earnings per share (EPS)."""
        stats = provider.get_key_stats(ticker)
        income = provider.get_income_statement(ticker)
        eps = stats.eps_ttm or income.eps_diluted
        return json.dumps(
            {
                "ticker": ticker.upper(),
                "eps_ttm": eps,
                "source": stats.source,
                "timestamp": stats.timestamp.isoformat(),
                "note": stats.note,
            }
        )

    def get_revenue(ticker: str) -> str:
        """Get trailing twelve-month revenue."""
        income = provider.get_income_statement(ticker)
        return json.dumps(
            {
                "ticker": ticker.upper(),
                "revenue": income.revenue,
                "period": income.period,
                "source": income.source,
                "timestamp": income.timestamp.isoformat(),
                "note": income.note,
            }
        )

    descriptions = {
        "get_income_statement": "Get the latest income statement for a ticker (revenue, gross profit, operating income, EBITDA, net income, EPS).",
        "get_balance_sheet": "Get the latest balance sheet for a ticker (total assets, liabilities, debt, cash, equity).",
        "get_cash_flow": "Get the latest cash flow statement for a ticker (operating/investing/financing cash flow, free cash flow).",
        "get_eps": "Get trailing earnings per share (EPS) for a ticker.",
        "get_revenue": "Get trailing twelve-month revenue for a ticker.",
    }

    return [
        StructuredTool.from_function(func=get_income_statement, name="get_income_statement", description=descriptions["get_income_statement"], args_schema=TickerInput),
        StructuredTool.from_function(func=get_balance_sheet, name="get_balance_sheet", description=descriptions["get_balance_sheet"], args_schema=TickerInput),
        StructuredTool.from_function(func=get_cash_flow, name="get_cash_flow", description=descriptions["get_cash_flow"], args_schema=TickerInput),
        StructuredTool.from_function(func=get_eps, name="get_eps", description=descriptions["get_eps"], args_schema=TickerInput),
        StructuredTool.from_function(func=get_revenue, name="get_revenue", description=descriptions["get_revenue"], args_schema=TickerInput),
    ]
