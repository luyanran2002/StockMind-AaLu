"""Valuation tools (deterministic computation over financial + market data, Phase 2)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools import analysis
from app.tools.base import metric_json
from app.tools.providers import MarketDataProvider


class TickerInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")


class DcfInput(TickerInput):
    growth_rate: float = Field(default=0.08, description="Annual FCF growth rate for projection years")
    discount_rate: float = Field(default=0.10, description="Discount rate (WACC)")
    terminal_growth: float = Field(default=0.03, description="Perpetual terminal growth rate")
    years: int = Field(default=5, description="Number of projection years", ge=1, le=20)


def build_valuation_tools(provider: MarketDataProvider) -> list[BaseTool]:
    def calculate_pe(ticker: str) -> str:
        """Calculate price-to-earnings (P/E) ratio from price and trailing EPS."""
        price_result = provider.get_stock_price(ticker)
        stats = provider.get_key_stats(ticker)
        eps = stats.eps_ttm or provider.get_income_statement(ticker).eps_diluted
        value = analysis.compute_pe(price_result.price, eps)
        return metric_json(
            "pe",
            value,
            {"ticker": ticker.upper(), "price": price_result.price, "eps": eps},
            price_result.source,
        )

    def calculate_ev_ebitda(ticker: str) -> str:
        """Calculate EV/EBITDA from market cap, total debt, cash and EBITDA."""
        stats = provider.get_key_stats(ticker)
        balance = provider.get_balance_sheet(ticker)
        income = provider.get_income_statement(ticker)
        value = analysis.compute_ev_ebitda(
            stats.market_cap, balance.total_debt, balance.cash_and_equivalents, income.ebitda
        )
        return metric_json(
            "ev_ebitda",
            value,
            {
                "ticker": ticker.upper(),
                "market_cap": stats.market_cap,
                "total_debt": balance.total_debt,
                "cash": balance.cash_and_equivalents,
                "ebitda": income.ebitda,
            },
            stats.source,
        )

    def calculate_fcf_yield(ticker: str) -> str:
        """Calculate free cash flow yield (FCF / market cap)."""
        cashflow = provider.get_cash_flow(ticker)
        market_cap = provider.get_key_stats(ticker).market_cap
        value = analysis.compute_fcf_yield(cashflow.free_cash_flow, market_cap)
        return metric_json(
            "fcf_yield",
            value,
            {"ticker": ticker.upper(), "free_cash_flow": cashflow.free_cash_flow, "market_cap": market_cap},
            cashflow.source,
        )

    def calculate_dcf(
        ticker: str,
        growth_rate: float = 0.08,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.03,
        years: int = 5,
    ) -> str:
        """Calculate a simplified two-stage discounted cash flow (DCF) intrinsic value."""
        cashflow = provider.get_cash_flow(ticker)
        shares = provider.get_key_stats(ticker).shares_outstanding
        value = analysis.compute_dcf(
            cashflow.free_cash_flow,
            shares,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            years=years,
        )
        return metric_json(
            "dcf",
            value,
            {"ticker": ticker.upper(), "free_cash_flow": cashflow.free_cash_flow, "shares_outstanding": shares},
            cashflow.source,
            "Simplified DCF: assumptions are inputs, not facts; see value.assumptions.",
        )

    return [
        StructuredTool.from_function(
            func=calculate_pe,
            name="calculate_pe",
            description="Calculate the price-to-earnings (P/E) ratio of a stock.",
            args_schema=TickerInput,
        ),
        StructuredTool.from_function(
            func=calculate_ev_ebitda,
            name="calculate_ev_ebitda",
            description="Calculate the enterprise-value-to-EBITDA multiple of a stock.",
            args_schema=TickerInput,
        ),
        StructuredTool.from_function(
            func=calculate_fcf_yield,
            name="calculate_fcf_yield",
            description="Calculate the free cash flow yield (FCF / market cap) of a stock.",
            args_schema=TickerInput,
        ),
        StructuredTool.from_function(
            func=calculate_dcf,
            name="calculate_dcf",
            description="Calculate a simplified discounted cash flow (DCF) intrinsic value per share.",
            args_schema=DcfInput,
        ),
    ]
