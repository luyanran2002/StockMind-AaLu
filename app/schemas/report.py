"""Structured output schema for the final research report.

The LLM produces free-form reasoning during the ReAct loop, but the final
deliverable is validated against this Pydantic model so downstream consumers
always receive a predictable shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StockResearchReport(BaseModel):
    """The canonical, structured output of a StockMind research run."""

    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")
    summary: str = Field(..., description="Concise executive summary of the research")
    market_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of price action, volume and market behaviour",
    )
    financial_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of income statement, balance sheet and cash flow",
    )
    technical_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of technical indicators",
    )
    news_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of recent news and catalysts",
    )
    valuation_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of valuation metrics",
    )
    risk_analysis: str = Field(
        default="Not covered in this phase.",
        description="Analysis of volatility, drawdown and other risks",
    )
    bull_case: list[str] = Field(default_factory=list, description="Arguments in favour")
    bear_case: list[str] = Field(default_factory=list, description="Arguments against")
    key_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Grounding, evidence-backed metrics"
    )
    uncertainty: list[str] = Field(
        default_factory=list, description="Explicitly stated uncertainties and limitations"
    )
    conclusion: str = Field(..., description="Final assessment")

