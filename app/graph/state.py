"""Explicit agent state.

All execution-relevant state lives here and is threaded through LangGraph as a
TypedDict. Nodes never communicate through globals; they return partial state
updates that LangGraph merges via the declared reducers.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.schemas.report import StockResearchReport


class StockAgentState(TypedDict, total=False):
    """The full state of a StockMind research run."""

    # -- user / task ---------------------------------------------------------
    user_query: str
    ticker: str | None

    # -- conversation + tool history -----------------------------------------
    # `add_messages` is LangGraph's reducer for accumulating chat history.
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    tool_call_signatures: list[str]

    # -- domain data buckets (populated as phases add more tools) ------------
    market_data: dict[str, Any]
    financial_data: dict[str, Any]
    news_data: list[dict[str, Any]]
    technical_data: dict[str, Any]
    valuation_data: dict[str, Any]
    risk_analysis: dict[str, Any]

    # -- output --------------------------------------------------------------
    final_report: str | None
    final_output: StockResearchReport | None
    errors: list[str]
    metadata: dict[str, Any]

    # -- control-flow / operational fields -----------------------------------
    iteration_count: int
    max_iterations: int
    language: str  # "en" | "zh"
    status: str  # "running" | "finished" | "max_iterations" | "error"


def create_initial_state(
    user_query: str,
    *,
    ticker: str | None = None,
    max_iterations: int = 8,
    language: str = "en",
    metadata: dict[str, Any] | None = None,
) -> StockAgentState:
    """Build a fresh state dict for a new research run."""

    return StockAgentState(
        user_query=user_query,
        ticker=ticker,
        messages=[HumanMessage(content=user_query)],
        tool_calls=[],
        observations=[],
        tool_call_signatures=[],
        market_data={},
        financial_data={},
        news_data=[],
        technical_data={},
        valuation_data={},
        risk_analysis={},
        final_report=None,
        final_output=None,
        errors=[],
        metadata=metadata or {},
        iteration_count=0,
        max_iterations=max_iterations,
        language=language,
        status="running",
    )
