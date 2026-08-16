"""Offline demo model.

``ScriptedReActDemoModel`` emulates the LLM side of a ReAct loop with a fixed
sequence of tool calls. Its final message is a marker that tells the graph to
assemble a grounded report from the *actual* tool observations (see
``app/reporting/assembler.py``), so the offline demo still quotes real numbers
and sources instead of canned prose.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


GROUNDED_REPORT_MARKER = "__ASSEMBLE_GROUNDED_REPORT__"


class ScriptedReActDemoModel(BaseChatModel):
    """Returns a fixed sequence of messages regardless of the input prompt.

    This is a test/demo double — never use it to reason about real stocks.
    """

    messages: Iterator[BaseMessage]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            message = next(self.messages)
        except StopIteration as exc:
            raise RuntimeError("ScriptedReActDemoModel ran out of scripted messages.") from exc
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "scripted-react-demo"


def build_demo_model(ticker: str = "NVDA", language: str = "en") -> ScriptedReActDemoModel:
    """A demo model scripted to research ``ticker`` through the Phase 2 tools."""
    symbol = ticker.upper()
    sequence = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_historical_prices",
                    "args": {"ticker": symbol, "period": "6mo", "interval": "1d"},
                    "id": f"demo-{symbol}-1a",
                },
                {
                    "name": "search_company_news",
                    "args": {"ticker": symbol, "query": "recent", "limit": 5},
                    "id": f"demo-{symbol}-1b",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_stock_price", "args": {"ticker": symbol}, "id": f"demo-{symbol}-2a"},
                {"name": "get_income_statement", "args": {"ticker": symbol}, "id": f"demo-{symbol}-2b"},
                {"name": "get_cash_flow", "args": {"ticker": symbol}, "id": f"demo-{symbol}-2c"},
                {"name": "get_balance_sheet", "args": {"ticker": symbol}, "id": f"demo-{symbol}-2d"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculate_pe", "args": {"ticker": symbol}, "id": f"demo-{symbol}-3a"},
                {"name": "calculate_ev_ebitda", "args": {"ticker": symbol}, "id": f"demo-{symbol}-3b"},
                {"name": "calculate_fcf_yield", "args": {"ticker": symbol}, "id": f"demo-{symbol}-3c"},
                {
                    "name": "calculate_dcf",
                    "args": {"ticker": symbol, "growth_rate": 0.08, "discount_rate": 0.10, "terminal_growth": 0.03, "years": 5},
                    "id": f"demo-{symbol}-3d",
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculate_ma", "args": {"ticker": symbol, "period": "3mo", "window": 20}, "id": f"demo-{symbol}-4a"},
                {"name": "calculate_rsi", "args": {"ticker": symbol, "period": "6mo", "window": 14}, "id": f"demo-{symbol}-4b"},
                {"name": "calculate_macd", "args": {"ticker": symbol, "period": "6mo"}, "id": f"demo-{symbol}-4c"},
                {"name": "calculate_volatility", "args": {"ticker": symbol, "period": "1y"}, "id": f"demo-{symbol}-4d"},
                {"name": "calculate_drawdown", "args": {"ticker": symbol, "period": "1y"}, "id": f"demo-{symbol}-4e"},
            ],
        ),
        AIMessage(content=GROUNDED_REPORT_MARKER),
    ]
    return ScriptedReActDemoModel(messages=iter(sequence))


def build_nvda_demo_model(language: str = "en") -> ScriptedReActDemoModel:
    """Backward-compatible alias for the NVDA demo."""
    return build_demo_model("NVDA", language)
