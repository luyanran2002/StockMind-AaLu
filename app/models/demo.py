"""Offline demo model.

``ScriptedReActDemoModel`` emulates the LLM side of a ReAct loop with a fixed
sequence of messages. It exists only so the example scripts can run end-to-end
with no API key and no network access, while still exercising the real
tool-execution and state-management machinery for any ticker and language.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.schemas.report import StockResearchReport


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


def _demo_report(ticker: str, language: str) -> StockResearchReport:
    symbol = ticker.upper()
    if language == "zh":
        return StockResearchReport(
            ticker=symbol,
            summary=(
                f"基于（模拟的）价格、历史、新闻、财务、技术与估值数据，{symbol} 的近期走势"
                "主要受消息面催化驱动，而非单一基本面冲击。本次为使用明确标注为模拟数据的研究流程演示。"
            ),
            market_analysis=(
                "模拟价格显示近一个月呈震荡、偏弱走势，下跌日成交量有所放大，当前价格接近近期模拟区间的下沿。"
            ),
            financial_analysis=(
                "模拟财务数据显示营收与净利润为正，EBITDA 为正、自由现金流为正，资产负债表处于模拟的正常区间。"
            ),
            technical_analysis=(
                "模拟技术指标显示 RSI 处于中性区间，波动率上升，回撤幅度处于历史模拟范围的较高位置。"
            ),
            news_analysis=(
                "模拟新闻呈现多空交织：产品路线图带来乐观情绪，但监管审查与宏观担忧对半导体板块构成压力。"
            ),
            valuation_analysis=(
                "模拟估值指标（P/E、EV/EBITDA、FCF 收益率、DCF）显示估值处于中性水平；DCF 基于输入假设计算。"
            ),
            risk_analysis=(
                "主要风险来自波动率与回撤上升，以及监管与宏观层面的不确定性。所有数字均为模拟数据。"
            ),
            bull_case=[
                "产品路线图与资本开支计划支撑长期叙事。",
                "近期回调使价格回到模拟区间的相对低位。",
            ],
            bear_case=[
                "监管审查带来短期不确定性。",
                "宏观担忧拖累整个半导体板块。",
            ],
            key_metrics={
                "data_source": "mock (simulated)",
                "note": "所有数字均为模拟数据，不得视为真实市场数据。",
            },
            uncertainty=[
                "本次运行的所有数据均为模拟（mock provider），不构成任何真实结论。",
                "估值与技术指标基于模拟数据计算，仅供流程演示。",
            ],
            conclusion=(
                f"演示表明 Agent 能够自主收集 {symbol} 的价格、财务、新闻、技术与估值数据并生成结构化报告。"
                "真实分析请将数据源切换为 yfinance 并提供 LLM API key。"
            ),
        )

    return StockResearchReport(
        ticker=symbol,
        summary=(
            f"Based on simulated price, financial, news, technical and valuation data, {symbol}'s "
            "recent move appears driven by headline catalysts rather than a single fundamental shock. "
            "This is a demonstration of the research workflow using clearly-labelled simulated data."
        ),
        market_analysis=(
            "Simulated price bars show choppy, sideways-to-lower action over the last month, with "
            "elevated volume on down days. The current price sits near the lower end of the recent "
            "simulated range."
        ),
        financial_analysis=(
            "Simulated financials show positive revenue and net income, positive EBITDA and positive "
            "free cash flow, with a balance sheet within a normal simulated range."
        ),
        technical_analysis=(
            "Simulated technicals show an RSI in the neutral zone, rising volatility, and drawdown "
            "near the higher end of the simulated historical range."
        ),
        news_analysis=(
            "Simulated headlines point to mixed catalysts: product-roadmap optimism offset by "
            "regulatory-review uncertainty and broader macro concerns for semiconductor names."
        ),
        valuation_analysis=(
            "Simulated valuation metrics (P/E, EV/EBITDA, FCF yield, DCF) place the stock in a "
            "neutral range; the DCF is computed from input assumptions."
        ),
        risk_analysis=(
            "Key risks are elevated volatility and drawdown plus regulatory and macro uncertainty. "
            "All figures are simulated."
        ),
        bull_case=[
            "Strong product roadmap and capex plans support a constructive long-term narrative.",
            "Recent pullback brings the price to the lower end of its simulated range.",
        ],
        bear_case=[
            "Regulatory review adds near-term uncertainty.",
            "Macro concerns weigh on the broader semiconductor complex.",
        ],
        key_metrics={
            "data_source": "mock (simulated)",
            "note": "All figures are simulated and must not be treated as real market data.",
        },
        uncertainty=[
            "All data in this run is simulated (mock provider); no conclusions are real.",
            "Valuation and technical figures are computed from simulated data for workflow demonstration only.",
        ],
        conclusion=(
            f"The demo shows that the agent can autonomously collect price, financial, news, technical "
            f"and valuation data for {symbol} and synthesise a structured report. For real analysis, "
            "switch the data provider to yfinance and supply an LLM API key."
        ),
    )


def build_demo_model(ticker: str = "NVDA", language: str = "en") -> ScriptedReActDemoModel:
    """A demo model scripted to research ``ticker`` through the Phase 2 tools."""
    symbol = ticker.upper()
    report_json = json.dumps(_demo_report(symbol, language).model_dump())
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
                {"name": "calculate_rsi", "args": {"ticker": symbol, "period": "6mo", "window": 14}, "id": f"demo-{symbol}-3d"},
            ],
        ),
        AIMessage(content=report_json),
    ]
    return ScriptedReActDemoModel(messages=iter(sequence))


def build_nvda_demo_model(language: str = "en") -> ScriptedReActDemoModel:
    """Backward-compatible alias for the NVDA demo."""
    return build_demo_model("NVDA", language)
