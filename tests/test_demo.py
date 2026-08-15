from app.agents.react_agent import StockMindAgent
from app.models.demo import build_demo_model
from app.tools import build_tools
from app.tools.providers import MockMarketDataProvider


def _agent(ticker, language):
    return StockMindAgent(
        llm=build_demo_model(ticker, language),
        tools=build_tools(MockMarketDataProvider(seed=1)),
        language=language,
    )


def test_demo_runs_full_phase2_loop():
    agent = _agent("AMD", "en")
    state = agent.run("research AMD", ticker="AMD")
    assert state["final_output"].ticker == "AMD"
    # The scripted demo calls 10 distinct Phase 2 tools.
    assert len(state["observations"]) == 10
    assert state["errors"] == []


def test_demo_supports_chinese_and_other_tickers():
    agent = _agent("TSM", "zh")
    report = agent.invoke("研究 TSM", ticker="TSM")
    assert report.ticker == "TSM"
    assert any("\u4e00" <= ch <= "\u9fff" for ch in report.summary)

