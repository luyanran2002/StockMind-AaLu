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
    report = state["final_output"]
    assert report.ticker == "AMD"
    # The scripted demo calls 15 distinct Phase 2 tools.
    assert len(state["observations"]) == 15
    assert state["errors"] == []
    # The report is assembled from real observations, not canned prose.
    assert report.key_metrics["price"] is not None
    assert "data_sources" in report.key_metrics
    assert any("mock" in source for source in report.key_metrics["data_sources"])


def test_demo_supports_chinese_and_other_tickers():
    agent = _agent("TSM", "zh")
    report = agent.invoke("研究 TSM", ticker="TSM")
    assert report.ticker == "TSM"
    assert any("\u4e00" <= ch <= "\u9fff" for ch in report.summary)


def test_report_has_precise_generated_at():
    agent = _agent("NVDA", "en")
    report = agent.invoke("research NVDA", ticker="NVDA")
    assert report.generated_at is not None
    assert "T" in report.generated_at
