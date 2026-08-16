from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agents.react_agent import StockMindAgent
from app.graph.workflow import AgentConfig
from app.models.demo import GROUNDED_REPORT_MARKER
from app.tools import build_tools
from app.tools.providers import MockMarketDataProvider


def _make_agent(messages, max_iterations=8):
    llm = GenericFakeChatModel(messages=iter(messages))
    tools = build_tools(MockMarketDataProvider(seed=1))
    return StockMindAgent(
        llm=llm,
        tools=tools,
        config=AgentConfig(max_iterations=max_iterations, tool_timeout_seconds=5, tool_retries=0),
    )


def _final_json(ticker="NVDA"):
    import json

    from app.schemas.report import StockResearchReport

    return json.dumps(
        StockResearchReport(ticker=ticker, summary="ok", conclusion="ok").model_dump()
    )


def test_agent_calls_tool_then_finalizes():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": "c1"}],
        ),
        AIMessage(content=_final_json()),
    ]
    agent = _make_agent(messages)
    state = agent.run("Analyze NVDA price", ticker="NVDA")
    assert state["final_output"].ticker == "NVDA"
    assert len(state["observations"]) == 1
    assert state["observations"][0]["tool"] == "get_stock_price"
    assert state["status"] == "finished"


def test_max_iterations_terminates_gracefully():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": f"c{i}"}],
        )
        for i in range(20)
    ]
    agent = _make_agent(messages, max_iterations=3)
    state = agent.run("Analyze NVDA", ticker="NVDA")
    assert state["status"] == "finished"  # finalize always closes the run
    assert state["final_output"].ticker == "NVDA"
    assert state["iteration_count"] == 4  # one past the limit triggers termination


def test_duplicate_tool_call_skipped():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": "c1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": "c2"}],
        ),
        AIMessage(content=_final_json()),
    ]
    agent = _make_agent(messages)
    state = agent.run("Analyze NVDA", ticker="NVDA")
    # Only the first identical call produced an observation.
    assert len(state["observations"]) == 1


def test_unknown_tool_is_recorded_as_error_but_does_not_crash():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "no_such_tool", "args": {}, "id": "c1"}],
        ),
        AIMessage(content=_final_json()),
    ]
    agent = _make_agent(messages)
    state = agent.run("Analyze NVDA", ticker="NVDA")
    assert state["final_output"].ticker == "NVDA"
    assert any("Unknown tool" in e for e in state["errors"])


def test_no_observations_does_not_leak_internal_marker():
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=GROUNDED_REPORT_MARKER)]))
    agent = StockMindAgent(
        llm=llm,
        tools=build_tools(MockMarketDataProvider(seed=1)),
        config=AgentConfig(tool_timeout_seconds=5, tool_retries=0),
    )
    report = agent.invoke("Analyze NVDA", ticker="NVDA")
    assert GROUNDED_REPORT_MARKER not in report.summary
    assert GROUNDED_REPORT_MARKER not in report.conclusion
    assert report.summary
