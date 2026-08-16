import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agents.react_agent import StockMindAgent
from app.graph.checkpoint import build_checkpointer
from app.graph.workflow import AgentConfig, _backoff_delay
from app.observability.tracing import TraceCollector
from app.schemas.report import StockResearchReport
from app.tools import build_tools
from app.tools.providers import MockMarketDataProvider


def _final_json(ticker="NVDA"):
    return json.dumps(
        StockResearchReport(ticker=ticker, summary="ok", conclusion="ok").model_dump()
    )


def _scripted_agent():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NVDA"}, "id": "c1"}],
        ),
        AIMessage(content=_final_json()),
    ]
    return StockMindAgent(
        llm=GenericFakeChatModel(messages=iter(messages)),
        tools=build_tools(MockMarketDataProvider(seed=1)),
        config=AgentConfig(tool_timeout_seconds=5, tool_retries=0),
    )


def test_async_invoke_matches_sync():
    agent = _scripted_agent()
    sync_report = agent.invoke("Analyze NVDA", ticker="NVDA")
    async_report = asyncio.run(agent.ainvoke("Analyze NVDA", ticker="NVDA"))
    assert sync_report.ticker == "NVDA"
    assert async_report.ticker == "NVDA"


def test_async_run_returns_state():
    agent = _scripted_agent()
    state = asyncio.run(agent.arun("Analyze NVDA", ticker="NVDA"))
    assert state["final_output"].ticker == "NVDA"
    assert state["status"] == "finished"


def test_checkpoint_state_is_retrievable():
    agent = _scripted_agent()
    asyncio.run(agent.ainvoke("Analyze NVDA", ticker="NVDA", thread_id="ckpt-1"))
    snapshot = asyncio.run(agent.aget_state("ckpt-1"))
    assert snapshot.values["status"] == "finished"
    assert snapshot.values["final_output"].ticker == "NVDA"


def test_resume_accumulates_messages():
    agent = _scripted_agent()
    first = asyncio.run(agent.arun("First query", ticker="NVDA", thread_id="resume-1"))
    second = asyncio.run(agent.arun("Follow-up query", ticker="NVDA", thread_id="resume-1"))
    assert len(second["messages"]) > len(first["messages"])


def test_sqlite_checkpointer():
    async def _run():
        async with build_checkpointer("sqlite", ":memory:") as saver:
            agent = StockMindAgent(
                llm=GenericFakeChatModel(
                    messages=iter(
                        [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": "get_stock_price", "args": {"ticker": "AMD"}, "id": "s1"}
                                ],
                            ),
                            AIMessage(content=_final_json("AMD")),
                        ]
                    )
                ),
                tools=build_tools(MockMarketDataProvider(seed=1)),
                config=AgentConfig(tool_timeout_seconds=5, tool_retries=0),
                checkpointer=saver,
            )
            report = await agent.ainvoke("Analyze AMD", ticker="AMD", thread_id="sql-1")
            assert report.ticker == "AMD"
            snapshot = await agent.aget_state("sql-1")
            assert snapshot.values["status"] == "finished"

    asyncio.run(_run())


def test_backoff_grows_and_is_capped():
    config = AgentConfig(retry_backoff_seconds=0.5, retry_max_backoff_seconds=2.0, retry_jitter=0.0)
    delays = [_backoff_delay(attempt, config) for attempt in range(6)]
    assert delays[0] == pytest.approx(0.5)
    assert delays[1] == pytest.approx(1.0)
    assert all(delay <= 2.0 for delay in delays)
    assert delays[3] == pytest.approx(2.0)


def test_trace_summary_dedupes_repeated_errors():
    tracer = TraceCollector()
    tracer.start_run("query")
    for _ in range(3):
        tracer.record(
            event_type="tool_call",
            tool_name="get_stock_price",
            error="YFRateLimitError: Too Many Requests",
        )
    summary = tracer.summarize()
    assert summary.count("YFRateLimitError") == 1
