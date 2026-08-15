"""Public facade for the StockMind ReAct agent."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.graph.state import StockAgentState, create_initial_state
from app.graph.workflow import AgentConfig, build_agent_graph
from app.models.providers import get_chat_model
from app.observability.tracing import TraceCollector
from app.schemas.report import StockResearchReport
from app.tools import build_tools, get_data_provider


class StockMindAgent:
    """High-level entry point wiring LLM + tools + graph + tracing."""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        config: AgentConfig | None = None,
        tracer: TraceCollector | None = None,
        language: str | None = None,
    ) -> None:
        self.llm = llm or get_chat_model()
        self.tools = tools or build_tools(get_data_provider())
        explicit_language = language or os.getenv("STOCKMIND_LANGUAGE")
        resolved_language = (explicit_language or "en").strip().lower()
        if config is None:
            self.config = AgentConfig(
                max_iterations=int(os.getenv("STOCKMIND_MAX_ITERATIONS", "8")),
                tool_timeout_seconds=float(os.getenv("STOCKMIND_TOOL_TIMEOUT_SECONDS", "15")),
                tool_retries=int(os.getenv("STOCKMIND_TOOL_RETRIES", "1")),
                language=resolved_language,
            )
        else:
            self.config = config
            if explicit_language:
                self.config.language = resolved_language
        self.language = self.config.language
        trace_dir = os.getenv("STOCKMIND_TRACE_DIR") or None
        self.tracer = tracer or TraceCollector(log_dir=trace_dir)
        self.graph = build_agent_graph(self.llm, self.tools, self.config, self.tracer)

    def run(
        self,
        user_query: str,
        *,
        ticker: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StockAgentState:
        """Run the full ReAct loop and return the final state."""
        meta = dict(metadata or {})
        run_id = self.tracer.start_run(user_query)
        meta.setdefault("run_id", run_id)

        state = create_initial_state(
            user_query,
            ticker=ticker,
            max_iterations=self.config.max_iterations,
            language=self.config.language,
            metadata=meta,
        )
        result = self.graph.invoke(state)
        self.tracer.record(
            event_type="run_end",
            latency_ms=self.tracer.total_latency_ms(),
            final_status=result.get("status"),
        )
        return result

    def invoke(
        self,
        user_query: str,
        *,
        ticker: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StockResearchReport:
        """Run the agent and return only the structured final report."""
        result = self.run(user_query, ticker=ticker, metadata=metadata)
        report = result.get("final_output")
        if report is None:
            raise RuntimeError("Agent completed without producing a final report.")
        return report
