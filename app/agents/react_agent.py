"""Public facade for the StockMind ReAct agent (async-first).

Sync methods (``run``/``invoke``) are thin wrappers around their async
counterparts (``arun``/``ainvoke``), so callers can use whichever style fits.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.graph.checkpoint import build_checkpointer
from app.graph.state import StockAgentState, create_initial_state
from app.graph.workflow import AgentConfig, build_agent_graph
from app.models.providers import get_chat_model
from app.observability.tracing import TraceCollector
from app.schemas.report import StockResearchReport
from app.tools import build_tools, get_data_provider


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


class StockMindAgent:
    """High-level entry point wiring LLM + tools + graph + tracing + checkpointer."""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        config: AgentConfig | None = None,
        tracer: TraceCollector | None = None,
        language: str | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.llm = llm or get_chat_model()
        if tools is None:
            self.data_provider = get_data_provider()
            self.tools = build_tools(self.data_provider)
        else:
            self.data_provider = None
            self.tools = tools

        explicit_language = language or os.getenv("STOCKMIND_LANGUAGE")
        resolved_language = (explicit_language or "en").strip().lower()
        if config is None:
            self.config = AgentConfig(
                max_iterations=_env_int("STOCKMIND_MAX_ITERATIONS", 8),
                tool_timeout_seconds=_env_float("STOCKMIND_TOOL_TIMEOUT_SECONDS", 15.0),
                tool_retries=_env_int("STOCKMIND_TOOL_RETRIES", 1),
                retry_backoff_seconds=_env_float("STOCKMIND_RETRY_BACKOFF_SECONDS", 0.5),
                retry_max_backoff_seconds=_env_float("STOCKMIND_RETRY_MAX_BACKOFF_SECONDS", 8.0),
                retry_jitter=_env_float("STOCKMIND_RETRY_JITTER", 0.1),
                language=resolved_language,
            )
        else:
            self.config = config
            if explicit_language:
                self.config.language = resolved_language
        self.language = self.config.language

        trace_dir = os.getenv("STOCKMIND_TRACE_DIR") or None
        self.tracer = tracer or TraceCollector(
            log_dir=trace_dir, model=getattr(self.llm, "model_name", None)
        )
        if tracer is not None:
            tracer.model = getattr(self.llm, "model_name", None)

        self.checkpointer = self._resolve_checkpointer(checkpointer)
        self.graph = build_agent_graph(
            self.llm, self.tools, self.config, self.tracer, checkpointer=self.checkpointer
        )

    @staticmethod
    def _resolve_checkpointer(checkpointer: Any | None) -> Any:
        if checkpointer is None or checkpointer == "memory":
            return build_checkpointer("memory")
        if checkpointer == "sqlite":
            raise ValueError(
                "SQLite checkpoints use an async context manager. Enter it yourself, e.g.:\n"
                "    from app.graph.checkpoint import build_checkpointer\n"
                "    async with build_checkpointer('sqlite', 'checkpoints.sqlite') as saver:\n"
                "        agent = StockMindAgent(checkpointer=saver)"
            )
        return checkpointer  # assume a ready-to-use checkpointer instance

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def arun(
        self,
        user_query: str,
        *,
        ticker: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StockAgentState:
        """Run the full ReAct loop asynchronously and return the final state."""
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
        config = self._config(thread_id or run_id)
        result = await self.graph.ainvoke(state, config=config)
        self.tracer.record(
            event_type="run_end",
            latency_ms=self.tracer.total_latency_ms(),
            final_status=result.get("status"),
        )
        return result

    async def ainvoke(
        self,
        user_query: str,
        *,
        ticker: str | None = None,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StockResearchReport:
        """Run the agent asynchronously and return the structured final report."""
        result = await self.arun(user_query, ticker=ticker, thread_id=thread_id, metadata=metadata)
        report = result.get("final_output")
        if report is None:
            raise RuntimeError("Agent completed without producing a final report.")
        return report

    async def aget_state(self, thread_id: str) -> Any:
        """Return the persisted checkpoint snapshot for ``thread_id``."""
        return await self.graph.aget_state(self._config(thread_id))

    # -- sync wrappers ---------------------------------------------------------
    def run(self, user_query: str, **kwargs: Any) -> StockAgentState:
        return asyncio.run(self.arun(user_query, **kwargs))

    def invoke(self, user_query: str, **kwargs: Any) -> StockResearchReport:
        return asyncio.run(self.ainvoke(user_query, **kwargs))

    def get_state(self, thread_id: str) -> Any:
        return asyncio.run(self.aget_state(thread_id))
