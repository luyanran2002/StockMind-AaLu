"""The LangGraph ReAct execution loop.

Nodes:
  agent     -> LLM reasons and either emits tool calls or a final answer.
  tools     -> Executes tool calls with timeout, retry and duplicate detection.
  finalize  -> Produces a validated :class:`StockResearchReport`.

The loop is bounded by ``max_iterations``. Tool failures are recorded as
observations rather than crashing the run (graceful degradation).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from app.agents.prompts import get_finalize_prompt, get_system_prompt
from app.graph.state import StockAgentState
from app.observability.tracing import TraceCollector
from app.schemas.report import StockResearchReport
from app.tools.base import tool_call_signature


@dataclass
class AgentConfig:
    """Runtime limits for the ReAct loop."""

    max_iterations: int = 8
    tool_timeout_seconds: float = 15.0
    tool_retries: int = 1
    language: str = "en"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_token_usage(message: BaseMessage) -> dict[str, Any] | None:
    usage = getattr(message, "usage_metadata", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return getattr(usage, "model_dump", lambda: None)()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _infer_ticker(state: StockAgentState) -> str:
    if state.get("ticker"):
        return state["ticker"].upper()
    for call in reversed(state.get("tool_calls", [])):
        args = call.get("args") or {}
        ticker = args.get("ticker")
        if ticker:
            return str(ticker).upper()
    return "UNKNOWN"


def _build_agent_messages(state: StockAgentState, language: str) -> list[BaseMessage]:
    return [SystemMessage(content=get_system_prompt(language)), *list(state.get("messages", []))]


def _make_agent_node(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig,
    tracer: TraceCollector,
):
    # Real providers support native tool binding; fake/test models do not, so we
    # fall back to the bare model (the fake returns pre-scripted tool calls).
    try:
        llm_with_tools = llm.bind_tools(tools)
    except NotImplementedError:
        llm_with_tools = llm

    def agent_node(state: StockAgentState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1
        updates: dict[str, Any] = {"iteration_count": iteration}

        if iteration > config.max_iterations:
            final_report = state.get("final_report") or (
                "Research stopped after reaching the maximum number of iterations. "
                "The answer below is based on the evidence collected so far."
            )
            tracer.record(
                event_type="agent_step",
                agent_step=iteration,
                final_status="max_iterations",
            )
            return {
                **updates,
                "messages": [AIMessage(content=final_report)],
                "final_report": final_report,
                "status": "max_iterations",
            }

        start = time.perf_counter()
        try:
            response = llm_with_tools.invoke(_build_agent_messages(state, config.language))
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:  # LLM outage -> degrade gracefully
            error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - start) * 1000.0
            tracer.record(
                event_type="agent_step",
                agent_step=iteration,
                latency_ms=latency_ms,
                error=error,
            )
            final_report = (
                f"The reasoning model failed at step {iteration} ({error}). "
                "Returning the evidence collected so far."
            )
            return {
                **updates,
                "messages": [AIMessage(content=final_report)],
                "final_report": final_report,
                "errors": state.get("errors", []) + [error],
                "status": "error",
            }

        tracer.record(
            event_type="agent_step",
            agent_step=iteration,
            latency_ms=latency_ms,
            token_usage=_extract_token_usage(response),
        )

        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            return {**updates, "messages": [response], "status": "running"}

        content = _stringify(getattr(response, "content", ""))
        return {
            **updates,
            "messages": [response],
            "final_report": content,
            "status": "finished",
        }

    return agent_node


def _invoke_tool_with_timeout(tool: BaseTool, args: dict[str, Any], timeout_seconds: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tool.invoke, args)
        return future.result(timeout=timeout_seconds)


def _make_tools_node(
    tools: list[BaseTool],
    config: AgentConfig,
    tracer: TraceCollector,
):
    tool_map = {tool.name: tool for tool in tools}

    def _run_with_retry(
        tool: BaseTool, args: dict[str, Any], agent_step: int
    ) -> tuple[str | None, str | None]:
        last_error: str | None = None
        for attempt in range(config.tool_retries + 1):
            start = time.perf_counter()
            try:
                raw = _invoke_tool_with_timeout(tool, args, config.tool_timeout_seconds)
                output = _stringify(raw)
                tracer.record(
                    event_type="tool_call",
                    agent_step=agent_step,
                    tool_name=tool.name,
                    tool_input=args,
                    tool_output=output,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
                return output, None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tracer.record(
                    event_type="tool_call",
                    agent_step=agent_step,
                    tool_name=tool.name,
                    tool_input=args,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    error=last_error,
                )
        tracer.record(
            event_type="tool_result",
            agent_step=agent_step,
            tool_name=tool.name,
            error=last_error,
        )
        return None, last_error

    def tools_node(state: StockAgentState) -> dict[str, Any]:
        last = state.get("messages", [])[-1] if state.get("messages") else None
        calls = getattr(last, "tool_calls", None) or []
        agent_step = state.get("iteration_count", 0)

        tool_messages: list[ToolMessage] = []
        observations: list[dict[str, Any]] = []
        errors: list[str] = []
        new_signatures: list[str] = []
        new_tool_calls: list[dict[str, Any]] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{agent_step}_{len(tool_messages)}"
            signature = tool_call_signature(name, args)
            new_tool_calls.append({"name": name, "args": args, "id": call_id, "step": agent_step})

            if signature in state.get("tool_call_signatures", []):
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {"duplicate": True, "message": "Identical tool call already executed; skipped."}
                        ),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            new_signatures.append(signature)
            tool = tool_map.get(name)
            if tool is None:
                error = f"Unknown tool requested: {name!r}"
                errors.append(error)
                tool_messages.append(
                    ToolMessage(content=json.dumps({"error": error}), tool_call_id=call_id, name=name)
                )
                continue

            output, error = _run_with_retry(tool, args, agent_step)
            if error:
                errors.append(error)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"tool": name, "error": error}),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
            else:
                observations.append(
                    {
                        "tool": name,
                        "args": args,
                        "result": output,
                        "timestamp": _utc_now_iso(),
                    }
                )
                tool_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))

        return {
            "messages": tool_messages,
            "tool_calls": state.get("tool_calls", []) + new_tool_calls,
            "observations": state.get("observations", []) + observations,
            "errors": state.get("errors", []) + errors,
            "tool_call_signatures": state.get("tool_call_signatures", []) + new_signatures,
        }

    return tools_node


def _fallback_text(state: StockAgentState) -> str:
    observations = state.get("observations", [])
    if not observations:
        return "No tool observations were collected before the run ended."
    lines = [f"Collected {len(observations)} tool observation(s):"]
    for obs in observations:
        lines.append(f"- {obs['tool']}: {obs['result']}")
    return "\n".join(lines)


def _fallback_report(ticker: str, text: str, note: str) -> StockResearchReport:
    return StockResearchReport(
        ticker=ticker,
        summary=text,
        market_analysis="Not covered in this phase.",
        financial_analysis="Not covered in this phase.",
        technical_analysis="Not covered in this phase.",
        news_analysis="Not covered in this phase.",
        valuation_analysis="Not covered in this phase.",
        risk_analysis="Not covered in this phase.",
        bull_case=[],
        bear_case=[],
        key_metrics={},
        uncertainty=[note],
        conclusion=text,
    )


def _generate_report(
    llm: BaseChatModel, state: StockAgentState, tracer: TraceCollector
) -> StockResearchReport:
    ticker = _infer_ticker(state)
    final_report = state.get("final_report") or _fallback_text(state)

    start = time.perf_counter()
    # 1) Prefer native structured output.
    try:
        structured_llm = llm.with_structured_output(StockResearchReport)
        messages: list[BaseMessage] = [
            SystemMessage(content=get_finalize_prompt(state.get("language", "en"))),
            *list(state.get("messages", [])),
            HumanMessage(content=f"Final reasoning so far:\n{final_report}"),
        ]
        report = structured_llm.invoke(messages)
        if not isinstance(report, StockResearchReport):
            raise TypeError(f"Structured output returned {type(report).__name__}, not StockResearchReport")
        report = report.model_copy(update={"ticker": report.ticker or ticker})
        tracer.record(
            event_type="finalize",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            final_status="finished",
        )
        return report
    except Exception as exc:
        tracer.record(
            event_type="finalize_fallback",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tool_output=(
                f"Structured output unavailable ({type(exc).__name__}); "
                "fell back to JSON/text report."
            ),
        )

    # 2) The final answer may already be valid JSON (e.g. the offline demo).
    try:
        return StockResearchReport.model_validate_json(final_report)
    except Exception:
        pass

    # 3) Last resort: wrap the raw reasoning in a minimal, honest report.
    return _fallback_report(
        ticker,
        final_report,
        "Structured output was unavailable; the report was generated from raw agent reasoning.",
    )


def _make_finalize_node(llm: BaseChatModel, tracer: TraceCollector):
    def finalize_node(state: StockAgentState) -> dict[str, Any]:
        report = _generate_report(llm, state, tracer)
        return {"final_output": report, "status": "finished"}

    return finalize_node


def _route_after_agent(state: StockAgentState) -> str:
    if state.get("status") in ("finished", "max_iterations", "error"):
        return "finalize"
    return "tools"


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig | None = None,
    tracer: TraceCollector | None = None,
):
    """Compile the Phase 1 ReAct graph."""
    config = config or AgentConfig()
    tracer = tracer or TraceCollector()

    graph = StateGraph(StockAgentState)
    graph.add_node("agent", _make_agent_node(llm, tools, config, tracer))
    graph.add_node("tools", _make_tools_node(tools, config, tracer))
    graph.add_node("finalize", _make_finalize_node(llm, tracer))

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
