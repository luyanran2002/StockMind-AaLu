"""Lightweight tracing for observability.

Every run gets a ``run_id`` and records structured events (agent steps, tool
calls, finalisation, errors) with latency and token usage. Events are kept
in-memory for a human-readable summary and optionally appended to JSONL on disk.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceEvent(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: str
    agent_step: int | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None
    error: str | None = None
    final_status: str | None = None


class TraceCollector:
    """Collects trace events for a run and can render a summary."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self.events: list[TraceEvent] = []
        self.run_id: str | None = None
        self._run_started_at: float | None = None
        self.log_dir = Path(log_dir) if log_dir else None

    def start_run(self, user_query: str) -> str:
        self.run_id = uuid.uuid4().hex[:12]
        self._run_started_at = time.perf_counter()
        self.events = []
        self.record(event_type="run_start", tool_input={"user_query": user_query})
        return self.run_id

    def record(
        self,
        *,
        event_type: str,
        agent_step: int | None = None,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: str | None = None,
        latency_ms: float | None = None,
        token_usage: dict[str, Any] | None = None,
        error: str | None = None,
        final_status: str | None = None,
    ) -> None:
        if self.run_id is None:
            raise RuntimeError("TraceCollector.start_run() must be called before record()")
        event = TraceEvent(
            run_id=self.run_id,
            event_type=event_type,
            agent_step=agent_step,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=latency_ms,
            token_usage=token_usage,
            error=error,
            final_status=final_status,
        )
        self.events.append(event)
        self._write_jsonl(event)

    def total_latency_ms(self) -> float:
        if self._run_started_at is None:
            return 0.0
        return (time.perf_counter() - self._run_started_at) * 1000.0

    def summarize(self) -> str:
        """Render a human-readable run summary, mirroring the target format."""
        if not self.events:
            return "No trace events recorded."

        lines = [f"Run #{self.run_id}", ""]
        lines.append(f"Total latency: {self.total_latency_ms() / 1000.0:.2f}s")
        lines.append("")
        lines.append("Steps:")
        index = 1
        for event in self.events:
            if event.event_type == "agent_step":
                label = "llm_reason"
                detail = ""
            elif event.event_type == "tool_call":
                label = event.tool_name or "tool"
                detail = ""
            elif event.event_type == "tool_result":
                continue
            elif event.event_type == "finalize":
                label = "final_llm_call"
                detail = ""
            elif event.event_type == "finalize_fallback":
                label = "finalize_fallback"
                detail = ""
            elif event.event_type == "run_end":
                label = f"run_end (status={event.final_status})"
                detail = ""
            else:
                continue
            latency = f"  {event.latency_ms / 1000.0:.2f}s" if event.latency_ms is not None else ""
            lines.append(f"{index}. {label:<26}{latency}")
            index += 1
        errors = [e for e in self.events if e.error]
        if errors:
            lines.append("")
            lines.append("Errors:")
            for e in errors:
                lines.append(f"  - {e.tool_name or e.event_type}: {e.error}")
        return "\n".join(lines)

    def _write_jsonl(self, event: TraceEvent) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"trace_{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
