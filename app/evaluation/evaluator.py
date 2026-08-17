"""Evaluation framework (Phase 4).

Runs a dataset of questions through an agent and measures task success, tool
selection, data grounding, numerical accuracy, answer quality, latency, token
usage and cost. It is provider-agnostic and works offline with the demo model +
mock provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.demo import GROUNDED_REPORT_MARKER
from app.observability.cost import estimate_cost_usd
from app.schemas.report import StockResearchReport

_NOT_COVERED = "Not covered in this phase."


@dataclass
class RunMetrics:
    question: str
    ticker: str
    success: bool
    grounded: bool
    tool_recall: float
    numerical_ok: bool
    quality: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class EvaluationReport:
    runs: list[RunMetrics] = field(default_factory=list)

    def _rate(self, values: list[bool]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def task_success_rate(self) -> float:
        return self._rate([r.success for r in self.runs])

    def tool_selection_accuracy(self) -> float:
        return sum(r.tool_recall for r in self.runs) / len(self.runs) if self.runs else 0.0

    def data_grounding_rate(self) -> float:
        return self._rate([r.grounded for r in self.runs])

    def numerical_accuracy(self) -> float:
        return self._rate([r.numerical_ok for r in self.runs])

    def avg_quality(self) -> float:
        return sum(r.quality for r in self.runs) / len(self.runs) if self.runs else 0.0

    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.runs) / len(self.runs) if self.runs else 0.0

    def avg_tokens(self) -> float:
        return (
            sum(r.input_tokens + r.output_tokens for r in self.runs) / len(self.runs)
            if self.runs
            else 0.0
        )

    def avg_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.runs) / len(self.runs) if self.runs else 0.0

    def summary(self) -> str:
        lines = [
            "EVALUATION REPORT",
            "=" * 40,
            f"Runs: {len(self.runs)}",
            f"Task Success Rate:      {self.task_success_rate():.0%}",
            f"Tool Selection Accuracy:{self.tool_selection_accuracy():.0%}",
            f"Data Grounding Rate:    {self.data_grounding_rate():.0%}",
            f"Numerical Accuracy:     {self.numerical_accuracy():.0%}",
            f"Answer Quality (avg):   {self.avg_quality():.0%}",
            f"Average Latency:        {self.avg_latency_ms() / 1000.0:.2f}s",
            f"Average Tokens:         {self.avg_tokens():.0f}",
            f"Average Cost:           ${self.avg_cost_usd():.6f}",
        ]
        return "\n".join(lines)


class Evaluator:
    """Run a dataset through per-ticker agents and score the results."""

    def __init__(self, agent_factory: Callable[[str | None], Any]) -> None:
        # agent_factory(ticker) -> a fresh StockMindAgent for that ticker.
        self.agent_factory = agent_factory

    async def evaluate(self, dataset: list[dict[str, Any]]) -> EvaluationReport:
        runs: list[RunMetrics] = []
        for index, item in enumerate(dataset):
            question = item["question"]
            ticker = item.get("ticker")
            agent = self.agent_factory(ticker)
            state = await agent.arun(
                question, ticker=ticker, thread_id=f"eval-{index}"
            )
            runs.append(self._metrics(agent, state, item))
        return EvaluationReport(runs=runs)

    def _metrics(self, agent: Any, state: dict[str, Any], item: dict[str, Any]) -> RunMetrics:
        report = state.get("final_output")
        tracer = agent.tracer

        expected = item.get("expected_tools", [])
        called = {obs.get("tool") for obs in state.get("observations", [])}
        called |= {call.get("name") for call in state.get("tool_calls", [])}
        recall = (
            sum(1 for tool in expected if tool in called) / len(expected)
            if expected
            else 1.0
        )

        return RunMetrics(
            question=item["question"],
            ticker=item.get("ticker", "UNKNOWN"),
            success=self._success(state, report),
            grounded=self._grounded(report),
            tool_recall=recall,
            numerical_ok=self._numerical_ok(agent, report, item),
            quality=self._quality(report),
            latency_ms=tracer.total_latency_ms(),
            input_tokens=tracer.total_input_tokens,
            output_tokens=tracer.total_output_tokens,
            cost_usd=estimate_cost_usd(
                tracer.model, tracer.total_input_tokens, tracer.total_output_tokens
            ),
        )

    @staticmethod
    def _success(state: dict[str, Any], report: Any) -> bool:
        if not isinstance(report, StockResearchReport):
            return False
        if state.get("status") != "finished":
            return False
        if not report.summary or GROUNDED_REPORT_MARKER in report.summary:
            return False
        return True

    @staticmethod
    def _grounded(report: Any) -> bool:
        if not isinstance(report, StockResearchReport):
            return False
        sources = (report.key_metrics or {}).get("data_sources") or []
        return bool(sources)

    def _numerical_ok(self, agent: Any, report: Any, item: dict[str, Any]) -> bool:
        if not isinstance(report, StockResearchReport):
            return False
        price = (report.key_metrics or {}).get("price")
        if price is None:
            return False
        provider = getattr(agent, "data_provider", None)
        if provider is None:
            return True
        try:
            truth = provider.get_stock_price(item.get("ticker", "UNKNOWN")).price
        except Exception:
            return True
        return abs(float(price) - float(truth)) / max(abs(float(truth)), 1e-9) < 0.02

    @staticmethod
    def _quality(report: Any) -> float:
        if not isinstance(report, StockResearchReport):
            return 0.0
        sections = [
            report.market_analysis,
            report.news_analysis,
            report.financial_analysis,
            report.technical_analysis,
            report.valuation_analysis,
            report.risk_analysis,
        ]
        filled = sum(1 for s in sections if s and s != _NOT_COVERED)
        score = filled / len(sections)
        if not report.summary or not report.conclusion:
            score *= 0.5
        return score
