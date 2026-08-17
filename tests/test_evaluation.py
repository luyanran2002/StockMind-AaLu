import asyncio

import pytest

from app.agents.react_agent import StockMindAgent
from app.evaluation import Evaluator
from app.models.demo import build_demo_model
from app.observability.cost import estimate_cost_usd, token_counts


def _agent_factory(ticker):
    return StockMindAgent(llm=build_demo_model(ticker or "NVDA", "en"), language="en")


def test_cost_estimation():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost_usd("unknown-model", 1000, 1000) == 0.0


def test_token_counts():
    assert token_counts({"input_tokens": 10, "output_tokens": 5}) == (10, 5)
    assert token_counts({"prompt_tokens": 7, "completion_tokens": 3}) == (7, 3)
    assert token_counts(None) == (0, 0)


def test_evaluation_metrics():
    dataset = [
        {
            "question": "What is NVDA's current price and recent trend?",
            "ticker": "NVDA",
            "expected_tools": ["get_stock_price", "get_historical_prices"],
        },
        {
            "question": "Summarize AMD's risk profile.",
            "ticker": "AMD",
            "expected_tools": ["calculate_volatility", "calculate_drawdown"],
        },
    ]
    report = asyncio.run(Evaluator(_agent_factory).evaluate(dataset))
    assert len(report.runs) == 2
    assert report.task_success_rate() == 1.0
    assert report.tool_selection_accuracy() == 1.0
    assert report.data_grounding_rate() == 1.0
    assert report.numerical_accuracy() == 1.0
    assert report.avg_quality() > 0.9
    assert "Task Success Rate" in report.summary()

