"""Phase 4 evaluation CLI.

Run:

    python examples/evaluate.py --lang en

Without an API key it evaluates using the offline demo model + mock provider;
with a key it uses a real LLM and (if configured) a real data provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.agents.react_agent import StockMindAgent
from app.evaluation import Evaluator
from app.models.demo import build_demo_model

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "evaluation" / "datasets" / "questions.json"
)


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _agent_factory(language: str):
    if _has_api_key():
        return lambda ticker: StockMindAgent(language=language)
    return lambda ticker: StockMindAgent(
        llm=build_demo_model(ticker or "NVDA", language), language=language
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="StockMind evaluation")
    parser.add_argument("--lang", choices=["en", "zh"], default=os.getenv("STOCKMIND_LANGUAGE", "en"))
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    report = await Evaluator(_agent_factory(args.lang)).evaluate(dataset)
    print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())

