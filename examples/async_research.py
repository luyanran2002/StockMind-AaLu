"""Async + checkpointing demo (Phase 3).

Run:

    python examples/async_research.py --ticker NVDA --lang en

Demonstrates ``ainvoke``, SQLite checkpointing and resuming a run by
``thread_id``. Without an API key it uses the offline scripted demo model.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.agents.react_agent import StockMindAgent
from app.graph.checkpoint import build_checkpointer
from app.models.demo import build_demo_model
from app.utils import format_local_time

CHECKPOINT_DB = Path(__file__).resolve().parents[1] / "checkpoints" / "checkpoints.sqlite"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Async + checkpoint demo")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--lang", choices=["en", "zh"], default=os.getenv("STOCKMIND_LANGUAGE", "en"))
    args = parser.parse_args()
    ticker = args.ticker.upper()

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with build_checkpointer("sqlite", str(CHECKPOINT_DB)) as saver:
        agent = StockMindAgent(
            llm=build_demo_model(ticker, args.lang),
            language=args.lang,
            checkpointer=saver,
        )

        thread_id = f"{ticker.lower()}-run"
        report = await agent.ainvoke(
            f"Give me a research snapshot of {ticker}.", ticker=ticker, thread_id=thread_id
        )
        print(f"[first run] {report.ticker} generated_at={format_local_time(report.generated_at)}")
        print(f"[first run] summary: {report.summary[:120]}...")

        snapshot = await agent.aget_state(thread_id)
        print(f"[checkpoint] status={snapshot.values['status']}  messages={len(snapshot.values['messages'])}")

        follow_up = await agent.ainvoke(
            f"Anything else worth flagging for {ticker}?", ticker=ticker, thread_id=thread_id
        )
        snapshot2 = await agent.aget_state(thread_id)
        print(f"[resumed] status={snapshot2.values['status']}  messages={len(snapshot2.values['messages'])}")


if __name__ == "__main__":
    asyncio.run(main())
