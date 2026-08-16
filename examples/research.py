"""General multi-stock, bilingual research CLI.

Run from the repository root:

    python examples/research.py --tickers NVDA,AMD,TSM --lang zh

* If OPENAI_API_KEY or ANTHROPIC_API_KEY is set, a real LLM drives each loop.
* Otherwise an offline scripted demo model drives each loop.
* Data defaults to the deterministic ``mock`` provider (clearly labelled as
  simulated). Set ``STOCKMIND_DATA_PROVIDER=yfinance`` for real data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.agents.react_agent import StockMindAgent
from app.models.demo import build_demo_model
from app.schemas.report import StockResearchReport
from app.tools.providers import PriceBar
from app.utils import format_local_time
from app.visualization.charts import ascii_sparkline, render_price_chart

CHARTS_DIR = Path(__file__).resolve().parents[1] / "charts"


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _default_query(ticker: str, language: str) -> str:
    if language == "zh":
        return (
            f"请对 {ticker} 做一份完整研究报告：当前价格、近期走势、最新新闻、"
            "财务数据（营收/利润/现金流/资产负债表）、估值（P/E、EV/EBITDA、FCF 收益率、DCF）"
            "以及技术面（均线、RSI、MACD、波动率、回撤）。请引用数据来源并区分数据与解读。"
        )
    return (
        f"Give me a full research report on {ticker}: current price, recent price action, "
        "latest news, financials (revenue/income/cash flow/balance sheet), valuation "
        "(P/E, EV/EBITDA, FCF yield, DCF) and technicals (MA, RSI, MACD, volatility, drawdown). "
        "Cite data sources and distinguish data from interpretation."
    )


def _print_report(report: StockResearchReport, language: str) -> None:
    label = {
        "en": {
            "title": "STOCKMIND RESEARCH REPORT",
            "generated_at": "GENERATED AT",
            "summary": "SUMMARY",
            "market": "MARKET ANALYSIS",
            "news": "NEWS ANALYSIS",
            "financial": "FINANCIAL ANALYSIS",
            "technical": "TECHNICAL ANALYSIS",
            "valuation": "VALUATION ANALYSIS",
            "risk": "RISK ANALYSIS",
            "bull": "BULL CASE",
            "bear": "BEAR CASE",
            "metrics": "KEY METRICS",
            "uncertainty": "UNCERTAINTY",
            "conclusion": "CONCLUSION",
        },
        "zh": {
            "title": "STOCKMIND 研究报告",
            "generated_at": "生成时间",
            "summary": "摘要",
            "market": "市场分析",
            "news": "新闻分析",
            "financial": "财务分析",
            "technical": "技术分析",
            "valuation": "估值分析",
            "risk": "风险分析",
            "bull": "看多理由",
            "bear": "看空理由",
            "metrics": "关键指标",
            "uncertainty": "不确定性",
            "conclusion": "结论",
        },
    }[language]

    print("=" * 70)
    print(f"{label['title']} — {report.ticker}")
    print("=" * 70)
    print(f"{label['generated_at']}: {format_local_time(report.generated_at)}")
    print(f"\n{label['summary']}\n  {report.summary}")
    for key in ("market", "news", "financial", "technical", "valuation", "risk"):
        print(f"\n{label[key]}\n  {getattr(report, f'{key}_analysis')}")
    print(f"\n{label['bull']}")
    for point in report.bull_case:
        print(f"  + {point}")
    print(f"\n{label['bear']}")
    for point in report.bear_case:
        print(f"  - {point}")
    print(f"\n{label['metrics']}")
    for key, value in report.key_metrics.items():
        print(f"  {key}: {value}")
    print(f"\n{label['uncertainty']}")
    for item in report.uncertainty:
        print(f"  ? {item}")
    print(f"\n{label['conclusion']}\n  {report.conclusion}")
    print("=" * 70)


def _extract_bars(state: dict) -> list[PriceBar]:
    """Pull historical price bars out of the collected tool observations."""
    for obs in reversed(state.get("observations", [])):
        if obs.get("tool") != "get_historical_prices":
            continue
        try:
            payload = json.loads(obs["result"])
            return [PriceBar(**bar) for bar in payload.get("bars", [])]
        except (TypeError, ValueError, KeyError):
            continue
    return []


def _render_chart(state: dict, ticker: str, language: str) -> None:
    bars = _extract_bars(state)
    if not bars:
        print("\n(no historical bars collected — chart skipped)")
        return
    closes = [bar.close for bar in bars]
    print(f"\nPRICE TREND (last {len(closes)} bars)")
    print(f"  {ascii_sparkline(closes)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = render_price_chart(
        bars, ticker, CHARTS_DIR / f"{ticker}_{stamp}.png", language=language
    )
    print(f"  Chart saved: {chart_path}")


def run_ticker(ticker: str, language: str, query: str | None = None) -> None:
    symbol = ticker.upper()
    question = query or _default_query(symbol, language)

    if _has_api_key():
        agent = StockMindAgent(language=language)
    else:
        agent = StockMindAgent(llm=build_demo_model(symbol, language), language=language)

    state = agent.run(question, ticker=symbol)
    report = state["final_output"]
    _print_report(report, language)
    _render_chart(state, symbol, language)

    print("\n--- TRACE ---")
    print(agent.tracer.summarize())


def main() -> None:
    parser = argparse.ArgumentParser(description="StockMind multi-stock research CLI")
    parser.add_argument(
        "--tickers",
        default=os.getenv("STOCKMIND_TICKERS", "NVDA"),
        help="Comma-separated tickers, e.g. NVDA,AMD,TSM",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default=os.getenv("STOCKMIND_LANGUAGE", "en"),
        help="Report language",
    )
    parser.add_argument("--query", default=None, help="Optional query override (use {ticker} placeholder)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    for ticker in tickers:
        query = args.query.format(ticker=ticker) if args.query else None
        run_ticker(ticker, args.lang, query)
        print()


if __name__ == "__main__":
    main()
