# StockMind

A **ReAct-based stock research agent** built with LangGraph, LangChain, Pydantic
and Python. The goal is not a stock chatbot, but an agent with a real runtime:
autonomous tool selection, explicit state management, deterministic
computation, observability and evaluation hooks.

> Language: **English** | [简体中文](README.zh-CN.md)

> **Current scope (Phase 1 + Phase 2):** the core ReAct loop, a full tool set
> (market, financial, news, technical, valuation), state management, iteration
> limits, tool error handling, structured final output, bilingual reports
> (English / 简体中文), multi-ticker runs, basic tracing and unit tests.
> Backtesting, portfolio optimisation and evaluation datasets are intentionally
> deferred to later phases.

---

## Architecture

```
User Query
   │
   ▼
┌─────────────┐   tool_calls?   ┌──────────┐
│    agent    │ ──────────────▶ │  tools   │
│ (LLM reason)│ ◀────────────── │ (execute)│
└─────────────┘    observation  └──────────┘
   │ no more tool calls /
   │ max iterations / error
   ▼
┌──────────────┐
│   finalize   │ ──▶ StockResearchReport (Pydantic)
└──────────────┘
```

The loop is implemented as a LangGraph `StateGraph` (not the prebuilt
`create_react_agent`) so that state, iteration guards and error semantics are
explicit and testable.

### Responsibilities

| Layer | Responsibility |
| --- | --- |
| LLM (`app/models`, `app/agents/prompts`) | Reasoning, tool selection, interpretation, summarisation |
| Tools (`app/tools`) | Data retrieval + deterministic computation |
| Analysis (`app/tools/analysis.py`) | Pure, side-effect-free financial/technical math |
| Providers (`app/tools/providers.py`) | Raw data retrieval (mock + yfinance + akshare) |
| Report assembly (`app/reporting`) | Deterministic, evidence-grounded report fallback |
| LangGraph (`app/graph`) | State, control flow, iteration/termination logic |
| Pydantic (`app/schemas`) | Structured data contracts |
| Tracing (`app/observability`) | Run/step/tool observability |

The LLM is **not** allowed to compute numbers (P/E, RSI, MACD, DCF, growth,
volatility, drawdown, etc.). Those belong in tools, which call the pure
functions in `app/tools/analysis.py`. Every tool result carries `source`,
`timestamp` and `data_period` so the final report can distinguish observed data
from interpretation.

---

## Tools (Phase 2)

| Category | Tools |
| --- | --- |
| Market | `get_stock_price`, `get_historical_prices` |
| Financial | `get_income_statement`, `get_balance_sheet`, `get_cash_flow`, `get_eps`, `get_revenue` |
| News | `search_company_news`, `search_market_news` |
| Technical | `calculate_ma`, `calculate_rsi`, `calculate_macd`, `calculate_volatility`, `calculate_drawdown` |
| Valuation | `calculate_pe`, `calculate_ev_ebitda`, `calculate_fcf_yield`, `calculate_dcf` |

---

## Directory layout

```
app/
├── utils.py                # time helpers
├── agents/
│   ├── react_agent.py      # StockMindAgent facade
│   └── prompts.py          # language-aware prompts
├── graph/
│   ├── state.py            # StockAgentState (TypedDict)
│   ├── workflow.py         # async LangGraph ReAct loop
│   └── checkpoint.py       # memory + SQLite checkpointer
├── models/
│   ├── providers.py        # OpenAI / Anthropic abstraction
│   └── demo.py             # offline scripted demo model
├── reporting/
│   └── assembler.py        # evidence-grounded report fallback
├── tools/
│   ├── base.py             # provenance + metric serialization
│   ├── analysis.py         # pure deterministic computation
│   ├── providers.py        # MarketDataProvider (mock + yfinance + akshare)
│   ├── market.py           # price + historical tools
│   ├── financial.py        # financial statement tools
│   ├── news.py             # company + market news tools
│   ├── technical.py        # technical indicator tools
│   └── valuation.py        # valuation tools
├── observability/
│   └── tracing.py          # TraceCollector
├── schemas/
│   └── report.py           # StockResearchReport
└── visualization/
    └── charts.py           # terminal sparkline + PNG charts
tests/
examples/
    research.py             # multi-stock, bilingual CLI
    nvda_research.py        # quick NVDA demo
    async_research.py       # async + checkpoint resume demo
promptlog/
    __main__.py             # prompt journal CLI
    journal.py              # storage + rendering
    prompts.jsonl / prompts.md
```

---

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in an API key if you have one
```

Optional: real market data via yfinance.

```bash
pip install -e ".[real-data]"
```

---

## Run the demo

Quick NVDA demo:

```bash
python examples/nvda_research.py --lang en
```

Multi-stock, bilingual:

```bash
python examples/research.py --tickers NVDA,AMD,TSM --lang zh
python examples/research.py --tickers AAPL --lang en
```

* With `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set, a real LLM drives each loop.
* Without a key, an offline scripted demo model drives the loop through the full
  tool set, for any ticker and language.
* Data defaults to the deterministic `mock` provider, which is **clearly
  labelled as simulated**. Set `STOCKMIND_DATA_PROVIDER=yfinance` (Yahoo) or
  `STOCKMIND_DATA_PROVIDER=akshare` (Eastmoney, China-friendly, ~15 min delayed)
  to fetch real data (requires network + optional dependency). akshare pulls US
  price/history from Sina, financial statements from Eastmoney and valuation
  from Baidu (best-effort parsing).

Each report shows a **precise generation time** (`generated_at`, ISO 8601 with
timezone), and a price/MA/volume/RSI/MACD **chart** with a dashed
**linear-extrapolation trend** (labelled *not a prediction*) is saved to
`charts/` (a terminal sparkline is also printed). The chart is generated with
matplotlib.

## Run the tests

```bash
pytest
```

---

## Async & checkpointing

The agent is async-first; sync `run`/`invoke` are thin wrappers over `arun`/
`ainvoke`.

```python
import asyncio
from app.agents.react_agent import StockMindAgent

async def main():
    agent = StockMindAgent()  # InMemorySaver by default
    report = await agent.ainvoke("Analyze NVDA", ticker="NVDA", thread_id="run-1")
    snapshot = await agent.aget_state("run-1")  # read persisted checkpoint

asyncio.run(main())
```

SQLite persistence (resumable across processes):

```python
from app.graph.checkpoint import build_checkpointer

async with build_checkpointer("sqlite", "checkpoints.sqlite") as saver:
    agent = StockMindAgent(checkpointer=saver)
    await agent.ainvoke("Analyze NVDA", ticker="NVDA", thread_id="run-1")
    await agent.ainvoke("Follow-up?", ticker="NVDA", thread_id="run-1")  # resumes
```

Run the bundled demo:

```bash
python examples/async_research.py --ticker NVDA --lang en
```

---

## Prompt journal

A standalone prompt recorder lives in [`promptlog/`](promptlog/):

```bash
python -m promptlog add "your prompt" --tag finance
python -m promptlog list --limit 20
python -m promptlog export
```

See [promptlog/README.md](promptlog/README.md).

---

## Configuration

See [.env.example](.env.example). Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `STOCKMIND_LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `STOCKMIND_LLM_MODEL` | `gpt-4o-mini` | model name |
| `STOCKMIND_DATA_PROVIDER` | `mock` | `mock`, `yfinance` or `akshare` |
| `STOCKMIND_LANGUAGE` | `en` | `en` or `zh` |
| `STOCKMIND_MAX_ITERATIONS` | `8` | ReAct iteration cap |
| `STOCKMIND_TOOL_TIMEOUT_SECONDS` | `15` | per-tool timeout |
| `STOCKMIND_TOOL_RETRIES` | `1` | retries after a tool failure |
| `STOCKMIND_RETRY_BACKOFF_SECONDS` | `0.5` | initial backoff for tool retries |
| `STOCKMIND_RETRY_MAX_BACKOFF_SECONDS` | `8` | max backoff for tool retries |
| `STOCKMIND_RETRY_JITTER` | `0.1` | jitter fraction for backoff |
| `STOCKMIND_TRACE_DIR` | *(empty)* | JSONL trace output directory |

API keys are read from the environment by the underlying SDKs and are never
stored in code.

---

## Design notes

* **State over globals** — every node returns partial `StockAgentState` updates;
  `messages` uses LangGraph's `add_messages` reducer for correct accumulation.
* **Data retrieval vs. computation** — providers fetch raw data; pure functions
  in `analysis.py` do the math; tools compose the two. The LLM only interprets.
* **Reliability guards** — `max_iterations`, per-tool timeout (`asyncio.wait_for`),
  retry policy, duplicate-tool-call detection, and graceful degradation: a
  failed tool becomes an error observation, never a crash.
* **Structured output with fallback** — `finalize` prefers
  `llm.with_structured_output(StockResearchReport)`, then JSON parsing, then a
  deterministic evidence-grounded assembly from tool observations, so the
  offline demo still quotes real numbers and sources.
* **Precise timing** — every report carries `generated_at` (ISO 8601,
  microsecond precision, with timezone).
* **Bilingual** — `language` is part of the run config/state; prompts carry a
  language directive while the structured-output schema stays stable.
* **Provenance** — mock data is explicitly labelled; the system prompt forbids
  fabrication and demands sources.

---

## Roadmap

- [x] **Phase 1** — ReAct loop + market/news tools.
- [x] **Phase 2** — financial, technical, news and valuation tools.
- [x] **Phase 3** — async execution, checkpointing, retries/backoff hardening.
- [ ] **Phase 4** — richer tracing, evaluation datasets, cost tracking.
- [ ] **Phase 5** — historical evaluation / backtesting of signals.
