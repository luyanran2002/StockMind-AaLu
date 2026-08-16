# StockMind

一个基于 **ReAct（推理 + 行动）** 的股票研究 Agent，使用 LangGraph、LangChain、
Pydantic 与 Python 构建。目标不是做一个「股票聊天机器人」，而是构建一个拥有
真正运行时的 Agent：自主工具选择、显式状态管理、确定性计算、可观测性与评估能力。

> **当前范围（Phase 1 + Phase 2）：** 核心 ReAct 循环、完整工具集
> （市场、财务、新闻、技术面、估值）、状态管理、迭代上限、工具错误处理、
> 结构化最终输出、双语报告（English / 简体中文）、多股票运行、基础追踪与单元测试。
> 回测、组合优化与评估数据集有意留到后续阶段。

> 语言：**简体中文** | [English](README.md)

---

## 架构

```
用户问题
   │
   ▼
┌─────────────┐  有工具调用?  ┌──────────┐
│    agent    │ ────────────▶ │  tools   │
│ (LLM 推理)  │ ◀──────────── │ (执行)   │
└─────────────┘   观察结果    └──────────┘
   │ 无更多工具调用 /
   │ 达到迭代上限 / 出错
   ▼
┌──────────────┐
│   finalize   │ ──▶ StockResearchReport (Pydantic)
└──────────────┘
```

循环使用 LangGraph `StateGraph` 手工实现（而非预置的 `create_react_agent`），
以便显式、可测试地控制状态、迭代上限与错误语义。

### 职责划分

| 层 | 职责 |
| --- | --- |
| LLM（`app/models`、`app/agents/prompts`） | 推理、工具选择、解读、总结 |
| 工具（`app/tools`） | 数据获取 + 确定性计算 |
| 分析（`app/tools/analysis.py`） | 无副作用、纯函数的财务/技术数学计算 |
| 数据源（`app/tools/providers.py`） | 原始数据获取（mock + yfinance） |
| 报告组装（`app/reporting`） | 确定性、基于证据的报告兜底 |
| LangGraph（`app/graph`） | 状态、控制流、迭代/终止逻辑 |
| Pydantic（`app/schemas`） | 结构化数据契约 |
| 追踪（`app/observability`） | 运行/步骤/工具可观测性 |

LLM **不允许**自行计算数字（P/E、RSI、MACD、DCF、增长率、波动率、回撤等）。
这些必须由工具完成，工具调用 `app/tools/analysis.py` 中的纯函数。每个工具结果
都携带 `source`、`timestamp` 与 `data_period`，使最终报告能够区分「观测数据」
与「解读」。

---

## 工具（Phase 2）

| 类别 | 工具 |
| --- | --- |
| 市场 | `get_stock_price`、`get_historical_prices` |
| 财务 | `get_income_statement`、`get_balance_sheet`、`get_cash_flow`、`get_eps`、`get_revenue` |
| 新闻 | `search_company_news`、`search_market_news` |
| 技术面 | `calculate_ma`、`calculate_rsi`、`calculate_macd`、`calculate_volatility`、`calculate_drawdown` |
| 估值 | `calculate_pe`、`calculate_ev_ebitda`、`calculate_fcf_yield`、`calculate_dcf` |

---

## 目录结构

```
app/
├── utils.py                # 时间工具
├── agents/
│   ├── react_agent.py      # StockMindAgent 门面
│   └── prompts.py          # 支持多语言的提示词
├── graph/
│   ├── state.py            # StockAgentState（TypedDict）
│   ├── workflow.py         # 异步 LangGraph ReAct 循环
│   └── checkpoint.py       # memory + SQLite checkpointer
├── models/
│   ├── providers.py        # OpenAI / Anthropic 抽象
│   └── demo.py             # 离线脚本化演示模型
├── reporting/
│   └── assembler.py        # 基于证据的报告兜底
├── tools/
│   ├── base.py             # 溯源 + 指标序列化
│   ├── analysis.py         # 纯确定性计算
│   ├── providers.py        # MarketDataProvider（mock + yfinance）
│   ├── market.py           # 价格 + 历史工具
│   ├── financial.py        # 财务报表工具
│   ├── news.py             # 公司 + 市场新闻工具
│   ├── technical.py        # 技术指标工具
│   └── valuation.py        # 估值工具
├── observability/
│   └── tracing.py          # TraceCollector
├── schemas/
│   └── report.py           # StockResearchReport
└── visualization/
    └── charts.py           # 终端走势线 + PNG 图表
tests/
examples/
    research.py             # 多股票、双语 CLI
    nvda_research.py        # NVDA 快速演示
    async_research.py       # 异步 + checkpoint 续跑演示
promptlog/
    __main__.py             # prompt 记录 CLI
    journal.py              # 存储与渲染
    prompts.jsonl / prompts.md
```

---

## 安装

需要 Python 3.11+。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 如有 API key 请填入
```

可选：通过 yfinance 获取真实行情数据。

```bash
pip install -e ".[real-data]"
```

---

## 运行演示

NVDA 快速演示：

```bash
python examples/nvda_research.py --lang zh
```

多股票、双语：

```bash
python examples/research.py --tickers NVDA,AMD,TSM --lang zh
python examples/research.py --tickers AAPL --lang en
```

* 设置了 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY` 时，由真实 LLM 驱动每一次循环。
* 未设置 key 时，由离线脚本化演示模型驱动完整工具链，支持任意股票与语言。
* 数据默认使用确定性的 `mock` 数据源，并**明确标注为模拟数据**。设置
  `STOCKMIND_DATA_PROVIDER=yfinance` 可获取真实数据（需要网络 + 可选依赖）。

每份报告都会显示**精确的生成时间**（`generated_at`，ISO 8601 含时区），并生成
价格/均线/成交量/RSI/MACD **图表**保存到 `charts/`（终端同时打印一条走势线）。
图表使用 matplotlib 生成。

## 运行测试

```bash
pytest
```

---

## 异步与 checkpoint

Agent 以异步为优先；同步的 `run`/`invoke` 只是 `arun`/`ainvoke` 的薄封装。

```python
import asyncio
from app.agents.react_agent import StockMindAgent

async def main():
    agent = StockMindAgent()  # 默认使用 InMemorySaver
    report = await agent.ainvoke("分析 NVDA", ticker="NVDA", thread_id="run-1")
    snapshot = await agent.aget_state("run-1")  # 读取已持久化的 checkpoint

asyncio.run(main())
```

SQLite 持久化（跨进程可恢复）：

```python
from app.graph.checkpoint import build_checkpointer

async with build_checkpointer("sqlite", "checkpoints.sqlite") as saver:
    agent = StockMindAgent(checkpointer=saver)
    await agent.ainvoke("分析 NVDA", ticker="NVDA", thread_id="run-1")
    await agent.ainvoke("还有什么补充吗？", ticker="NVDA", thread_id="run-1")  # 续跑
```

运行内置演示：

```bash
python examples/async_research.py --ticker NVDA --lang zh
```

---

## Prompt 记录

独立的 prompt 记录框架位于 [`promptlog/`](promptlog/)：

```bash
python -m promptlog add "你的 prompt" --tag finance
python -m promptlog list --limit 20
python -m promptlog export
```

详见 [promptlog/README.md](promptlog/README.md)。

---

## 配置

参见 [.env.example](.env.example)。关键变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `STOCKMIND_LLM_PROVIDER` | `openai` | `openai` 或 `anthropic` |
| `STOCKMIND_LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `STOCKMIND_DATA_PROVIDER` | `mock` | `mock` 或 `yfinance` |
| `STOCKMIND_LANGUAGE` | `en` | `en` 或 `zh` |
| `STOCKMIND_MAX_ITERATIONS` | `8` | ReAct 迭代上限 |
| `STOCKMIND_TOOL_TIMEOUT_SECONDS` | `15` | 单工具超时 |
| `STOCKMIND_TOOL_RETRIES` | `1` | 工具失败后的重试次数 |
| `STOCKMIND_RETRY_BACKOFF_SECONDS` | `0.5` | 工具重试的初始退避时间 |
| `STOCKMIND_RETRY_MAX_BACKOFF_SECONDS` | `8` | 工具重试的最大退避时间 |
| `STOCKMIND_RETRY_JITTER` | `0.1` | 退避抖动比例 |
| `STOCKMIND_TRACE_DIR` | *(空)* | JSONL 追踪输出目录 |

API key 由底层 SDK 从环境变量读取，绝不写入代码。

---

## 设计说明

* **状态优于全局变量** — 每个节点返回 `StockAgentState` 的部分更新；
  `messages` 使用 LangGraph 的 `add_messages` reducer 正确累积。
* **数据获取与计算分离** — 数据源获取原始数据；`analysis.py` 中的纯函数负责计算；
  工具把两者组合起来。LLM 只做解读。
* **可靠性保障** — `max_iterations`、单工具超时（`asyncio.wait_for`）、重试策略、重复工具调用
  检测与优雅降级：失败的工具会变成一条错误观察，而不会导致崩溃。
* **结构化输出 + 兜底** — `finalize` 优先使用
  `llm.with_structured_output(StockResearchReport)`，其次尝试 JSON 解析，最后退化为
  基于工具观测的确定性组装报告；因此离线演示也能引用真实数字与来源。
* **精确时间** — 每份报告都携带 `generated_at`（ISO 8601，微秒精度，含时区）。
* **双语** — `language` 是运行配置/状态的一部分；提示词携带语言指令，而结构化
  输出的 schema 保持稳定。
* **溯源** — mock 数据被明确标注；系统提示禁止编造数据并要求注明来源。

---

## 路线图

- [x] **Phase 1** — ReAct 循环 + 市场/新闻工具。
- [x] **Phase 2** — 财务、技术面、新闻与估值工具。
- [x] **Phase 3** — 异步执行、检查点、重试/退避加固。
- [ ] **Phase 4** — 更丰富的追踪、评估数据集、成本统计。
- [ ] **Phase 5** — 历史评估 / 信号回测。
