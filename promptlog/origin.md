# Role

你是一名资深 AI Agent Engineer / MLE，擅长 Python、LangChain、LangGraph、ReAct Agent、LLM Tool Calling、金融数据分析、Agent Evaluation 和 Production Agent Systems。

我想从零开发一个个人项目：

**StockMind — A ReAct-based Stock Research Agent**

项目目标不是简单做一个“股票聊天机器人”，而是构建一个真正具有 Agent Runtime、Tool Calling、State Management、Observability 和 Evaluation 能力的股票研究 Agent。

---

# 1. 核心架构

请使用 **ReAct（Reasoning + Acting）架构**作为 Agent 的核心执行模式。

基本循环：

```text
User Question
      ↓
   ReAct Agent
      ↓
  Thought / Decision
      ↓
   Tool Call
      ↓
 Tool Observation
      ↓
  ReAct Agent
      ↓
   Tool Call / Final Answer
```

Agent 不应该仅依赖 LLM 的已有知识回答股票问题，而应该主动决定什么时候需要调用工具。

例如用户：

> Analyze NVDA and tell me whether its current valuation looks expensive.

Agent 应该能够自主决定调用：

```text
get_stock_price()
get_financials()
get_earnings()
calculate_valuation()
search_news()
calculate_technical_indicators()
```

然后根据工具返回的数据继续推理。

---

# 2. 技术栈

优先使用：

* Python 3.11+
* LangChain
* LangGraph
* Pydantic
* pandas
* numpy
* asyncio
* SQLite
* FastAPI（如果需要 API）
* Streamlit 或 React（如果需要前端）

LLM provider 应该通过 abstraction 设计，不要把 OpenAI API 写死在业务逻辑中。

优先支持：

```text
OpenAI
Anthropic
```

未来可以扩展其他 provider。

---

# 3. Agent 的核心 State

使用 LangGraph StateGraph 管理 Agent state。

设计一个清晰的 `StockAgentState`，至少包含：

```python
class StockAgentState(TypedDict):
    user_query: str
    ticker: str | None
    messages: list
    tool_calls: list
    observations: list
    market_data: dict
    financial_data: dict
    news_data: list
    technical_data: dict
    valuation_data: dict
    risk_analysis: dict
    final_report: str | None
    errors: list
    metadata: dict
```

不要让不同 Agent/node 通过全局变量共享数据。

所有重要执行状态都应该显式存在于 State 中。

---

# 4. ReAct Agent

实现一个 ReAct-style Agent。

Agent 应该遵循：

```text
1. Understand the user query
2. Identify missing information
3. Select the appropriate tool
4. Execute the tool
5. Inspect the observation
6. Decide whether additional tools are needed
7. Repeat until sufficient evidence is collected
8. Produce the final answer
```

Agent 必须支持：

* multi-step reasoning
* tool calling
* tool errors
* retries
* max iteration limit
* timeout
* early termination
* structured final output

不要让 Agent 无限循环。

例如：

```python
max_iterations = 8
```

如果超过最大 iteration，必须安全终止并返回当前能够支持的结果。

---

# 5. Tools

设计一个统一 Tool abstraction。

至少实现以下工具：

### Market Data

```text
get_stock_price
get_historical_prices
get_volume
get_market_cap
```

### Financial Data

```text
get_income_statement
get_balance_sheet
get_cash_flow
get_eps
get_revenue
```

### News

```text
search_company_news
search_market_news
```

### Technical Analysis

```text
calculate_ma
calculate_rsi
calculate_macd
calculate_volatility
calculate_drawdown
```

### Valuation

```text
calculate_pe
calculate_ev_ebitda
calculate_fcf_yield
calculate_dcf
```

### Risk

```text
calculate_volatility
calculate_max_drawdown
calculate_beta
```

---

# 6. 非常重要：LLM 不负责确定性计算

不要让 LLM 自己计算：

```text
P/E
RSI
Moving Average
DCF
Revenue Growth
Sharpe Ratio
Volatility
Max Drawdown
```

这些必须由 Python / pandas / numpy 等 deterministic tools 完成。

架构应该是：

```text
Raw Data
   ↓
Python Tool
   ↓
Deterministic Calculation
   ↓
Structured Result
   ↓
LLM Interpretation
```

LLM 的职责主要是：

```text
reasoning
tool selection
comparison
interpretation
summarization
report generation
```

而不是承担精确计算。

---

# 7. 数据可靠性

这是一个金融研究项目，因此不要允许 Agent 无依据地编造数据。

所有重要金融事实必须尽可能来自工具。

每个数据结果应该记录：

```python
{
    "value": ...,
    "source": ...,
    "timestamp": ...,
    "data_period": ...,
}
```

最终报告应该能够区分：

```text
Observed Data
Calculated Metrics
Agent Interpretation
Uncertainty
```

不要把 LLM 的推测包装成事实。

---

# 8. Agent Workflow

第一版使用一个核心 ReAct Agent，而不是一开始就做十几个 Agent。

推荐：

```text
                 User
                  ↓
             Query Parser
                  ↓
              ReAct Agent
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
     Market    Financial    News
      Tool       Tool       Tool
       │          │          │
       └──────────┼──────────┘
                  ↓
            Technical Tool
                  ↓
            Valuation Tool
                  ↓
              Risk Tool
                  ↓
            ReAct Agent
                  ↓
            Final Report
```

Agent 应该动态决定工具调用顺序，而不是简单硬编码：

```python
market()
financial()
news()
technical()
valuation()
risk()
```

---

# 9. Example

用户输入：

> Why has NVDA fallen recently, and is the current valuation attractive?

Agent 应该可能产生类似这样的 execution：

```text
User
 ↓
ReAct Agent
 ↓
Need recent price movement
 ↓
get_historical_prices(NVDA)
 ↓
Observation
 ↓
Need recent catalysts
 ↓
search_company_news(NVDA)
 ↓
Observation
 ↓
Need financial context
 ↓
get_income_statement(NVDA)
 ↓
Observation
 ↓
Need valuation
 ↓
calculate_pe(NVDA)
calculate_fcf_yield(NVDA)
 ↓
Observation
 ↓
Need risk assessment
 ↓
calculate_volatility(NVDA)
calculate_drawdown(NVDA)
 ↓
Observation
 ↓
Final Analysis
```

注意：以上只是示例，Agent 实际应该根据问题动态选择工具。

---

# 10. Final Answer Schema

最终不要让 Agent 返回一大段没有结构的文本。

定义 Pydantic schema：

```python
class StockResearchReport(BaseModel):
    ticker: str
    summary: str
    market_analysis: str
    financial_analysis: str
    technical_analysis: str
    news_analysis: str
    valuation_analysis: str
    risk_analysis: str
    bull_case: list[str]
    bear_case: list[str]
    key_metrics: dict
    uncertainty: list[str]
    conclusion: str
```

最终输出必须符合这个 schema。

---

# 11. ReAct 的安全限制

实现：

```text
max_iterations
timeout
tool_timeout
retry_policy
duplicate_tool_call_detection
error_handling
```

如果工具失败：

```text
Tool
 ↓
Error
 ↓
Retry
 ↓
Still failed?
 ↓
Alternative tool / graceful degradation
```

不要因为单个数据源失败导致整个 Agent crash。

---

# 12. Observability

给 Agent 增加 tracing。

记录：

```text
run_id
timestamp
user_query
agent_step
tool_name
tool_input
tool_output
latency
token_usage
error
final_status
```

最终可以看到：

```text
Run #1023

Total latency: 4.82s

Steps:
1. get_historical_prices    0.42s
2. search_company_news      0.81s
3. get_financials           0.51s
4. calculate_valuation      0.03s
5. calculate_risk           0.02s
6. final_llm_call           1.82s
```

这部分非常重要，因为这个项目不仅要展示 AI，还要展示 production-oriented engineering。

---

# 13. Evaluation

不要只通过人工测试判断 Agent 好不好。

建立一个 evaluation dataset，例如：

```text
questions.json
```

包含：

```text
Why did NVDA decline recently?
Is AMD expensive relative to NVDA?
What are the major risks for TSM?
Compare AVGO and NVDA.
```

评价：

```text
Tool Selection Accuracy
Data Grounding
Numerical Accuracy
Answer Quality
Task Success Rate
Latency
Token Usage
Cost
```

最终输出：

```text
Task Success Rate: 87%
Tool Selection Accuracy: 91%
Numerical Accuracy: 98%
Average Latency: 4.1s
Average Tokens: 2,134
```

---

# 14. Backtesting / Historical Evaluation

第二阶段加入历史评估。

例如：

```text
At date T
      ↓
Agent analyzes stock
      ↓
Agent produces:
Bullish / Neutral / Bearish
      ↓
Evaluate after:
5 days
20 days
60 days
```

记录：

```text
prediction
confidence
actual_return
max_drawdown
```

不要声称 Agent 可以预测未来股票价格。

目标是研究：

> Whether the Agent's research process and signals have historically contained useful information.

---

# 15. Project Structure

建议：

```text
stockmind/
│
├── app/
│   ├── agents/
│   │   ├── react_agent.py
│   │   └── prompts.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   └── workflow.py
│   │
│   ├── tools/
│   │   ├── market.py
│   │   ├── financial.py
│   │   ├── news.py
│   │   ├── technical.py
│   │   ├── valuation.py
│   │   └── risk.py
│   │
│   ├── models/
│   │   └── providers.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── datasets/
│   │
│   ├── observability/
│   │   └── tracing.py
│   │
│   └── schemas/
│       └── report.py
│
├── tests/
├── examples/
├── README.md
├── pyproject.toml
└── .env.example
```

---

# 16. Development Strategy

不要一次性实现整个系统。

严格按照以下阶段开发：

### Phase 1

实现：

```text
LLM
↓
ReAct Agent
↓
2–3 Tools
↓
Final Answer
```

首先证明 Agent 能够自主调用工具。

### Phase 2

增加：

```text
Financial Tools
Technical Tools
News Tools
Valuation Tools
```

### Phase 3

加入：

```text
LangGraph State
Retry
Timeout
Checkpoint
Async
```

### Phase 4

加入：

```text
Tracing
Evaluation
Cost Tracking
```

### Phase 5

加入：

```text
Historical Evaluation
Backtesting
```

---

# 17. Coding Requirements

代码必须：

* 使用 type hints
* 使用 Pydantic 管理 structured data
* 避免 global state
* 模块之间低耦合
* Tool 与 Agent 分离
* 数据获取与数据分析分离
* LLM reasoning 与 deterministic computation 分离
* 为核心模块编写 unit tests
* 对外部 API 使用 timeout 和 retry
* 使用 environment variables 管理 API keys
* 不把 secrets 写进代码
* 提供清晰 README
* 每完成一个阶段都确保项目可以运行

不要为了“看起来像 LangChain”而过度抽象。

优先保证：

```text
Correctness
Reliability
Testability
Observability
```

---

# 18. 最重要的设计原则

这个项目的核心不是：

> “让 LLM 给出股票买卖建议。”

而是：

> **构建一个能够自主决定研究步骤、调用外部工具、验证数据、进行确定性计算、维护状态并最终生成 evidence-grounded financial research report 的 ReAct Agent。**

因此请始终遵循：

```text
LLM → Reasoning / Planning / Interpretation

Tools → Data Retrieval / Deterministic Computation

LangGraph → State / Control Flow / Execution

Python → Numerical Analysis

Pydantic → Structured Data

Tracing → Observability

Evaluation → Measurement
```

---

# 19. 你的第一项任务

现在不要直接生成整个项目。

首先完成 **Phase 1**：

```text
User
 ↓
ReAct Agent
 ↓
get_stock_price
 ↓
get_historical_prices
 ↓
search_company_news
 ↓
Final Answer
```

要求：

1. 创建完整项目目录。
2. 创建 `StockAgentState`。
3. 创建 3 个基础 tools。
4. 使用 LangGraph 构建 ReAct execution loop。
5. 实现 max iteration 和 tool error handling。
6. 使用 Pydantic 定义最终输出。
7. 添加最基本的 tracing/logging。
8. 编写 unit tests。
9. 创建一个可以直接运行的 `examples/nvda_research.py`。
10. README 解释整个 Agent architecture。

**不要在 Phase 1 实现 valuation、DCF、portfolio optimization 或 backtesting。**

完成 Phase 1 后，再根据实际代码结构决定 Phase 2 的架构。

在写代码之前，先解释你准备采用的 architecture 和关键设计选择，然后开始实现。
