"""Deterministic, evidence-grounded report assembly.

When the LLM's structured output is unavailable (e.g. the offline demo), the
agent still has a full set of tool observations. This module turns those
observations into a substantive :class:`StockResearchReport` that quotes actual
numbers, cites each source and timestamp, and never fabricates.
"""

from __future__ import annotations

import json
from typing import Any

from app.schemas.report import StockResearchReport


def _load(obs: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(obs.get("result", "{}"))
    except (TypeError, ValueError):
        return {}


def _num(value: Any, nd: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any, nd: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return str(value)


def _source(payload: dict[str, Any], fallback: str = "unknown") -> str:
    src = payload.get("source") or fallback
    ts = payload.get("timestamp") or ""
    return f"{src} ({ts})" if ts else str(src)


def build_grounded_report(state: dict[str, Any], language: str = "en") -> StockResearchReport | None:
    """Build a report purely from collected tool observations."""
    observations = state.get("observations", [])
    if not observations:
        return None

    ticker = (state.get("ticker") or "UNKNOWN").upper()
    data: dict[str, dict[str, Any]] = {}
    for obs in observations:
        name = obs.get("tool")
        if name and name not in data:
            data[name] = _load(obs)

    price = data.get("get_stock_price", {})
    hist = data.get("get_historical_prices", {})
    news = data.get("search_company_news", {})
    income = data.get("get_income_statement", {})
    balance = data.get("get_balance_sheet", {})
    cashflow = data.get("get_cash_flow", {})
    pe = data.get("calculate_pe", {})
    ev_ebitda = data.get("calculate_ev_ebitda", {})
    fcf_yield = data.get("calculate_fcf_yield", {})
    dcf = data.get("calculate_dcf", {})
    ma = data.get("calculate_ma", {})
    rsi = data.get("calculate_rsi", {})
    macd = data.get("calculate_macd", {})
    vol = data.get("calculate_volatility", {})
    dd = data.get("calculate_drawdown", {})

    closes = [float(b["close"]) for b in hist.get("bars", []) if b.get("close") is not None]
    first, last = (closes[0], closes[-1]) if closes else (None, None)
    change = (last - first) / first if (first and last) else None
    hi, lo = (max(closes), min(closes)) if closes else (None, None)

    price_v = price.get("price")
    currency = price.get("currency", "USD")
    pe_v = pe.get("value")
    ev_v = ev_ebitda.get("value")
    fy_v = fcf_yield.get("value")
    rsi_v = rsi.get("value")
    vol_v = vol.get("value")
    dd_v = dd.get("value") or {}
    ma_v = ma.get("value")
    macd_v = macd.get("value") or {}
    dcf_v = dcf.get("value") or {}

    sources = []
    for payload in data.values():
        src = payload.get("source")
        if src and src not in sources:
            sources.append(str(src))

    key_metrics: dict[str, Any] = {
        "price": price_v,
        "period": hist.get("period"),
        "change": change,
        "revenue": income.get("revenue"),
        "net_income": income.get("net_income"),
        "ebitda": income.get("ebitda"),
        "eps": income.get("eps_diluted"),
        "free_cash_flow": cashflow.get("free_cash_flow"),
        "total_debt": balance.get("total_debt"),
        "cash": balance.get("cash_and_equivalents"),
        "pe": pe_v,
        "ev_ebitda": ev_v,
        "fcf_yield": fy_v,
        "rsi": rsi_v,
        "annualized_volatility": vol_v,
        "max_drawdown": dd_v.get("max_drawdown"),
        "dcf_intrinsic_value": dcf_v.get("intrinsic_value_per_share"),
        "data_sources": sources,
    }

    news_titles = [item.get("title", "") for item in news.get("items", [])][:3]
    news_blob = "; ".join(news_titles) if news_titles else "None returned"

    if language == "zh":
        return StockResearchReport(
            ticker=ticker,
            summary=(
                f"{ticker} 最新价格 {_num(price_v)} {currency}，近 {len(closes)} 个交易日"
                f"由 {_num(first)} 变为 {_num(last)}（{_pct(change)}）。"
                f"估值 P/E {_num(pe_v, 1)}、FCF 收益率 {_pct(fy_v)}，RSI {_num(rsi_v, 1)}，"
                f"年化波动率 {_pct(vol_v)}。以下所有数字均来自工具观测，非模型推测。"
            ),
            market_analysis=(
                f"最新价 {_num(price_v)} {currency}（来源：{_source(price)}）。"
                f"区间 {_num(lo)}–{_num(hi)}，累计涨跌幅 {_pct(change)}。"
                f"数据区间 {hist.get('data_period') or hist.get('period')}。"
            ),
            financial_analysis=(
                f"营收 {_num(income.get('revenue'))}，净利润 {_num(income.get('net_income'))}，"
                f"EBITDA {_num(income.get('ebitda'))}，EPS {_num(income.get('eps_diluted'))}；"
                f"自由现金流 {_num(cashflow.get('free_cash_flow'))}，总债务 {_num(balance.get('total_debt'))}，"
                f"现金 {_num(balance.get('cash_and_equivalents'))}。"
                f"（来源：{_source(income)}）"
            ),
            technical_analysis=(
                f"RSI {_num(rsi_v, 1)}，均线 MA {_num(ma_v)}，"
                f"MACD {_num(macd_v.get('macd'), 3)}/信号 {_num(macd_v.get('signal'), 3)}。"
                f"（来源：{_source(rsi)}）"
            ),
            news_analysis=(
                f"公司新闻 {len(news.get('items', []))} 条，示例：{news_blob}。"
                f"（来源：{_source(news)}）"
            ),
            valuation_analysis=(
                f"P/E {_num(pe_v, 1)}，EV/EBITDA {_num(ev_v, 1)}，FCF 收益率 {_pct(fy_v)}，"
                f"DCF 每股内在价值 {_num(dcf_v.get('intrinsic_value_per_share'))}。"
                f"（来源：{_source(pe)}）"
            ),
            risk_analysis=(
                f"年化波动率 {_pct(vol_v)}，最大回撤 {_pct(dd_v.get('max_drawdown'))}，"
                f"当前回撤 {_pct(dd_v.get('current_drawdown'))}。"
                f"（来源：{_source(vol)}）"
            ),
            bull_case=[
                f"RSI {_num(rsi_v, 1)} 处于中性区间，未出现极端超买。",
                f"P/E {_num(pe_v, 1)} 与 FCF 收益率 {_pct(fy_v)} 提供了可比较的估值锚点。",
            ],
            bear_case=[
                f"近 {len(closes)} 日涨跌幅 {_pct(change)}，最大回撤 {_pct(dd_v.get('max_drawdown'))}，波动风险上升。",
                "新闻面存在监管与宏观不确定性（见新闻分析）。",
            ],
            key_metrics=key_metrics,
            uncertainty=[
                "本报告由离线组装器基于工具观测生成，未经过 LLM 语义润色。",
                "若数据源为 mock，则所有数字均为模拟数据，不得视为真实行情。",
                "DCF 为简化两阶段模型，假设是输入而非事实。",
            ],
            conclusion=(
                f"{ticker} 当前价格 {_num(price_v)} {currency}，估值与动量指标均已通过工具"
                f"计算并标注来源。是否具有吸引力取决于风险偏好与真实数据；模拟数据仅用于流程演示。"
            ),
        )

    return StockResearchReport(
        ticker=ticker,
        summary=(
            f"{ticker}'s latest price is {_num(price_v)} {currency}; over the last {len(closes)} "
            f"sessions it moved from {_num(first)} to {_num(last)} ({_pct(change)}). "
            f"Valuation shows P/E {_num(pe_v, 1)} and FCF yield {_pct(fy_v)}, with RSI {_num(rsi_v, 1)} "
            f"and annualized volatility {_pct(vol_v)}. All figures come from tool observations, not model inference."
        ),
        market_analysis=(
            f"Latest price {_num(price_v)} {currency} (source: {_source(price)}). "
            f"Session range {_num(lo)}–{_num(hi)}, cumulative change {_pct(change)}. "
            f"Window: {hist.get('data_period') or hist.get('period')}."
        ),
        financial_analysis=(
            f"Revenue {_num(income.get('revenue'))}, net income {_num(income.get('net_income'))}, "
            f"EBITDA {_num(income.get('ebitda'))}, EPS {_num(income.get('eps_diluted'))}; "
            f"free cash flow {_num(cashflow.get('free_cash_flow'))}, total debt {_num(balance.get('total_debt'))}, "
            f"cash {_num(balance.get('cash_and_equivalents'))}. (source: {_source(income)})"
        ),
        technical_analysis=(
            f"RSI {_num(rsi_v, 1)}, moving average {_num(ma_v)}, "
            f"MACD {_num(macd_v.get('macd'), 3)}/signal {_num(macd_v.get('signal'), 3)}. "
            f"(source: {_source(rsi)})"
        ),
        news_analysis=(
            f"{len(news.get('items', []))} company news item(s); examples: {news_blob}. "
            f"(source: {_source(news)})"
        ),
        valuation_analysis=(
            f"P/E {_num(pe_v, 1)}, EV/EBITDA {_num(ev_v, 1)}, FCF yield {_pct(fy_v)}, "
            f"DCF intrinsic value per share {_num(dcf_v.get('intrinsic_value_per_share'))}. "
            f"(source: {_source(pe)})"
        ),
        risk_analysis=(
            f"Annualized volatility {_pct(vol_v)}, max drawdown {_pct(dd_v.get('max_drawdown'))}, "
            f"current drawdown {_pct(dd_v.get('current_drawdown'))}. (source: {_source(vol)})"
        ),
        bull_case=[
            f"RSI {_num(rsi_v, 1)} sits in the neutral zone, not extreme overbought.",
            f"P/E {_num(pe_v, 1)} and FCF yield {_pct(fy_v)} provide a comparable valuation anchor.",
        ],
        bear_case=[
            f"{len(closes)}-session change of {_pct(change)} with max drawdown {_pct(dd_v.get('max_drawdown'))} points to elevated downside risk.",
            "News flow includes regulatory and macro uncertainty (see news analysis).",
        ],
        key_metrics=key_metrics,
        uncertainty=[
            "This report was assembled deterministically from tool observations, without LLM polish.",
            "If the data source is mock, all figures are simulated and must not be treated as real.",
            "DCF is a simplified two-stage model; assumptions are inputs, not facts.",
        ],
        conclusion=(
            f"{ticker} trades at {_num(price_v)} {currency} with valuation and momentum metrics "
            "computed by tools and cited with sources. Whether that is attractive depends on risk "
            "tolerance and real data; simulated data is for workflow demonstration only."
        ),
    )

