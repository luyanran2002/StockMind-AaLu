"""Pure, deterministic financial computation.

These functions are deliberately free of I/O: they take raw numbers and return
computed values. The LLM never performs these calculations; tools call these
functions and attach provenance to the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_ma(closes: list[float], window: int) -> float | None:
    """Latest simple moving average."""
    if not closes or window <= 0 or len(closes) < window:
        return None
    return float(np.mean(closes[-window:]))


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Latest Relative Strength Index (Wilder smoothing)."""
    if len(closes) < period + 1 or period <= 0:
        return None
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def compute_rsi_series(closes: list[float], period: int = 14) -> list[float | None]:
    """Full RSI series (Wilder smoothing); ``None`` before enough history exists."""
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= period or period <= 0:
        return result
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())

    def _rsi(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0
        return float(100 - 100 / (1 + gain / loss))

    result[period] = _rsi(avg_gain, avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i + 1] = _rsi(avg_gain, avg_loss)
    return result


def _ema(values: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def compute_macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, float] | None:
    """Latest MACD line, signal line and histogram."""
    if len(closes) < slow + signal:
        return None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    return {
        "macd": round(macd_line[-1], 6),
        "signal": round(signal_line[-1], 6),
        "histogram": round(macd_line[-1] - signal_line[-1], 6),
    }


def compute_macd_series(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """Full MACD line, signal line and histogram series."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def compute_annualized_volatility(closes: list[float], window: int | None = None) -> float | None:
    """Annualized volatility of daily (log-free) returns."""
    values = closes[-window:] if window else closes
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    return float(np.std(returns, ddof=1) * np.sqrt(252))


def compute_drawdown(closes: list[float]) -> dict[str, float] | None:
    """Max drawdown (peak-to-trough) and current drawdown, as positive fractions."""
    if not closes:
        return None
    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        peak = max(peak, price)
        max_dd = max(max_dd, (peak - price) / peak)
    current_dd = (peak - closes[-1]) / peak if peak else 0.0
    return {"max_drawdown": round(max_dd, 6), "current_drawdown": round(current_dd, 6)}


def compute_pe(price: float | None, eps: float | None) -> float | None:
    """Price / earnings. Returns None when EPS is missing or non-positive."""
    if price is None or eps is None or eps <= 0:
        return None
    return round(price / eps, 3)


def compute_ev_ebitda(
    market_cap: float | None,
    total_debt: float | None,
    cash: float | None,
    ebitda: float | None,
) -> float | None:
    """Enterprise value / EBITDA."""
    if market_cap is None or total_debt is None or cash is None or ebitda is None or ebitda <= 0:
        return None
    return round((market_cap + total_debt - cash) / ebitda, 3)


def compute_fcf_yield(free_cash_flow: float | None, market_cap: float | None) -> float | None:
    """Free cash flow yield (FCF / market cap)."""
    if free_cash_flow is None or market_cap is None or market_cap <= 0:
        return None
    return round(free_cash_flow / market_cap, 6)


def compute_dcf(
    free_cash_flow: float | None,
    shares_outstanding: float | None,
    *,
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 5,
) -> dict[str, Any] | None:
    """Simple two-stage DCF. Returns intrinsic value per share plus assumptions.

    This intentionally ignores net-debt adjustments and uses FCF directly as the
    cash-flow proxy — a simplification that must be disclosed to the user.
    """
    if (
        free_cash_flow is None
        or shares_outstanding is None
        or free_cash_flow <= 0
        or shares_outstanding <= 0
        or discount_rate <= terminal_growth
        or years <= 0
    ):
        return None

    pv_sum = 0.0
    cash_flow = free_cash_flow
    for year in range(1, years + 1):
        cash_flow = free_cash_flow * (1 + growth_rate) ** year
        pv_sum += cash_flow / (1 + discount_rate) ** year

    terminal_value = (
        cash_flow * (1 + terminal_growth) / (discount_rate - terminal_growth)
    ) / (1 + discount_rate) ** years
    enterprise_value = pv_sum + terminal_value
    per_share = enterprise_value / shares_outstanding

    return {
        "intrinsic_value_per_share": round(per_share, 2),
        "assumptions": {
            "free_cash_flow": round(free_cash_flow, 2),
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "projection_years": years,
            "note": "Simplified DCF: uses FCF as cash-flow proxy and ignores net-debt adjustments.",
        },
    }
