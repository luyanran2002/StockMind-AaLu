"""Terminal + image chart rendering for research reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.tools.analysis import compute_macd_series, compute_rsi_series, project_linear_trend
from app.tools.providers import PriceBar

_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_sparkline(closes: Sequence[float], width: int = 64) -> str:
    """Render a compact terminal sparkline of closing prices."""
    data = list(closes)
    if not data:
        return "(no data)"
    if len(data) > width:
        data = data[-width:]
    lo, hi = min(data), max(data)
    if hi == lo:
        return "▄" * len(data)
    scale = (len(_BLOCKS) - 1) / (hi - lo)
    return "".join(_BLOCKS[int((v - lo) * scale)] for v in data)


def render_price_chart(
    bars: Sequence[PriceBar],
    ticker: str,
    output_path: str | Path,
    *,
    language: str = "en",
    ma_windows: tuple[int, ...] = (20, 50),
    forecast_horizon: int = 20,
    watermark: str | None = None,
) -> Path:
    """Render a multi-panel PNG chart (price/MA, volume, RSI, MACD)."""
    if not bars:
        raise ValueError("No price bars to chart")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    closes = [float(bar.close) for bar in bars]
    dates = [bar.date for bar in bars]
    volumes = [float(bar.volume) for bar in bars]
    x = list(range(len(closes)))
    series = pd.Series(closes)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1, 1, 1]},
    )
    ax_price, ax_volume, ax_rsi, ax_macd = axes

    ax_price.plot(x, closes, color="black", linewidth=1.0, label="Close")
    for window in ma_windows:
        if len(closes) >= window:
            ax_price.plot(series.rolling(window).mean(), linewidth=1.0, label=f"MA{window}")
    projection = project_linear_trend(closes, horizon=forecast_horizon)
    if projection:
        future_x = list(range(len(closes), len(closes) + len(projection)))
        ax_price.plot(
            future_x,
            projection,
            linestyle="--",
            color="tab:orange",
            linewidth=1.4,
            label="Linear extrapolation (not a prediction)",
        )
        band = _projection_band(closes)
        ax_price.fill_between(
            future_x,
            np.asarray(projection) - band,
            np.asarray(projection) + band,
            color="tab:orange",
            alpha=0.15,
        )
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_title(f"{ticker} — Research chart")
    ax_price.grid(alpha=0.25)
    if watermark:
        ax_price.text(
            0.99,
            0.01,
            watermark,
            transform=ax_price.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
            alpha=0.7,
        )

    ax_volume.bar(x, volumes, color="tab:blue", alpha=0.55, label="Volume")
    ax_volume.legend(loc="upper left", fontsize=8)
    ax_volume.grid(alpha=0.25)

    if len(closes) > 14:
        rsi = compute_rsi_series(closes)
        ax_rsi.plot(x, rsi, color="purple", linewidth=1.0, label="RSI (14)")
        ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.7)
        ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.7)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(alpha=0.25)

    if len(closes) >= 35:
        macd_line, signal_line, hist = compute_macd_series(closes)
        colors = ["tab:green" if h >= 0 else "tab:red" for h in hist]
        ax_macd.plot(x, macd_line, color="tab:blue", linewidth=1.0, label="MACD")
        ax_macd.plot(x, signal_line, color="tab:orange", linewidth=1.0, label="Signal")
        ax_macd.bar(x, hist, color=colors, alpha=0.5, label="Histogram")
        ax_macd.axhline(0, color="black", linewidth=0.7)
        ax_macd.legend(loc="upper left", fontsize=8)
    ax_macd.grid(alpha=0.25)

    if len(dates) > 12:
        step = max(1, len(dates) // 8)
        ticks = x[::step]
        tick_labels = [dates[i] for i in ticks]
    else:
        ticks = x
        tick_labels = dates
    ax_macd.set_xticks(ticks)
    ax_macd.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def _projection_band(closes: list[float], window: int = 60) -> float:
    """1-sigma residual band for the linear trend, used as a naive uncertainty."""
    data = closes[-window:] if len(closes) > window else closes
    x = np.arange(len(data), dtype=float)
    y = np.asarray(data, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    return float(np.std(residuals))
