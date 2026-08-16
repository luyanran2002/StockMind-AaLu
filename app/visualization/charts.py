"""Terminal + image chart rendering for research reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.tools.analysis import compute_macd_series, compute_rsi_series
from app.tools.providers import PriceBar

_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_sparkline(closes: Sequence[float], width: int = 48) -> str:
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


def _configure_cjk_font(language: str) -> None:
    if language == "zh":
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Hiragino Sans GB",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False


def render_price_chart(
    bars: Sequence[PriceBar],
    ticker: str,
    output_path: str | Path,
    *,
    language: str = "en",
    ma_windows: tuple[int, ...] = (20, 50),
) -> Path:
    """Render a multi-panel PNG chart (price/MA, volume, RSI, MACD)."""
    if not bars:
        raise ValueError("No price bars to chart")

    _configure_cjk_font(language)
    closes = [float(bar.close) for bar in bars]
    dates = [bar.date for bar in bars]
    volumes = [float(bar.volume) for bar in bars]
    x = list(range(len(closes)))
    series = pd.Series(closes)

    label = {
        "en": {"close": "Close", "volume": "Volume", "title": "Research chart"},
        "zh": {"close": "收盘价", "volume": "成交量", "title": "研究图表"},
    }.get(language, {"close": "Close", "volume": "Volume", "title": "Research chart"})

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )
    ax_price, ax_volume, ax_rsi, ax_macd = axes

    ax_price.plot(x, closes, color="black", linewidth=1.0, label=label["close"])
    for window in ma_windows:
        if len(closes) >= window:
            ax_price.plot(series.rolling(window).mean(), linewidth=1.0, label=f"MA{window}")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_title(f"{ticker} — {label['title']}")
    ax_price.grid(alpha=0.25)

    ax_volume.bar(x, volumes, color="tab:blue", alpha=0.55, label=label["volume"])
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

