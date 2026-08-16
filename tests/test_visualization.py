from app.tools.providers import PriceBar
from app.visualization.charts import ascii_sparkline, render_price_chart


def test_sparkline_returns_string():
    assert isinstance(ascii_sparkline([1, 2, 3, 4, 5]), str)


def test_sparkline_flat_series():
    assert ascii_sparkline([5.0, 5.0, 5.0]) == "▄▄▄"


def test_render_price_chart_writes_png(tmp_path):
    bars = [
        PriceBar(
            date=f"d{i + 1}",
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000 + i,
        )
        for i in range(60)
    ]
    out = render_price_chart(bars, "TEST", tmp_path / "chart.png", language="en")
    assert out.exists()
    assert out.stat().st_size > 0

