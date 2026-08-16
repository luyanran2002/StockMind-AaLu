import math

import pytest

from app.tools import analysis


def test_compute_ma():
    assert analysis.compute_ma([1, 2, 3, 4, 5], window=3) == 4.0
    assert analysis.compute_ma([1, 2], window=3) is None


def test_compute_rsi_all_gains_is_100():
    closes = list(range(1, 17))  # 15 strictly rising bars
    assert analysis.compute_rsi(closes, period=14) == 100.0


def test_compute_rsi_series_length_and_leading_none():
    closes = list(range(1, 31))
    series = analysis.compute_rsi_series(closes, period=14)
    assert len(series) == len(closes)
    assert all(v is None for v in series[:14])
    assert all(v is not None for v in series[14:])


def test_compute_macd_returns_structure():
    closes = [10 + i for i in range(50)]
    result = analysis.compute_macd(closes)
    assert set(result) == {"macd", "signal", "histogram"}
    assert result["histogram"] == pytest.approx(result["macd"] - result["signal"], abs=1e-6)


def test_compute_macd_series_lengths():
    closes = [10 + i for i in range(60)]
    macd, signal, hist = analysis.compute_macd_series(closes)
    assert len(macd) == len(signal) == len(hist) == len(closes)


def test_compute_volatility_constant_series_is_zero():
    assert analysis.compute_annualized_volatility([10.0, 10.0, 10.0, 10.0]) == 0.0


def test_compute_drawdown():
    result = analysis.compute_drawdown([10.0, 8.0, 12.0])
    assert result["max_drawdown"] == pytest.approx(0.2)
    assert result["current_drawdown"] == pytest.approx(0.0)


def test_compute_pe():
    assert analysis.compute_pe(100.0, 10.0) == 10.0
    assert analysis.compute_pe(100.0, 0.0) is None
    assert analysis.compute_pe(100.0, None) is None


def test_compute_ev_ebitda():
    assert analysis.compute_ev_ebitda(1000.0, 200.0, 100.0, 100.0) == 11.0
    assert analysis.compute_ev_ebitda(1000.0, 200.0, 100.0, 0.0) is None


def test_compute_fcf_yield():
    assert analysis.compute_fcf_yield(50.0, 1000.0) == pytest.approx(0.05)


def test_compute_dcf_valid_and_invalid():
    result = analysis.compute_dcf(
        100.0, 100.0, growth_rate=0.10, discount_rate=0.12, terminal_growth=0.03, years=5
    )
    assert result is not None
    assert result["intrinsic_value_per_share"] > 0
    assert "growth_rate" in result["assumptions"]

    assert analysis.compute_dcf(
        100.0, 100.0, growth_rate=0.10, discount_rate=0.05, terminal_growth=0.06, years=5
    ) is None
