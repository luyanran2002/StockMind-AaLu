import akshare
import pandas as pd

from app.tools.providers import AkShareMarketDataProvider


def _daily_df():
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-14"), "open": 960.0, "high": 975.0, "low": 955.0, "close": 971.66, "volume": 1_000_000.0},
            {"date": pd.Timestamp("2026-08-15"), "open": 971.66, "high": 985.0, "low": 968.0, "close": 980.0, "volume": 1_200_000.0},
        ]
    )


def _report_df(symbol):
    if symbol == "综合损益表":
        return pd.DataFrame(
            [
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "营业总收入", "AMOUNT": 30e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "归母净利润", "AMOUNT": 4e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "EBITDA", "AMOUNT": 8e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "稀释每股收益", "AMOUNT": 4.0},
                {"REPORT_DATE": "2024-12-31", "ITEM_NAME": "营业总收入", "AMOUNT": 20e9},
            ]
        )
    if symbol == "资产负债表":
        return pd.DataFrame(
            [
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "资产总计", "AMOUNT": 50e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "总债务", "AMOUNT": 10e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "现金及现金等价物", "AMOUNT": 5e9},
            ]
        )
    if symbol == "现金流量表":
        return pd.DataFrame(
            [
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "经营活动产生的现金流量净额", "AMOUNT": 7e9},
                {"REPORT_DATE": "2025-12-31", "ITEM_NAME": "购建固定资产、无形资产和其他长期资产支付的现金", "AMOUNT": -2e9},
            ]
        )
    return pd.DataFrame()


def _valuation_df(symbol, indicator, **kwargs):
    if indicator == "总市值":
        value = 100e9
    else:
        value = 20.0
    return pd.DataFrame({"date": ["2026-08-15"], "value": [value]})


def _provider(monkeypatch):
    monkeypatch.setattr(akshare, "stock_us_daily", lambda **kwargs: _daily_df())
    monkeypatch.setattr(akshare, "stock_financial_us_report_em", lambda **kwargs: _report_df(kwargs["symbol"]))
    monkeypatch.setattr(akshare, "stock_us_valuation_baidu", _valuation_df)
    return AkShareMarketDataProvider()


def test_akshare_price_uses_last_close(monkeypatch):
    result = _provider(monkeypatch).get_stock_price("MU")
    assert result.price == 980.0
    assert result.source == "akshare/sina"


def test_akshare_historical(monkeypatch):
    result = _provider(monkeypatch).get_historical_prices("MU", "1mo", "1d")
    assert len(result.bars) == 2
    assert result.bars[-1].close == 980.0


def test_akshare_income_statement(monkeypatch):
    result = _provider(monkeypatch).get_income_statement("MU")
    assert result.revenue == 30e9
    assert result.eps_diluted == 4.0
    assert result.ebitda == 8e9


def test_akshare_balance_sheet(monkeypatch):
    result = _provider(monkeypatch).get_balance_sheet("MU")
    assert result.total_debt == 10e9
    assert result.cash_and_equivalents == 5e9


def test_akshare_cash_flow_fcf(monkeypatch):
    result = _provider(monkeypatch).get_cash_flow("MU")
    assert result.free_cash_flow == 5e9  # OCF 7e9 - capex 2e9


def test_akshare_key_stats(monkeypatch):
    result = _provider(monkeypatch).get_key_stats("MU")
    assert result.market_cap == 100e9
    assert result.eps_ttm == 49.0  # price 980 / PE 20

