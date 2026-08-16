import akshare
import pandas as pd

from app.tools.providers import AkShareMarketDataProvider


def _spot_df():
    return pd.DataFrame(
        [
            {
                "代码": "105.MU",
                "名称": "Micron",
                "最新价": 971.66,
                "市盈率": 18.5,
                "总市值": 2.0e12,
            }
        ]
    )


def _hist_df():
    return pd.DataFrame(
        [
            {"日期": "2026-08-14", "开盘": 960.0, "收盘": 971.66, "最高": 975.0, "最低": 955.0, "成交量": 1_000_000.0},
            {"日期": "2026-08-15", "开盘": 971.66, "收盘": 980.0, "最高": 985.0, "最低": 968.0, "成交量": 1_200_000.0},
        ]
    )


def _provider(monkeypatch):
    monkeypatch.setattr(akshare, "stock_us_spot_em", _spot_df)
    monkeypatch.setattr(akshare, "stock_us_hist", lambda **kwargs: _hist_df())
    return AkShareMarketDataProvider()


def test_akshare_price(monkeypatch):
    provider = _provider(monkeypatch)
    result = provider.get_stock_price("MU")
    assert result.price == 971.66
    assert result.source == "akshare/eastmoney"


def test_akshare_historical(monkeypatch):
    provider = _provider(monkeypatch)
    result = provider.get_historical_prices("MU", "1mo", "1d")
    assert len(result.bars) == 2
    assert result.bars[-1].close == 980.0


def test_akshare_key_stats(monkeypatch):
    provider = _provider(monkeypatch)
    stats = provider.get_key_stats("MU")
    assert stats.market_cap == 2.0e12
    assert stats.eps_ttm == round(971.66 / 18.5, 2)


def test_akshare_unknown_ticker(monkeypatch):
    provider = _provider(monkeypatch)
    try:
        provider.get_stock_price("ZZZZ")
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown ticker")

