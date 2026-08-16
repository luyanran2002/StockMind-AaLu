import akshare
import pandas as pd

from app.tools.providers import AkShareMarketDataProvider


def _daily_df():
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-14"),
                "open": 960.0,
                "high": 975.0,
                "low": 955.0,
                "close": 971.66,
                "volume": 1_000_000.0,
            },
            {
                "date": pd.Timestamp("2026-08-15"),
                "open": 971.66,
                "high": 985.0,
                "low": 968.0,
                "close": 980.0,
                "volume": 1_200_000.0,
            },
        ]
    )


def _provider(monkeypatch):
    monkeypatch.setattr(akshare, "stock_us_daily", lambda **kwargs: _daily_df())
    return AkShareMarketDataProvider()


def test_akshare_price_uses_last_close(monkeypatch):
    provider = _provider(monkeypatch)
    result = provider.get_stock_price("MU")
    assert result.price == 980.0
    assert result.source == "akshare/sina"


def test_akshare_historical(monkeypatch):
    provider = _provider(monkeypatch)
    result = provider.get_historical_prices("MU", "1mo", "1d")
    assert len(result.bars) == 2
    assert result.bars[-1].close == 980.0
    assert result.bars[0].date == "2026-08-14"


def test_akshare_key_stats_empty(monkeypatch):
    provider = _provider(monkeypatch)
    stats = provider.get_key_stats("MU")
    assert stats.market_cap is None
    assert stats.eps_ttm is None

