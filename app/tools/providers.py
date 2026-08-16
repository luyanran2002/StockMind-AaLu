"""Data-provider abstraction.

Tools depend only on the :class:`MarketDataProvider` interface. Concrete
providers are swappable via configuration:

* :class:`MockMarketDataProvider` — deterministic, offline, clearly labelled as
  simulated data. Used for tests and demos.
* :class:`YFinanceMarketDataProvider` — real data via ``yfinance`` (optional
  dependency, requires network access).
"""

from __future__ import annotations

import hashlib
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import utc_now


def _as_float(value: Any) -> float | None:
    """Coerce a provider value to float, returning None for missing/NaN values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


class StockPriceResult(BaseModel):
    ticker: str
    price: float
    currency: str
    source: str
    timestamp: datetime
    data_period: str | None = Field(default=None)
    note: str | None = Field(default=None)


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalPricesResult(BaseModel):
    ticker: str
    period: str
    interval: str
    bars: list[PriceBar]
    source: str
    timestamp: datetime
    data_period: str | None = Field(default=None)
    note: str | None = Field(default=None)


class NewsItem(BaseModel):
    title: str
    source: str
    published_at: str | None = Field(default=None)
    url: str | None = Field(default=None)
    summary: str | None = Field(default=None)


class NewsResult(BaseModel):
    ticker: str
    query: str
    items: list[NewsItem]
    source: str
    timestamp: datetime
    note: str | None = Field(default=None)


class IncomeStatement(BaseModel):
    ticker: str
    period: str = Field(default="TTM")
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    net_income: float | None = None
    eps_diluted: float | None = None
    source: str
    timestamp: datetime
    note: str | None = None


class BalanceSheet(BaseModel):
    ticker: str
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_debt: float | None = None
    cash_and_equivalents: float | None = None
    total_equity: float | None = None
    source: str
    timestamp: datetime
    note: str | None = None


class CashFlow(BaseModel):
    ticker: str
    operating_cash_flow: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    free_cash_flow: float | None = None
    source: str
    timestamp: datetime
    note: str | None = None


class KeyStats(BaseModel):
    ticker: str
    eps_ttm: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None
    beta: float | None = None
    source: str
    timestamp: datetime
    note: str | None = None


class MarketDataProvider(ABC):
    """Interface all market/news data providers must implement."""

    @abstractmethod
    def get_stock_price(self, ticker: str) -> StockPriceResult:
        """Return the latest available price for ``ticker``."""

    @abstractmethod
    def get_historical_prices(
        self, ticker: str, period: str, interval: str
    ) -> HistoricalPricesResult:
        """Return historical price bars for ``ticker``."""

    @abstractmethod
    def search_company_news(self, ticker: str, query: str, limit: int) -> NewsResult:
        """Return recent company/market news for ``ticker``."""

    @abstractmethod
    def search_market_news(self, query: str, limit: int) -> NewsResult:
        """Return recent market-wide news (not tied to a single ticker)."""

    @abstractmethod
    def get_income_statement(self, ticker: str) -> IncomeStatement:
        """Return the latest income statement for ``ticker``."""

    @abstractmethod
    def get_balance_sheet(self, ticker: str) -> BalanceSheet:
        """Return the latest balance sheet for ``ticker``."""

    @abstractmethod
    def get_cash_flow(self, ticker: str) -> CashFlow:
        """Return the latest cash flow statement for ``ticker``."""

    @abstractmethod
    def get_key_stats(self, ticker: str) -> KeyStats:
        """Return key valuation/ownership statistics for ``ticker``."""


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic, offline provider that is *explicitly* simulated."""

    SOURCE = "mock"
    NOTE = "Simulated data for demonstration only — not real market data."

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _rng(self, ticker: str, salt: str) -> random.Random:
        digest = hashlib.sha256(f"{ticker.upper()}:{salt}:{self.seed}".encode()).hexdigest()
        return random.Random(int(digest, 16) % (2**32))

    def _base_price(self, ticker: str) -> float:
        rng = self._rng(ticker, "base")
        return round(rng.uniform(20.0, 800.0), 2)

    def _current_price(self, ticker: str) -> float:
        rng = self._rng(ticker, "price")
        return round(self._base_price(ticker) * rng.uniform(0.98, 1.02), 2)

    def get_stock_price(self, ticker: str) -> StockPriceResult:
        symbol = ticker.upper().strip()
        price = self._current_price(symbol)
        return StockPriceResult(
            ticker=symbol,
            price=price,
            currency="USD",
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period="latest close",
            note=self.NOTE,
        )

    def get_historical_prices(
        self, ticker: str, period: str, interval: str
    ) -> HistoricalPricesResult:
        symbol = ticker.upper().strip()
        rng = self._rng(symbol, "history")
        days = self._period_to_days(period)
        current = self._current_price(symbol)
        today = datetime.now(timezone.utc).date()

        # Build closes backwards from the current price so the final bar equals
        # get_stock_price(), keeping the simulated dataset internally consistent.
        closes = [0.0] * days
        closes[-1] = current
        for i in range(days - 2, -1, -1):
            drift = rng.uniform(-0.03, 0.03)
            closes[i] = round(max(1.0, closes[i + 1] / (1 + drift)), 2)

        bars: list[PriceBar] = []
        for i, close in enumerate(closes):
            open_ = close * rng.uniform(0.99, 1.01)
            high = max(open_, close) * rng.uniform(1.0, 1.02)
            low = min(open_, close) * rng.uniform(0.98, 1.0)
            volume = int(rng.uniform(1_000_000, 80_000_000))
            bars.append(
                PriceBar(
                    date=(today - timedelta(days=days - i - 1)).isoformat(),
                    open=round(open_, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=close,
                    volume=volume,
                )
            )
        return HistoricalPricesResult(
            ticker=symbol,
            period=period,
            interval=interval,
            bars=bars,
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period=f"last {days} calendar days",
            note=self.NOTE,
        )

    def search_company_news(self, ticker: str, query: str, limit: int) -> NewsResult:
        symbol = ticker.upper().strip()
        rng = self._rng(symbol, "news")
        templates = [
            "{t} reports quarterly results in line with market expectations",
            "Analysts adjust price targets for {t} on demand outlook",
            "{t} announces new product roadmap and capex plans",
            "Regulatory review adds uncertainty to {t} outlook",
            "Macro concerns weigh on semiconductor names including {t}",
        ]
        items: list[NewsItem] = []
        for i in range(max(1, min(limit, len(templates)))):
            title = rng.choice(templates).format(t=symbol)
            items.append(
                NewsItem(
                    title=f"[MOCK] {title}",
                    source="simulated-feed",
                    published_at=(datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                    url=None,
                    summary="Simulated headline for demonstration only.",
                )
            )
        return NewsResult(
            ticker=symbol,
            query=query or "",
            items=items,
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    def search_market_news(self, query: str, limit: int) -> NewsResult:
        rng = self._rng("MARKET", "news")
        templates = [
            "Equity indices mixed as investors weigh inflation data",
            "Treasury yields edge higher ahead of central bank meeting",
            "Sector rotation continues as growth names lag value",
            "Oil prices move on supply outlook",
            "Markets focus on upcoming earnings season",
        ]
        items: list[NewsItem] = []
        for i in range(max(1, min(limit, len(templates)))):
            title = rng.choice(templates)
            items.append(
                NewsItem(
                    title=f"[MOCK] {title}",
                    source="simulated-feed",
                    published_at=(datetime.now(timezone.utc) - timedelta(days=i)).isoformat(),
                    url=None,
                    summary="Simulated market headline for demonstration only.",
                )
            )
        return NewsResult(
            ticker="MARKET",
            query=query or "",
            items=items,
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    def _financials(self, ticker: str) -> dict[str, Any]:
        """Deterministic, internally-consistent simulated financial statements."""
        symbol = ticker.upper().strip()
        rng = self._rng(symbol, "financials")
        revenue = round(rng.uniform(3e9, 120e9), 2)
        net_margin = rng.uniform(0.08, 0.35)
        net_income = round(revenue * net_margin, 2)
        shares = round(rng.uniform(5e8, 3e10), 0)
        eps = round(net_income / shares, 2)
        price = self._current_price(symbol)
        market_cap = round(price * shares, 2)
        ebitda = round(revenue * rng.uniform(0.20, 0.50), 2)
        gross_profit = round(revenue * rng.uniform(0.40, 0.75), 2)
        operating_income = round(revenue * rng.uniform(0.15, 0.40), 2)
        total_assets = round(revenue * rng.uniform(0.80, 2.00), 2)
        total_liabilities = round(total_assets * rng.uniform(0.30, 0.70), 2)
        total_debt = round(total_assets * rng.uniform(0.05, 0.25), 2)
        cash = round(total_assets * rng.uniform(0.05, 0.30), 2)
        total_equity = round(total_assets - total_liabilities, 2)
        operating_cash_flow = round(net_income * rng.uniform(1.0, 1.6), 2)
        capex = round(revenue * rng.uniform(0.02, 0.08), 2)
        free_cash_flow = round(operating_cash_flow - capex, 2)
        beta = round(rng.uniform(0.8, 1.8), 2)
        return {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "ebitda": ebitda,
            "net_income": net_income,
            "eps_diluted": eps,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "total_debt": total_debt,
            "cash_and_equivalents": cash,
            "total_equity": total_equity,
            "operating_cash_flow": operating_cash_flow,
            "investing_cash_flow": -capex,
            "financing_cash_flow": round(-net_income * rng.uniform(0.2, 0.8), 2),
            "free_cash_flow": free_cash_flow,
            "eps_ttm": eps,
            "shares_outstanding": shares,
            "market_cap": market_cap,
            "beta": beta,
        }

    def get_income_statement(self, ticker: str) -> IncomeStatement:
        symbol = ticker.upper().strip()
        f = self._financials(symbol)
        return IncomeStatement(
            ticker=symbol,
            period="TTM",
            revenue=f["revenue"],
            gross_profit=f["gross_profit"],
            operating_income=f["operating_income"],
            ebitda=f["ebitda"],
            net_income=f["net_income"],
            eps_diluted=f["eps_diluted"],
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    def get_balance_sheet(self, ticker: str) -> BalanceSheet:
        symbol = ticker.upper().strip()
        f = self._financials(symbol)
        return BalanceSheet(
            ticker=symbol,
            total_assets=f["total_assets"],
            total_liabilities=f["total_liabilities"],
            total_debt=f["total_debt"],
            cash_and_equivalents=f["cash_and_equivalents"],
            total_equity=f["total_equity"],
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    def get_cash_flow(self, ticker: str) -> CashFlow:
        symbol = ticker.upper().strip()
        f = self._financials(symbol)
        return CashFlow(
            ticker=symbol,
            operating_cash_flow=f["operating_cash_flow"],
            investing_cash_flow=f["investing_cash_flow"],
            financing_cash_flow=f["financing_cash_flow"],
            free_cash_flow=f["free_cash_flow"],
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    def get_key_stats(self, ticker: str) -> KeyStats:
        symbol = ticker.upper().strip()
        f = self._financials(symbol)
        return KeyStats(
            ticker=symbol,
            eps_ttm=f["eps_ttm"],
            shares_outstanding=f["shares_outstanding"],
            market_cap=f["market_cap"],
            beta=f["beta"],
            source=self.SOURCE,
            timestamp=utc_now(),
            note=self.NOTE,
        )

    @staticmethod
    def _period_to_days(period: str) -> int:
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        return mapping.get(period, 30)


class YFinanceMarketDataProvider(MarketDataProvider):
    """Real market data via ``yfinance``. Lazily imported so the base install
    has no network dependency."""

    SOURCE = "yfinance"

    def _yf(self) -> Any:
        try:
            import yfinance  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "yfinance is not installed. Install it with: pip install -e '.[real-data]'"
            ) from exc
        return yfinance

    def get_stock_price(self, ticker: str) -> StockPriceResult:
        symbol = ticker.upper().strip()
        t = self._yf().Ticker(symbol)
        price = None
        currency = "USD"
        # fast_info hits a single, lighter endpoint and is usually more reliable.
        try:
            fast = t.fast_info
            price = fast.last_price
            currency = fast.currency or "USD"
        except Exception:
            price = None
        if price is None:
            info = t.info or {}
            price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
            )
            currency = info.get("currency", "USD")
        if price is None:
            raise ValueError(f"No price found for ticker {symbol!r}")
        return StockPriceResult(
            ticker=symbol,
            price=float(price),
            currency=currency,
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period="latest close",
        )

    def get_historical_prices(
        self, ticker: str, period: str, interval: str
    ) -> HistoricalPricesResult:
        symbol = ticker.upper().strip()
        df = self._yf().Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No historical data found for ticker {symbol!r}")
        bars = [
            PriceBar(
                date=str(index.date()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for index, row in df.iterrows()
        ]
        return HistoricalPricesResult(
            ticker=symbol,
            period=period,
            interval=interval,
            bars=bars,
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period=period,
        )

    def search_company_news(self, ticker: str, query: str, limit: int) -> NewsResult:
        symbol = ticker.upper().strip()
        raw = self._yf().Search(symbol, news_count=limit, include_research=False).news or []
        items = [
            NewsItem(
                title=str(item.get("title", "Untitled")),
                source=str(item.get("publisher", "unknown")),
                published_at=item.get("providerPublishTime"),
                url=item.get("link"),
                summary=None,
            )
            for item in raw[:limit]
        ]
        return NewsResult(
            ticker=symbol,
            query=query or "",
            items=items,
            source=self.SOURCE,
            timestamp=utc_now(),
        )

    def search_market_news(self, query: str, limit: int) -> NewsResult:
        raw = self._yf().Search("stock market", news_count=limit, include_research=False).news or []
        items = [
            NewsItem(
                title=str(item.get("title", "Untitled")),
                source=str(item.get("publisher", "unknown")),
                published_at=item.get("providerPublishTime"),
                url=item.get("link"),
                summary=None,
            )
            for item in raw[:limit]
        ]
        return NewsResult(
            ticker="MARKET",
            query=query or "",
            items=items,
            source=self.SOURCE,
            timestamp=utc_now(),
        )

    def _info(self, ticker: str) -> dict[str, Any]:
        return self._yf().Ticker(ticker).info or {}

    def get_income_statement(self, ticker: str) -> IncomeStatement:
        symbol = ticker.upper().strip()
        info = self._info(symbol)
        return IncomeStatement(
            ticker=symbol,
            period="TTM",
            revenue=_as_float(info.get("totalRevenue")),
            gross_profit=_as_float(info.get("grossProfits")),
            operating_income=_as_float(info.get("operatingIncome")),
            ebitda=_as_float(info.get("ebitda")),
            net_income=_as_float(info.get("netIncomeToCommon")),
            eps_diluted=_as_float(info.get("trailingEps") or info.get("dilutedEPSTtm")),
            source=self.SOURCE,
            timestamp=utc_now(),
        )

    def get_balance_sheet(self, ticker: str) -> BalanceSheet:
        symbol = ticker.upper().strip()
        info = self._info(symbol)
        return BalanceSheet(
            ticker=symbol,
            total_assets=_as_float(info.get("totalAssets")),
            total_liabilities=_as_float(info.get("totalLiab")),
            total_debt=_as_float(info.get("totalDebt")),
            cash_and_equivalents=_as_float(info.get("totalCash")),
            total_equity=_as_float(info.get("totalStockholderEquity")),
            source=self.SOURCE,
            timestamp=utc_now(),
        )

    def get_cash_flow(self, ticker: str) -> CashFlow:
        symbol = ticker.upper().strip()
        info = self._info(symbol)
        return CashFlow(
            ticker=symbol,
            operating_cash_flow=_as_float(info.get("operatingCashflow")),
            investing_cash_flow=_as_float(info.get("investingCashflow")),
            financing_cash_flow=_as_float(info.get("financingCashflow")),
            free_cash_flow=_as_float(info.get("freeCashflow")),
            source=self.SOURCE,
            timestamp=utc_now(),
        )

    def get_key_stats(self, ticker: str) -> KeyStats:
        symbol = ticker.upper().strip()
        info = self._info(symbol)
        return KeyStats(
            ticker=symbol,
            eps_ttm=_as_float(info.get("trailingEps")),
            shares_outstanding=_as_float(info.get("sharesOutstanding")),
            market_cap=_as_float(info.get("marketCap")),
            beta=_as_float(info.get("beta")),
            source=self.SOURCE,
            timestamp=utc_now(),
        )


class AkShareMarketDataProvider(MarketDataProvider):
    """US stock data via ``akshare`` (Sina), reachable from mainland China.

    Uses a single-ticker daily endpoint (``stock_us_daily``) so lookups are
    fast and don't require fetching the full market list. Covers latest daily
    close and history. Key stats / financials / news are not exposed by this
    endpoint, so those degrade to N/A.
    """

    SOURCE = "akshare/sina"
    NOTE = "US stock daily data via akshare (Sina); ~15 min delayed, not real-time."

    def _ak(self) -> Any:
        try:
            import akshare  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "akshare is not installed. Install it with: pip install -e '.[real-data]'"
            ) from exc
        return akshare

    @staticmethod
    def _period_days(period: str) -> int:
        return {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }.get(period, 180)

    def _daily(self, ticker: str, adjust: str) -> Any:
        return self._ak().stock_us_daily(symbol=ticker.upper().strip(), adjust=adjust)

    def get_stock_price(self, ticker: str) -> StockPriceResult:
        symbol = ticker.upper().strip()
        df = self._daily(symbol, adjust="")
        if df is None or getattr(df, "empty", True):
            raise ValueError(f"No price found for ticker {symbol!r}")
        price = _as_float(df.iloc[-1].get("close"))
        if price is None:
            raise ValueError(f"No price found for ticker {symbol!r}")
        return StockPriceResult(
            ticker=symbol,
            price=round(price, 2),
            currency="USD",
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period="latest daily close",
            note=self.NOTE,
        )

    def get_historical_prices(
        self, ticker: str, period: str, interval: str
    ) -> HistoricalPricesResult:
        symbol = ticker.upper().strip()
        df = self._daily(symbol, adjust="qfq")
        if df is None or getattr(df, "empty", True):
            raise ValueError(f"No historical data found for ticker {symbol!r}")
        df = df.tail(self._period_days(period))
        bars: list[PriceBar] = []
        for _, row in df.iterrows():
            volume = _as_float(row.get("volume")) or 0
            date_val = row.get("date") or row.get("index") or ""
            bars.append(
                PriceBar(
                    date=str(date_val)[:10],
                    open=float(_as_float(row.get("open")) or 0.0),
                    high=float(_as_float(row.get("high")) or 0.0),
                    low=float(_as_float(row.get("low")) or 0.0),
                    close=float(_as_float(row.get("close")) or 0.0),
                    volume=int(volume),
                )
            )
        return HistoricalPricesResult(
            ticker=symbol,
            period=period,
            interval=interval,
            bars=bars,
            source=self.SOURCE,
            timestamp=utc_now(),
            data_period=period,
            note=self.NOTE,
        )

    def get_key_stats(self, ticker: str) -> KeyStats:
        return KeyStats(
            ticker=ticker.upper().strip(),
            source=self.SOURCE,
            timestamp=utc_now(),
            note="Market cap / EPS not available via akshare daily endpoint.",
        )

    def get_income_statement(self, ticker: str) -> IncomeStatement:
        return IncomeStatement(
            ticker=ticker.upper().strip(),
            source=self.SOURCE,
            timestamp=utc_now(),
            note="US income statement not available via akshare.",
        )

    def get_balance_sheet(self, ticker: str) -> BalanceSheet:
        return BalanceSheet(
            ticker=ticker.upper().strip(),
            source=self.SOURCE,
            timestamp=utc_now(),
            note="US balance sheet not available via akshare.",
        )

    def get_cash_flow(self, ticker: str) -> CashFlow:
        return CashFlow(
            ticker=ticker.upper().strip(),
            source=self.SOURCE,
            timestamp=utc_now(),
            note="US cash flow not available via akshare.",
        )

    def search_company_news(self, ticker: str, query: str, limit: int) -> NewsResult:
        return NewsResult(
            ticker=ticker.upper().strip(),
            query=query or "",
            items=[],
            source=self.SOURCE,
            timestamp=utc_now(),
            note="Company news not available via akshare.",
        )

    def search_market_news(self, query: str, limit: int) -> NewsResult:
        return NewsResult(
            ticker="MARKET",
            query=query or "",
            items=[],
            source=self.SOURCE,
            timestamp=utc_now(),
            note="Market news not available via akshare.",
        )


def get_data_provider(provider: str | None = None) -> MarketDataProvider:
    """Resolve the active data provider from an argument or environment."""
    name = (provider or os.getenv("STOCKMIND_DATA_PROVIDER") or "mock").strip().lower()
    if name == "mock":
        return MockMarketDataProvider()
    if name == "yfinance":
        return YFinanceMarketDataProvider()
    if name == "akshare":
        return AkShareMarketDataProvider()
    raise ValueError(f"Unsupported data provider: {name!r} (expected 'mock', 'yfinance' or 'akshare')")
