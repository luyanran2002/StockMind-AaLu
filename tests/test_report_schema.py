from pydantic import ValidationError
import pytest

from app.schemas.report import StockResearchReport


def test_minimal_report_valid():
    report = StockResearchReport(ticker="NVDA", summary="s", conclusion="c")
    assert report.ticker == "NVDA"
    assert report.market_analysis == "Not covered in this phase."


def test_report_requires_ticker_summary_conclusion():
    with pytest.raises(ValidationError):
        StockResearchReport(ticker="NVDA")


def test_report_roundtrip_json():
    report = StockResearchReport(ticker="NVDA", summary="s", conclusion="c")
    reparsed = StockResearchReport.model_validate_json(report.model_dump_json())
    assert reparsed == report

