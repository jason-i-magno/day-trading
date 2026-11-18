"""Parsing of EDGAR full-text search hits."""

from datetime import date

from screener.edgar import _parse_hit


def _hit(**overrides):
    base = {
        "items": ["2.02", "9.01"],
        "display_names": ["DICK'S SPORTING GOODS, INC.  (DKS)  (CIK 0001089063)"],
        "file_date": "2026-08-25",
        "adsh": "0001089063-26-000042",
    }
    return {**base, **overrides}


def test_extracts_ticker_and_cik():
    event = _parse_hit(_hit())
    assert event.ticker == "DKS"
    assert event.cik == "0001089063"
    assert event.filing_date == date(2026, 8, 25)
    assert event.company == "DICK'S SPORTING GOODS, INC."


def test_multi_ticker_filer_takes_common_stock():
    # Issuers with preferred or warrant lines list several symbols; the first
    # is the common stock, which is what we would actually trade.
    event = _parse_hit(_hit(display_names=["BOX INC  (BOX, BXCAP)  (CIK 0001372612)"]))
    assert event.ticker == "BOX"


def test_skips_filings_without_item_202():
    # The Item 2.02 phrase can appear in an 8-K that is not an earnings release.
    assert _parse_hit(_hit(items=["8.01"])) is None


def test_skips_filers_without_a_ticker():
    # Trusts and wholly-owned subsidiaries file 8-Ks but have nothing to trade.
    assert _parse_hit(_hit(display_names=["SOME TRUST  (CIK 0000319654)"])) is None


def test_skips_unparseable_date():
    assert _parse_hit(_hit(file_date="not-a-date")) is None
