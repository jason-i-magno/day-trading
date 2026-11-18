"""Delisting classification and coverage accounting."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtest import coverage
from screener.backtest.delistings import CompanyStatus, _latest


def _status(**overrides) -> CompanyStatus:
    base = dict(cik="0000000001", name="Test Co", has_ticker=False,
                delisting_date=None, deregistration_date=None,
                last_filing_date=date(2024, 1, 1), renamed=False, sic="")
    return CompanyStatus(**{**base, **overrides})


# --- classification -----------------------------------------------------------

def test_a_live_ticker_outranks_an_old_form_25():
    """Form 25 delists a *security*, not a company. Apple filed one for a note
    series while the common stock kept trading; treating that as a delisting
    misclassifies healthy large-cap issuers."""
    status = _status(has_ticker=True, delisting_date=date(2019, 3, 14))
    assert status.classification == "still listed"


def test_renamed_is_reported_only_for_live_companies():
    assert _status(has_ticker=True, renamed=True).classification == "renamed"
    # A dead company that also happens to have former names is still delisted.
    assert _status(has_ticker=False, renamed=True,
                   delisting_date=date(2024, 1, 1)).classification == "delisted"


def test_delisted_requires_no_current_ticker():
    assert _status(delisting_date=date(2024, 12, 18)).classification == "delisted"


def test_deregistered_without_a_delisting_notice():
    assert _status(deregistration_date=date(2024, 5, 1)).classification == "deregistered"


def test_no_ticker_and_no_forms_is_its_own_bucket():
    assert _status().classification == "no ticker on file"


def test_latest_picks_the_final_notice():
    """A company may delist several securities over its life; the last notice is
    the one corresponding to the common stock going away."""
    assert _latest(["2019-03-14", "2024-12-18", "2021-09-03"]) == date(2024, 12, 18)
    assert _latest([]) is None


# --- coverage accounting ------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows)


def test_tradeable_losses_count_only_delistings_well_past_the_window():
    """An event whose company delisted two years later traded perfectly normally
    at the time -- dropping it is a pure loss, and the whole point of the count."""
    frame = _frame([
        {"classification": "delisted", "days_to_delisting": 800, "public_float": 1e9},
        {"classification": "delisted", "days_to_delisting": 400, "public_float": 1e9},
        {"classification": "delisted", "days_to_delisting": 10, "public_float": 1e9},
        {"classification": "delisted", "days_to_delisting": -5, "public_float": 1e9},
    ])
    report = coverage.summarise(frame)
    assert report.tradeable_losses == 2       # 800d and 400d
    assert report.total_dropped == 4


def test_delistings_before_the_event_are_bucketed_separately():
    """A delisting dated before the earnings filing means the symbol was reused
    or the data is inconsistent; it must not be counted as a tradeable loss."""
    frame = _frame([{"classification": "delisted", "days_to_delisting": -30,
                     "public_float": 1e8}])
    report = coverage.summarise(frame)
    assert report.delisting_lag["before the event"] == 1
    assert report.tradeable_losses == 0


def test_classification_counts_cover_every_dropped_event():
    frame = _frame([
        {"classification": "delisted", "days_to_delisting": 100, "public_float": 1e8},
        {"classification": "renamed", "days_to_delisting": np.nan, "public_float": np.nan},
        {"classification": "still listed", "days_to_delisting": np.nan, "public_float": np.nan},
    ])
    report = coverage.summarise(frame)
    assert report.events_by_class.sum() == report.total_dropped == 3


def test_float_buckets_separate_untradeable_microcaps():
    """Size decides whether any hidden effect is reachable with small retail positions,
    so the smallest bucket has to be distinguishable."""
    frame = _frame([
        {"classification": "delisted", "days_to_delisting": 200, "public_float": 1e7},
        {"classification": "delisted", "days_to_delisting": 200, "public_float": 5e8},
        {"classification": "delisted", "days_to_delisting": 200, "public_float": 5e9},
    ])
    report = coverage.summarise(frame)
    assert report.float_distribution["<$50M"] == 1
    assert report.float_distribution["$300M-1B"] == 1
    assert report.float_distribution[">$2B"] == 1


def test_dropped_events_are_those_without_cached_prices(monkeypatch):
    from screener.edgar import EarningsEvent

    monkeypatch.setattr(coverage.prices, "cached_tickers", lambda: {"AAA"})
    events = [
        EarningsEvent("AAA", "1", "A Co", date(2024, 1, 1), "x", ("2.02",)),
        EarningsEvent("BBB", "2", "B Co", date(2024, 1, 1), "y", ("2.02",)),
    ]
    assert [e.ticker for e in coverage.dropped_events(events)] == ["BBB"]
