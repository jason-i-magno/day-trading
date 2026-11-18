"""Forward-test scoring: the one uncontaminated evaluation in the project."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener import forward


def _series(values, start="2026-06-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.fixture
def prices(monkeypatch):
    """Patch the price download; yields a setter taking {symbol: close series}."""
    def setup(mapping):
        monkeypatch.setattr(forward, "_download", lambda tickers, start: mapping)
    return setup


def _pick(ticker, logged):
    return {"logged_at": f"{logged}T22:00:00+00:00", "ticker": ticker}


# --- log parsing --------------------------------------------------------------

def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "picks.jsonl"
    path.write_text('{"ticker": "AAA", "logged_at": "2026-06-01T00:00:00+00:00"}\n'
                    "not json\n"
                    "\n"
                    '{"ticker": "BBB", "logged_at": "2026-06-02T00:00:00+00:00"}\n')
    assert [p["ticker"] for p in forward.load_picks(path)] == ["AAA", "BBB"]


def test_missing_log_is_not_an_error(tmp_path):
    assert forward.load_picks(tmp_path / "nope.jsonl") == []


# --- entry timing -------------------------------------------------------------

def test_entry_is_the_first_close_after_logging():
    """The screen runs after the close, so the logged day's close was already
    unreachable. Entering on it would be look-ahead."""
    closes = _series([10, 11, 12, 13])          # Mon..Thu
    log_date = closes.index[0].date()           # logged Monday
    idx = closes.index.searchsorted(pd.Timestamp(log_date), side="right")
    assert closes.index[idx].date() == closes.index[1].date()


def test_abnormal_return_is_measured_against_the_benchmark(prices):
    stock = _series([100, 100, 110])
    bench = _series([100, 100, 105])
    prices({"AAA": stock, "SPY": bench})
    scored = forward.score([_pick("AAA", stock.index[0].date().isoformat())], horizon=1)
    assert scored[0].matured
    assert scored[0].abnormal_return == pytest.approx(0.05)


# --- nothing may be dropped silently ------------------------------------------

def test_a_pick_logged_after_the_last_close_stays_open(prices):
    """Its entry bar has not printed yet. Reporting nothing would read as 'no
    picks logged' and make the whole log look empty."""
    closes = _series([10, 11])
    prices({"AAA": closes, "SPY": closes})
    scored = forward.score([_pick("AAA", "2026-12-31")], horizon=20)
    assert len(scored) == 1 and not scored[0].matured


def test_a_pick_with_no_price_data_stays_visible(prices):
    prices({"SPY": _series([100, 101, 102])})
    scored = forward.score([_pick("GONE", "2026-06-01")], horizon=1)
    assert len(scored) == 1 and not scored[0].matured


def test_a_total_download_failure_reports_every_pick_as_open(prices):
    prices({})
    picks = [_pick("AAA", "2026-06-01"), _pick("BBB", "2026-06-01")]
    scored = forward.score(picks, horizon=20)
    assert len(scored) == 2
    assert not any(s.matured for s in scored)


def test_an_immature_pick_is_open_not_missing(prices):
    closes = _series([10, 11, 12])
    prices({"AAA": closes, "SPY": closes})
    scored = forward.score([_pick("AAA", closes.index[0].date().isoformat())], horizon=20)
    assert len(scored) == 1 and not scored[0].matured


# --- reporting ----------------------------------------------------------------

def test_report_counts_open_and_matured_separately(prices):
    closes = _series([100] * 30)
    prices({"AAA": closes, "SPY": closes})
    picks = [_pick("AAA", closes.index[0].date().isoformat()),
             _pick("AAA", closes.index[-1].date().isoformat())]
    text = forward.format_report(forward.score(picks, horizon=5), 5)
    assert "2 picks logged" in text
    assert "1 matured" in text and "1 still open" in text


def test_report_flags_repeated_tickers(prices):
    """The same name picked twice in its drift window is one position held
    longer, not two independent observations."""
    closes = _series(list(np.linspace(100, 120, 40)))
    prices({"AAA": closes, "SPY": _series([100] * 40)})
    picks = [_pick("AAA", closes.index[0].date().isoformat()),
             _pick("AAA", closes.index[2].date().isoformat())]
    text = forward.format_report(forward.score(picks, horizon=5), 5)
    assert "picked more than once" in text


def test_report_always_carries_the_interpretation_warning(prices):
    closes = _series([100, 105, 110])
    prices({"AAA": closes, "SPY": _series([100, 100, 100])})
    text = forward.format_report(
        forward.score([_pick("AAA", closes.index[0].date().isoformat())], horizon=1), 1)
    assert "naive" in text and "null in this size band" in text


# --- log deduplication --------------------------------------------------------

def test_an_event_is_never_logged_twice(tmp_path):
    """An earnings event stays in the entry window for up to ten trading days, so
    a weekly run would otherwise record one position as several observations."""
    from dataclasses import replace as dc_replace
    from datetime import date as _date

    from screener.config import ScreenConfig
    from screener.edgar import EarningsEvent
    from screener.market import MarketSnapshot
    from screener.report import append_log
    from screener.screen import Candidate, ScreenResult

    event = EarningsEvent("AAA", "1", "A Co", _date(2026, 6, 1), "0001-26-000001", ("2.02",))
    snapshot = MarketSnapshot("AAA", 10.0, 5e8, 2e6, 0.08, 0.004, 2, "NMS")
    result = ScreenResult(candidates=[Candidate(event=event, snapshot=snapshot)])

    path = tmp_path / "picks.jsonl"
    assert append_log(result, ScreenConfig(), path) == 1
    assert append_log(result, ScreenConfig(), path) == 0      # same event again
    assert len(path.read_text().strip().splitlines()) == 1

    # A genuinely new event still gets through.
    other = dc_replace(event, accession="0001-26-000002")
    result2 = ScreenResult(candidates=[Candidate(event=other, snapshot=snapshot)])
    assert append_log(result2, ScreenConfig(), path) == 1
