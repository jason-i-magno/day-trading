"""Filter funnel behaviour, driven by synthetic data."""

from datetime import date

import pandas as pd
import pytest

from screener import screen
from screener.config import ScreenConfig
from screener.edgar import EarningsEvent
from screener.screen import run_screen

EVENT_DAY = date(2026, 8, 20)


def _frame(tickers, closes_by_ticker):
    """Build a yfinance-shaped OHLCV frame with a flat SPY benchmark."""
    idx = pd.bdate_range(end="2026-08-28", periods=40)
    blocks = {}
    for ticker in [*tickers, "SPY"]:
        closes = closes_by_ticker.get(ticker, [100.0] * len(idx))
        series = pd.Series(closes, index=idx, dtype=float)
        blocks[ticker] = pd.DataFrame(
            {
                "Open": series,
                "High": series * 1.01,
                "Low": series * 0.99,
                "Close": series,
                "Volume": pd.Series([500_000.0] * len(idx), index=idx),
            }
        )
    return pd.concat(blocks, axis=1)


def _closes_with_jump(pct, idx_len=40, jump_at=-6):
    """Flat at 100, then a one-day jump of `pct` that holds."""
    closes = [100.0] * idx_len
    for i in range(idx_len + jump_at, idx_len):
        closes[i] = 100.0 * (1 + pct)
    return closes


@pytest.fixture
def wire(monkeypatch):
    """Patch the network layer; yields a setup(events, closes, profiles) helper."""

    def setup(events, closes, profiles, throttled=()):
        monkeypatch.setattr(screen, "find_earnings_events", lambda *a, **k: events)
        monkeypatch.setattr(
            screen, "download_history", lambda t: _frame([e.ticker for e in events], closes)
        )
        monkeypatch.setattr(
            screen, "fetch_profiles", lambda t, **k: (profiles, set(throttled))
        )

    return setup


def _event(ticker):
    return EarningsEvent(
        ticker=ticker,
        cik="0000000001",
        company=f"{ticker} Inc",
        filing_date=EVENT_DAY,
        accession="0000000001-26-000001",
        items=("2.02",),
    )


def _profile(cap, exchange="NMS"):
    return {"market_cap": cap, "exchange": exchange}


def test_strong_reaction_in_band_survives(wire):
    wire([_event("AAA")], {"AAA": _closes_with_jump(0.10)}, {"AAA": _profile(800e6)})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert [c.ticker for c in result.candidates] == ["AAA"]
    assert result.candidates[0].snapshot.abnormal_return == pytest.approx(0.10, abs=1e-6)


def test_weak_reaction_is_rejected(wire):
    wire([_event("AAA")], {"AAA": _closes_with_jump(0.005)}, {"AAA": _profile(800e6)})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert result.candidates == []
    assert result.rejections["reaction too weak"] == 1


@pytest.mark.parametrize(
    "cap, reason",
    [(50e6, "market cap outside band"), (50e9, "market cap outside band")],
)
def test_market_cap_band_excludes_both_tails(wire, cap, reason):
    wire([_event("AAA")], {"AAA": _closes_with_jump(0.10)}, {"AAA": _profile(cap)})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert result.rejections[reason] == 1


def test_otc_listings_are_excluded(wire):
    wire([_event("AAA")], {"AAA": _closes_with_jump(0.10)}, {"AAA": _profile(800e6, "PNK")})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert result.rejections["OTC / pink sheets"] == 1


def test_missing_profile_is_counted_not_raised(wire):
    wire([_event("AAA")], {"AAA": _closes_with_jump(0.10)}, {})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert result.rejections["no market data"] == 1


def test_candidates_rank_by_reaction_strength(wire):
    events = [_event("AAA"), _event("BBB")]
    closes = {"AAA": _closes_with_jump(0.05), "BBB": _closes_with_jump(0.15)}
    wire(events, closes, {"AAA": _profile(800e6), "BBB": _profile(900e6)})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert [c.ticker for c in result.candidates] == ["BBB", "AAA"]


def test_funnel_accounts_for_every_event(wire):
    events = [_event("AAA"), _event("BBB"), _event("CCC")]
    closes = {
        "AAA": _closes_with_jump(0.10),
        "BBB": _closes_with_jump(0.001),
        "CCC": _closes_with_jump(0.10),
    }
    profiles = {"AAA": _profile(800e6), "BBB": _profile(800e6), "CCC": _profile(50e9)}
    wire(events, closes, profiles)
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert result.events_found == 3
    assert len(result.candidates) + sum(result.rejections.values()) == 3


def test_funnel_counts_survivors_cut_by_the_candidate_cap(wire):
    # A heavy earnings week: more survivors than the cap. The ones ranked out
    # must still appear in the funnel, or it stops adding up to the event count.
    events = [_event(t) for t in ("AAA", "BBB", "CCC")]
    closes = {t: _closes_with_jump(0.10) for t in ("AAA", "BBB", "CCC")}
    wire(events, closes, {t: _profile(800e6) for t in ("AAA", "BBB", "CCC")})
    result = run_screen(ScreenConfig(max_candidates=2), as_of=date(2026, 8, 28))
    assert len(result.candidates) == 2
    assert result.rejections["ranked below top 2"] == 1
    assert len(result.candidates) + sum(result.rejections.values()) == 3


def test_throttled_ticker_is_not_counted_as_missing_data(wire):
    # Rate-limited means "never screened", not "dead symbol" -- keep them apart.
    events = [_event("AAA"), _event("BBB")]
    closes = {"AAA": _closes_with_jump(0.10), "BBB": _closes_with_jump(0.10)}
    wire(events, closes, {"AAA": _profile(800e6)}, throttled={"BBB"})
    result = run_screen(ScreenConfig(), as_of=date(2026, 8, 28))
    assert [c.ticker for c in result.candidates] == ["AAA"]
    assert result.rejections == {screen.THROTTLED: 1}
