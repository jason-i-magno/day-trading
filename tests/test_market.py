"""Signal and liquidity measures."""

import numpy as np
import pandas as pd

from screener.market import _abnormal_return, corwin_schultz_spread


def _series(values, start="2026-08-17"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_abnormal_return_subtracts_the_market():
    closes = _series([100, 100, 110, 110])
    benchmark = _series([100, 100, 105, 105])
    # Stock +10% across the window, market +5%, so the company-specific move is 5%.
    abnormal, _ = _abnormal_return(closes, benchmark, pd.Timestamp("2026-08-18"))
    assert abnormal == pytest_approx(0.05)


def test_window_captures_an_after_close_filing():
    # Filed after the bell on day 1: the tape does not move until day 2.
    closes = _series([100, 100, 120, 120])
    benchmark = _series([100, 100, 100, 100])
    abnormal, _ = _abnormal_return(closes, benchmark, pd.Timestamp("2026-08-18"))
    assert abnormal == pytest_approx(0.20)


def test_window_captures_a_before_open_filing():
    # Filed pre-market on day 1: the move lands the same session.
    closes = _series([100, 120, 120, 120])
    benchmark = _series([100, 100, 100, 100])
    abnormal, _ = _abnormal_return(closes, benchmark, pd.Timestamp("2026-08-18"))
    assert abnormal == pytest_approx(0.20)


def test_days_since_event_counts_bars_after_the_reaction():
    closes = _series([100, 100, 110, 110, 110, 110])
    benchmark = _series([100] * 6)
    _, days = _abnormal_return(closes, benchmark, pd.Timestamp("2026-08-18"))
    assert days == 3


def test_event_before_available_history_is_rejected():
    closes = _series([100, 101, 102])
    benchmark = _series([100, 100, 100])
    abnormal, days = _abnormal_return(closes, benchmark, pd.Timestamp("2026-01-01"))
    assert days == -1
    assert np.isnan(abnormal)


def test_spread_estimator_ranks_wide_above_tight():
    # The absolute level is unreliable, but a name whose intraday range is
    # consistently wider must not estimate tighter.
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.5, 40))
    tight_h, tight_l = _series(mid * 1.001), _series(mid * 0.999)
    wide_h, wide_l = _series(mid * 1.02), _series(mid * 0.98)
    assert corwin_schultz_spread(wide_h, wide_l) > corwin_schultz_spread(tight_h, tight_l)


def test_spread_estimator_needs_enough_bars():
    assert np.isnan(corwin_schultz_spread(_series([10, 11]), _series([9, 10])))


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)
