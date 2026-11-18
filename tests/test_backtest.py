"""Backtest mechanics, especially the parts that could fake a result."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtest import analyze, panel, shares
from screener.backtest.events import _month_bounds


# --- event collection windows -------------------------------------------------

def test_month_bounds_splits_on_calendar_months():
    windows = _month_bounds(date(2024, 1, 15), date(2024, 3, 10))
    assert windows == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),   # leap year
        (date(2024, 3, 1), date(2024, 3, 10)),
    ]


def test_month_bounds_handles_a_year_boundary():
    windows = _month_bounds(date(2023, 12, 20), date(2024, 1, 5))
    assert windows[0] == (date(2023, 12, 20), date(2023, 12, 31))
    assert windows[1] == (date(2024, 1, 1), date(2024, 1, 5))


# --- point-in-time shares -----------------------------------------------------

def test_shares_uses_filed_date_not_measurement_date():
    # Measured 2024-03-31 but not filed until 2024-05-05: an April event must
    # not see it, or the backtest is using information that was not public.
    history = [(date(2024, 1, 30), 100.0), (date(2024, 5, 5), 200.0)]
    assert shares.shares_as_of(history, date(2024, 4, 15)) == 100.0
    assert shares.shares_as_of(history, date(2024, 5, 6)) == 200.0


def test_shares_excludes_a_filing_made_the_same_day():
    # Same-day filings are not safely tradeable information; require strictly before.
    history = [(date(2024, 5, 5), 200.0)]
    assert shares.shares_as_of(history, date(2024, 5, 5)) is None


def test_shares_returns_none_before_any_filing():
    assert shares.shares_as_of([(date(2024, 5, 5), 200.0)], date(2020, 1, 1)) is None


# --- forward returns ----------------------------------------------------------

def _series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_forward_return_is_market_adjusted():
    closes = _series([100, 110, 110, 110, 110, 110])
    bench = _series([100, 105, 105, 105, 105, 105])
    # Stock +10%, market +5% over the same bars, so the abnormal part is 5%.
    assert panel._forward_return(closes, bench, 0, 1) == pytest.approx(0.05)


def test_forward_return_is_nan_past_the_end_of_history():
    closes = _series([100, 101, 102])
    assert np.isnan(panel._forward_return(closes, closes, 1, 60))


# --- statistics ---------------------------------------------------------------

def _frame(rows):
    frame = pd.DataFrame(rows)
    frame["entry_month"] = pd.PeriodIndex(pd.to_datetime(frame["entry_date"]), freq="M")
    return frame


def test_monthly_clustering_ignores_within_month_sample_size():
    """Two months of identical means must give the same t-stat regardless of how
    many correlated events sit inside each month -- that is the whole point of
    clustering, and treating events as independent would inflate significance."""
    small = _frame([{"entry_date": d, "x": v}
                    for d, v in [("2024-01-10", 0.02), ("2024-02-10", 0.04)]])
    large = _frame([{"entry_date": d, "x": v}
                    for d in ["2024-01-10"] * 50 for v in [0.02]]
                   + [{"entry_date": d, "x": v} for d in ["2024-02-10"] * 50 for v in [0.04]])
    assert analyze._monthly_stat(small, "x").t_stat == pytest.approx(
        analyze._monthly_stat(large, "x").t_stat
    )


def test_monthly_stat_reports_event_and_month_counts():
    frame = _frame([{"entry_date": "2024-01-10", "x": 0.01},
                    {"entry_date": "2024-01-20", "x": 0.03},
                    {"entry_date": "2024-02-10", "x": 0.02}])
    stat = analyze._monthly_stat(frame, "x")
    assert stat.n_events == 3
    assert stat.n_months == 2
    assert stat.mean == pytest.approx(0.02)  # mean of monthly means: (0.02 + 0.02)/2


def test_quintiles_are_formed_within_each_month():
    """A month where every stock fell must still produce a full quintile spread,
    otherwise a bad month would land wholly in Q1 and confound drift with beta."""
    rows = [{"entry_date": "2024-01-10", "reaction": r, "fwd_20": 0.0}
            for r in np.linspace(-0.20, -0.01, 10)]
    rows += [{"entry_date": "2024-02-10", "reaction": r, "fwd_20": 0.0}
             for r in np.linspace(0.01, 0.20, 10)]
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    for month in grouped["entry_month"].unique():
        chunk = grouped[grouped["entry_month"] == month]
        assert set(chunk["quintile"]) == {1, 2, 3, 4, 5}


def test_spread_is_zero_when_reaction_carries_no_information():
    rng = np.random.default_rng(7)
    rows = []
    for month in range(1, 13):
        for _ in range(40):
            rows.append({
                "entry_date": f"2024-{month:02d}-10",
                "reaction": rng.normal(),
                "fwd_20": rng.normal(0, 0.05),   # independent of the reaction
            })
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    stat = analyze._spread_stat(grouped, "fwd_20")
    assert abs(stat.t_stat) < 1.96


def test_spread_detects_a_planted_drift():
    rng = np.random.default_rng(7)
    rows = []
    for month in range(1, 13):
        for _ in range(40):
            reaction = rng.normal()
            rows.append({
                "entry_date": f"2024-{month:02d}-10",
                "reaction": reaction,
                "fwd_20": 0.03 * reaction + rng.normal(0, 0.02),
            })
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    stat = analyze._spread_stat(grouped, "fwd_20")
    assert stat.mean > 0 and stat.t_stat > 1.96


def test_screen_filters_exclude_out_of_band_events():
    from screener.config import ScreenConfig

    frame = pd.DataFrame([
        {"price": 20.0, "dollar_volume": 5e6, "market_cap": 8e8},    # keep
        {"price": 2.0, "dollar_volume": 5e6, "market_cap": 8e8},     # price
        {"price": 20.0, "dollar_volume": 1e4, "market_cap": 8e8},    # volume
        {"price": 20.0, "dollar_volume": 5e6, "market_cap": 5e10},   # cap
    ])
    assert len(analyze.apply_screen_filters(frame, ScreenConfig())) == 1


def test_cost_sensitivity_reduces_return_monotonically():
    from screener.backtest.panel import HORIZONS

    rows = [{"entry_date": f"2024-{m:02d}-10", "reaction": r, f"fwd_{HORIZONS[2]}": 0.05}
            for m in range(1, 13) for r in np.linspace(-1, 1, 10)]
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    table = analyze.cost_sensitivity(grouped, HORIZONS[2])
    assert table["net_per_trade"].is_monotonic_decreasing


# --- transient error handling -------------------------------------------------

def test_fetch_retries_transient_server_errors(monkeypatch):
    """EDGAR returns intermittent 500s; a multi-year sweep makes thousands of
    requests, so one must not end the run."""
    import httpx
    from screener import edgar

    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status
            self.request = httpx.Request("GET", "https://example.test")

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=self.request, response=self)

        def json(self):
            return {"ok": True}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(500 if calls["n"] < 3 else 200)

    monkeypatch.setenv("SEC_USER_AGENT", "test test@example.com")
    monkeypatch.setattr(edgar.httpx, "get", fake_get)
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    monkeypatch.setattr(edgar.cache, "get", lambda k, ttl: None)
    monkeypatch.setattr(edgar.cache, "put", lambda k, v: None)

    assert edgar._fetch("https://example.test/retry", {}) == {"ok": True}
    assert calls["n"] == 3


def test_fetch_does_not_retry_client_errors(monkeypatch):
    """A 404 for a missing XBRL concept will never succeed; retrying wastes the
    rate-limit budget on thousands of companies that simply do not report it."""
    import httpx
    from screener import edgar

    calls = {"n": 0}

    class NotFound:
        status_code = 404
        request = httpx.Request("GET", "https://example.test")

        def raise_for_status(self):
            raise httpx.HTTPStatusError("nope", request=self.request, response=self)

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return NotFound()

    monkeypatch.setenv("SEC_USER_AGENT", "test test@example.com")
    monkeypatch.setattr(edgar.httpx, "get", fake_get)
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    monkeypatch.setattr(edgar.cache, "get", lambda k, ttl: None)
    monkeypatch.setattr(edgar.cache, "put", lambda k, v: None)

    with pytest.raises(httpx.HTTPStatusError):
        edgar._fetch("https://example.test/missing", {})
    assert calls["n"] == 1


def test_placebo_finds_nothing_when_there_is_nothing():
    """The permutation null must not flag a signal in pure noise."""
    rng = np.random.default_rng(3)
    rows = [{"entry_date": f"2024-{m:02d}-10", "reaction": rng.normal(),
             "fwd_20": rng.normal(0, 0.05)}
            for m in range(1, 13) for _ in range(40)]
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    result = analyze.placebo_spread(grouped, 20, n_permutations=50)
    assert result["p_value"] > 0.05


def test_placebo_flags_a_planted_signal():
    rng = np.random.default_rng(3)
    rows = []
    for month in range(1, 13):
        for _ in range(40):
            reaction = rng.normal()
            rows.append({"entry_date": f"2024-{month:02d}-10", "reaction": reaction,
                         "fwd_20": 0.05 * reaction + rng.normal(0, 0.01)})
    grouped = analyze.add_groupings(pd.DataFrame(rows))
    result = analyze.placebo_spread(grouped, 20, n_permutations=50)
    assert result["p_value"] < 0.05
    assert result["observed"] > abs(result["placebo_mean"])


# --- data quality guards ------------------------------------------------------

def test_panel_rejects_a_series_with_a_non_positive_close():
    """Yahoo's split adjustment can emit negative closes on sub-penny names.
    CODQL carried -0.1177, producing a -745% 'return' that no long position can
    experience. Such a series is corrupt, not extreme, and must be dropped."""
    from screener.edgar import EarningsEvent

    idx = pd.bdate_range("2024-01-01", periods=140)
    closes = pd.Series(1.0, index=idx)
    closes.iloc[100] = -0.12
    bars = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": 1e6}, index=idx)
    bench = pd.Series(100.0, index=idx)
    event = EarningsEvent("BAD", "1", "Bad Co", idx[100].date(), "x", ("2.02",))
    assert panel._event_row(event, bars, bench, [(date(2020, 1, 1), 1e6)]) is None


def test_panel_accepts_an_all_positive_series():
    from screener.edgar import EarningsEvent

    idx = pd.bdate_range("2024-01-01", periods=140)
    closes = pd.Series(np.linspace(10, 12, 140), index=idx)
    bars = pd.DataFrame({"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
                         "Close": closes, "Volume": 1e6}, index=idx)
    bench = pd.Series(100.0, index=idx)
    event = EarningsEvent("OK", "1", "Ok Co", idx[100].date(), "x", ("2.02",))
    assert panel._event_row(event, bars, bench, [(date(2020, 1, 1), 1e6)]) is not None


def test_winsorize_clips_the_tails_at_the_stated_percentiles():
    frame = pd.DataFrame({"fwd_20": list(np.linspace(-0.5, 0.5, 200)) + [666.0, -50.0]})
    out = analyze.winsorize(frame, ["fwd_20"], tails=0.01)
    assert out["fwd_20"].max() < 1.0
    assert out["fwd_20"].min() > -1.0


def test_winsorize_leaves_the_centre_of_the_distribution_alone():
    """Clipping must not shift the bulk of the data, or it would bias the very
    means it is meant to protect."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"fwd_20": rng.normal(0.01, 0.1, 5000)})
    out = analyze.winsorize(frame, ["fwd_20"], tails=0.01)
    assert out["fwd_20"].median() == pytest.approx(frame["fwd_20"].median())
    inner = frame["fwd_20"].between(frame["fwd_20"].quantile(0.05),
                                    frame["fwd_20"].quantile(0.95))
    assert (out.loc[inner, "fwd_20"] == frame.loc[inner, "fwd_20"]).all()


def test_winsorize_skips_columns_with_too_little_data():
    frame = pd.DataFrame({"fwd_20": [0.1, 99.0, -0.1]})
    out = analyze.winsorize(frame, ["fwd_20"])
    assert out["fwd_20"].max() == 99.0
