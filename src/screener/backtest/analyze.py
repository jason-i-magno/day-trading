"""Statistics on the event panel.

Two design choices carry most of the weight here.

First, quintiles rather than a single long bucket. Post-earnings drift predicts
that strong reactions keep going *and* weak ones keep falling; if every quintile
drifts the same direction we are looking at market exposure, not drift. The
long-short spread is the clean test because it is market-neutral by construction.

Second, standard errors are clustered by calendar month. Holding periods of 20-60
trading days overlap heavily, so treating events as independent would shrink the
standard errors and manufacture significance. Averaging within a month first and
testing the monthly series is the conservative alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ScreenConfig
from .panel import HORIZONS

COST_SCENARIOS_BPS = (0, 10, 25, 50, 100)
N_QUINTILES = 5


@dataclass(frozen=True)
class Stat:
    """A mean return with its clustered t-statistic."""

    mean: float
    t_stat: float
    n_events: int
    n_months: int

    @property
    def significant(self) -> bool:
        # Two-sided 5% on a t-distribution with plenty of months.
        return abs(self.t_stat) > 1.96


def apply_screen_filters(panel: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    """Restrict the panel to what the live screen would have been allowed to see."""
    mask = (
        panel["price"].ge(config.min_price)
        & panel["dollar_volume"].ge(config.min_avg_dollar_volume)
        & panel["market_cap"].between(config.min_market_cap, config.max_market_cap)
    )
    return panel[mask].copy()


def _monthly_stat(frame: pd.DataFrame, column: str) -> Stat:
    """Mean of `column`, with the t-stat computed across monthly means."""
    values = frame[["entry_month", column]].dropna()
    if values.empty:
        return Stat(np.nan, np.nan, 0, 0)
    monthly = values.groupby("entry_month")[column].mean()
    n_months = len(monthly)
    if n_months < 2:
        return Stat(float(monthly.mean()), np.nan, len(values), n_months)
    standard_error = monthly.std(ddof=1) / np.sqrt(n_months)
    mean = float(monthly.mean())
    t_stat = mean / standard_error if standard_error > 0 else np.nan
    return Stat(mean, float(t_stat), len(values), n_months)


WINSOR_TAILS = 0.01


def winsorize(frame: pd.DataFrame, columns: list[str], tails: float = WINSOR_TAILS) -> pd.DataFrame:
    """Clip return columns at their tail percentiles.

    Sub-dollar stocks generate arithmetically valid but economically meaningless
    returns -- a $0.0009 quote moving one tick is +66,000%, and a handful of such
    events dominate the mean of a 15,000-event quintile. Clipping at the 1st and
    99th percentiles is standard practice in cross-sectional asset pricing and
    keeps a few microcaps from deciding the result.
    """
    frame = frame.copy()
    for column in columns:
        if column in frame:
            values = frame[column].dropna()
            if len(values) > 100:
                low, high = values.quantile(tails), values.quantile(1 - tails)
                frame[column] = frame[column].clip(low, high)
    return frame


def add_groupings(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the entry month and the within-month reaction quintile.

    Quintiles are formed *within* each month so the comparison is always between
    stocks that reported around the same time, rather than letting a quiet quarter
    land entirely in one bucket. Returns are winsorized first so that extreme
    microcap prints cannot drive a quintile mean.
    """
    frame = winsorize(frame, [f"fwd_{h}" for h in HORIZONS] + ["reaction"])
    frame["entry_month"] = pd.PeriodIndex(pd.to_datetime(frame["entry_date"]), freq="M")
    frame["quintile"] = (
        frame.groupby("entry_month")["reaction"]
        .transform(lambda s: pd.qcut(s, N_QUINTILES, labels=False, duplicates="drop") + 1
                   if s.notna().sum() >= N_QUINTILES else np.nan)
    )
    return frame.dropna(subset=["quintile"])


def quintile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return by reaction quintile and horizon, plus the Q5-Q1 spread."""
    records = []
    for horizon in HORIZONS:
        column = f"fwd_{horizon}"
        row: dict[str, object] = {"horizon": horizon}
        for quintile in range(1, N_QUINTILES + 1):
            stat = _monthly_stat(frame[frame["quintile"] == quintile], column)
            row[f"Q{quintile}"] = stat.mean
        spread = _spread_stat(frame, column)
        row["Q5-Q1"] = spread.mean
        row["t"] = spread.t_stat
        row["n"] = spread.n_events
        records.append(row)
    return pd.DataFrame(records).set_index("horizon")


def _spread_stat(frame: pd.DataFrame, column: str) -> Stat:
    """Long-short spread, tested on the monthly series of Q5 minus Q1."""
    values = frame[["entry_month", "quintile", column]].dropna()
    if values.empty:
        return Stat(np.nan, np.nan, 0, 0)
    monthly = (
        values.groupby(["entry_month", "quintile"])[column].mean().unstack("quintile")
    )
    if N_QUINTILES not in monthly.columns or 1 not in monthly.columns:
        return Stat(np.nan, np.nan, len(values), 0)
    spread = (monthly[N_QUINTILES] - monthly[1]).dropna()
    if len(spread) < 2:
        return Stat(float(spread.mean()) if len(spread) else np.nan, np.nan, len(values), len(spread))
    standard_error = spread.std(ddof=1) / np.sqrt(len(spread))
    mean = float(spread.mean())
    return Stat(mean, float(mean / standard_error) if standard_error > 0 else np.nan,
                len(values), len(spread))


def cost_sensitivity(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Top-quintile return after round-trip costs, and a rough annualisation.

    The annualised column assumes capital recycles into non-overlapping trades at
    this horizon. It ignores the fact that qualifying events do not arrive evenly,
    so treat it as an upper bound rather than a forecast.
    """
    column = f"fwd_{horizon}"
    top = frame[frame["quintile"] == N_QUINTILES]
    stat = _monthly_stat(top, column)
    trades_per_year = 252 / horizon
    records = []
    for bps in COST_SCENARIOS_BPS:
        net = stat.mean - bps / 10_000
        records.append(
            {
                "round_trip_bps": bps,
                "net_per_trade": net,
                "annualised": (1 + net) ** trades_per_year - 1 if net > -1 else np.nan,
            }
        )
    return pd.DataFrame(records).set_index("round_trip_bps")


def yearly_stability(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Long-short spread year by year -- an effect present in only one year is noise."""
    column = f"fwd_{horizon}"
    records = []
    for year, chunk in frame.groupby(frame["entry_month"].dt.year):
        stat = _spread_stat(chunk, column)
        records.append({"year": year, "spread": stat.mean, "t": stat.t_stat, "n": stat.n_events})
    return pd.DataFrame(records).set_index("year")


def size_breakdown(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Spread by market-cap band, to test the literature's claim that drift is
    strongest where analyst coverage is thinnest."""
    bands = [(0, 3e8, "<300M"), (3e8, 1e9, "300M-1B"), (1e9, 2e9, "1-2B"),
             (2e9, 1e10, "2-10B"), (1e10, np.inf, ">10B")]
    records = []
    for low, high, label in bands:
        chunk = frame[frame["market_cap"].between(low, high, inclusive="left")]
        stat = _spread_stat(chunk, f"fwd_{horizon}")
        records.append({"band": label, "spread": stat.mean, "t": stat.t_stat, "n": stat.n_events})
    return pd.DataFrame(records).set_index("band")


def placebo_spread(
    frame: pd.DataFrame, horizon: int, n_permutations: int = 200, seed: int = 0
) -> dict[str, float]:
    """Compare the real Q5-Q1 spread against shuffled reactions.

    Reactions are permuted *within* each month, destroying any link between the
    announcement move and what followed while leaving every other feature of the
    data intact -- the same stocks, months, and return distribution. If the real
    spread does not stand out against this null, whatever the quintile table
    shows is an artefact of the pipeline rather than a signal in the market.

    Returns the observed spread, the placebo mean and standard deviation, and an
    empirical p-value: the share of shuffles that beat the real result.
    """
    column = f"fwd_{horizon}"
    observed = _spread_stat(frame, column).mean
    if not np.isfinite(observed):
        return {"observed": np.nan, "placebo_mean": np.nan,
                "placebo_std": np.nan, "p_value": np.nan}

    rng = np.random.default_rng(seed)
    working = frame[["entry_month", "reaction", column]].dropna().copy()
    placebo: list[float] = []
    for _ in range(n_permutations):
        shuffled = working.copy()
        shuffled["reaction"] = (
            working.groupby("entry_month")["reaction"]
            .transform(lambda s: rng.permutation(s.to_numpy()))
        )
        shuffled = add_groupings_precomputed(shuffled)
        stat = _spread_stat(shuffled, column)
        if np.isfinite(stat.mean):
            placebo.append(stat.mean)

    if not placebo:
        return {"observed": observed, "placebo_mean": np.nan,
                "placebo_std": np.nan, "p_value": np.nan}

    array = np.asarray(placebo)
    # Two-sided: how often does chance produce a spread at least this large?
    p_value = float((np.abs(array) >= abs(observed)).mean())
    return {
        "observed": observed,
        "placebo_mean": float(array.mean()),
        "placebo_std": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
        "p_value": p_value,
    }


def add_groupings_precomputed(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-quintile a frame that already carries `entry_month`."""
    frame = frame.copy()
    frame["quintile"] = (
        frame.groupby("entry_month")["reaction"]
        .transform(lambda s: pd.qcut(s, N_QUINTILES, labels=False, duplicates="drop") + 1
                   if s.notna().sum() >= N_QUINTILES else np.nan)
    )
    return frame.dropna(subset=["quintile"])


def add_groupings_by(frame: pd.DataFrame, signal: str, extra_keys: list[str] | None = None
                     ) -> pd.DataFrame:
    """Quintile `signal` within month, optionally splitting further by `extra_keys`.

    The text panel needs this: 10-Q and 10-K similarity scores live on different
    scales (quarterlies are shorter and vary more), so ranking them together
    would sort by form type rather than by how much the company actually changed.
    """
    frame = winsorize(frame, [f"fwd_{h}" for h in HORIZONS])
    frame = frame.copy()
    frame["entry_month"] = pd.PeriodIndex(pd.to_datetime(frame["entry_date"]), freq="M")
    keys = ["entry_month"] + (extra_keys or [])
    frame["quintile"] = (
        frame.groupby(keys)[signal]
        .transform(lambda s: pd.qcut(s, N_QUINTILES, labels=False, duplicates="drop") + 1
                   if s.notna().sum() >= N_QUINTILES else np.nan)
    )
    return frame.dropna(subset=["quintile"])
