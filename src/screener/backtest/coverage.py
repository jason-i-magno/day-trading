"""How much of the backtest's missing data is actually survivorship?

The backtest drops every event whose ticker has no prices. Those drops get
treated as one undifferentiated bias, but they are not the same thing:

  - A company that **delisted long after** the event traded normally at the time.
    Excluding it is a pure loss of a valid observation -- real survivorship.
  - A company that **delisted right after** the event is a genuinely hard case;
    its holding-period return is partly unobservable at any price.
  - A company that **still trades under a different symbol** is not survivorship
    at all. It is a symbol-matching failure, and it is fixable for free.

Separating these decides whether paying for survivorship-free price data is
worth it, so the split is the whole point of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..edgar import EarningsEvent
from . import delistings, prices

# The longest holding period the backtest measures is 60 trading days; ~90
# calendar days covers it with room for weekends and holidays.
TRADEABLE_WINDOW_DAYS = 90


@dataclass(frozen=True)
class CoverageReport:
    events_by_class: pd.Series
    delisting_lag: pd.Series
    float_distribution: pd.Series
    tradeable_losses: int
    total_dropped: int


def dropped_events(events: list[EarningsEvent]) -> list[EarningsEvent]:
    """Events the backtest could not use because the ticker has no prices."""
    cached = prices.cached_tickers()
    return [e for e in events if e.ticker not in cached]


def classify(events: list[EarningsEvent], verbose: bool = True) -> pd.DataFrame:
    """Attach listing status to each dropped event, one lookup per company."""
    by_cik: dict[str, list[EarningsEvent]] = {}
    for event in events:
        by_cik.setdefault(event.cik, []).append(event)

    rows: list[dict] = []
    for index, (cik, cik_events) in enumerate(sorted(by_cik.items()), 1):
        status = delistings.fetch_status(cik)
        for event in cik_events:
            if status is None:
                rows.append({"ticker": event.ticker, "cik": cik,
                             "filing_date": event.filing_date, "company": event.company,
                             "classification": "not in EDGAR", "delisting_date": None,
                             "days_to_delisting": np.nan, "sic": ""})
                continue
            lag = ((status.delisting_date - event.filing_date).days
                   if status.delisting_date else np.nan)
            rows.append({
                "ticker": event.ticker, "cik": cik, "filing_date": event.filing_date,
                "company": event.company, "classification": status.classification,
                "delisting_date": status.delisting_date, "days_to_delisting": lag,
                "sic": status.sic,
            })
        if verbose and index % 250 == 0:
            print(f"  status {index}/{len(by_cik)} companies", flush=True)
    return pd.DataFrame(rows)


def add_public_float(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Look up public float for delisted companies, as a market-cap proxy.

    Only delisted companies are worth the request budget: for everything else we
    either have prices already or the company still trades.
    """
    frame = frame.copy()
    frame["public_float"] = np.nan
    targets = frame[frame["classification"] == "delisted"]
    unique = targets.drop_duplicates("cik")

    lookups: dict[str, float | None] = {}
    for index, (_, row) in enumerate(unique.iterrows(), 1):
        lookups[row["cik"]] = delistings.public_float(row["cik"], row["filing_date"])
        if verbose and index % 250 == 0:
            print(f"  float {index}/{len(unique)} companies", flush=True)

    mask = frame["classification"] == "delisted"
    frame.loc[mask, "public_float"] = frame.loc[mask, "cik"].map(lookups)
    return frame


def summarise(frame: pd.DataFrame) -> CoverageReport:
    by_class = frame["classification"].value_counts()

    delisted = frame[frame["classification"] == "delisted"]
    lag = delisted["days_to_delisting"].dropna()
    buckets = pd.cut(
        lag,
        bins=[-np.inf, 0, 30, 90, 365, 730, np.inf],
        labels=["before the event", "0-30d", "30-90d", "90d-1y", "1-2y", ">2y"],
    ).value_counts().sort_index()

    floats = delisted["public_float"].dropna()
    float_buckets = pd.cut(
        floats,
        bins=[0, 5e7, 3e8, 1e9, 2e9, np.inf],
        labels=["<$50M", "$50-300M", "$300M-1B", "$1-2B", ">$2B"],
    ).value_counts().sort_index() if len(floats) else pd.Series(dtype=int)

    # Events whose company kept trading well past the whole holding period:
    # these are observations the backtest lost for no good reason.
    tradeable = int((lag > TRADEABLE_WINDOW_DAYS).sum())

    return CoverageReport(
        events_by_class=by_class,
        delisting_lag=buckets,
        float_distribution=float_buckets,
        tradeable_losses=tradeable,
        total_dropped=len(frame),
    )
