"""Price, liquidity, and event-reaction measures from free daily data.

The screen needs three things per candidate: is it in the right size and
liquidity band, how hard did it react to earnings, and can it be traded without
the spread eating the move. All three come from daily OHLCV bars plus a
market-cap lookup -- no paid feed required.
"""

from __future__ import annotations

import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from .config import BENCHMARK_TICKER

# Enough history for a stable liquidity and spread estimate.
HISTORY_DAYS = 120
_SPREAD_WINDOW = 20

# Yahoo throttles bursts of profile lookups. In peak earnings season 1,500+
# tickers pass through here, and before these retries existed the throttle hit
# a few hundred tickers in and silently dropped the rest -- alphabetically,
# since lookups run in sorted order. The 2026-11-04 and 11-11 runs logged only
# A-H tickers because of it. The limit resets in ~20s.
PROFILE_BACKOFF = 20.0
PROFILE_MAX_STALLS = 4


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the screen needs to know about one candidate's tape."""

    ticker: str
    price: float
    market_cap: float
    avg_dollar_volume: float
    abnormal_return: float
    spread_estimate: float
    days_since_event: int
    exchange: str


def download_history(tickers: list[str]) -> pd.DataFrame:
    """Fetch daily bars for `tickers` plus the benchmark, as a ticker/field frame."""
    symbols = sorted({*tickers, BENCHMARK_TICKER})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = yf.download(
            symbols,
            period=f"{HISTORY_DAYS}d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    if isinstance(frame.columns, pd.MultiIndex):
        return frame
    # A single successful ticker comes back flat; normalise to the same shape.
    return pd.concat({symbols[0]: frame}, axis=1)


def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, YFRateLimitError) or "Too Many Requests" in str(exc)


def fetch_profiles(
    tickers: list[str], workers: int = 8
) -> tuple[dict[str, dict], set[str]]:
    """Look up market cap and listing venue for each ticker, in parallel.

    Returns `(profiles, throttled)`. Tickers that fail lookup outright are
    omitted rather than raised on: delisted and renamed symbols are routine in
    a universe drawn straight from EDGAR. A rate-limited lookup is different --
    it says nothing about the ticker -- so it is retried after a backoff, and
    anything Yahoo is still refusing after PROFILE_MAX_STALLS rounds without
    progress comes back in `throttled`, to be counted as lost coverage rather
    than mistaken for a dead symbol.
    """
    profiles: dict[str, dict] = {}
    pending = list(tickers)
    stalls = 0
    while pending:
        # Once one lookup is throttled, stop sending more this round: every
        # further request fails anyway and only prolongs the limit.
        throttled_now = threading.Event()

        def one(ticker: str) -> tuple[str, dict | None, bool]:
            if throttled_now.is_set():
                return ticker, None, True
            try:
                info = yf.Ticker(ticker).fast_info
                return ticker, {
                    "market_cap": info.get("marketCap"),
                    "exchange": info.get("exchange") or "",
                }, False
            except Exception as exc:
                if _is_rate_limit(exc):
                    throttled_now.set()
                    return ticker, None, True
                return ticker, None, False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, pending))

        found = {t: p for t, p, _ in results if p is not None}
        profiles.update(found)
        pending = [t for t, _, retry in results if retry]
        if not pending:
            break
        stalls = 0 if found else stalls + 1
        if stalls >= PROFILE_MAX_STALLS:
            return profiles, set(pending)
        time.sleep(PROFILE_BACKOFF * (stalls + 1))
    return profiles, set()


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> float:
    """Estimate the effective bid-ask spread from daily high/low ranges.

    Implements the Corwin-Schultz (2012) estimator, which infers the spread
    from the observation that a two-day high-low range contains one day of
    spread while two single-day ranges contain two. Free daily bars carry no
    bid/ask, so this stands in for the quoted spread we cannot see.

    Includes the paper's overnight adjustment: an unadjusted gap between
    sessions inflates the two-day range and gets misread as spread, which
    badly overstates the cost on gappy names -- exactly the earnings reactions
    this screen selects for.

    Even adjusted, the estimator runs high in absolute terms and is best read
    as a *relative* liquidity ranking, not a literal cost input. Use it to
    prefer the tighter of two candidates, not to predict a fill.

    Returns a round-trip fraction of price (0.004 == 40bp), or NaN if there is
    not enough clean data.
    """
    high, low = high.dropna(), low.dropna()
    if len(high) < 3 or len(low) < 3:
        return float("nan")

    high, low = high.tail(_SPREAD_WINDOW), low.tail(_SPREAD_WINDOW)
    denominator = 3 - 2 * math.sqrt(2)
    estimates: list[float] = []

    for i in range(len(high) - 1):
        h1, l1 = high.iloc[i], low.iloc[i]
        h2, l2 = high.iloc[i + 1], low.iloc[i + 1]
        if min(h1, l1, h2, l2) <= 0:
            continue

        # Overnight adjustment: shift the second session back onto the first
        # when the two do not overlap, so a gap is not counted as spread.
        if l2 > h1:
            gap = l2 - h1
            h2, l2 = h2 - gap, l2 - gap
        elif h2 < l1:
            gap = l1 - h2
            h2, l2 = h2 + gap, l2 + gap

        beta = math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2
        gamma = math.log(max(h1, h2) / min(l1, l2)) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denominator - math.sqrt(gamma / denominator)
        spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
        # The estimator is noisy trade-by-trade and can go negative; the paper
        # treats negatives as zero and averages across pairs.
        estimates.append(max(spread, 0.0))

    return sum(estimates) / len(estimates) if estimates else float("nan")


def _abnormal_return(
    closes: pd.Series, benchmark: pd.Series, event_day: pd.Timestamp
) -> tuple[float, int]:
    """Return the benchmark-adjusted reaction to the announcement, and its age.

    The window spans the close before the filing to the close after it, which
    captures the move whether the release landed before the open or after the
    bell. Subtracting the benchmark separates the company's news from the day's
    market direction.
    """
    index = closes.index
    position = index.searchsorted(event_day)
    if position == 0 or position >= len(index):
        return float("nan"), -1

    before = index[position - 1]
    after = index[min(position + 1, len(index) - 1)]
    if before not in benchmark.index or after not in benchmark.index:
        return float("nan"), -1

    stock_return = closes[after] / closes[before] - 1
    market_return = benchmark[after] / benchmark[before] - 1
    days_elapsed = len(index) - 1 - index.searchsorted(after)
    return float(stock_return - market_return), int(days_elapsed)


def build_snapshot(
    ticker: str, frame: pd.DataFrame, profile: dict, event_date: date
) -> MarketSnapshot | None:
    """Assemble one candidate's measures, or None if its data is unusable."""
    if ticker not in frame.columns.get_level_values(0):
        return None

    bars = frame[ticker].dropna(how="all")
    benchmark = frame[BENCHMARK_TICKER]["Close"].dropna()
    if len(bars) < 25 or benchmark.empty:
        return None

    closes = bars["Close"].dropna()
    if closes.empty:
        return None

    abnormal, days_since = _abnormal_return(
        closes, benchmark, pd.Timestamp(event_date)
    )
    if days_since < 0 or not math.isfinite(abnormal):
        return None

    dollar_volume = (bars["Close"] * bars["Volume"]).tail(60).mean()
    market_cap = profile.get("market_cap")
    if not market_cap or not math.isfinite(float(market_cap)):
        return None

    return MarketSnapshot(
        ticker=ticker,
        price=float(closes.iloc[-1]),
        market_cap=float(market_cap),
        avg_dollar_volume=float(dollar_volume),
        abnormal_return=abnormal,
        spread_estimate=corwin_schultz_spread(bars["High"], bars["Low"]),
        days_since_event=days_since,
        exchange=str(profile.get("exchange", "")),
    )
