"""Turn earnings events into a analysable panel: one row per event.

Every column here is computed from information available *before* it would have
been acted on. Entry is the close of D+1 -- the same point the live screen enters,
after the initial reaction has played out -- so the reaction used for ranking and
the return being measured never overlap.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..edgar import EarningsEvent
from . import prices, shares

BENCHMARK = "SPY"
HORIZONS = (5, 10, 20, 40, 60)
_ADV_WINDOW = 60


def _forward_return(closes: pd.Series, bench: pd.Series, i: int, horizon: int) -> float:
    """Benchmark-adjusted return from bar `i` to bar `i + horizon`."""
    j = i + horizon
    if j >= len(closes):
        return np.nan
    entry_date, exit_date = closes.index[i], closes.index[j]
    if entry_date not in bench.index or exit_date not in bench.index:
        return np.nan
    stock = closes.iloc[j] / closes.iloc[i] - 1
    market = bench[exit_date] / bench[entry_date] - 1
    return float(stock - market)


def _event_row(
    event: EarningsEvent,
    bars: pd.DataFrame,
    bench: pd.Series,
    share_history: list[tuple[date, float]],
) -> dict | None:
    closes = bars["Close"].dropna()
    if len(closes) < _ADV_WINDOW + max(HORIZONS) // 2:
        return None

    # Yahoo's split adjustment occasionally produces non-positive closes on
    # sub-penny names -- CODQL carried a close of -0.1177, which yields a
    # "return" of -745% and swamps a quintile mean on its own. A price that is
    # not strictly positive is corrupt, not extreme, so drop the series.
    if (closes <= 0).any():
        return None

    position = closes.index.searchsorted(pd.Timestamp(event.filing_date))
    # Need one bar before the filing and two after: D-1 for the pre-event mark,
    # D+1 as the entry.
    if position < _ADV_WINDOW or position + 1 >= len(closes):
        return None

    pre, entry_idx = position - 1, position + 1
    pre_date, entry_date = closes.index[pre], closes.index[entry_idx]
    if pre_date not in bench.index or entry_date not in bench.index:
        return None

    # Announcement reaction: D-1 close to D+1 close, which captures the move
    # whether the release landed before the open or after the bell.
    stock_move = closes.iloc[entry_idx] / closes.iloc[pre] - 1
    market_move = bench[entry_date] / bench[pre_date] - 1
    reaction = float(stock_move - market_move)

    window = bars.iloc[pre - _ADV_WINDOW : pre]
    dollar_volume = float((window["Close"] * window["Volume"]).mean())
    price = float(closes.iloc[pre])

    outstanding = shares.shares_as_of(share_history, event.filing_date)
    market_cap = float(outstanding) * price if outstanding else np.nan

    row = {
        "ticker": event.ticker,
        "cik": event.cik,
        "filing_date": event.filing_date,
        "entry_date": entry_date.date(),
        "reaction": reaction,
        "price": price,
        "dollar_volume": dollar_volume,
        "market_cap": market_cap,
    }
    for horizon in HORIZONS:
        row[f"fwd_{horizon}"] = _forward_return(closes, bench, entry_idx, horizon)
    return row


def build_panel(events: list[EarningsEvent], verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the event panel and a coverage report.

    The coverage counts matter as much as the panel: events dropped for missing
    prices are exactly where survivorship bias would creep back in, so the caller
    needs to see how many there were.
    """
    bench_bars = prices.load(BENCHMARK)
    if bench_bars is None:
        raise RuntimeError(f"{BENCHMARK} prices not cached; run the price fetch first")
    bench = bench_bars["Close"].dropna()

    by_ticker: dict[str, list[EarningsEvent]] = {}
    for event in events:
        by_ticker.setdefault(event.ticker, []).append(event)

    rows: list[dict] = []
    # Drop reasons must sum with `kept` to `events`; diagnostics do not drop
    # anything and are tracked separately so the arithmetic stays checkable.
    coverage = {"events": len(events), "tickers": len(by_ticker),
                "dropped_no_prices": 0, "dropped_bad_window": 0, "kept": 0,
                "diag_events_without_shares": 0}

    for index, (ticker, ticker_events) in enumerate(sorted(by_ticker.items()), 1):
        bars = prices.load(ticker)
        if bars is None:
            coverage["dropped_no_prices"] += len(ticker_events)
            continue
        share_history = shares.shares_history(ticker_events[0].cik)
        if not share_history:
            coverage["diag_events_without_shares"] += len(ticker_events)
        for event in ticker_events:
            row = _event_row(event, bars, bench, share_history)
            if row is None:
                coverage["dropped_bad_window"] += 1
            else:
                rows.append(row)
                coverage["kept"] += 1
        if verbose and index % 200 == 0:
            print(f"  panel {index}/{len(by_ticker)} tickers  rows={len(rows)}", flush=True)

    panel = pd.DataFrame(rows)
    return panel, coverage
