"""Persistent local price store.

A backtest needs years of daily bars for thousands of tickers. Yahoo rate-limits
hard on bulk access (HTTP 429 within a single large batch), so the rule here is:
download each ticker's full history once, keep it on disk forever, and never ask
twice. Tickers that genuinely have no data are remembered as misses so repeated
runs do not re-attempt them.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

STORE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "prices"
MISSES_FILE = STORE_DIR / "_misses.json"

# Yahoo tolerates modest batches; larger ones trip the crumb rate limiter.
BATCH_SIZE = 20
BATCH_PAUSE = 1.5
MAX_RETRIES = 4
RETRY_BACKOFF = 20.0
CONFIRM_PAUSE = 2.0

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _path(ticker: str) -> Path:
    # Ticker symbols contain characters that are awkward in filenames.
    safe = ticker.replace("/", "-").replace(".", "_")
    return STORE_DIR / f"{safe}.parquet"


def _load_misses() -> set[str]:
    if not MISSES_FILE.exists():
        return set()
    try:
        return set(json.loads(MISSES_FILE.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_misses(misses: set[str]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    MISSES_FILE.write_text(json.dumps(sorted(misses)))


def cached_tickers() -> set[str]:
    if not STORE_DIR.exists():
        return set()
    return {p.stem.replace("-", "/").replace("_", ".") for p in STORE_DIR.glob("*.parquet")}


def load(ticker: str) -> pd.DataFrame | None:
    """Read one ticker's cached bars, or None if not stored."""
    path = _path(ticker)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _download_batch(batch: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Download one batch, backing off through Yahoo's rate limiter."""
    for attempt in range(MAX_RETRIES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.download(
                batch, start=start, end=end, group_by="ticker",
                auto_adjust=True, progress=False, threads=False,
            )
        if frame is not None and not frame.empty:
            if not isinstance(frame.columns, pd.MultiIndex):
                frame = pd.concat({batch[0]: frame}, axis=1)
            return frame
        # An empty frame for a whole batch means throttling far more often than
        # it means every symbol is dead, so wait longer and try again.
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return None


def _download_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Confirm one ticker individually. Returns its bars, or None if truly absent."""
    for attempt in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = yf.download(ticker, start=start, end=end, progress=False,
                                auto_adjust=True, threads=False)
        if frame is not None and not frame.empty:
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            bars = frame[_OHLCV].dropna(how="all")
            if bars["Close"].notna().sum() >= 30:
                return bars
            return None
        if attempt == 0:
            # Short pause only: this is a per-ticker confirmation, not a batch
            # retry, so the long rate-limit backoff would cost minutes per dead
            # symbol for no benefit.
            time.sleep(CONFIRM_PAUSE)
    return None


def ensure(tickers: list[str], start: str, end: str, verbose: bool = True) -> dict[str, int]:
    """Fetch and store any tickers not already cached.

    Returns counts of what happened, so callers can report how much of the
    requested universe actually has data -- a number that matters, because
    silently dropping unpriceable tickers reintroduces survivorship bias.
    """
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    misses = _load_misses()
    have = cached_tickers()
    todo = sorted(set(tickers) - have - misses)

    stats = {"cached": len(set(tickers) & have), "known_missing": len(set(tickers) & misses),
             "fetched": 0, "failed": 0, "recovered": 0}
    if not todo:
        return stats

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        frame = _download_batch(batch, start, end)
        if frame is None:
            # A whole batch coming back empty means throttling far more often
            # than it means twenty dead symbols. Count it, but do NOT record
            # misses -- that would permanently blacklist live tickers and fake
            # a survivorship story. A later run retries them.
            stats["failed"] += len(batch)
        else:
            unresolved: list[str] = []
            for ticker in batch:
                bars = None
                if ticker in frame.columns.get_level_values(0):
                    bars = frame[ticker][_OHLCV].dropna(how="all")
                if bars is None or bars["Close"].notna().sum() < 30:
                    unresolved.append(ticker)
                else:
                    bars.to_parquet(_path(ticker))
                    stats["fetched"] += 1

            # A ticker absent from an otherwise successful batch is NOT proof it
            # is dead: Yahoo routinely returns partial batches. Spot checks found
            # 8 of 10 such "misses" (DJCO, WSFS, CLS, ...) had full history when
            # asked for individually. Confirm one at a time before blacklisting,
            # or the fetcher manufactures its own survivorship bias.
            for ticker in unresolved:
                bars = _download_single(ticker, start, end)
                if bars is None:
                    misses.add(ticker)
                    stats["failed"] += 1
                else:
                    bars.to_parquet(_path(ticker))
                    stats["fetched"] += 1
                    stats["recovered"] = stats.get("recovered", 0) + 1
        if verbose:
            done = min(i + BATCH_SIZE, len(todo))
            print(f"  prices {done}/{len(todo)}  fetched={stats['fetched']} failed={stats['failed']}",
                  flush=True)
        _save_misses(misses)
        time.sleep(BATCH_PAUSE)

    return stats
