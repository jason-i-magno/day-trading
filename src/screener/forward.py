"""Score the forward-test log.

The point of `--log` is that picks are recorded *before* outcomes exist, so this
is the one evaluation in the project that cannot be contaminated by hindsight or
by a model having seen the future. Everything else here has been measured on
history; this measures the screen going forward.

What it cannot do is establish significance. A weekly run producing a handful of
picks yields a few hundred observations a year, clustered into a dozen or so
months, against a signal the backtest already measured as null in this size band.
The honest use is as a record that the screen behaved as expected, not as
evidence that it works. The report says so.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"
DEFAULT_HORIZON = 20


@dataclass(frozen=True)
class Scored:
    ticker: str
    logged_at: date
    entry_date: date
    entry_price: float
    exit_date: date | None
    exit_price: float | None
    abnormal_return: float | None

    @property
    def matured(self) -> bool:
        return self.abnormal_return is not None


def load_picks(path: Path) -> list[dict]:
    """Read the append-only JSONL log, skipping malformed lines."""
    if not path.exists():
        return []
    picks: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            picks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return picks


def _download(tickers: list[str], start: date) -> dict[str, pd.Series]:
    symbols = sorted({*tickers, BENCHMARK})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = yf.download(
            symbols, start=start.isoformat(), group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    if frame is None or frame.empty:
        return {}
    if not isinstance(frame.columns, pd.MultiIndex):
        frame = pd.concat({symbols[0]: frame}, axis=1)
    out: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol in frame.columns.get_level_values(0):
            closes = frame[symbol]["Close"].dropna()
            if len(closes):
                out[symbol] = closes
    return out


def score(picks: list[dict], horizon: int = DEFAULT_HORIZON) -> list[Scored]:
    """Compute each pick's benchmark-adjusted return over `horizon` trading days.

    Entry is the close of the first trading day *after* the pick was logged: the
    screen runs after the close, so that is the first price actually reachable.
    """
    if not picks:
        return []

    logged = [datetime.fromisoformat(p["logged_at"]).date() for p in picks]
    # Pad the window back: a pick logged today would otherwise request start ==
    # end, which Yahoo rejects outright, silently discarding every pending pick.
    prices = _download([p["ticker"] for p in picks], min(logged) - timedelta(days=7))
    benchmark = prices.get(BENCHMARK)
    if benchmark is None:
        # No market data at all -- report every pick as still open rather than
        # returning nothing, which would read as "no picks logged".
        return [Scored(p["ticker"], d, d, float("nan"), None, None, None)
                for p, d in zip(picks, logged)]

    results: list[Scored] = []
    for pick, log_date in zip(picks, logged):
        closes = prices.get(pick["ticker"])
        if closes is None or closes.empty:
            # Delisted, renamed, or simply not returned; keep it visible.
            results.append(Scored(pick["ticker"], log_date, log_date,
                                  float("nan"), None, None, None))
            continue

        entry_idx = closes.index.searchsorted(pd.Timestamp(log_date), side="right")
        if entry_idx >= len(closes):
            # Logged after the most recent close: the entry bar has not printed
            # yet. Still open, not missing.
            results.append(Scored(pick["ticker"], log_date, log_date,
                                  float("nan"), None, None, None))
            continue
        entry_date = closes.index[entry_idx]
        exit_idx = entry_idx + horizon

        if exit_idx >= len(closes) or entry_date not in benchmark.index:
            results.append(Scored(pick["ticker"], log_date, entry_date.date(),
                                  float(closes.iloc[entry_idx]), None, None, None))
            continue

        exit_date = closes.index[exit_idx]
        if exit_date not in benchmark.index:
            results.append(Scored(pick["ticker"], log_date, entry_date.date(),
                                  float(closes.iloc[entry_idx]), None, None, None))
            continue

        stock = closes.iloc[exit_idx] / closes.iloc[entry_idx] - 1
        market = benchmark[exit_date] / benchmark[entry_date] - 1
        results.append(Scored(
            pick["ticker"], log_date, entry_date.date(), float(closes.iloc[entry_idx]),
            exit_date.date(), float(closes.iloc[exit_idx]), float(stock - market),
        ))
    return results


def format_report(scored: list[Scored], horizon: int) -> str:
    matured = [s for s in scored if s.matured]
    pending = [s for s in scored if not s.matured]

    lines = [
        f"Forward test - {date.today().isoformat()}",
        "=" * 66,
        f"{len(scored)} picks logged  |  {len(matured)} matured  |  {len(pending)} still open",
        f"horizon: {horizon} trading days, returns in excess of {BENCHMARK}",
        "",
    ]

    if not matured:
        lines.append("Nothing has matured yet. Check back once picks are older than")
        lines.append(f"{horizon} trading days (about {horizon // 5} weeks).")
        return "\n".join(lines)

    returns = np.array([s.abnormal_return for s in matured])
    mean = returns.mean()
    standard_error = returns.std(ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else np.nan

    lines.append("MATURED PICKS")
    lines.append(f"{'TICKER':<8}{'LOGGED':>12}{'ENTRY':>12}{'EXIT':>12}{'vs SPY':>10}")
    lines.append("-" * 66)
    for s in sorted(matured, key=lambda x: x.logged_at):
        lines.append(f"{s.ticker:<8}{s.logged_at.isoformat():>12}{s.entry_date.isoformat():>12}"
                     f"{s.exit_date.isoformat():>12}{s.abnormal_return:>9.1%}")

    lines += [
        "",
        f"mean abnormal return : {mean:+.2%}",
        f"median               : {np.median(returns):+.2%}",
        f"hit rate (> 0)       : {(returns > 0).mean():.0%}",
        f"best / worst         : {returns.max():+.1%} / {returns.min():+.1%}",
    ]
    if len(returns) > 1 and standard_error > 0:
        lines.append(f"naive t-statistic    : {mean / standard_error:.2f}")

    repeats = pd.Series([s.ticker for s in matured]).value_counts()
    duplicated = repeats[repeats > 1]
    if len(duplicated):
        lines.append("")
        lines.append(f"note: {len(duplicated)} ticker(s) picked more than once "
                     f"({', '.join(duplicated.index[:5])}) - overlapping positions in "
                     "the same name are not independent observations.")

    lines += [
        "",
        "READ THIS BEFORE INTERPRETING THE NUMBER ABOVE.",
        "The t-statistic is naive: it ignores that holding periods overlap and that",
        "picks cluster in earnings season, so it overstates significance. At a few",
        "hundred picks a year it cannot reach a real threshold either way. The",
        "backtest already measured this signal as null in this size band",
        "(results/backtest_2026-08-30_corrected.txt); a positive number here is far",
        "more likely to be noise than a discovery. This log exists to be an honest",
        "record, not a verdict.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="forward", description="Score the forward-test pick log."
    )
    parser.add_argument("--log", type=Path, default=Path("picks.jsonl"))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="holding period in trading days")
    args = parser.parse_args(argv)

    picks = load_picks(args.log)
    if not picks:
        print(f"no picks in {args.log}")
        return 0
    print(format_report(score(picks, args.horizon), args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
