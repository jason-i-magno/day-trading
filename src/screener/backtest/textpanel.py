"""Build an event panel keyed on how much a company rewrote its own filing.

One row per periodic report that has a comparable report from ~a year earlier.
The signal is the cosine similarity between the two documents' word frequencies:
low similarity means the company changed its language a lot.

Filings are matched **year-over-year within the same form type** rather than to
whatever came before. A 10-Q compared against the previous quarter's 10-Q would
mostly measure seasonal wording, not news; the same quarter a year earlier is the
natural control.

Entry is the close of the first trading day after the filing, matching the
earnings panel, so the signal is always known before the return being measured.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import filings, prices, shares
from .panel import BENCHMARK, HORIZONS, _forward_return

_ADV_WINDOW = 60
# A comparable report should sit about a year back; quarters drift by a few weeks.
YEAR_LAG_DAYS = 365
LAG_TOLERANCE_DAYS = 75


@dataclass(frozen=True)
class FilingPair:
    current: filings.Filing
    prior: filings.Filing


# Function words dominate raw term frequency: two filings from any company score
# ~0.999 cosine and the measure has no cross-sectional spread left to rank on.
STOPWORDS = frozenset("""
the of and to in a is are was were for on with as by that this it its an be been being
has have had will would may might can could shall should not no nor or if than then
from at we our us their they he she which who whom what when where all any both each
other such only own same so too very s t don now
""".split())


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine between two word-frequency vectors, stopwords removed, in [0, 1]."""
    a = Counter({w: v for w, v in a.items() if w not in STOPWORDS})
    b = Counter({w: v for w, v in b.items() if w not in STOPWORDS})
    if not a or not b:
        return float("nan")
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    dot = sum(a[word] * b[word] for word in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: Counter, b: Counter) -> float:
    """Vocabulary overlap between two filings, in [0, 1].

    The primary measure. Frequency-weighted cosine compresses everything into
    the top few percent of its range even after stopword removal (observed std
    0.043); vocabulary overlap spreads roughly twice as wide (std 0.086), which
    is what makes a quintile sort meaningful. It also matches the underlying
    idea more directly: what matters is whether the company started saying
    something it was not saying before.
    """
    left, right = set(a), set(b)
    union = left | right
    if not union:
        return float("nan")
    return len(left & right) / len(union)


def pair_filings(index: list[filings.Filing]) -> list[FilingPair]:
    """Match each filing to the same-form filing from roughly a year earlier."""
    pairs: list[FilingPair] = []
    by_form: dict[str, list[filings.Filing]] = {}
    for filing in index:
        by_form.setdefault(filing.form, []).append(filing)

    for form_filings in by_form.values():
        for current in form_filings:
            target = current.filing_date - timedelta(days=YEAR_LAG_DAYS)
            candidates = [
                f for f in form_filings
                if f.filing_date < current.filing_date
                and abs((f.filing_date - target).days) <= LAG_TOLERANCE_DAYS
            ]
            if candidates:
                best = min(candidates, key=lambda f: abs((f.filing_date - target).days))
                pairs.append(FilingPair(current=current, prior=best))
    return pairs


def _row(pair: FilingPair, ticker: str, similarity: float, bars: pd.DataFrame,
         bench: pd.Series, share_history: list[tuple[date, float]]) -> dict | None:
    closes = bars["Close"].dropna()
    if len(closes) < _ADV_WINDOW + 5 or (closes <= 0).any():
        return None

    position = closes.index.searchsorted(pd.Timestamp(pair.current.filing_date))
    entry_idx = position + 1
    if position < _ADV_WINDOW or entry_idx >= len(closes):
        return None

    entry_date = closes.index[entry_idx]
    if entry_date not in bench.index:
        return None

    window = bars.iloc[position - _ADV_WINDOW : position]
    outstanding = shares.shares_as_of(share_history, pair.current.filing_date)
    price = float(closes.iloc[position])

    row = {
        "ticker": ticker,
        "cik": pair.current.cik,
        "form": pair.current.form,
        "filing_date": pair.current.filing_date,
        "prior_filing_date": pair.prior.filing_date,
        "entry_date": entry_date.date(),
        "similarity": similarity,
        "price": price,
        "dollar_volume": float((window["Close"] * window["Volume"]).mean()),
        "market_cap": float(outstanding) * price if outstanding else np.nan,
    }
    for horizon in HORIZONS:
        row[f"fwd_{horizon}"] = _forward_return(closes, bench, entry_idx, horizon)
    return row


# Prices are cached from mid-2020, and a row needs 60 trading days of history
# before the filing. Pairs whose current filing predates that can never produce a
# row, so they are dropped before any document is fetched -- EDGAR returns well
# over a decade of filings per company, and fetching the unusable ones roughly
# doubled the collection.
MIN_CURRENT_FILING_DATE = date(2020, 9, 1)


def build_text_panel(
    tickers_by_cik: dict[str, str], verbose: bool = True,
    min_date: date = MIN_CURRENT_FILING_DATE,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the similarity panel for the given companies.

    `tickers_by_cik` maps CIK to the ticker whose prices we hold. Coverage counts
    are returned alongside, for the same reason the earnings panel reports them:
    silent drops are where bias hides.
    """
    bench_bars = prices.load(BENCHMARK)
    if bench_bars is None:
        raise RuntimeError(f"{BENCHMARK} prices not cached")
    bench = bench_bars["Close"].dropna()

    rows: list[dict] = []
    coverage = {"companies": len(tickers_by_cik), "pairs_found": 0, "text_missing": 0,
                "unusable_window": 0, "kept": 0}

    for index, (cik, ticker) in enumerate(sorted(tickers_by_cik.items()), 1):
        bars = prices.load(ticker)
        if bars is None:
            continue
        pairs = [q for q in pair_filings(filings.filing_index(cik))
                 if q.current.filing_date >= min_date]
        coverage["pairs_found"] += len(pairs)
        if not pairs:
            continue
        # Warm the cache for every document this company needs before scoring,
        # so fetches overlap instead of running one at a time.
        filings.prefetch_texts([p.current for p in pairs] + [p.prior for p in pairs])
        share_history = shares.shares_history(cik)

        cache: dict[str, Counter | None] = {}

        def counts(filing: filings.Filing) -> Counter | None:
            if filing.key not in cache:
                text = filings.document_text(filing)
                cache[filing.key] = filings.token_counts(text) if text else None
            return cache[filing.key]

        for pair in pairs:
            current, prior = counts(pair.current), counts(pair.prior)
            if current is None or prior is None:
                coverage["text_missing"] += 1
                continue
            similarity = jaccard_similarity(current, prior)
            cosine = cosine_similarity(current, prior)
            if not math.isfinite(similarity):
                coverage["text_missing"] += 1
                continue
            row = _row(pair, ticker, similarity, bars, bench, share_history)
            if row is not None:
                row["cosine"] = cosine
            if row is None:
                coverage["unusable_window"] += 1
            else:
                rows.append(row)
                coverage["kept"] += 1

        if verbose and index % 25 == 0:
            print(f"  text panel {index}/{len(tickers_by_cik)} companies  rows={len(rows)}",
                  flush=True)

    return pd.DataFrame(rows), coverage
