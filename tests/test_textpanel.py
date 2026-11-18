"""Filing-similarity signal: pairing, metrics, and form-aware ranking."""

from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtest import analyze, textpanel
from screener.backtest.filings import Filing, token_counts


def _filing(form: str, day: str) -> Filing:
    return Filing("1", form, date.fromisoformat(day), f"acc-{form}-{day}", "doc.htm")


# --- pairing ------------------------------------------------------------------

def test_pairs_match_the_same_quarter_a_year_earlier():
    index = [_filing("10-Q", d) for d in
             ["2023-05-01", "2023-08-01", "2023-11-01", "2024-05-01", "2024-08-01"]]
    pairs = {p.current.filing_date: p.prior.filing_date for p in textpanel.pair_filings(index)}
    assert pairs[date(2024, 5, 1)] == date(2023, 5, 1)
    assert pairs[date(2024, 8, 1)] == date(2023, 8, 1)


def test_a_filing_is_never_paired_with_a_later_one():
    """The prior document must already exist when the current one is filed, or
    the signal is built from information that did not yet exist."""
    index = [_filing("10-K", d) for d in ["2022-03-01", "2023-03-01", "2024-03-01"]]
    for pair in textpanel.pair_filings(index):
        assert pair.prior.filing_date < pair.current.filing_date


def test_forms_are_never_compared_across_types():
    index = [_filing("10-K", "2023-03-01"), _filing("10-Q", "2024-03-05")]
    assert textpanel.pair_filings(index) == []


def test_a_missing_year_leaves_the_filing_unpaired():
    # Two years apart is outside the tolerance; no comparable document exists.
    index = [_filing("10-K", "2022-03-01"), _filing("10-K", "2024-03-01")]
    assert textpanel.pair_filings(index) == []


def test_the_closest_candidate_to_one_year_wins():
    index = [_filing("10-Q", "2023-03-01"), _filing("10-Q", "2023-05-01"),
             _filing("10-Q", "2024-04-20")]
    pairs = textpanel.pair_filings(index)
    current = [p for p in pairs if p.current.filing_date == date(2024, 4, 20)][0]
    assert current.prior.filing_date == date(2023, 5, 1)


# --- similarity metrics -------------------------------------------------------

def test_jaccard_is_one_for_identical_vocabulary():
    counts = Counter({"risk": 3, "factor": 1})
    assert textpanel.jaccard_similarity(counts, counts) == 1.0


def test_jaccard_is_zero_for_disjoint_vocabulary():
    assert textpanel.jaccard_similarity(Counter({"a": 1}), Counter({"b": 1})) == 0.0


def test_jaccard_ignores_frequency_and_tracks_vocabulary():
    """Restating the same words more often is not a language change; introducing
    new words is. The measure has to distinguish those."""
    base = Counter({"revenue": 1, "risk": 1})
    louder = Counter({"revenue": 900, "risk": 900})
    expanded = Counter({"revenue": 1, "risk": 1, "litigation": 1, "impairment": 1})
    assert textpanel.jaccard_similarity(base, louder) == 1.0
    assert textpanel.jaccard_similarity(base, expanded) == pytest.approx(0.5)


def test_cosine_discards_stopwords():
    """Function words dominate raw term frequency and flatten every score toward
    1.0; two filings sharing only stopwords must not look similar."""
    a = Counter({"the": 500, "of": 400, "revenue": 1})
    b = Counter({"the": 500, "of": 400, "litigation": 1})
    assert textpanel.cosine_similarity(a, b) == 0.0


def test_token_counts_drop_numbers():
    """Quarterly figures change in every filing by construction; leaving them in
    would measure arithmetic rather than wording."""
    counts = token_counts("Revenue was 54,794 and 35,424 in 2026")
    assert "revenue" in counts
    assert not any(ch.isdigit() for word in counts for ch in word)


# --- form-aware ranking -------------------------------------------------------

def test_quintiles_are_formed_within_each_form_type():
    """10-Q similarity runs systematically lower than 10-K similarity, so a
    combined sort would rank by form rather than by how much text changed."""
    rows = []
    for i in range(10):
        rows.append({"entry_date": "2024-03-10", "form": "10-K",
                     "similarity": 0.90 + i * 0.005, "fwd_20": 0.0})
        rows.append({"entry_date": "2024-03-10", "form": "10-Q",
                     "similarity": 0.70 + i * 0.005, "fwd_20": 0.0})
    grouped = analyze.add_groupings_by(pd.DataFrame(rows), "similarity", ["form"])
    for form in ("10-K", "10-Q"):
        chunk = grouped[grouped["form"] == form]
        assert set(chunk["quintile"]) == {1, 2, 3, 4, 5}


def test_combined_ranking_would_confound_form_with_signal():
    """Demonstrates the failure the previous test guards against."""
    rows = []
    for i in range(10):
        rows.append({"entry_date": "2024-03-10", "form": "10-K",
                     "similarity": 0.90 + i * 0.005, "fwd_20": 0.0})
        rows.append({"entry_date": "2024-03-10", "form": "10-Q",
                     "similarity": 0.70 + i * 0.005, "fwd_20": 0.0})
    naive = analyze.add_groupings_by(pd.DataFrame(rows), "similarity")
    # Every 10-Q lands in the bottom half purely because of its form.
    assert naive[naive["form"] == "10-Q"]["quintile"].max() <= 3
