"""Fetch and clean periodic-report text from EDGAR.

Tests a different hypothesis from the rest of this repo. The price-reaction
signal is exhausted; this measures whether a company *changing its own language*
between reports predicts what the stock does next -- the "Lazy Prices" effect
(Cohen, Malloy & Nguyen, Journal of Finance 2020). The premise is that firms
rewrite risk factors and MD&A when something has changed, and that few investors
read closely enough to notice.

Two extraction details matter:

  * Modern filings are inline XBRL: machine-readable facts live in hidden header
    blocks that a naive strip turns into tag soup. Those blocks are removed.
  * Numbers are dropped. Quarterly figures change every filing by construction,
    so leaving them in would measure arithmetic rather than language.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from ..config import sec_user_agent
from ..edgar import _fetch, _limiter

TEXT_DIR = Path(__file__).resolve().parents[3] / ".cache" / "filing_text"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

PERIODIC_FORMS = ("10-K", "10-Q")
_WORD = re.compile(r"[a-z]{2,}")


@dataclass(frozen=True)
class Filing:
    cik: str
    form: str
    filing_date: date
    accession: str
    document: str

    @property
    def key(self) -> str:
        return f"{self.cik}_{self.accession.replace('-', '')}"

    @property
    def url(self) -> str:
        return DOC_URL.format(
            cik=int(self.cik), accession=self.accession.replace("-", ""), document=self.document
        )


def filing_index(cik: str) -> list[Filing]:
    """Every 10-K and 10-Q the company has filed, oldest first."""
    try:
        payload = _fetch(SUBMISSIONS_URL.format(cik=int(cik)), {})
    except Exception:
        return []

    recent = payload.get("filings", {}).get("recent", {})
    rows: list[Filing] = []
    for form, filed, accession, document in zip(
        recent.get("form", []), recent.get("filingDate", []),
        recent.get("accessionNumber", []), recent.get("primaryDocument", []),
    ):
        if form not in PERIODIC_FORMS or not document:
            continue
        try:
            rows.append(Filing(cik, form, date.fromisoformat(filed), accession, document))
        except ValueError:
            continue
    return sorted(rows, key=lambda f: f.filing_date)


# Regex extraction rather than a DOM parse. Filings run to several megabytes and
# BeautifulSoup made the collection CPU-bound at ~150% of one core; stripping with
# regex is ~10x faster and agrees with the parsed text on 97.9% of vocabulary,
# which is well inside the noise of the similarity measure itself.
#
# Order matters: the inline-XBRL header and hidden blocks must go before general
# tag stripping, or their machine-readable facts survive as tag soup in the text.
_IX_BLOCK = re.compile(r"<ix:(header|hidden)\b.*?</ix:\1>", re.I | re.S)
_HIDDEN_BLOCK = re.compile(r"<(div|span|td)[^>]*display\s*:\s*none[^>]*>.*?</\1>", re.I | re.S)
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.I)


def _clean(html: str) -> str:
    """Strip a filing down to readable prose."""
    html = _IX_BLOCK.sub(" ", html)
    html = _HIDDEN_BLOCK.sub(" ", html)
    html = _SCRIPT.sub(" ", html)
    html = _TAG.sub(" ", html)
    html = _ENTITY.sub(" ", html)
    return re.sub(r"\s+", " ", html).strip()


def document_text(filing: Filing) -> str | None:
    """Cleaned text for one filing, cached gzipped on disk.

    Raw filings run to ~1MB each; storing the cleaned text compressed keeps a
    multi-thousand-filing corpus to a manageable size and makes re-runs free.
    """
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    path = TEXT_DIR / f"{filing.key}.txt.gz"
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            pass

    _limiter.wait()
    try:
        response = httpx.get(
            filing.url,
            headers={"User-Agent": sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=90.0,
            follow_redirects=True,
        )
        if response.status_code != 200:
            return None
        text = _clean(response.text)
    except Exception:
        return None

    if len(text) < 2000:
        # Too short to be a real periodic report -- usually a cover page or an
        # exhibit-only document.
        return None
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return text


def token_counts(text: str) -> Counter:
    """Lowercase alphabetic word frequencies.

    Numbers are excluded deliberately: every quarterly report restates its
    figures, so including them would swamp genuine wording changes.
    """
    return Counter(_WORD.findall(text.lower()))


def prefetch_texts(batch: list[Filing], workers: int = 12) -> None:
    """Warm the on-disk text cache for many filings at once.

    Serial fetching ran at ~0.6 documents/second, an order of magnitude below the
    SEC rate limit -- the wait was bandwidth and HTML parsing, not throttling.
    The shared limiter is thread-safe, so requests can overlap while the whole
    process stays inside the 8 req/s budget.
    """
    from concurrent.futures import ThreadPoolExecutor

    pending = [f for f in batch if not (TEXT_DIR / f"{f.key}.txt.gz").exists()]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(document_text, pending))
