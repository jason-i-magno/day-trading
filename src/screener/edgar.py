"""SEC EDGAR client: find earnings announcements.

A company reports earnings by filing an 8-K carrying Item 2.02, "Results of
Operations and Financial Condition". EDGAR's full-text search exposes those
item codes directly, so earnings detection here is authoritative rather than a
guess based on headlines or calendar scraping.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from . import cache
from .config import SEC_MAX_REQUESTS_PER_SECOND, sec_user_agent

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
EARNINGS_ITEM = "2.02"
EARNINGS_PHRASE = '"Results of Operations and Financial Condition"'

# Filings are immutable once accepted, so a long TTL is safe. The only thing
# that changes is which filings exist, and a re-run past the TTL picks those up.
_FILING_TTL = 6 * 3600

# "DICK'S SPORTING GOODS, INC.  (DKS)  (CIK 0001089063)" -> DKS
# Some issuers list several tickers, e.g. "(BOX, BXCAP)"; the first is the common stock.
_DISPLAY_NAME_RE = re.compile(r"\(([A-Z0-9.\-]+(?:,\s*[A-Z0-9.\-]+)*)\)\s*\(CIK\s*(\d+)\)")


@dataclass(frozen=True)
class EarningsEvent:
    """One company's earnings announcement, as filed with the SEC."""

    ticker: str
    cik: str
    company: str
    filing_date: date
    accession: str
    items: tuple[str, ...]

    @property
    def edgar_url(self) -> str:
        acc = self.accession.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{acc}/"


class _RateLimiter:
    """Spaces requests to stay inside SEC's fair-access policy."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval


_limiter = _RateLimiter(SEC_MAX_REQUESTS_PER_SECOND)


# EDGAR intermittently returns 500s and 503s under load. A multi-year sweep makes
# thousands of requests, so hitting one is near-certain and must not end the run.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = 3.0


def _fetch(url: str, params: dict[str, str]) -> dict:
    key = f"{url}?{sorted(params.items())}"
    if (hit := cache.get(key, _FILING_TTL)) is not None:
        return hit

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        _limiter.wait()
        try:
            response = httpx.get(
                url,
                params=params,
                headers={"User-Agent": sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
                timeout=30.0,
            )
            if response.status_code in _TRANSIENT_STATUS:
                raise httpx.HTTPStatusError(
                    f"transient {response.status_code}", request=response.request, response=response
                )
            response.raise_for_status()
            payload = response.json()
            cache.put(key, payload)
            return payload
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # A genuine 4xx (bad CIK, missing concept) will never succeed on retry.
            if status is not None and status not in _TRANSIENT_STATUS:
                raise
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_SECONDS * (2**attempt))

    raise RuntimeError(f"EDGAR request failed after {_MAX_ATTEMPTS} attempts: {url}") from last_error


def _parse_hit(source: dict) -> EarningsEvent | None:
    """Convert one search hit into an EarningsEvent, or None if unusable.

    Hits are skipped when they carry no Item 2.02 (the phrase can appear in a
    filing that is not itself an earnings release) or when the filer has no
    ticker, which is the case for trusts and wholly-owned subsidiaries.
    """
    items = tuple(source.get("items") or ())
    if EARNINGS_ITEM not in items:
        return None

    names = source.get("display_names") or []
    if not names:
        return None
    match = _DISPLAY_NAME_RE.search(names[0])
    if not match:
        return None

    ticker = match.group(1).split(",")[0].strip()
    try:
        filing_date = date.fromisoformat(source["file_date"])
    except (KeyError, ValueError):
        return None

    return EarningsEvent(
        ticker=ticker,
        cik=match.group(2),
        company=names[0].split("  (")[0].strip(),
        filing_date=filing_date,
        accession=source.get("adsh", ""),
        items=items,
    )


def find_earnings_events(lookback_days: int, end: date | None = None) -> list[EarningsEvent]:
    """Return earnings 8-Ks filed in the trailing `lookback_days` window.

    Results are deduplicated per ticker, keeping the most recent filing -- a
    company that amends or supplements its release should appear once.
    """
    end = end or date.today()
    start = end - timedelta(days=lookback_days)

    events: list[EarningsEvent] = []
    offset = 0
    while True:
        payload = _fetch(
            FTS_URL,
            {
                "q": EARNINGS_PHRASE,
                "forms": "8-K",
                "startdt": start.isoformat(),
                "enddt": end.isoformat(),
                "from": str(offset),
            },
        )
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        events.extend(e for h in hits if (e := _parse_hit(h.get("_source", {}))))
        offset += len(hits)
        if offset >= payload.get("hits", {}).get("total", {}).get("value", 0):
            break

    latest: dict[str, EarningsEvent] = {}
    for event in events:
        existing = latest.get(event.ticker)
        if existing is None or event.filing_date > existing.filing_date:
            latest[event.ticker] = event
    return sorted(latest.values(), key=lambda e: e.filing_date, reverse=True)
