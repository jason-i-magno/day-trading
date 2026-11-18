"""Point-in-time shares outstanding, for look-ahead-free market caps.

Using today's market cap to filter a 2022 event would be look-ahead bias of the
worst kind: it selects companies that are $300M-$2B *now*, which is conditioned
on everything that happened since. Instead we take shares outstanding from the
most recent SEC filing that existed *before* the event, and multiply by the
price on the day.

The `filed` date is what matters, not the `end` date -- a figure measured on
March 31 but filed on May 5 was not public knowledge in April.
"""

from __future__ import annotations

from datetime import date

from .. import cache
from ..edgar import _fetch

CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json"
_TTL = 30 * 24 * 3600


def shares_history(cik: str) -> list[tuple[date, float]]:
    """Return (filed_date, shares) pairs for one company, oldest first.

    An empty list means the company does not report the concept -- common for
    foreign private issuers and some trusts.
    """
    key = f"shares:{cik}"
    cached = cache.get(key, _TTL)
    if cached is not None:
        return [(date.fromisoformat(d), v) for d, v in cached]

    try:
        payload = _fetch(CONCEPT_URL.format(cik=int(cik)), {})
    except Exception:
        cache.put(key, [])
        return []

    rows: list[tuple[str, float]] = []
    for entry in payload.get("units", {}).get("shares", []):
        filed, value = entry.get("filed"), entry.get("val")
        if filed and value:
            rows.append((filed, float(value)))
    rows.sort()
    cache.put(key, rows)
    return [(date.fromisoformat(d), v) for d, v in rows]


def shares_as_of(history: list[tuple[date, float]], as_of: date) -> float | None:
    """Most recently *filed* share count strictly before `as_of`."""
    best: float | None = None
    for filed, value in history:
        if filed < as_of:
            best = value
        else:
            break
    return best
