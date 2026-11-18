"""Historical earnings event collection.

EDGAR full-text search caps each query's result window, so a multi-year sweep is
chunked into months. Every filing is a point-in-time record -- a company that has
since delisted still appears in the month it reported, which is what keeps the
event universe free of survivorship bias at this stage.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..edgar import EARNINGS_PHRASE, FTS_URL, EarningsEvent, _fetch, _parse_hit

EVENTS_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "events"


def _month_bounds(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-month windows."""
    windows: list[tuple[date, date]] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1)
        window_end = min(next_month.fromordinal(next_month.toordinal() - 1), end)
        windows.append((max(cursor, start), window_end))
        cursor = next_month
    return windows


def _collect_month(start: date, end: date) -> list[dict]:
    """Page through one month of earnings 8-Ks, caching the raw result."""
    EVENTS_CACHE.mkdir(parents=True, exist_ok=True)
    path = EVENTS_CACHE / f"{start.isoformat()}_{end.isoformat()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    rows: list[dict] = []
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
        for hit in hits:
            event = _parse_hit(hit.get("_source", {}))
            if event:
                rows.append(
                    {
                        "ticker": event.ticker,
                        "cik": event.cik,
                        "company": event.company,
                        "filing_date": event.filing_date.isoformat(),
                        "accession": event.accession,
                    }
                )
        offset += len(hits)
        total = payload.get("hits", {}).get("total", {}).get("value", 0)
        # The search window caps how deep paging can go; months stay well inside it.
        if offset >= total or offset >= 9900:
            break

    path.write_text(json.dumps(rows))
    return rows


def collect_events(start: date, end: date, verbose: bool = True) -> list[EarningsEvent]:
    """Return every earnings 8-K filed between `start` and `end`."""
    events: list[EarningsEvent] = []
    windows = _month_bounds(start, end)
    for index, (window_start, window_end) in enumerate(windows, 1):
        rows = _collect_month(window_start, window_end)
        for row in rows:
            events.append(
                EarningsEvent(
                    ticker=row["ticker"],
                    cik=row["cik"],
                    company=row["company"],
                    filing_date=date.fromisoformat(row["filing_date"]),
                    accession=row["accession"],
                    items=("2.02",),
                )
            )
        if verbose:
            print(
                f"  events {index}/{len(windows)}  {window_start:%Y-%m}  "
                f"+{len(rows):>4}  total={len(events)}",
                flush=True,
            )
    return events
