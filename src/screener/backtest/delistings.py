"""Why is a company missing from the price data?

The backtest drops every event whose ticker has no price history, and those are
overwhelmingly companies that stopped trading. That is textbook survivorship
bias, but its *severity* depends on something we can find out for free: when did
each company actually stop trading, relative to the earnings event?

A company that delisted three years after an event contributed a perfectly
normal holding window at the time. Excluding it loses a valid observation for no
reason. A company that delisted the week after is a different story.

SEC filings answer this directly:
  Form 25 / 25-NSE   notification of removal from listing -- the delisting itself
  Form 15-12B/12G    deregistration -- the company going dark afterwards
  empty tickers[]    EDGAR no longer associates a trading symbol with the filer
  formerNames        the company renamed, so the old symbol simply moved
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .. import cache
from ..edgar import _fetch

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FLOAT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityPublicFloat.json"
)

DELISTING_FORMS = ("25-NSE", "25")
DEREGISTRATION_FORMS = ("15-12B", "15-12G", "15F-12B", "15F-12G")

_TTL = 30 * 24 * 3600


@dataclass(frozen=True)
class CompanyStatus:
    """What EDGAR knows about whether a company still trades."""

    cik: str
    name: str
    has_ticker: bool
    delisting_date: date | None
    deregistration_date: date | None
    last_filing_date: date | None
    renamed: bool
    sic: str

    @property
    def classification(self) -> str:
        """Bucket the company into why its prices are missing.

        A current ticker on file outranks everything else. Form 25 delists a
        *security*, not necessarily a company -- Apple filed one in 2019 for a
        note series while the common stock kept trading -- so treating any Form
        25 as a delisting misclassifies large healthy issuers.
        """
        if self.has_ticker:
            return "renamed" if self.renamed else "still listed"
        if self.delisting_date:
            return "delisted"
        if self.deregistration_date:
            # Went dark without an exchange delisting notice: typically a small
            # filer deregistering, or a company that was never exchange-listed.
            return "deregistered"
        return "no ticker on file"


def _latest(dates: list[str]) -> date | None:
    """Most recent of the given filing dates.

    A company may delist several securities over its life; the last such notice
    is the one that corresponds to the common stock going away.
    """
    parsed = sorted(date.fromisoformat(d) for d in dates if d)
    return parsed[-1] if parsed else None


def fetch_status(cik: str) -> CompanyStatus | None:
    """Look up one company's listing status. None if EDGAR has no such filer."""
    key = f"status:{cik}"
    if (hit := cache.get(key, _TTL)) is not None:
        if not hit:
            return None
        return CompanyStatus(
            cik=hit["cik"], name=hit["name"], has_ticker=hit["has_ticker"],
            delisting_date=date.fromisoformat(hit["delisting_date"]) if hit["delisting_date"] else None,
            deregistration_date=(date.fromisoformat(hit["deregistration_date"])
                                 if hit["deregistration_date"] else None),
            last_filing_date=date.fromisoformat(hit["last_filing_date"]) if hit["last_filing_date"] else None,
            renamed=hit["renamed"], sic=hit["sic"],
        )

    try:
        payload = _fetch(SUBMISSIONS_URL.format(cik=int(cik)), {})
    except Exception:
        cache.put(key, {})
        return None

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])

    delisting = _latest([d for f, d in zip(forms, dates) if f in DELISTING_FORMS])
    deregistration = _latest([d for f, d in zip(forms, dates) if f in DEREGISTRATION_FORMS])

    status = CompanyStatus(
        cik=cik,
        name=payload.get("name", ""),
        has_ticker=bool(payload.get("tickers")),
        delisting_date=delisting,
        deregistration_date=deregistration,
        last_filing_date=date.fromisoformat(dates[0]) if dates else None,
        renamed=bool(payload.get("formerNames")),
        sic=str(payload.get("sicDescription", "")),
    )
    cache.put(key, {
        "cik": status.cik, "name": status.name, "has_ticker": status.has_ticker,
        "delisting_date": status.delisting_date.isoformat() if status.delisting_date else None,
        "deregistration_date": (status.deregistration_date.isoformat()
                                if status.deregistration_date else None),
        "last_filing_date": status.last_filing_date.isoformat() if status.last_filing_date else None,
        "renamed": status.renamed, "sic": status.sic,
    })
    return status


def public_float(cik: str, before: date) -> float | None:
    """Most recently reported public float filed before `before`.

    Public float -- the market value of shares held by non-affiliates -- appears
    on every 10-K cover page. It is the only free market-cap proxy available for
    a company whose prices have vanished, which is exactly the population we need
    to size here.
    """
    key = f"float:{cik}"
    rows = cache.get(key, _TTL)
    if rows is None:
        try:
            payload = _fetch(FLOAT_URL.format(cik=int(cik)), {})
            rows = [
                (entry["filed"], float(entry["val"]))
                for unit in payload.get("units", {}).values()
                for entry in unit
                if entry.get("filed") and entry.get("val")
            ]
            rows.sort()
        except Exception:
            rows = []
        cache.put(key, rows)

    best: float | None = None
    for filed, value in rows:
        if date.fromisoformat(filed) < before:
            best = value
        else:
            break
    return best
