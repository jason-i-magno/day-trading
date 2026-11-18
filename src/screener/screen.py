"""The screen itself: earnings events in, ranked candidates out.

Deliberately keeps a full rejection funnel. A screen that only reports what
survived is impossible to debug and easy to fool yourself with -- if 176 events
become 2 candidates you need to know which filter did that before you trust
the 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .config import ScreenConfig
from .edgar import EarningsEvent, find_earnings_events
from .market import MarketSnapshot, build_snapshot, download_history, fetch_profiles

# A ticker Yahoo refused to look up. Separate from "no market data" because it
# says nothing about the ticker, only that this run's coverage is incomplete.
THROTTLED = "rate-limited by Yahoo"


@dataclass(frozen=True)
class Candidate:
    """An earnings event that cleared every filter."""

    event: EarningsEvent
    snapshot: MarketSnapshot

    @property
    def ticker(self) -> str:
        return self.event.ticker


@dataclass
class ScreenResult:
    """Ranked candidates plus the funnel that produced them."""

    candidates: list[Candidate]
    rejections: dict[str, int] = field(default_factory=dict)
    events_found: int = 0
    as_of: date = field(default_factory=date.today)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


def run_screen(config: ScreenConfig | None = None, as_of: date | None = None) -> ScreenResult:
    """Find recent earnings announcements and rank the tradeable ones.

    The signal is post-earnings-announcement drift: a strong benchmark-adjusted
    reaction tends to keep going for weeks. We deliberately skip day 0-1, when
    spreads are widest and the initial repricing is still happening, and enter
    into the drift instead of the reaction.
    """
    config = config or ScreenConfig()
    as_of = as_of or date.today()

    events = find_earnings_events(config.lookback_days, end=as_of)
    result = ScreenResult(candidates=[], events_found=len(events), as_of=as_of)
    if not events:
        return result

    by_ticker: dict[str, EarningsEvent] = {e.ticker: e for e in events}
    tickers = sorted(by_ticker)

    frame = download_history(tickers)
    profiles, throttled = fetch_profiles(tickers)

    survivors: list[Candidate] = []
    for ticker in tickers:
        event = by_ticker[ticker]
        profile = profiles.get(ticker)
        if ticker in throttled:
            result.reject(THROTTLED)
            continue
        if not profile:
            result.reject("no market data")
            continue
        if not profile.get("market_cap"):
            result.reject("no market cap")
            continue

        snapshot = build_snapshot(ticker, frame, profile, event.filing_date)
        if snapshot is None:
            result.reject("incomplete price history")
            continue

        if snapshot.exchange in config.exclude_exchanges:
            result.reject("OTC / pink sheets")
        elif not config.min_market_cap <= snapshot.market_cap <= config.max_market_cap:
            result.reject("market cap outside band")
        elif snapshot.price < config.min_price:
            result.reject("price below minimum")
        elif snapshot.avg_dollar_volume < config.min_avg_dollar_volume:
            result.reject("insufficient dollar volume")
        elif not (
            config.min_days_since_event <= snapshot.days_since_event <= config.max_days_since_event
        ):
            result.reject("outside drift entry window")
        elif snapshot.abnormal_return < config.min_abnormal_return:
            result.reject("reaction too weak")
        else:
            survivors.append(Candidate(event=event, snapshot=snapshot))

    # Strongest reaction first; a tighter spread estimate breaks ties, since
    # between two equal signals the cheaper one to trade wins.
    survivors.sort(
        key=lambda c: (-c.snapshot.abnormal_return, c.snapshot.spread_estimate or 0)
    )
    result.candidates = survivors[: config.max_candidates]
    # Cleared every filter but ranked out. Counted so the funnel still adds up
    # to the event count in a heavy week, when the cap is what binds.
    for _ in survivors[config.max_candidates :]:
        result.reject(f"ranked below top {config.max_candidates}")
    return result
