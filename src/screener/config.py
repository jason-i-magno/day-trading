"""Screen thresholds and runtime settings.

Every number that shapes the screen lives here so the whole strategy is
auditable in one place. Changing a threshold changes the experiment, so
treat edits to this file as a new hypothesis rather than a tweak.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# SEC requires a User-Agent identifying the requester. Requests without a real
# contact get throttled or blocked outright.
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

# SEC fair-access limit is 10 requests/second. We stay meaningfully under it.
SEC_MAX_REQUESTS_PER_SECOND = 8.0

BENCHMARK_TICKER = "SPY"


@dataclass(frozen=True)
class ScreenConfig:
    """Filters defining the candidate universe and the entry signal.

    Defaults target post-earnings-announcement drift (PEAD): the tendency of
    stocks that surprise on earnings to keep drifting in the surprise direction
    for weeks afterward. The literature finds the effect concentrated in
    smaller, less-covered names -- but spreads widen as size falls, so the caps
    below aim at the band where the effect survives trading costs.
    """

    # --- Universe ---------------------------------------------------------
    # Below ~$300M spreads typically eat the drift; above ~$2B analyst coverage
    # is dense enough that the underreaction is largely arbitraged away.
    min_market_cap: float = 300_000_000
    max_market_cap: float = 2_000_000_000

    # Sub-$5 stocks carry margin restrictions and proportionally wider spreads.
    min_price: float = 5.0

    # Enough turnover that a small position does not itself move the price.
    min_avg_dollar_volume: float = 1_000_000

    # --- Event window -----------------------------------------------------
    # How far back to look for earnings 8-Ks.
    lookback_days: int = 10

    # Drift entry window, in trading days since the announcement. Day 0-1 is
    # the reaction itself (worst spreads); the drift is what comes after.
    min_days_since_event: int = 1
    max_days_since_event: int = 10

    # --- Signal -----------------------------------------------------------
    # Announcement-window return in excess of the benchmark. This is the
    # surprise proxy: the market's own reaction, which sidesteps the need for
    # paid analyst-estimate data.
    min_abnormal_return: float = 0.03

    # --- Execution --------------------------------------------------------
    # Spread is NOT a gate. Daily-bar spread estimators (Corwin-Schultz,
    # Abdi-Ranaldo) proved too noisy to threshold on: tested against known
    # liquid names they disagree by an order of magnitude and both mistake
    # volatility for spread. The screen therefore gates liquidity on measured
    # dollar volume and reports the spread estimate only as a relative ranking
    # hint. Check the real quoted spread in your broker before entering.
    spread_warn_threshold: float = 0.01

    # --- Housekeeping -----------------------------------------------------
    max_candidates: int = 25
    exclude_exchanges: tuple[str, ...] = field(default_factory=lambda: ("PNK", "OTC"))


def sec_user_agent() -> str:
    """Return the SEC User-Agent, or explain how to set it."""
    ua = os.environ.get(SEC_USER_AGENT_ENV, "").strip()
    if not ua:
        raise RuntimeError(
            f"{SEC_USER_AGENT_ENV} is not set. The SEC requires a contact string "
            f'identifying who is making requests, e.g.:\n\n'
            f'  export {SEC_USER_AGENT_ENV}="your-name your-email@example.com"\n\n'
            "Requests without one are throttled or blocked."
        )
    return ua
