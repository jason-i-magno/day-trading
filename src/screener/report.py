"""Human-readable output, and the append-only pick log.

The log is the point of the whole exercise. Every run records what it selected
*before* the outcome is known, so that months from now there is an honest
record to score -- rather than a memory of which calls felt good.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from .config import ScreenConfig
from .screen import THROTTLED, ScreenResult


def _millions(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    return f"{value / 1e6:.0f}M"


def format_report(result: ScreenResult, config: ScreenConfig) -> str:
    lines: list[str] = []
    lines.append(f"Earnings drift screen - {result.as_of.isoformat()}")
    lines.append("=" * 78)
    lines.append(
        f"{result.events_found} earnings 8-Ks in the last {config.lookback_days} days"
        f"  ->  {len(result.candidates)} candidates"
    )
    throttled = result.rejections.get(THROTTLED, 0)
    if throttled:
        lines.append(
            f"WARNING: Yahoo rate-limited {throttled} lookups even after retries. This\n"
            "run's coverage is incomplete -- those tickers were never screened."
        )
    lines.append("")

    if result.candidates:
        header = (
            f"{'TICKER':<8}{'PRICE':>9}{'MKT CAP':>10}{'$VOL/DAY':>11}"
            f"{'REACTION':>10}{'DAYS':>6}{'SPREAD~':>9}"
        )
        lines.append(header)
        lines.append("-" * 78)
        for candidate in result.candidates:
            s = candidate.snapshot
            spread = "n/a" if s.spread_estimate != s.spread_estimate else f"{s.spread_estimate * 1e4:.0f}bp"
            flag = " !" if s.spread_estimate == s.spread_estimate and s.spread_estimate > config.spread_warn_threshold else ""
            lines.append(
                f"{s.ticker:<8}{s.price:>9.2f}{_millions(s.market_cap):>10}"
                f"{_millions(s.avg_dollar_volume):>11}{s.abnormal_return:>9.1%}"
                f"{s.days_since_event:>6}{spread:>9}{flag}"
            )
        lines.append("")
        lines.append("Company names")
        for candidate in result.candidates:
            lines.append(f"  {candidate.ticker:<8} {candidate.event.company}")
    else:
        lines.append("No candidates cleared the filters.")

    if result.rejections:
        lines.append("")
        lines.append("Rejection funnel")
        for reason, count in sorted(result.rejections.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {count:>4}  {reason}")

    lines.append("")
    lines.append(
        "REACTION is the benchmark-adjusted move across the announcement, used as a\n"
        "surprise proxy. SPREAD~ is a daily-bar estimate, reliable only for ranking --\n"
        "check the real quoted spread before trading.\n"
        "\n"
        "THIS SCREEN HAS BEEN BACKTESTED AND SHOWS NO EDGE IN THIS SIZE BAND.\n"
        "78,632 events, 2021-2026: drift is real but concentrated below $300M, and\n"
        "the long leg turns negative once $1M/day of volume is required. See\n"
        "results/backtest_2026-08-30_corrected.txt. Output is a forward-test record,\n"
        "not a buy list."
    )
    return "\n".join(lines)


def _logged_accessions(path: Path) -> set[str]:
    """Accession numbers already recorded, so an event is never logged twice."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            accession = json.loads(line).get("accession")
        except json.JSONDecodeError:
            continue
        if accession:
            seen.add(accession)
    return seen


def append_log(result: ScreenResult, config: ScreenConfig, path: Path) -> int:
    """Append this run's picks to a JSONL log. Returns rows written.

    An earnings event stays inside the drift entry window for up to ten trading
    days, so a weekly run would otherwise log the same event several times and
    count one position as several independent observations. Events already in
    the log are skipped, keyed on the SEC accession number.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = datetime.now(timezone.utc).isoformat()
    already = _logged_accessions(path)
    written = 0
    with path.open("a") as handle:
        for candidate in result.candidates:
            if candidate.event.accession in already:
                continue
            written += 1
            handle.write(
                json.dumps(
                    {
                        "logged_at": stamped,
                        "as_of": result.as_of.isoformat(),
                        "ticker": candidate.ticker,
                        "company": candidate.event.company,
                        "filing_date": candidate.event.filing_date.isoformat(),
                        "accession": candidate.event.accession,
                        "edgar_url": candidate.event.edgar_url,
                        "snapshot": asdict(candidate.snapshot),
                        "config": asdict(config),
                    },
                    default=str,
                )
                + "\n"
            )
    return written
