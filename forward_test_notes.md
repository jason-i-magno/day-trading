# Forward-test log notes

Known defects in specific runs of `picks.jsonl`. The log itself is append-only
and is never edited; anything a scorer needs to know about a run goes here.

## 2026-11-04 and 2026-11-11 — partial coverage (Yahoo rate limit)

Yahoo throttled the market-cap lookups partway through each run, and
`fetch_profiles` treated every throttled ticker as having no data. Lookups run
in alphabetical order, so **both runs only screened tickers starting A–H**.

| Run | Events | Lost to throttling | Picks logged |
|---|---|---|---|
| 2026-11-04 | 1,473 | 802 (54%) | 19, all A–H |
| 2026-11-11 | 1,631 | 959 (59%) | 25, all A–H |

The picks that *were* logged are valid: recorded before outcomes, same
pre-registered config, and a ticker's first letter carries no information about
its returns. Keep them. What was lost is sample size, plus the I–Z names that
would have competed for the 25-pick cap.

**Not backfilled, deliberately.** Re-running those dates now would use today's
market caps and log picks after their outcomes were known — exactly the
contamination the forward test exists to avoid.

The 2026-10-28 run is fine (2 of 789 missing, tickers span A–W).

## Funnel did not count the 25-pick cap (all runs through 2026-11-11)

Survivors ranked below the cap were dropped without a funnel entry, so the
funnel came up short whenever the cap was reached: 6 events on 2026-10-28, 41
on 2026-11-11. This only affected the report, not which picks were logged.

## Fixed 2026-11-17, before the 2026-11-18 run

- `fetch_profiles` retries throttled lookups with backoff, and pauses the rest
  of a round as soon as one lookup is refused. Tickers still refused after
  retries are counted as `rate-limited by Yahoo`, not `no market data`, and
  the report prints a WARNING. If that warning appears, add the run here.
- Survivors cut by the cap are counted as `ranked below top 25`.
- A profile with no market cap is counted as `no market cap` instead of
  `incomplete price history`.

Thresholds in `config.py` are unchanged, so the log continues as one record.
