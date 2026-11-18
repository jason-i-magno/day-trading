# Earnings drift screener

Finds companies that just reported earnings, filters to a size and liquidity
band where post-earnings drift plausibly survives trading costs, and ranks them
by how hard the stock reacted.

**Backtested. The drift is real; it is not tradeable in this size band.**
See `results/backtest_2026-08-30_corrected.txt`.

Across 78,632 earnings events (2021-2026), reaction quintiles are cleanly
monotonic and the Q5-Q1 spread reaches +2.06% at 60 days (t = 4.18, placebo
p = 0.000). But the effect is almost entirely sub-$300M, it is long-short (you
would have to short illiquid microcaps to capture it), and the long-only leg
turns **negative** once you require $1M/day of volume. This screener's own
universe ($300M-$2B) shows nothing: placebo p = 0.81.

That is roughly what a 1968 anomaly should look like -- arbitraged out of the
liquid names, surviving only below the friction floor.

## Setup

```bash
uv sync
export SEC_USER_AGENT="your-name your-email@example.com"
```

The SEC requires a contact string identifying who is making requests. Requests
without one get throttled or blocked.

## Usage

```bash
uv run python -m screener.cli                      # default screen
uv run python -m screener.cli --min-reaction 0.05  # stronger surprises only
uv run python -m screener.cli --log picks.jsonl    # record picks for later scoring
```

Sample output:

```
Earnings drift screen - 2026-08-29
==============================================================================
176 earnings 8-Ks in the last 12 days  ->  4 candidates

TICKER      PRICE   MKT CAP   $VOL/DAY  REACTION  DAYS  SPREAD~
------------------------------------------------------------------------------
OOMA        22.91      629M         8M    11.2%     1    105bp !
SCSC        56.55     1.15B        13M     6.1%     5     66bp
FLXS        81.80      333M         5M     5.7%     8     71bp
HOV        129.77      760M        18M     3.4%     5    117bp !

Rejection funnel
   145  market cap outside band
    13  reaction too weak
     4  price below minimum
     4  outside drift entry window
```

`!` marks a wide spread estimate. `REACTION` is the move across the
announcement in excess of SPY. `DAYS` is trading days since the reaction.

## Second signal: filing-language change

```bash
uv run python -m screener.backtest.text_cli
```

Tests whether companies that rewrite their 10-K/10-Q language underperform
("Lazy Prices", Cohen, Malloy & Nguyen 2020). 11,847 filing pairs across 600
companies, scored by vocabulary overlap against the same-form filing a year
earlier.

**Also null.** The full sample looks significant (Q5-Q1 = +1.69%, placebo
p = 0.005), but it is a COVID artifact:

| Sample | n | Q5-Q1 | placebo p |
|---|---|---|---|
| full 2020-2026 | 11,460 | +1.69% | 0.005 |
| 2020-2021 only | 2,094 | +4.75% | 0.000 |
| 2023 onward | 7,466 | +0.70% | 0.475 |

In 2020-2021 firms rewrote risk factors to add pandemic language, so "how much
did the filing change" proxied for COVID exposure during a period when COVID
exposure predicted returns. From 2023 onward the signal is indistinguishable
from noise.

See `results/text_backtest_2026-08-30.txt`.

## How it works

1. **Find earnings** — SEC EDGAR full-text search for 8-K filings carrying
   Item 2.02, the item code for an earnings release. Authoritative, free, no
   calendar scraping.
2. **Measure the reaction** — return from the close before the filing to the
   close after, minus SPY over the same window. The two-day span catches the
   move whether the company reported before the open or after the bell.
3. **Filter** — $300M–$2B market cap, $5+ price, $1M+ average daily dollar
   volume, and an entry window of 1–10 trading days after the reaction. Day 0–1
   is skipped: that's when spreads are widest and the initial repricing is
   still happening.
4. **Rank** — strongest reaction first.

Every threshold lives in `src/screener/config.py`.

## Backtest

```bash
uv run python -m screener.backtest.cli              # analyse the cached panel
uv run python -m screener.backtest.cli --rebuild    # recollect (slow: ~1-2 hours)
uv run python -m screener.backtest.cli --horizon 40
```

Collects every earnings 8-K over a multi-year window, measures what happened
after each one, and asks whether the reaction predicted the drift.

Events are sorted into **reaction quintiles within each month**, so the
comparison is always between stocks reporting at the same time. The headline
number is the **Q5-Q1 spread**: strongest reactions minus weakest. That spread
is market-neutral by construction, so it isolates the signal from market
direction. Standard errors are clustered by month, because 20-60 day holding
periods overlap and treating events as independent would manufacture
significance.

The report also breaks results down by year (an effect present in only one year
is noise), by market-cap band, and across round-trip cost assumptions from 0 to
100bp.

### Known data limitations

- **Survivorship.** Yahoo deletes delisted tickers' history entirely, so
  companies that were acquired or went under are simply unavailable. Around 15%
  of the historical universe is missing for this reason. The direction of the
  bias is ambiguous (acquisitions exit at a premium, bankruptcies at zero) but
  the magnitude is real. The backtest prints its coverage numbers; read them.
- **No intraday data.** Everything is daily closes, so the backtest cannot model
  slippage within a session or the actual spread paid.
- **Cost assumptions are scenarios, not measurements.** The cost table shows what
  the result would be at various round-trip costs; it does not know what you
  would actually pay.

## Forward test

A weekly cron job logs picks with a timestamp *before* outcomes exist -- the one
evaluation here that hindsight cannot contaminate.

```bash
cp .env.example .env      # put your real SEC contact string in it

systemctl --user list-timers earnings-forward-test.timer   # next scheduled run
systemctl --user start earnings-forward-test.service       # run now
./run_weekly_screen.sh                                     # or run it by hand
uv run python -m screener.forward                          # score what has matured
```

Runs weekly via a systemd user timer with `Persistent=true`, so a run missed
because the machine was asleep fires on next boot rather than being lost.

Entry is the close of the first trading day after logging. Picks are deduplicated
on SEC accession so one earnings event is never counted as several positions.
The screen's thresholds are pre-registered: changing `config.py` while the log
accumulates invalidates it.

Expect nothing. The backtest already measured this signal as null in this size
band, and a few hundred picks a year cannot reach significance either way. It
costs a cron entry and no capital.

## Tests

```bash
uv run pytest
```

No network access required — all tests run against synthetic data.
