# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A screener for **post-earnings-announcement drift (PEAD)** — the documented tendency of stocks that surprise on earnings to keep drifting in the surprise direction for weeks. It finds companies that just reported, filters to a size and liquidity band where the effect plausibly survives trading costs, and ranks them by reaction strength.

This is a **research tool, not a trading system**. Nothing here has been backtested yet. See "Validation status" below before treating output as actionable.

## Commands

```bash
export SEC_USER_AGENT="your-name your-email@example.com"   # required, see below
uv sync                                                     # install deps

uv run python -m screener.cli                               # run the screen
uv run python -m screener.cli --lookback-days 20 --min-reaction 0.05
uv run python -m screener.cli --log picks.jsonl             # append picks to log
uv run python -m screener.cli --clear-cache                 # force fresh API data
uv run python -m screener.cli --as-of 2026-08-01            # run as of a past date

uv run pytest                                               # all tests (no network)
uv run pytest tests/test_market.py::test_window_captures_an_after_close_filing
uv run pytest -q -k spread
```

`SEC_USER_AGENT` is mandatory — the SEC throttles or blocks requests without a contact string, and the code raises a clear error rather than failing mysteriously.

## Architecture

Data flows one direction, `edgar → market → screen → report`:

- **`config.py`** — every threshold that shapes the screen, in one dataclass. Changing a number here changes the experiment; treat edits as a new hypothesis, not a tweak.
- **`edgar.py`** — finds earnings announcements via SEC EDGAR full-text search. Rate-limited to 8 req/s (SEC's limit is 10).
- **`market.py`** — daily OHLCV from yfinance, market cap via `fast_info`, plus the event-reaction and spread measures.
- **`screen.py`** — applies filters and tracks a full rejection funnel.
- **`report.py`** — terminal table plus the append-only JSONL pick log.
- **`cache.py`** — disk cache under `.cache/`, 6h TTL on SEC responses. Iterating on the screen re-runs it constantly; the cache keeps that fast and inside fair-access limits.

### Earnings detection is authoritative, not heuristic

A company reports earnings by filing an **8-K carrying Item 2.02** ("Results of Operations and Financial Condition"). EDGAR's full-text search returns those item codes inline in `_source.items`, so `_parse_hit` filters on the actual item code rather than guessing from headlines or scraping a calendar. Don't replace this with a news feed or an earnings-calendar API — it would be strictly less reliable.

The full-text query matches the Item 2.02 heading phrase, then the item code is verified. Precision is high; recall may miss the occasional filing that phrases the heading unusually.

### The event window spans D−1 → D+1, deliberately

Companies file earnings either **before the open** or **after the close**, and EDGAR only gives a filing *date*, not a time. A naive day-0 return silently misses roughly half of all reactions.

Real example from development: OOMA filed Wed 2026-08-26 and was flat that day (20.59 → 20.53), then jumped Thursday to 23.04 — it filed after the bell. Day-0 return would read −0.3% instead of +11.9%. SCSC is the mirror case, filing pre-market and moving the same session.

The two-day window catches both. `tests/test_market.py` locks in both cases; don't narrow this window.

Reaction is measured **in excess of SPY**, so a stock that rose 3% on a day the market rose 3% scores zero. That separates company news from market direction.

### Spread estimators are diagnostics, never gates

This was tested and found wanting — don't re-break it:

| Ticker | Corwin-Schultz | Abdi-Ranaldo | Reality |
|---|---|---|---|
| AAPL | 60bp | 25bp | ~1bp |
| MSFT | 44bp | 117bp | ~1bp |
| DKS | 65bp | 0bp | ~5bp |

Both estimators mistake volatility for spread and disagree by an order of magnitude. Corwin-Schultz at least *ranks* monotonically (SPY < MSFT < AAPL < DKS < MBUU < KOSS), so it survives as a relative hint printed in the report. Liquidity is gated on **measured dollar volume**, which is directly observed rather than inferred.

The Corwin-Schultz overnight adjustment is implemented and barely changes results — consecutive daily ranges usually overlap, so gaps aren't the bias source. Keep it; it's correct, just not sufficient.

### The rejection funnel is load-bearing

`ScreenResult.rejections` counts why every event was dropped, and `test_funnel_accounts_for_every_event` asserts candidates + rejections equals the input count. A screen that reports only survivors is impossible to debug and easy to fool yourself with. When 176 events become 4 candidates, you need to know which filter did that. Preserve this when adding filters.

## Backtest (`src/screener/backtest/`)

```bash
uv run python -m screener.backtest.cli                  # uses cached panel
uv run python -m screener.backtest.cli --rebuild        # recollect from scratch (slow)
uv run python -m screener.backtest.cli --horizon 40
```

`events.py` → `prices.py` / `shares.py` → `panel.py` → `analyze.py`. One row per earnings event, then quintile statistics.

### Data sourcing is the hard part, and it's fragile

- **Yahoo rate-limits hard.** A 1,984-ticker batch returns HTTP 429 and *silently reports live tickers as delisted* — AAPL and ABBV both came back empty during development. Never trust a bulk-download failure as evidence a ticker is dead. `prices.py` batches at 20 with backoff and a permanent parquet cache under `.cache/prices/`; the limit resets in ~20s.
- **Stooq is not a fallback.** It now serves a proof-of-work bot challenge. Don't build against it.
- **EDGAR is the reliable leg.** Rate-limited to 8 req/s, cached, and never fails in bulk the way Yahoo does.

### Three biases the design specifically defends against

1. **Market-cap look-ahead.** Filtering a 2022 event on *today's* market cap selects companies that are $300M–$2B now, which is conditioned on everything that happened since. `shares.py` pulls point-in-time shares outstanding from EDGAR `companyconcept` and keys off the **`filed` date, not the `end` date** — a figure measured March 31 but filed May 5 was not public in April. Tests pin this.
2. **Survivorship.** The event universe comes from historical EDGAR filings, so delisted companies are present. They then drop out when Yahoo has no prices, which quietly reintroduces the bias — so `build_panel` returns a coverage dict counting exactly what was lost, and the CLI prints it. Read that number before trusting any result.
3. **Overlapping windows.** A 20–60 day hold means events overlap heavily and are not independent. Standard errors are clustered by entry month (`_monthly_stat` averages within month, then tests the monthly series). `test_monthly_clustering_ignores_within_month_sample_size` locks this in — without it, packing more correlated events into a month would fake significance.

### Quintiles, not a single bucket

Events are sorted into reaction quintiles **within each month**, so comparisons are always between stocks reporting at the same time. PEAD predicts strong reactions keep rising *and* weak ones keep falling; if all five quintiles drift the same direction, that's market exposure, not drift. The **Q5−Q1 spread** is the real test because it's market-neutral by construction.

Two tests validate the instrument itself: `test_spread_is_zero_when_reaction_carries_no_information` (no false positive on noise) and `test_spread_detects_a_planted_drift` (has the power to find a real effect). If either breaks, no result from this pipeline means anything.

## Validation status: the drift is real, and it is not tradeable here

Authoritative output: **`results/backtest_2026-08-30_corrected.txt`**. 91,866 events, 2021-01 to 2026-05, **78,632 usable**.

An earlier run in `results/backtest_2026-08-30.txt` reported a flat null. **That result was wrong** — see "The bug that faked a null" below. Don't cite it.

### What the corrected panel shows

Across all events, quintiles are cleanly monotonic and the spread grows with horizon:

| Horizon | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 | t |
|---|---|---|---|---|---|---|---|
| 20d | −0.92 | −0.86 | −0.60 | −0.54 | −0.08 | **+0.84** | 2.62 |
| 40d | −2.72 | −1.93 | −1.51 | −1.44 | −1.07 | **+1.65** | 4.15 |
| 60d | −3.86 | −3.18 | −2.30 | −2.17 | −1.80 | **+2.06** | 4.18 |

Placebo p = **0.000** (observed +0.84% against a shuffled null of +0.01% ± 0.22%). This is a genuine effect, not a pipeline artefact.

Caveat on the 40/60-day rows: a 60-day hold spans roughly three months, so monthly clustering under-corrects there. The 20-day result (t = 2.62) is the cleanest.

### Why it still doesn't help

The effect is almost entirely sub-$300M, and it dies on contact with tradeability:

| Filter | n | Q5−Q1 | t | **Q5 long-only** |
|---|---|---|---|---|
| all sub-$300M | 16,715 | 2.44% | 3.63 | +1.77% |
| price ≥ $5 | 8,834 | 2.84% | 3.43 | +1.15% |
| + $500k/day volume | 4,219 | 2.77% | 2.05 | +0.79% |
| + $1M/day volume | 2,608 | 2.00% | 1.16 | **−0.42%** |

Two things kill it. The spread is **long-short** — capturing it means shorting Q1 microcaps, where borrow is expensive or unavailable. And the long-only leg, which is all a retail account can realistically take, goes **negative** once you require $1M/day of volume.

The screen's own universe ($300M–$2B) shows nothing at all: Q5−Q1 = +0.12%, t = 0.19, placebo **p = 0.81**.

Sub-$300M liquidity is why: median daily dollar volume $402k, only 29.8% above $1M/day, 47.1% priced under $5.

This is what you'd expect of an anomaly published in 1968. The tradeable version was arbitraged away; what survives sits below the friction floor.

### The bug that faked a null

Two data faults, both mine, flattened the original result:

1. **Corrupt prices.** Yahoo's split adjustment emitted a **negative close** (−0.1177 on CODQL), yielding a −745% "return" — impossible for a long position. `panel.py` now rejects any series containing a non-positive close.
2. **No winsorization.** A $0.0009 stock moving one tick is +66,000%. A handful of such prints dominated 15,000-event quintile means and destroyed the ordering — Q3 spiked to +1.97% while its neighbours sat near zero. `analyze.py` now winsorizes returns at 1%/99%, standard practice in cross-sectional asset pricing.

A third fault inflated the apparent survivorship problem: the price fetcher treated "absent from a partial Yahoo batch" as "delisted", blacklisting ~900 live tickers. Spot checks found 8 of 10 sampled misses had full history. `prices.py` now confirms each absentee individually before blacklisting. Dropped events fell from 22.2% to 14.4%.

**Lesson for future work here: check the return distribution before trusting any aggregate.** A single bad print can hide a real effect, and the pipeline's own null-and-power tests will still pass while it does.

## Second signal tested: filing-language change ("Lazy Prices") — also null

`src/screener/backtest/` `filings.py` → `textpanel.py` → `text_cli.py`. Output in **`results/text_backtest_2026-08-30.txt`**.

Tests Cohen, Malloy & Nguyen (*JF* 2020): companies that rewrite their 10-K/10-Q language underperform, because few investors read closely enough to notice. Signal is Jaccard vocabulary overlap between a filing and the same-form filing ~a year earlier. **11,847 filing pairs, 600 companies, 16,677 documents.**

```bash
uv run python -m screener.backtest.text_cli
uv run python -m screener.backtest.text_cli --horizon 40 --signal cosine
```

### The headline looked real. It wasn't.

| Sample | n | Q5−Q1 | t | placebo p |
|---|---|---|---|---|
| full 2020–2026 | 11,460 | +1.69% | 2.87 | **0.005** |
| 2020–2021 only | 2,094 | +4.75% | 3.31 | **0.000** |
| excluding 2020–2021 | 9,366 | +0.80% | 1.36 | 0.220 |
| **2023 onward** | 7,466 | +0.70% | 1.01 | **0.475** |

Essentially all of it is COVID. In 2020–2021 companies rewrote risk factors en masse to add pandemic language; how much a firm rewrote proxied for COVID exposure, and COVID exposure predicted returns. The signal is that confound. From 2023 onward it sits at the dead centre of the null distribution — p = 0.475 is not "nearly significant", it is noise.

**The full-sample p = 0.005 would have been reportable.** The year-by-year breakdown is what caught it. Keep that panel in any future signal test.

Secondary findings, for completeness: the effect (such as it is) is carried entirely by Q1, the most-rewritten filings, at −1.36% (t = −2.33) — a **short** signal, and shorting these names is impractical. It also dies at liquidity like everything else here: Q5−Q1 falls to +0.10% (t = 0.14) once you require $1M/day volume, and to −1.25% in the $300M–$2B band.

### Implementation notes that cost real time

- **Extraction is regex, not BeautifulSoup, deliberately.** Filings run to several megabytes; a DOM parse made collection CPU-bound at 150% of one core (17 docs/min). Regex stripping is ~10x faster and agrees on 97.9% of vocabulary. Collection went to 392 docs/min. Don't "improve" this back to a parser.
- **Never mix extractors within a corpus.** Two extraction methods disagree on ~2% of vocabulary, which registers as fake dissimilarity. Changing the extractor means re-extracting everything.
- **Numbers are stripped from tokens.** Every quarterly report restates its figures; leaving them in measures arithmetic, not language.
- **Stopwords are removed and Jaccard is primary.** Raw term-frequency cosine scores ~0.999 for every pair (std 0.009) — no cross-sectional spread to rank on. Jaccard spreads ~10x wider (std 0.102).
- **Quintiles rank within month × form.** 10-Q similarity runs systematically below 10-K similarity, so a combined sort ranks by form type rather than by signal. `test_combined_ranking_would_confound_form_with_signal` demonstrates the failure.
- **Filings before 2020-09 are skipped before fetching.** EDGAR returns 10+ years per company; without the filter, 53% of fetched documents produced no usable row.

## Forward test (running)

A scheduled job runs the screen weekly and appends picks to `picks.jsonl` **before outcomes exist**. This is the only evaluation here that cannot be contaminated by hindsight or by a model having seen the future.

Scheduled by a **systemd user timer**, not cron:

```bash
systemctl --user list-timers earnings-forward-test.timer   # next run
systemctl --user start earnings-forward-test.service       # run now
journalctl --user -u earnings-forward-test.service -n 20   # what happened
systemctl --user disable --now earnings-forward-test.timer # stop it

./run_weekly_screen.sh                        # run by hand; appends to forward_test.log
uv run python -m screener.forward             # score it
uv run python -m screener.forward --horizon 40
```

Units live in `~/.config/systemd/user/earnings-forward-test.{service,timer}`. The
timer sets **`Persistent=true`** deliberately: cron silently drops a run if the
machine is asleep at the scheduled moment, and a forward-test log that quietly
develops gaps is worth much less than one that does not. With `Persistent`, a
missed window fires on the next boot or login instead.

User lingering is off (`loginctl show-user $USER -p Linger`), so the timer needs
an active login session. `Persistent=true` covers the gap, and given the screen
looks back 10 days a few hours' delay changes nothing. Enable lingering only if
the machine is routinely powered on with nobody logged in.

`.env` holds `SEC_USER_AGENT` (gitignored; see `.env.example`).

**Read `forward_test_notes.md` before scoring.** It records runs with known
defects — the 2026-11-04 and 11-11 runs only screened A–H tickers because of a
Yahoo rate limit. Picks are never edited or backfilled; defects are noted there.

**The thresholds in `config.py` are pre-registered.** Changing them while the log accumulates destroys its value — the entire point is that the rules were fixed in advance. If you change them, start a new log file.

Design points worth keeping:

- **Entry is the close of the first trading day *after* the pick was logged.** The screen runs after the close, so the logged day's close was never reachable; entering on it would be look-ahead.
- **Picks are deduplicated on SEC accession.** An event stays inside the 1–10 day entry window for weeks, so a weekly run would otherwise log one position several times and count it as independent observations.
- **Nothing is ever silently dropped.** Three separate paths (no price data, logged after the last close, not yet matured) all report the pick as *open* rather than skipping it. An early version dropped all three, which made a populated log read as "0 picks logged". Tests pin each path.
- **`forward.py` prints a warning with every result.** The t-statistic it reports is naive — overlapping holds, earnings-season clustering — and at a few hundred picks a year it cannot reach a real threshold either way. Given the backtest measured this signal as null in this band, a positive number here is far more likely noise than discovery.

Expected outcome: nothing. It costs a cron entry and no capital, and in six months there is an honest forward record either way.

## Where this leaves the project

Two independent signals tested properly, both null for a retail account:

1. **Price reaction (PEAD)** — real, monotonic, placebo p = 0.000, but sub-$300M only and the long leg goes negative at $1M/day volume.
2. **Filing-language change** — not real outside the COVID window.

The infrastructure is signal-agnostic and validated: swapping in a third signal means writing one scoring function. But note the pattern in both results — anomalies published in 1968 and 2020 survive only below the liquidity floor. That is what an efficient-enough market looks like from the outside, and a third attempt should expect the same unless it tests something genuinely un-mined.

Do not tune thresholds against these panels until something passes. The multiple-testing counters exist to make that visible.

### If you want to keep going

Do not tune thresholds against this panel and re-run until something passes — that manufactures a false positive, and the multiple-testing counter in the report exists to make that visible. Genuinely different directions:

1. **A different signal entirely** — the filing-text angle (LLM on 10-Q/10-K language diffs) was never the thing tested here. This backtest only rules out *price reaction* as a predictor.
2. **A different anomaly** with its own literature, tested the same way.
3. **Accept the null.** This is a real answer, obtained cheaply, and it's the one the base rates predicted.

### LLM look-ahead contamination

If step 3 involves having a model analyze historical news or filings, **the model already knows what happened next**. Backtests of LLM sentiment on pre-cutoff data measure recall, not prediction, and will look spectacular while meaning nothing.

Only two clean approaches: test strictly on data after the model's training cutoff, or forward-test from today. The `--log` JSONL exists for exactly this — it records picks with a timestamp *before* outcomes are known, so there's an honest record to score later. Never evaluate picks that weren't logged in advance.

## Context

The project is designed around a **small retail account used for learning**, not for deploying capital. That shapes several decisions:

- Small position sizes mean spread costs matter enormously — hence the liquidity band rather than chasing the wider theoretical edge in microcaps.
- Below $25k, the pattern-day-trader rule limits margin accounts to 3 day trades per 5 business days. This strategy holds for days-to-weeks and sidesteps that entirely, by design.
- Live trading at this size can never produce statistical significance. The research sample (thousands of historical events) is what answers whether the edge is real; live trades teach execution mechanics. Keep those two purposes separate and don't let live results masquerade as evidence.
