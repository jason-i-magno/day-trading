"""Backtest the filing-similarity signal and print the findings.

Reuses the earnings backtest's statistics wholesale -- month-clustered standard
errors, winsorized returns, the permutation placebo. Only the signal differs, so
a result here is comparable to the price-reaction result on the same terms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import ScreenConfig
from . import analyze
from .panel import HORIZONS

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / ".cache" / "text_panel.parquet"
# Printed relative to the repo, so saved output carries no local home path.
PANEL_SHOWN = PANEL_PATH.relative_to(ROOT)
SIGNAL = "similarity"


def _pct(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = (out[column] * 100).round(2)
    return out


def report(panel: pd.DataFrame, horizon: int, signal: str = SIGNAL) -> None:
    print("=" * 74)
    print("BACKTEST: filing-language change ('Lazy Prices')")
    print("=" * 74)
    print(f"\n{len(panel):,} filing pairs, "
          f"{panel['ticker'].nunique():,} companies, "
          f"{panel['filing_date'].min()} to {panel['filing_date'].max()}")
    print(f"forms: {dict(panel['form'].value_counts())}")
    s = panel[signal].dropna()
    print(f"{signal}: mean {s.mean():.3f}  std {s.std():.3f}  "
          f"p10 {s.quantile(.10):.3f}  p90 {s.quantile(.90):.3f}")

    grouped = analyze.add_groupings_by(panel, signal, ["form"])
    print(f"\nranked within month x form: {len(grouped):,} observations")
    print("\nQ1 = most changed language, Q5 = least changed.")
    print("Lazy Prices predicts Q1 underperforms, i.e. a POSITIVE Q5-Q1 spread.\n")

    print("--- mean forward return by similarity quintile (%) ---")
    table = analyze.quintile_table(grouped)
    print(_pct(table, [f"Q{i}" for i in range(1, 6)] + ["Q5-Q1"]).to_string())
    print("  t is on the Q5-Q1 spread, clustered by entry month.")

    print(f"\n--- PLACEBO: similarity shuffled within month, {horizon}-day hold ---")
    shuffled = grouped.rename(columns={signal: "reaction"})
    placebo = analyze.placebo_spread(shuffled, horizon, n_permutations=200)
    print(f"  observed {placebo['observed'] * 100:+.2f}%   "
          f"placebo {placebo['placebo_mean'] * 100:+.2f}% "
          f"+/- {placebo['placebo_std'] * 100:.2f}%   p={placebo['p_value']:.3f}")

    print(f"\n--- BY MARKET CAP: Q5-Q1 spread, {horizon}-day hold (%) ---")
    print(_pct(analyze.size_breakdown(grouped, horizon), ["spread"]).to_string())

    print(f"\n--- YEAR BY YEAR: Q5-Q1 spread, {horizon}-day hold (%) ---")
    print(_pct(analyze.yearly_stability(grouped, horizon), ["spread"]).to_string())

    print("\n--- TRADEABILITY ---")
    print(f"{'filter':<34}{'n':>8}{'Q5-Q1':>9}{'t':>8}{'Q5 only':>10}")
    config = ScreenConfig()
    filters = [
        ("all pairs", panel),
        ("price >= $5", panel[panel["price"] >= config.min_price]),
        ("+ $1M/day volume", panel[(panel["price"] >= config.min_price)
                                   & (panel["dollar_volume"] >= 1e6)]),
        ("+ $300M-$2B cap", panel[(panel["price"] >= config.min_price)
                                  & (panel["dollar_volume"] >= 1e6)
                                  & panel["market_cap"].between(config.min_market_cap,
                                                                config.max_market_cap)]),
    ]
    for label, subset in filters:
        if len(subset) < 300:
            print(f"{label:<34}{len(subset):>8,}   too few")
            continue
        chunk = analyze.add_groupings_by(subset, signal, ["form"])
        spread = analyze._spread_stat(chunk, f"fwd_{horizon}")
        top = analyze._monthly_stat(chunk[chunk["quintile"] == 5], f"fwd_{horizon}")
        print(f"{label:<34}{len(chunk):>8,}{spread.mean * 100:>8.2f}%"
              f"{spread.t_stat:>8.2f}{top.mean * 100:>9.2f}%")

    print("\n--- MULTIPLE TESTING ---")
    print(f"  {len(HORIZONS)} horizons plus size, year, and tradeability breakdowns.")
    print("  A real effect should hold across horizons and survive the placebo.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="text-backtest")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--signal", default=SIGNAL, choices=["similarity", "cosine"],
                        help="similarity = Jaccard (primary), cosine = robustness check")
    args = parser.parse_args(argv)

    if not PANEL_PATH.exists():
        print(f"no panel at {PANEL_SHOWN}; build it first", file=sys.stderr)
        return 1
    panel = pd.read_parquet(PANEL_PATH)
    if panel.empty:
        print("panel is empty", file=sys.stderr)
        return 1
    report(panel, args.horizon, args.signal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
