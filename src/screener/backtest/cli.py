"""Run the backtest and print the findings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import ScreenConfig
from . import analyze
from .events import collect_events
from .panel import HORIZONS, build_panel

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / ".cache" / "panel.parquet"
# Printed relative to the repo, so saved output carries no local home path.
PANEL_SHOWN = PANEL_PATH.relative_to(ROOT)


def _pct(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = (out[column] * 100).round(2)
    return out


def build(args) -> pd.DataFrame:
    """Collect events and build the panel, caching the result."""
    if PANEL_PATH.exists() and not args.rebuild:
        print(f"loading cached panel from {PANEL_SHOWN}")
        return pd.read_parquet(PANEL_PATH)

    print(f"collecting earnings events {args.start} -> {args.end}")
    events = collect_events(args.start, args.end)
    print(f"{len(events)} events collected\n")

    print("building panel")
    panel, coverage = build_panel(events)
    print("\ncoverage:")
    for key, value in coverage.items():
        print(f"  {key:>26}: {value:,}")
    dropped = coverage["events"] - coverage["kept"]
    if coverage["events"]:
        print(f"  {'TOTAL DROPPED':>26}: {dropped:,} ({dropped / coverage['events']:.1%})")
        missing = coverage["diag_events_without_shares"]
        print(f"  {'no market cap available':>26}: {missing:,} "
              f"({missing / coverage['events']:.1%}) - excluded by cap filter, not by coverage")

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)
    print(f"\npanel saved to {PANEL_SHOWN}")
    return panel


def report(panel: pd.DataFrame, config: ScreenConfig, horizon: int) -> None:
    print("\n" + "=" * 74)
    print("BACKTEST: post-earnings-announcement drift")
    print("=" * 74)

    universe = analyze.add_groupings(panel)
    print(f"\nfull event panel: {len(panel):,} events -> {len(universe):,} with quintiles")

    screened = analyze.add_groupings(analyze.apply_screen_filters(panel, config))
    print(
        f"after screen filters (${config.min_market_cap / 1e6:.0f}M-"
        f"${config.max_market_cap / 1e9:.1f}B cap, ${config.min_avg_dollar_volume / 1e6:.0f}M+ "
        f"volume, ${config.min_price:.0f}+ price): {len(screened):,} events"
    )

    for label, frame in (("ALL EVENTS", universe), ("SCREEN UNIVERSE", screened)):
        if frame.empty:
            continue
        print(f"\n--- {label}: mean forward return by reaction quintile (%) ---")
        table = analyze.quintile_table(frame)
        print(_pct(table, [f"Q{i}" for i in range(1, 6)] + ["Q5-Q1"]).to_string())
        print("  Q1 = weakest reaction, Q5 = strongest. t is on the Q5-Q1 spread,")
        print("  clustered by entry month. |t| > 1.96 is significant at 5%.")

    if screened.empty:
        print("\nscreen universe is empty; cannot run cost or stability analysis")
        return

    print(f"\n--- COST SENSITIVITY: top quintile, {horizon}-day hold ---")
    costs = analyze.cost_sensitivity(screened, horizon)
    print(_pct(costs, ["net_per_trade", "annualised"]).to_string())

    print(f"\n--- YEAR BY YEAR: Q5-Q1 spread, {horizon}-day hold (%) ---")
    print(_pct(analyze.yearly_stability(screened, horizon), ["spread"]).to_string())

    print(f"\n--- BY MARKET CAP: Q5-Q1 spread, {horizon}-day hold (%), all events ---")
    print(_pct(analyze.size_breakdown(universe, horizon), ["spread"]).to_string())

    print(f"\n--- PLACEBO: reactions shuffled within month, {horizon}-day hold ---")
    for label, frame in (("all events", universe), ("screen universe", screened)):
        placebo = analyze.placebo_spread(frame, horizon, n_permutations=200)
        print(
            f"  {label:>16}: observed {placebo['observed'] * 100:+.2f}%  "
            f"placebo {placebo['placebo_mean'] * 100:+.2f}% "
            f"+/- {placebo['placebo_std'] * 100:.2f}%  p={placebo['p_value']:.3f}"
        )
    print("  p is the share of shuffles producing a spread at least this large.")
    print("  A real effect should be rare under the null; p > 0.05 means it is not.")

    print("\n--- MULTIPLE TESTING ---")
    print(f"  This report evaluates {len(HORIZONS)} horizons x 2 universes = "
          f"{len(HORIZONS) * 2} primary tests, plus per-year and per-size breakdowns.")
    print("  At 5% significance, roughly 1 in 20 will look significant by chance.")
    print("  Treat any single starred result with suspicion unless it holds across")
    print("  horizons, years, and the placebo test.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtest", description="Backtest the drift signal.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 5, 15))
    parser.add_argument("--horizon", type=int, default=20, help="holding period in trading days")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the panel from scratch")
    parser.add_argument("--json", type=Path, help="also write summary stats here")
    args = parser.parse_args(argv)

    panel = build(args)
    if panel.empty:
        print("panel is empty", file=sys.stderr)
        return 1

    config = ScreenConfig()
    report(panel, config, args.horizon)

    if args.json:
        screened = analyze.add_groupings(analyze.apply_screen_filters(panel, config))
        args.json.write_text(
            json.dumps(
                {
                    "events": len(panel),
                    "screened": len(screened),
                    "quintiles": analyze.quintile_table(screened).to_dict(),
                },
                default=str,
                indent=2,
            )
        )
        print(f"\nsummary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
