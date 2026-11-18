"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from . import cache
from .config import ScreenConfig
from .report import append_log, format_report
from .screen import run_screen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Screen recent earnings announcements for post-earnings drift candidates.",
    )
    parser.add_argument("--lookback-days", type=int, help="days of earnings filings to scan")
    parser.add_argument("--min-market-cap", type=float, help="minimum market cap in dollars")
    parser.add_argument("--max-market-cap", type=float, help="maximum market cap in dollars")
    parser.add_argument("--min-reaction", type=float, help="minimum abnormal return, e.g. 0.03")
    parser.add_argument("--min-dollar-volume", type=float, help="minimum average daily dollar volume")
    parser.add_argument("--as-of", type=date.fromisoformat, help="run the screen as of YYYY-MM-DD")
    parser.add_argument("--log", type=Path, help="append picks to this JSONL file")
    parser.add_argument("--clear-cache", action="store_true", help="drop cached API responses first")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.clear_cache:
        print(f"cleared {cache.clear()} cached responses", file=sys.stderr)

    overrides = {
        k: v
        for k, v in {
            "lookback_days": args.lookback_days,
            "min_market_cap": args.min_market_cap,
            "max_market_cap": args.max_market_cap,
            "min_abnormal_return": args.min_reaction,
            "min_avg_dollar_volume": args.min_dollar_volume,
        }.items()
        if v is not None
    }
    config = replace(ScreenConfig(), **overrides)

    try:
        result = run_screen(config, as_of=args.as_of)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(result, config))
    if args.log:
        written = append_log(result, config, args.log)
        print(f"\nlogged {written} picks to {args.log}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
