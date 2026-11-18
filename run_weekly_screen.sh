#!/bin/bash
# Weekly forward-test run. Appends picks to picks.jsonl with a timestamp, before
# any outcome is known. Invoked by cron; safe to run by hand.
#
# The screen's thresholds are PRE-REGISTERED. Changing src/screener/config.py
# while this log is accumulating invalidates it -- the whole value of a forward
# test is that the rules were fixed in advance. Start a new log if you change them.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

# Your SEC contact string. The SEC throttles or blocks requests without one.
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi
if [ -z "${SEC_USER_AGENT:-}" ]; then
    echo "$(date -Is)  ERROR: SEC_USER_AGENT not set. Create .env containing:" >> forward_test.log
    echo "  SEC_USER_AGENT=\"your-name your-email@example.com\"" >> forward_test.log
    exit 1
fi

{
    echo "===== $(date -Is) ====="
    PYTHONPATH=src uv run python -m screener.cli --log picks.jsonl 2>&1
    echo
} >> forward_test.log 2>&1
