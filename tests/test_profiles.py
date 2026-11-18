"""Profile lookup under Yahoo's rate limiter, with the network faked out."""

import threading

import pytest
from yfinance.exceptions import YFRateLimitError

from screener import market


class _FakeYahoo:
    """Stands in for yf.Ticker. `throttle` maps ticker -> how many lookups fail
    with a rate limit before one succeeds; `dead` tickers fail outright."""

    def __init__(self, throttle=None, dead=()):
        self.throttle = dict(throttle or {})
        self.dead = set(dead)
        self.calls: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, ticker):
        fake = self

        class _Info:
            def get(self, key, default=None):
                if key != "marketCap":  # one lookup reads several keys
                    return {"exchange": "NMS"}[key]
                with fake._lock:
                    fake.calls[ticker] = fake.calls.get(ticker, 0) + 1
                    if fake.throttle.get(ticker, 0) > 0:
                        fake.throttle[ticker] -= 1
                        raise YFRateLimitError()
                if ticker in fake.dead:
                    raise KeyError("Quote not found")
                return {"marketCap": 800e6, "exchange": "NMS"}[key]

        class _Ticker:
            fast_info = _Info()

        return _Ticker()


@pytest.fixture
def yahoo(monkeypatch):
    monkeypatch.setattr(market, "PROFILE_BACKOFF", 0.0)

    def install(**kwargs):
        fake = _FakeYahoo(**kwargs)
        monkeypatch.setattr(market.yf, "Ticker", fake)
        return fake

    return install


def test_rate_limited_lookups_are_retried_not_dropped(yahoo):
    # The 2026-11 failure: throttled tickers were silently treated as missing.
    yahoo(throttle={"MMM": 1, "ZZZ": 2})
    profiles, throttled = market.fetch_profiles(["AAA", "MMM", "ZZZ"], workers=1)
    assert set(profiles) == {"AAA", "MMM", "ZZZ"}
    assert throttled == set()


def test_persistent_throttle_is_reported_separately(yahoo):
    yahoo(throttle={"ZZZ": 10**6})
    profiles, throttled = market.fetch_profiles(["AAA", "ZZZ"], workers=1)
    assert set(profiles) == {"AAA"}
    assert throttled == {"ZZZ"}


def test_dead_ticker_is_omitted_without_retrying(yahoo):
    fake = yahoo(dead={"DDD"})
    profiles, throttled = market.fetch_profiles(["AAA", "DDD"], workers=1)
    assert set(profiles) == {"AAA"}
    assert throttled == set()
    assert fake.calls["DDD"] == 1


def test_lookups_pause_after_the_first_throttle(yahoo):
    # Once Yahoo starts refusing, hammering it with the rest of the round only
    # extends the limit. Later tickers should wait for the retry instead.
    fake = yahoo(throttle={"AAA": 1})
    profiles, _ = market.fetch_profiles(["AAA", "BBB", "CCC"], workers=1)
    assert set(profiles) == {"AAA", "BBB", "CCC"}
    assert fake.calls == {"AAA": 2, "BBB": 1, "CCC": 1}
