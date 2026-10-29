"""A `BarStore` hands every session the same bars `set_bars` copies into one.

A sweep runs thousands of sessions over the same few series. `set_bars`
builds a native copy of a series for each, which on years of minute bars is
hundreds of megabytes a run; a store is that copy built once. A session over
a store must run exactly as one over copied bars, and must leave the store as
it found it for the next.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, BarStore
from raptorbt._raptorbt import PortfolioSession

MINUTE_NS = 60_000_000_000


def _bars(closes, start_ts=0):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return (
        np.arange(start_ts, start_ts + n * MINUTE_NS, MINUTE_NS, dtype=np.int64),
        closes - 0.5,
        closes + 1.0,
        closes - 1.0,
        closes,
        np.full(n, 1_000.0),
    )


SERIES = [
    _bars([100.0, 104.0, 97.0, 91.0, 99.0, 108.0, 102.0, 95.0, 101.0, 110.0]),
    _bars([50.0, 52.0, 48.5, 45.5, 49.5, 54.0, 51.0, 47.5, 50.5, 55.0], 30_000_000_000),
]


def _session(attach):
    session = PortfolioSession(config=BacktestConfig(initial_capital=100_000.0, fees=0.001))
    for index, bars in enumerate(SERIES):
        instrument = session.add_instrument(f"S{index}", direction=1)
        attach(session, instrument, bars)
    session.seal()
    return session


def _trade_out(session):
    events = []
    step = 0
    while session.current_event() is not None:
        applied = session.apply_current(
            entry=step % 4 == 0, exit=step % 4 == 2, size_mult=0.4
        )
        events.extend(repr(event) for event in applied)
        step += 1
    result = session.finish()
    trades = result.result.trades()
    assert trades, "the pattern never closed a trade"
    return (
        events,
        result.result.equity_curve().tolist(),
        [(trade.entry_price, trade.exit_price, trade.size) for trade in trades],
        result.metrics.end_value,
    )


def test_a_store_runs_as_copied_bars_and_outlives_its_sessions():
    expected = _trade_out(_session(lambda session, i, bars: session.set_bars(i, *bars)))
    stores = [BarStore(*bars) for bars in SERIES]

    for _ in range(3):
        session = _session(lambda session, i, _bars: session.set_bar_store(i, stores[i]))
        assert _trade_out(session) == expected
    assert [len(store) for store in stores] == [len(bars[0]) for bars in SERIES]


def test_a_store_refuses_what_set_bars_refuses():
    timestamps, open_, high, low, close, volume = SERIES[0]
    with pytest.raises(ValueError, match="share one length"):
        BarStore(timestamps, open_, high, low, close[:-1], volume)
    with pytest.raises(ValueError, match="close must be contiguous"):
        BarStore(timestamps, open_, high, low, np.repeat(close, 2)[::2], volume)

