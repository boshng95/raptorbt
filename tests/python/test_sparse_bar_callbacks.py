"""Optional sparse strategy callbacks still step every market bar."""

import numpy as np
import pytest

from raptorbt import BacktestConfig, KernelSession, Strategy, run_strategy_backtest
from raptorbt.strategy import orders


def _bars():
    timestamps = np.arange(5, dtype=np.int64) * 60_000_000_000
    close = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    return (timestamps, close, close, close, close, np.ones(5))


def test_masked_bar_callbacks_can_be_woken_by_the_strategy() -> None:
    class Sparse(Strategy):
        def __init__(self):
            super().__init__()
            self.calls = []

        def on_bar(self, ctx):
            self.calls.append(ctx.idx)
            self.bar_callback_required = ctx.idx == 1

    strategy = Sparse()
    sparse = run_strategy_backtest(
        strategy,
        *_bars(),
        bar_callback_mask=np.asarray([True, True, False, False, True]),
    )
    ordinary = run_strategy_backtest(Sparse(), *_bars())

    assert strategy.calls == [0, 1, 2, 4]
    assert np.array_equal(sparse.equity_curve(), ordinary.equity_curve())


def test_mask_must_match_primary_bars() -> None:
    with pytest.raises(ValueError, match="bar_callback_mask"):
        run_strategy_backtest(Strategy(), *_bars(), bar_callback_mask=[True])


def test_idle_batch_returns_at_a_fill_and_wakes_the_next_bar() -> None:
    assert hasattr(KernelSession, "step_until_event")

    class Waiting(Strategy):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.fills = []

        def on_bar(self, ctx):
            self.calls.append(ctx.idx)
            if ctx.idx == 0:
                self.submit_order(orders.Limit(side="buy", price=97.0, units=10.0))
            self.bar_callback_required = False

        def on_order_filled(self, ctx, event):
            self.fills.append(ctx.idx)
            self.bar_callback_required = True

    ts = np.arange(6, dtype=np.int64) * 60_000_000_000
    close = np.asarray([100.0, 100.0, 99.0, 97.0, 98.0, 99.0])
    low = np.asarray([99.0, 99.0, 98.0, 96.0, 97.5, 98.5])
    volume = np.full(6, 1_000.0)
    config = BacktestConfig()
    config.fees = 0.0
    strategy = Waiting()
    sparse = run_strategy_backtest(
        strategy,
        ts,
        close,
        close + 1.0,
        low,
        close,
        volume,
        config=config,
        bar_callback_mask=np.asarray([True, False, False, False, False, True]),
    )
    ordinary = run_strategy_backtest(
        Waiting(), ts, close, close + 1.0, low, close, volume, config=config
    )

    assert strategy.calls == [0, 4, 5]
    assert strategy.fills == [3]
    assert np.array_equal(sparse.equity_curve(), ordinary.equity_curve())


def test_full_idle_range_keeps_the_final_context_and_equity_curve() -> None:
    class Idle(Strategy):
        def on_stop(self, ctx):
            self.last_idx = ctx.idx

    strategy = Idle()
    sparse = run_strategy_backtest(strategy, *_bars(), bar_callback_mask=[False] * 5)
    ordinary = run_strategy_backtest(Idle(), *_bars())

    assert strategy.last_idx == 4
    assert np.array_equal(sparse.equity_curve(), ordinary.equity_curve())
