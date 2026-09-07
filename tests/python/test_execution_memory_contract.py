"""Resource controls fail explicitly and do not change accounting."""

import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders
from raptorbt.strategy.limits import ExecutionLimitExceeded, ExecutionLimits
from test_orders import _bars


class Reentrant(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Market(side="buy", units=1))

    def on_order_filled(self, ctx, event):
        self.submit_order(orders.Market(side="sell" if ctx.position else "buy", units=1))


def test_reentrant_fill_loop_fails_instead_of_retaining_unbounded_events():
    config = BacktestConfig()
    config.fees = 0
    with pytest.raises(ExecutionLimitExceeded, match="did not quiesce"):
        run_strategy_backtest(Reentrant(), **_bars([100., 101.]), config=config,
                              limits=ExecutionLimits(20))


def test_timestamp_budget_is_shared_across_passes_and_resets_on_progress():
    budget = ExecutionLimits(2).budget()
    budget.consume(1)
    budget.consume(1)
    budget.consume(2, 2)
    with pytest.raises(ExecutionLimitExceeded):
        budget.consume(2)


class RoundTrip(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Market(side="buy", units=1))
        elif ctx.idx == 2:
            self.submit_order(orders.Market(side="sell", units=1, reduce_only=True))


def test_discarding_output_curves_preserves_trades_and_metrics_exactly():
    full = BacktestConfig()
    compact = BacktestConfig()
    compact.retain_curves = False
    data = _bars([100., 110., 105., 106.])
    a = run_strategy_backtest(RoundTrip(), **data, config=full)
    b = run_strategy_backtest(RoundTrip(), **data, config=compact)
    assert len(a.equity_curve()) == 4
    assert len(b.equity_curve()) == len(b.drawdown_curve()) == len(b.returns()) == 0
    assert len(a.trades()) == len(b.trades()) == 1
    for field in ('entry_price', 'exit_price', 'pnl', 'fees'):
        assert getattr(a.trades()[0], field) == getattr(b.trades()[0], field)
    for field in ('total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
                  'win_rate_pct', 'total_trades', 'total_fees_paid', 'expectancy', 'exposure_pct'):
        np.testing.assert_equal(getattr(a.metrics, field), getattr(b.metrics, field))


def test_daily_performance_is_opt_in_and_retains_one_settled_mark_per_day():
    day = 86_400_000_000_000
    data = _bars([100.0, 110.0, 105.0, 106.0])
    data["timestamps"] = np.array([0, 1, day, day + 1], dtype=np.int64)
    default = BacktestConfig(retain_curves=False)
    compact = BacktestConfig(
        retain_curves=False,
        retain_daily_performance=True,
        performance_tz_transitions=[(-1, 0)],
    )
    absent = run_strategy_backtest(RoundTrip(), **data, config=default)
    retained = run_strategy_backtest(RoundTrip(), **data, config=compact)
    assert absent.daily_performance_timestamps().size == 0
    np.testing.assert_array_equal(
        retained.daily_performance_timestamps(), np.array([1, day + 1])
    )
    assert retained.daily_performance_equity().size == 2
    assert retained.daily_performance_equity()[-1] == retained.metrics.end_value
