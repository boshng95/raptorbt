//! Steppable single-instrument runner.
//!
//! Wraps an [`EngineKernel`] together with the per-bar bookkeeping the batch
//! engine previously kept as loop locals: equity/drawdown/return curves, peak
//! tracking, streaming trade metrics, the kill-switch feed, and end-of-data
//! finalization. Both the batch engine and per-bar drivers (e.g. the strategy
//! session exposed to Python) loop this type, so the accounting has exactly one
//! implementation.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, Direction, InstrumentConfig, Trade,
};
use crate::execution::orders::OrderRecord;
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::curve::InstantCurve;
use crate::portfolio::engine::{compute_backtest_metrics_with_config, PortfolioEngine};
use crate::portfolio::kernel::{
    BarColumns, EngineEvent, EngineKernel, IdleRun, KernelBar, StepInput,
};

/// Drives one instrument bar-by-bar and accumulates result curves.
///
/// Call [`SingleRunner::step`] once per bar in ascending index order, then
/// [`SingleRunner::finish`] to force-close any open position and compute
/// metrics.
#[derive(Debug)]
pub struct SingleRunner {
    kernel: EngineKernel,
    config: BacktestConfig,
    /// One sample per bar instant, taken once the instant has settled; see
    /// [`InstantCurve`].
    curve: InstantCurve,
    trades: Vec<Trade>,
    /// What became of every order, at its id. Kept apart from the book
    /// because one order is mutated across several events (accepted, then
    /// triggered, then filled in slices); the book supplies submission order
    /// at `finish`.
    order_events: Vec<Option<OrderRecord>>,
    streaming: StreamingMetrics,
    /// Equity marked at the most recent step's close: what the strategy and
    /// the kill-switch see, before anything the strategy then does at that
    /// instant.
    marked_equity: f64,
    /// High-water mark of `marked_equity`, which the kill-switch measures
    /// its drawdown from.
    risk_peak: f64,
    steps: usize,
    last_bar: Option<(usize, KernelBar)>,
}

impl SingleRunner {
    /// Build a runner from engine-level models and optional per-instrument config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: BacktestConfig,
        fee_model: FeeModel,
        slippage_model: SlippageModel,
        fill_price: FillPrice,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        let initial_capital = config.initial_capital;
        let risk = config.risk_gate();
        let kernel = EngineKernel::new(
            config.clone(),
            fee_model,
            slippage_model,
            fill_price,
            symbol,
            direction,
            inst_config,
        )
        .with_risk_gate(risk);

        let curve = InstantCurve::new(initial_capital)
            .recording(config.retain_curves || config.compute_metrics)
            .with_daily_performance(
                config.retain_daily_performance.then(|| config.performance_tz_transitions.clone()),
            );
        Self {
            kernel,
            config,
            curve,
            trades: Vec::new(),
            order_events: Vec::new(),
            streaming: StreamingMetrics::new(),
            marked_equity: initial_capital,
            risk_peak: initial_capital,
            steps: 0,
            last_bar: None,
        }
    }

    /// Build a runner deriving fee/slippage/fill models from the config, the
    /// same way [`PortfolioEngine::new`] does.
    pub fn from_config(
        config: BacktestConfig,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        let engine = PortfolioEngine::new(config);
        Self::new(
            engine.config.clone(),
            engine.fee_model.clone(),
            engine.slippage_model.clone(),
            engine.fill_price,
            symbol,
            direction,
            inst_config,
        )
    }

    /// Attach an instrument market definition; see [`EngineKernel::with_instrument`].
    pub fn with_instrument(mut self, spec: InstrumentSpec) -> Self {
        self.kernel.set_instrument(spec);
        self
    }

    /// Set the position policy; see [`EngineKernel::with_position_policy`].
    pub fn with_position_policy(
        mut self,
        policy: crate::portfolio::ledger::PositionPolicy,
    ) -> Self {
        self.kernel.set_position_policy(policy);
        self
    }

    /// Set the account mode; see [`EngineKernel::with_account_mode`].
    pub fn with_account_mode(mut self, account: crate::accounts::AccountMode) -> Self {
        self.kernel.set_account_mode(account);
        self
    }

    /// Advance one bar: delegate to the kernel, then account for the outcome.
    ///
    /// The returned events are the same ones the kernel produced; completed
    /// trades have already been recorded internally.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        self.advance(idx, bar, input, true)
    }

    /// Step bars from `start` for as long as the kernel stays idle on them,
    /// up to `end`, and return the first bar not stepped.
    ///
    /// Each bar stepped is stepped as [`Self::step`] would have, with its
    /// input built by [`BarColumns::input`], and produced no event. An idle
    /// bar changes nothing the account is worth bar to bar but the close
    /// it is marked at, so after the first -- which closes whatever instant
    /// the account last moved in -- a run whose timestamps rise and whose
    /// closes [`EngineKernel::idle_run`] can bound is taken in one piece:
    /// its book and extremes from its newest bars, its peak from its best
    /// close, and its samples only at the day boundaries the daily marks
    /// keep. A run that cannot be bounded is stepped bar by bar.
    pub fn step_idle_run(&mut self, bars: &BarColumns<'_>, start: usize, end: usize) -> usize {
        if start >= end || !self.kernel.idle_state() {
            return start;
        }
        let bar = bars.bar(start);
        if !self.kernel.idle_at(&bar) {
            return start;
        }
        self.advance(start, &bar, bars.input(start), true);
        let first = start + 1;
        if first == end {
            return end;
        }
        let rising = match bars.index {
            Some(index) => index.rising(start, end),
            None => bars.timestamp[start..end].windows(2).all(|pair| pair[0] < pair[1]),
        };
        if rising
            && !self.curve.records()
            && self.curve.open_instant() == Some(bars.timestamp[start])
            && !self.kernel.watches_drawdown()
        {
            let live = self.kernel.live_until(bars.timestamp, first, end);
            if live > first {
                if let Some(run) = self.kernel.idle_run(bars, first, live) {
                    self.take_idle_run(bars, first, live, &run);
                    return live;
                }
            }
        }
        for idx in first..end {
            let bar = bars.bar(idx);
            if !self.kernel.idle_at(&bar) {
                return idx;
            }
            self.advance(idx, &bar, bars.input(idx), true);
        }
        end
    }

    /// Step bars `first..end` idle in one piece; see [`Self::step_idle_run`].
    fn take_idle_run(&mut self, bars: &BarColumns<'_>, first: usize, end: usize, run: &IdleRun) {
        // Each bar closes the instant before it at the account the bar
        // before it left, which no bar of the run changes.
        let kernel = &self.kernel;
        self.curve
            .close_run(bars.timestamp, first - 1, end - 1, |idx| kernel.equity(bars.close[idx]));
        self.kernel.step_idle_bars(bars, first, end, run);
        let last = bars.bar(end - 1);
        self.marked_equity = self.kernel.equity(last.close);
        if let Some(peak) = run.peak_equity {
            if peak > self.risk_peak {
                self.risk_peak = peak;
            }
        }
        // `observe_equity` is idle too: no drawdown is watched.
        self.last_bar = Some((end - 1, last));
        self.steps += end - first;
    }

    /// Advance the venue clock to `bar` without a market print.
    ///
    /// A driver whose data has a gap still has to let the strategy act when
    /// one of its own clocks fires inside it -- a composite window closing,
    /// say -- and the venue matches what it submits against the standing
    /// book. Such a driver steps a degenerate bar held at the last close.
    /// The kernel treats it like any other bar, so orders match exactly as
    /// they would on a stepped bar; but no price traded, so it is not an
    /// instant of the equity curve. Whatever it fills belongs to the instant
    /// of the last real bar, still settling until the next one arrives --
    /// which is where the reference engine, whose timers fire between prints,
    /// books it too.
    pub fn step_clock(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        input: StepInput,
    ) -> Vec<EngineEvent> {
        self.advance(idx, bar, input, false)
    }

    fn advance(
        &mut self,
        idx: usize,
        bar: &KernelBar,
        input: StepInput,
        priced: bool,
    ) -> Vec<EngineEvent> {
        // A later priced bar means the previous one's instant has finished
        // settling, fills the strategy provoked off-schedule (`walk_book`) or
        // on a clock bar included.
        if priced && self.curve.closes(bar.timestamp) {
            if let Some((_, last)) = self.last_bar {
                self.curve.close(self.kernel.equity(last.close));
            }
        }
        let events = self.kernel.step(idx, bar, input);

        for event in &events {
            match event {
                EngineEvent::Exited { trade, .. } => {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push((**trade).clone());
                }
                // Fills and rejections carry what the book cannot: the price
                // and size of each slice, and the reason an order was
                // refused. Everything else about the order is read from the
                // book at `finish`.
                EngineEvent::OrderFilled { idx, order_id, price, size, .. } => {
                    self.order_fill(*order_id, *idx, *price, *size);
                }
                EngineEvent::OrderRejected { order_id, reason, .. } => {
                    self.order_reject(*order_id, reason);
                }
                _ => {}
            }
        }

        let equity = self.kernel.equity(bar.close);
        self.marked_equity = equity;
        if equity > self.risk_peak {
            self.risk_peak = equity;
        }

        // Feed the kill-switch after this bar is marked to market, so the
        // halt takes effect from the next bar's entry check onward.
        self.kernel.observe_equity(equity, self.risk_peak);

        if priced {
            self.curve.open_at(bar.timestamp);
        }
        self.last_bar = Some((idx, *bar));
        self.steps += 1;

        events
    }

    /// Settle resting orders off-schedule, at `ts_now`.
    ///
    /// A venue walks its books every time it drains a batch of commands, not
    /// only when a bar arrives. A driver that steps once per bar therefore
    /// under-fills an order the strategy placed on hearing that bar's own
    /// fills: it would sit until the next bar and fill against a range the
    /// strategy never saw. This is the single-instrument counterpart of
    /// [`PortfolioSession::walk_book`], and the same phase of the step: only
    /// resting orders are matched, against the last bar this runner stepped,
    /// dated to the instant of the walk.
    ///
    /// No equity point is sampled here: these fills belong to the instant of
    /// the last bar, which is sampled once it has settled -- when the next
    /// bar arrives, or at `finish`.
    pub fn walk_book(&mut self, ts_now: i64) -> Vec<EngineEvent> {
        let Some((idx, last)) = self.last_bar else { return Vec::new() };
        let bar = KernelBar { timestamp: ts_now, ..last };
        let events = self.kernel.walk_book(idx, &bar);

        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
        }

        events
    }

    /// The order with engine id `order_id`, which is its place in the book.
    fn order(&self, order_id: u64) -> Option<&crate::execution::orders::Order> {
        let order = self.kernel.order_book().get(order_id as usize)?;
        debug_assert_eq!(order.id, order_id);
        Some(order)
    }

    /// Fold one fill slice onto the order's record.
    ///
    /// The record is seeded from the book here rather than at submission, so
    /// an order is only materialised once something happened to it; the
    /// remaining orders are picked up wholesale at `finish`.
    fn order_fill(&mut self, order_id: u64, idx: usize, price: f64, size: f64) {
        // Zero the seed ONCE, as it is created: `record_fill` folds each
        // slice in, and the book's own `filled_qty` already holds the
        // total. Resetting on every slice instead would keep only the
        // last fill, which is exactly the partial-fill case this record
        // exists to describe.
        let seed = |record: &mut OrderRecord| {
            record.filled_qty = 0.0;
            record.avg_fill_price = None;
        };
        if let Some(record) = self.order_event(order_id, seed) {
            record.record_fill(idx, price, size);
        }
    }

    /// Record why an order was refused. The reason lives only in the event.
    fn order_reject(&mut self, order_id: u64, reason: &str) {
        if let Some(record) = self.order_event(order_id, |_| {}) {
            record.reject_reason = Some(reason.to_string());
        }
    }

    /// The order's record, seeded from the book by `seed` the first time
    /// something happens to it. Order ids are book positions, so the records
    /// sit at the same positions.
    fn order_event(
        &mut self,
        order_id: u64,
        seed: impl FnOnce(&mut OrderRecord),
    ) -> Option<&mut OrderRecord> {
        let slot = usize::try_from(order_id).ok()?;
        if !self.order_events.get(slot).is_some_and(Option::is_some) {
            let mut record = OrderRecord::from_order(self.order(order_id)?, self.kernel.symbol());
            seed(&mut record);
            if self.order_events.len() <= slot {
                self.order_events.resize_with(slot + 1, || None);
            }
            self.order_events[slot] = Some(record);
        }
        self.order_events[slot].as_mut()
    }

    /// Every order the run placed, in submission order.
    ///
    /// Reconciled against the book, not assembled from events alone: an order
    /// that rested and expired emits no fill and no rejection, and omitting
    /// it would report "nothing happened" for exactly the case a user is
    /// asking about. Fill and reject detail is layered on where it exists.
    fn order_log(&mut self) -> Vec<OrderRecord> {
        let mut events = std::mem::take(&mut self.order_events);
        let symbol = self.kernel.symbol();
        self.kernel
            .order_book()
            .iter()
            .map(|order| match events.get_mut(order.id as usize).and_then(Option::take) {
                Some(mut record) => {
                    // The book holds the final status; the events hold
                    // what happened along the way.
                    record.status = order.status.as_str();
                    record
                }
                None => OrderRecord::from_order(order, symbol),
            })
            .collect()
    }

    /// Force-close any open position and compute final metrics.
    pub fn finish(mut self) -> BacktestResult {
        // The last instant closes at the account's mark before the
        // end-of-data close, which books the lot's PnL in whole currency
        // units where the mark keeps it raw; see `EventSession::finish`.
        let final_equity = self
            .last_bar
            .map_or(self.config.initial_capital, |(_, bar)| self.kernel.equity(bar.close));
        if self.kernel.is_in_position() {
            if let Some((idx, bar)) = self.last_bar {
                for trade in self.kernel.finalize_all(idx, &bar) {
                    self.streaming.update(trade.return_pct / 100.0);
                    self.trades.push(trade);
                }
            }
        }

        // Built before the result takes ownership of the parts.
        let order_log = self.order_log();
        self.curve.close(final_equity);
        let mut curve = std::mem::replace(&mut self.curve, InstantCurve::new(0.0));
        let daily_performance = curve.take_daily_performance();
        let end_value = curve.last().unwrap_or(self.config.initial_capital);
        let mut curve = curve.into_parts();

        let metrics = if self.config.compute_metrics {
            compute_backtest_metrics_with_config(
                &curve.equity,
                &curve.drawdown,
                &curve.returns,
                &self.trades,
                &curve.timestamps,
                &self.config,
            )
        } else {
            BacktestMetrics::unmeasured(self.config.initial_capital, end_value)
        };

        if !self.config.retain_curves {
            curve = Default::default();
        }

        BacktestResult::new(metrics, curve.equity, curve.drawdown, self.trades, curve.returns)
            .with_timestamps(curve.timestamps)
            .with_orders(order_log)
            .with_daily_performance(daily_performance)
    }

    /// Mark-to-market equity after the most recent step, or initial capital
    /// before the first step.
    #[inline]
    pub fn equity(&self) -> f64 {
        self.marked_equity
    }

    /// Current uninvested cash.
    #[inline]
    pub fn cash(&self) -> f64 {
        self.kernel.cash()
    }

    /// Whether a position is currently open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.kernel.is_in_position()
    }

    /// Number of bars stepped so far.
    #[inline]
    pub fn bars_seen(&self) -> usize {
        self.steps
    }

    /// Mutable access to the underlying kernel, for callers that adjust
    /// position state between steps (e.g. programmatic stop updates).
    #[inline]
    pub fn kernel_mut(&mut self) -> &mut EngineKernel {
        &mut self.kernel
    }

    /// Shared access to the underlying kernel.
    #[inline]
    pub fn kernel(&self) -> &EngineKernel {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce};

    fn bar(timestamp: i64, close: f64) -> KernelBar {
        KernelBar {
            timestamp,
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1e9,
        }
    }

    #[test]
    fn a_fill_from_a_book_walk_is_in_its_own_instants_sample() {
        // The strategy answers bar 0's fills with an order the venue matches
        // off-schedule, at the same instant. Sampled straight after the step,
        // the curve would miss that fill; the instant is sampled once it has
        // settled, so it is in.
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let mut runner = SingleRunner::from_config(config, "AAA".into(), Direction::Long, None);
        runner.step(0, &bar(0, 100.0), StepInput::default());
        let marked = runner.equity();
        runner.kernel_mut().submit_order(
            OrderSide::Buy,
            QtySpec::Units(100.0),
            OrderKind::Limit { price: 101.0 },
            TimeInForce::Gtc,
            0,
            0,
            "on-fill".to_string(),
            None,
            None,
        );
        runner.walk_book(0);
        assert!(runner.is_in_position(), "the walk filled the order");
        let settled = runner.kernel().equity(100.0);
        assert_eq!(runner.equity(), marked, "sizing still sees the bar's own mark");

        runner.step(1, &bar(10, 110.0), StepInput::default());
        let result = runner.finish();

        assert_eq!(result.timestamps, vec![0, 10]);
        assert_eq!(result.equity_curve[0], settled);
        assert!(result.equity_curve[1] > settled, "the position gained into the close");
    }

    #[test]
    fn a_clock_bar_matches_but_is_not_an_instant() {
        // A composite window closes inside a data gap: the driver steps a bar
        // held at the last close so the strategy's order meets the standing
        // book. The order fills there, but nothing traded at that time, so
        // the fill is booked to the last real bar's instant and the curve has
        // no sample of its own for the clock.
        let config = BacktestConfig { fees: 0.001, ..BacktestConfig::default() };
        let mut runner = SingleRunner::from_config(config, "AAA".into(), Direction::Long, None);
        runner.step(0, &bar(0, 100.0), StepInput::default());
        let before = runner.kernel().equity(100.0);
        // Submitted on the timer, before the clock bar reaches the venue.
        runner.kernel_mut().submit_order_full(
            OrderSide::Buy,
            QtySpec::Units(100.0),
            OrderKind::Limit { price: 101.0 },
            TimeInForce::Gtc,
            1,
            5,
            "on-clock".to_string(),
            None,
            None,
            false,
            false,
            true,
            None,
        );
        let clock = KernelBar {
            timestamp: 5,
            open: 100.0,
            high: 100.0,
            low: 100.0,
            close: 100.0,
            volume: 0.0,
        };
        runner.step_clock(1, &clock, StepInput::default());
        assert!(runner.is_in_position(), "the clock bar matched the order");
        let settled = runner.kernel().equity(100.0);
        assert!(settled < before, "the fill paid its fee");

        runner.step(2, &bar(10, 110.0), StepInput::default());
        assert_eq!(runner.bars_seen(), 3, "the kernel stepped every bar");
        let result = runner.finish();

        assert_eq!(result.timestamps, vec![0, 10]);
        assert_eq!(result.equity_curve[0], settled);
        assert!(result.equity_curve[1] > settled, "the position gained into the close");
    }

    #[test]
    fn a_lot_open_at_the_end_closes_the_curve_at_its_mark() {
        // The end-of-data close books the lot's PnL as money, in whole
        // cents, where the mark keeps it raw: 250 units bought at 100.00
        // and marked at 101.00371 are 250.9275 up, which the close books as
        // 250.93. The run ends at the mark, as Nautilus's sampler values
        // the book it finishes with.
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let cents = InstrumentConfig { currency_precision: Some(2), ..InstrumentConfig::default() };
        let mut runner =
            SingleRunner::from_config(config, "AAA".into(), Direction::Long, Some(&cents))
                .with_account_mode(crate::accounts::AccountMode::Margin { leverage: 1.0 });
        let entry = StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() };
        runner.step(0, &bar(0, 100.0), entry);
        runner.step(1, &bar(10, 101.00371), StepInput::default());
        let marked = runner.kernel().equity(101.00371);
        assert_ne!(marked, (marked * 100.0).round() / 100.0, "the mark is not whole cents");

        let result = runner.finish();
        assert_eq!(result.trades.len(), 1, "the lot was closed at the end of data");
        assert_eq!(result.equity_curve.last(), Some(&marked));
        assert_eq!(result.metrics.end_value, marked);
    }

    /// Bought on the first of five bars twelve hours apart and held, so the
    /// run crosses two day boundaries and ends in the end-of-data close.
    fn held_across_days(config: BacktestConfig) -> BacktestResult {
        const HALF_DAY: i64 = 43_200_000_000_000;
        let mut runner = SingleRunner::from_config(config, "AAA".into(), Direction::Long, None);
        for (i, close) in [100.0, 104.0, 98.0, 101.0, 107.0].into_iter().enumerate() {
            let input = StepInput { entry: i == 0, ..StepInput::default() };
            runner.step(i, &bar(i as i64 * HALF_DAY, close), input);
        }
        runner.finish()
    }

    #[test]
    fn a_run_that_computes_no_metrics_settles_the_same() {
        let measured = BacktestConfig {
            fees: 0.001,
            retain_daily_performance: true,
            ..BacktestConfig::default()
        };
        let full = held_across_days(measured.clone());
        assert_eq!(full.metrics.total_trades, 1);
        assert_eq!(full.daily_performance.as_ref().map(|d| d.timestamps.len()), Some(3));
        for retain_curves in [false, true] {
            let lean = held_across_days(BacktestConfig {
                compute_metrics: false,
                retain_curves,
                ..measured.clone()
            });
            assert_eq!(lean.metrics.start_value, full.metrics.start_value);
            assert_eq!(lean.metrics.end_value, full.metrics.end_value);
            assert_eq!(lean.metrics.total_return_pct, full.metrics.total_return_pct);
            assert_eq!(lean.metrics.total_trades, 0, "nothing else is measured");
            assert_eq!(format!("{:?}", lean.trades), format!("{:?}", full.trades));
            assert_eq!(
                format!("{:?}", lean.daily_performance),
                format!("{:?}", full.daily_performance)
            );
            if retain_curves {
                assert_eq!(lean.equity_curve, full.equity_curve);
                assert_eq!(lean.drawdown_curve, full.drawdown_curve);
                assert_eq!(lean.returns, full.returns);
                assert_eq!(lean.timestamps, full.timestamps);
            } else {
                assert!(lean.equity_curve.is_empty() && lean.timestamps.is_empty());
            }
        }
    }
}
