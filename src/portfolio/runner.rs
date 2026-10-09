//! Steppable single-instrument runner.
//!
//! Wraps an [`EngineKernel`] together with the per-bar bookkeeping the batch
//! engine previously kept as loop locals: equity/drawdown/return curves, peak
//! tracking, streaming trade metrics, the kill-switch feed, and end-of-data
//! finalization. Both the batch engine and per-bar drivers (e.g. the strategy
//! session exposed to Python) loop this type, so the accounting has exactly one
//! implementation.

use std::collections::HashMap;

use crate::core::performance::DailyPerformanceCollector;
use crate::core::types::{BacktestConfig, BacktestResult, Direction, InstrumentConfig, Trade};
use crate::execution::orders::OrderRecord;
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::instruments::InstrumentSpec;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::curve::InstantCurve;
use crate::portfolio::engine::{compute_backtest_metrics_with_config, PortfolioEngine};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};

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
    /// What became of every order, keyed by id. A map because one order is
    /// mutated across several events (accepted, then triggered, then filled
    /// in slices); the book supplies submission order at `finish`.
    order_events: HashMap<u64, OrderRecord>,
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
    daily_performance_transitions: Option<Vec<(i64, i64)>>,
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

        let daily_performance_transitions =
            config.retain_daily_performance.then(|| config.performance_tz_transitions.clone());
        Self {
            kernel,
            config,
            curve: InstantCurve::new(initial_capital),
            trades: Vec::new(),
            order_events: HashMap::new(),
            streaming: StreamingMetrics::new(),
            marked_equity: initial_capital,
            risk_peak: initial_capital,
            steps: 0,
            last_bar: None,
            daily_performance_transitions,
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

    /// Fold one fill slice onto the order's record.
    ///
    /// The record is seeded from the book here rather than at submission, so
    /// an order is only materialised once something happened to it; the
    /// remaining orders are picked up wholesale at `finish`.
    fn order_fill(&mut self, order_id: u64, idx: usize, price: f64, size: f64) {
        let seed = self
            .kernel
            .order_book()
            .iter()
            .find(|o| o.id == order_id)
            .map(|o| OrderRecord::from_order(o, self.kernel.symbol()));
        if let Some(mut seed) = seed {
            // Zero the seed ONCE, as it is created: `record_fill` folds each
            // slice in, and the book's own `filled_qty` already holds the
            // total. Resetting on every slice instead would keep only the
            // last fill, which is exactly the partial-fill case this record
            // exists to describe.
            seed.filled_qty = 0.0;
            seed.avg_fill_price = None;
            let record = self.order_events.entry(order_id).or_insert(seed);
            record.record_fill(idx, price, size);
        }
    }

    /// Record why an order was refused. The reason lives only in the event.
    fn order_reject(&mut self, order_id: u64, reason: &str) {
        let seed = self
            .kernel
            .order_book()
            .iter()
            .find(|o| o.id == order_id)
            .map(|o| OrderRecord::from_order(o, self.kernel.symbol()));
        if let Some(seed) = seed {
            let record = self.order_events.entry(order_id).or_insert(seed);
            record.reject_reason = Some(reason.to_string());
        }
    }

    /// Every order the run placed, in submission order.
    ///
    /// Reconciled against the book, not assembled from events alone: an order
    /// that rested and expired emits no fill and no rejection, and omitting
    /// it would report "nothing happened" for exactly the case a user is
    /// asking about. Fill and reject detail is layered on where it exists.
    fn order_log(&self) -> Vec<OrderRecord> {
        self.kernel
            .order_book()
            .iter()
            .map(|order| match self.order_events.get(&order.id) {
                Some(seen) => {
                    let mut record = seen.clone();
                    // The book holds the final status; the event map holds
                    // what happened along the way.
                    record.status = order.status.as_str();
                    record
                }
                None => OrderRecord::from_order(order, self.kernel.symbol()),
            })
            .collect()
    }

    /// Force-close any open position and compute final metrics.
    pub fn finish(mut self) -> BacktestResult {
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
        // The last instant closes at the settled account: flat, so the
        // end-of-data close above is already in the balance.
        let final_equity = self
            .last_bar
            .map_or(self.config.initial_capital, |(_, bar)| self.kernel.equity(bar.close));
        self.curve.close(final_equity);
        let mut curve = std::mem::replace(&mut self.curve, InstantCurve::new(0.0)).into_parts();

        let metrics = compute_backtest_metrics_with_config(
            &curve.equity,
            &curve.drawdown,
            &curve.returns,
            &self.trades,
            &curve.timestamps,
            &self.config,
        );

        let daily_performance = self.daily_performance_transitions.take().map(|transitions| {
            let mut collector = DailyPerformanceCollector::new(transitions);
            for (&timestamp, &equity) in curve.timestamps.iter().zip(&curve.equity) {
                collector.observe(timestamp, equity);
            }
            collector.finish()
        });

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
}
