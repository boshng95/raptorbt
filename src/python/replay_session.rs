//! A crossover replay's whole bar loop, run without Python.
//!
//! The parity runner's account-coupled native lane executes a decision cursor
//! (`CrossoverRuleSession`) through a `SignalReplay` strategy, which
//! `run_strategy_backtest` drives bar by bar. The kernel already takes the
//! strategy's quiet bars itself (`KernelSession.step_rules_until`); every
//! other bar -- a composite that decides, a target being chased, a fill, a
//! flat account near its cash halt -- still crosses into Python and back.
//! This is that driver loop and that strategy, for the shape the rule session
//! covers: a long-only replay whose targets are flat or long, with no
//! policies, no borrow and no window callbacks.
//!
//! Like the rule session it is a transcription, not a model. Each branch
//! mirrors a line of `raptorbt/strategy/runner.py` or of the caller's
//! `SignalReplay`, in the same order. Money stays the caller's: sizing, price
//! snapping, affordability and the cash halt are its own Python functions,
//! called where the strategy calls them -- unless the caller hands over a
//! `MoneyModel`, which answers them in the kernel wherever it can prove its
//! answer is the function's. What the replay returns is the
//! ordered record the strategy would have booked -- every order as it was
//! sent, every fill and every unfilled end -- for the caller to book in turn,
//! or, given the caller's book, that record booked (`fill_ledger`).
//!
//! Where the strategy would take a branch this does not transcribe, the
//! replay stops with [`ReplayDeclined`] and returns nothing; the caller then
//! runs the strategy in Python from the start.

use crate::portfolio::kernel::EngineEvent;
use crate::portfolio::runner::SingleRunner;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::fill_ledger::{EndStatus, Ledger, OrderRecord};
use super::money::PyMoneyModel;
use super::rule_session::{money_as_double, QuietParams, RuleStepper, Target};
use super::strategy_bindings::{
    held_position, step_rules_until, step_until_event, submit_order_on, BarArrays,
};

create_exception!(
    raptorbt,
    ReplayDeclined,
    PyRuntimeError,
    "The replay reached a branch it does not transcribe; run the strategy in Python."
);

/// The bars [`Replay::reads_held_from`] first looks over for a dust close,
/// and the run it narrows one down to before reading each close.
const DUST_SPAN: usize = 4096;
const DUST_SPAN_LAST: usize = 64;

pub(crate) fn declined(why: &str) -> PyErr {
    ReplayDeclined::new_err(why.to_string())
}

fn side_of(quantity: f64) -> i32 {
    if quantity > 0.0 {
        1
    } else if quantity < 0.0 {
        -1
    } else {
        0
    }
}

/// The venue and instrument terms `SignalReplay` closes over.
pub(crate) struct Venue {
    /// `_median_step_ns`: a resting entry lives twelve of these.
    pub bar_ns: i64,
    /// `venue_supports_ioc`: a limit close is IOC there, DAY elsewhere.
    pub venue_ioc: bool,
    /// Whether `reference_order_type` sends an entry, and a close, at market.
    pub entry_market: bool,
    pub exit_market: bool,
    /// `1.0 + MARKET_ORDER_PRICE_ALLOWANCE`.
    pub market_markup: f64,
    pub size_increment: f64,
    pub minimum_quantity: f64,
    pub minimum_notional: f64,
    /// The run's lowest close, NaN only when every close is.
    pub lowest_close: f64,
    /// Client ids are `"{order_tag}-{seq}"`, counting from `first_order_seq`.
    pub order_tag: String,
    pub first_order_seq: u64,
    /// `ExecutionLimits.max_actions_per_timestamp`.
    pub max_actions_per_timestamp: usize,
}

/// The caller's money functions, each answering for the strategy.
pub(crate) struct Money<'a> {
    /// `(cash, close, reference) -> float`: `quantity`, with `strategy_cash`
    /// and the bar's close standing in for its context.
    pub units: &'a PyAny,
    /// `(reference, buy) -> float`: `snapped`.
    pub snap: &'a PyAny,
    /// `(cash, units, price) -> bool`: `entry_cost.can_afford`, given
    /// `strategy_cash`.
    pub affordable: &'a PyAny,
    /// `(cash, price) -> bool`: `cash_is_exhausted` with its cash given.
    pub exhausted: &'a PyAny,
    /// The same four in the kernel, where they answer; the functions above
    /// answer the rest.
    pub model: Option<&'a PyMoneyModel>,
}

/// One order `SignalReplay` queued, as `apply_commands` sends it.
struct Submit {
    client_id: String,
    /// Where the ledger keeps the order's terms.
    ledger_index: u32,
    buy: bool,
    market: bool,
    units: f64,
    limit_price: Option<f64>,
    tif: &'static str,
    expire_ns: Option<i64>,
    reduce_only: bool,
}

struct Replay<'a> {
    runner: &'a mut SingleRunner,
    bars: BarArrays<'a>,
    mask: &'a [bool],
    gate: Option<&'a [bool]>,
    rules: RuleStepper<'a>,
    quiet: QuietParams,
    venue: Venue,
    money: Money<'a>,

    // `SignalReplay`'s own state, under its own names.
    working_entry: bool,
    /// `_orders`: the kernel ids of the orders sent, rising, each with where
    /// the ledger keeps its terms.
    orders: Vec<(u64, u32)>,
    pending_target: Option<f64>,
    target_bars: i64,
    target_bar_ts: i64,
    pending_reference: f64,
    last_exit_ts: i64,
    cash_halted: bool,
    working_orders: i64,
    bar_callback_required: bool,
    /// `runtime.account.position_quantity`: the position as the runtime's
    /// last refresh read it, which only a processed callback and a fill do.
    account_quantity: f64,
    /// The last [`Self::reads_held_from`] asked: `(held, from, until)`.
    held_reads: (f64, usize, usize),

    // The driver's: `Strategy._order_seq`, the queued commands and the
    // per-timestamp action budget.
    order_seq: u64,
    commands: Vec<Submit>,
    budget_ts: Option<i64>,
    budget_actions: usize,

    ledger: Ledger,
}

/// Run every bar of a fresh session into *ledger*; see the module
/// documentation.
///
/// Its log holds, in the order the strategy produced them:
///
/// - `("submit", client_id, side, order_type, time_in_force, quantity,
///   price, expire_time_ns, reduce_only, decided_ns, arrived_on_ns,
///   policies_after_fill)`, where the strategy built its `_OrderTerms`;
/// - `("fill", client_id, now_ns, order_id, size, price, leaves,
///   gross_realized, commission)`, where it booked an `order_filled`;
/// - `("end", client_id, status, now_ns)`, where it booked an order the
///   venue canceled or expired.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay<'a>(
    runner: &'a mut SingleRunner,
    bars: BarArrays<'a>,
    mask: &'a [bool],
    gate: Option<&'a [bool]>,
    rules: RuleStepper<'a>,
    quiet: QuietParams,
    venue: Venue,
    money: Money<'a>,
    ledger: Ledger,
) -> PyResult<Ledger> {
    let order_seq = venue.first_order_seq;
    let mut replay = Replay {
        runner,
        bars,
        mask,
        gate,
        rules,
        quiet,
        venue,
        money,
        working_entry: false,
        orders: Vec::new(),
        pending_target: None,
        target_bars: 0,
        target_bar_ts: -1,
        pending_reference: 0.0,
        last_exit_ts: -1,
        cash_halted: false,
        working_orders: 0,
        // The first callback checks cash and establishes whether quiet bars
        // can be deferred.
        bar_callback_required: true,
        account_quantity: 0.0,
        held_reads: (0.0, 0, 0),
        order_seq,
        commands: Vec::new(),
        budget_ts: None,
        budget_actions: 0,
        ledger,
    };
    replay.run()?;
    Ok(replay.ledger)
}

impl Replay<'_> {
    // -- the driver: `run_strategy_backtest` -------------------------------

    fn run(&mut self) -> PyResult<()> {
        let n = self.bars.len();
        // The bar a fast-forward handed back, which the loop runs itself.
        let mut python_bar: Option<usize> = None;
        // Bars before `idle_until` were stepped by the kernel already.
        let mut idle_until = 0usize;
        let mut next = 0usize;
        loop {
            let mut i = next;
            if i < idle_until {
                i = idle_until;
            }
            if i >= n {
                break;
            }
            next = i + 1;
            if python_bar != Some(i) && !self.bar_callback_required {
                if let Some((halted, pending, until)) = self.fast_forward_rules(i) {
                    let (kind, last, events) = step_rules_until(
                        self.runner,
                        &self.bars,
                        self.mask,
                        &mut self.rules,
                        halted,
                        pending,
                        &self.quiet,
                        i,
                        until,
                    );
                    if kind == 2 {
                        python_bar = Some(last);
                        if last > i {
                            idle_until = last;
                            continue;
                        }
                    } else {
                        self.dispatch(last, events)?;
                        self.settle(last)?;
                        idle_until = last + 1;
                        continue;
                    }
                }
            }
            if !self.mask[i] && !self.bar_callback_required {
                let limit = self.mask[i..].iter().position(|&due| due).map_or(n, |at| i + at);
                let (last, events) = step_until_event(self.runner, &self.bars, i, limit);
                self.dispatch(last, events)?;
                self.settle(last)?;
                idle_until = last + 1;
                continue;
            }
            if self.mask[i] || self.bar_callback_required {
                self.on_bar(i)?;
            }
            self.apply_commands(i)?;
            let (bar, input) = self.bars.bar(i);
            let events = self.runner.step(i, &bar, input);
            self.dispatch(i, events)?;
            self.settle(i)?;
        }
        debug_assert!(!self.rules.holds_taken(), "a callback bar handed back was never taken");
        Ok(())
    }

    /// `apply_commands`: send what the strategy queued, and say whether it
    /// queued anything.
    fn apply_commands(&mut self, i: usize) -> PyResult<bool> {
        let commands = std::mem::take(&mut self.commands);
        let now = self.bars.ts[i];
        if self.budget_ts != Some(now) {
            self.budget_ts = Some(now);
            self.budget_actions = 0;
        }
        self.budget_actions += commands.len();
        if self.budget_actions > self.venue.max_actions_per_timestamp {
            return Err(declined("execution did not quiesce within its action budget"));
        }
        for command in &commands {
            let order_id = submit_order_on(
                self.runner.kernel_mut(),
                if command.buy { "buy" } else { "sell" },
                if command.market { "market" } else { "limit" },
                i,
                now,
                &command.client_id,
                Some(command.units),
                None,
                command.limit_price,
                None,
                command.tif,
                command.expire_ns,
                None,
                None,
                None,
                "price",
                0.0,
                false,
                command.reduce_only,
                false,
                None,
            )?;
            let at = match self.orders.last() {
                Some(&(last, _)) if last >= order_id => {
                    self.orders.partition_point(|&(id, _)| id < order_id)
                }
                _ => self.orders.len(),
            };
            self.orders.insert(at, (order_id, command.ledger_index));
        }
        Ok(!commands.is_empty())
    }

    /// `settle`: walk the book for orders the bar's own fills provoked.
    fn settle(&mut self, i: usize) -> PyResult<()> {
        let now = self.bars.ts[i];
        while self.apply_commands(i)? {
            let events = self.runner.walk_book(now);
            self.dispatch(i, events)?;
        }
        Ok(())
    }

    /// `dispatch_events`, into the hooks `SignalReplay` defines.
    fn dispatch(&mut self, at: usize, events: Vec<EngineEvent>) -> PyResult<()> {
        let mut order_fill_preceded = false;
        for event in events {
            match event {
                EngineEvent::Entered { .. } | EngineEvent::Exited { .. } => {
                    if !order_fill_preceded {
                        // `reject_unordered_close`.
                        return Err(declined("a position event with no order behind it"));
                    }
                    order_fill_preceded = false;
                }
                EngineEvent::EntryRejected { .. } => self.on_order_rejected(),
                EngineEvent::AlgoStarted { .. }
                | EngineEvent::AlgoCompleted { .. }
                | EngineEvent::MarginCall { .. } => {}
                EngineEvent::OrderAccepted { .. } | EngineEvent::OrderTriggered { .. } => {
                    order_fill_preceded = false;
                }
                EngineEvent::OrderFilled {
                    order_id,
                    price,
                    size,
                    commission,
                    leaves,
                    gross_realized,
                    ..
                } => {
                    self.on_order_filled(
                        at,
                        order_id,
                        price,
                        size,
                        commission,
                        leaves,
                        gross_realized,
                    )?;
                    order_fill_preceded = true;
                }
                EngineEvent::OrderCanceled { order_id, .. } => {
                    self.end_order(at, order_id, EndStatus::Canceled);
                    order_fill_preceded = false;
                }
                EngineEvent::OrderExpired { order_id, .. } => {
                    self.end_order(at, order_id, EndStatus::Expired);
                    order_fill_preceded = false;
                }
                EngineEvent::OrderRejected { .. } => {
                    self.on_order_rejected();
                    order_fill_preceded = false;
                }
            }
        }
        Ok(())
    }

    // -- the strategy: `SignalReplay` ---------------------------------------

    /// `fast_forward_rules`: whether the kernel may take quiet callbacks
    /// from bar `from`, as `(halted, pending_target, until)`: it takes those
    /// before `until`.
    ///
    /// A quiet callback is asked with the book as it stands, where a callback
    /// the strategy takes reads a held lot too small to trade as flat
    /// (`reads_dust_as_flat`). `SignalReplay` declines the whole run once the
    /// lot is dust at the run's lowest close; here the callbacks are taken
    /// up to the first bar whose close makes it dust, where the strategy
    /// takes them again.
    fn fast_forward_rules(&mut self, from: usize) -> Option<(bool, Option<f64>, usize)> {
        let n = self.bars.len();
        if self.cash_halted {
            return Some((true, self.pending_target, n));
        }
        if self.pending_target.is_some() || self.working_orders != 0 {
            return None;
        }
        let held = self.account_quantity;
        let until = if held != 0.0 { self.reads_held_from(held, from) } else { n };
        (until > from).then_some((false, None, until))
    }

    /// The first bar from `from` on whose close the lot `held` is dust, or
    /// the bar count where there is none.
    ///
    /// `is_dust` is monotone in the price, so a lot that is not dust at the
    /// lowest close of a run of bars is dust at none of them. The run's own
    /// lowest close settles most lots; for the rest, the stream's summaries
    /// (`BarIndex`) find the bar in time logarithmic in how far off it is.
    /// They summarize the kernel's closes, which are the ones the strategy
    /// reads dust at only when they are the same array; otherwise every bar
    /// is taken as one that may read dust.
    fn reads_held_from(&mut self, held: f64, from: usize) -> usize {
        let n = self.bars.len();
        if !self.is_dust(held, self.venue.lowest_close) {
            return n;
        }
        // No bar of `from..until` reads `held` as dust, and `until` does (or
        // ends the run), so the same holds from any bar before `until`.
        let (asked, start, until) = self.held_reads;
        if asked.to_bits() == held.to_bits() && start <= from && from < until {
            return until;
        }
        let closes = self.rules.primary_closes();
        if self.is_dust(held, closes[from]) {
            return from;
        }
        let index = match self.bars.index.as_deref() {
            Some(index) if std::ptr::eq(closes, self.bars.close) => index,
            _ => return from,
        };
        let dust_in = |start: usize, end: usize| {
            let lowest = index.extremes(self.bars.close, self.bars.high, self.bars.low, start, end);
            self.is_dust(held, lowest.lowest_close)
        };
        // Runs of doubling length from `from`, up to the first that holds a
        // dust bar, which halving then narrows to.
        let mut found = n;
        let (mut start, mut span) = (from, DUST_SPAN);
        while start < n {
            let end = n.min(start.saturating_add(span));
            if !dust_in(start, end) {
                (start, span) = (end, span.saturating_mul(2));
                continue;
            }
            let (mut low, mut high) = (start, end);
            while high - low > DUST_SPAN_LAST {
                let middle = low + (high - low) / 2;
                if dust_in(low, middle) {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            found = (low..high).find(|&idx| self.is_dust(held, closes[idx])).unwrap_or(low);
            break;
        }
        self.held_reads = (held, from, found);
        found
    }

    fn on_bar(&mut self, i: usize) -> PyResult<()> {
        if self.cash_halted {
            self.advance(i)?;
            // A halted replay acts on nothing: `not _defer_quiet_primary`.
            self.bar_callback_required = false;
            return Ok(());
        }
        let reconciled =
            if self.pending_target.is_some() { self.reconcile_target(i)? } else { false };
        let target = self.advance(i)?;
        if !reconciled {
            self.submit_native_intent(i, target)?;
        }
        let (has_position, net_position) = held_position(self.runner);
        let flat = !has_position;
        let cash_flat = flat || self.holds_flat_dust(i, has_position, net_position);
        let mut flat_cash_safe = false;
        if !self.cash_halted && cash_flat {
            let price = self.bars.close[i];
            let cash = self.strategy_cash()?;
            let q = self.quiet;
            let mut funds = cash * q.fraction;
            if funds > q.cap {
                funds = q.cap;
            }
            let need = q.units * (price * q.markup + q.slack) + q.fee;
            if funds < need {
                self.cash_halted = self.exhausted(cash, price)?;
            }
            flat_cash_safe = funds >= q.need_at_highest;
        }
        self.bar_callback_required = (self.pending_target.is_some()
            && !self.flat_target_rests_before_dispatch(i, has_position, net_position))
            || self.working_orders != 0
            || (cash_flat
                && !self.cash_halted
                && !flat_cash_safe
                && !self.flat_cash_holds_before_dispatch(i)?)
            || self.rules.visit_primary_after_fill()
            || (!flat
                && self.rules.held_reopens(i, has_position, net_position, self.pending_target));
        Ok(())
    }

    /// `RustRuleSession.advance`.
    fn advance(&mut self, i: usize) -> PyResult<Option<Target>> {
        let (mut has_position, mut quantity) = held_position(self.runner);
        // Level rules read dust themselves, on the venue's own lot.
        if !self.rules.level()
            && has_position
            && self.reads_dust_as_flat(quantity, self.rules.primary_close(i))
        {
            has_position = false;
            quantity = 0.0;
        }
        let (processed, target) =
            self.rules.advance(self.bars.ts[i], i, has_position, quantity, self.pending_target)?;
        if processed {
            self.refresh()?;
        }
        Ok(target)
    }

    /// `NativeExecutionSession._refresh`, as far as anything reads it.
    fn refresh(&mut self) -> PyResult<()> {
        let (has_position, net_position) = held_position(self.runner);
        self.account_quantity = if has_position { net_position } else { 0.0 };
        self.strategy_cash().map(|_| ())
    }

    fn submit_native_intent(&mut self, i: usize, target: Option<Target>) -> PyResult<()> {
        let Some(target) = target else {
            return Ok(());
        };
        let target_side = target.side;
        let (has_position, net_position) = held_position(self.runner);
        let held = if has_position { net_position } else { 0.0 };
        let mut held_side = side_of(held);
        if held_side != 0 && self.flat_on_dust(has_position, net_position, self.bars.close[i]) {
            held_side = 0;
        }
        if target_side == held_side {
            return Ok(());
        }
        let reference = target.reference.unwrap_or(self.bars.close[i]);
        let closing = held_side != 0 && target_side != held_side;
        let units = if target_side != 0 { self.quantity(i, reference)? } else { 0.0 };
        if closing {
            if target_side != 0 {
                return Err(declined("a reversal"));
            }
            self.pending_reference = reference;
            self.target_bars = 0;
            self.target_bar_ts = -1;
            self.retarget(Some(0.0));
            // A close the venue refused persists, and is reconciled.
            self.submit_exit(i, reference)?;
            return Ok(());
        }
        if target_side == 0 || self.working_entry || self.cash_halted {
            return Ok(());
        }
        if units <= 0.0 {
            return Ok(());
        }
        if target_side < 0 {
            return Err(declined("a short entry"));
        }
        self.submit_entry(i, reference, units, target.policies_after_fill)?;
        Ok(())
    }

    fn submit_exit(&mut self, i: usize, reference: f64) -> PyResult<bool> {
        let decided_ns = self.bars.ts[i];
        if let Some(gate) = self.gate {
            if !gate[i] {
                return Ok(false);
            }
        }
        let (_, net_position) = held_position(self.runner);
        if net_position < 0.0 {
            // A buy close takes `reference_order_type`'s buy side.
            return Err(declined("a short position"));
        }
        let buy = net_position <= 0.0;
        let limit_price = self.snap(reference, net_position < 0.0)?;
        let units = net_position.abs();
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        let market = self.venue.exit_market;
        let tif = if market {
            "gtc"
        } else if self.venue.venue_ioc {
            "ioc"
        } else {
            "day"
        };
        self.submit(
            Submit {
                client_id: String::new(),
                ledger_index: 0,
                buy,
                market,
                units,
                limit_price: (!market).then_some(limit_price),
                tif,
                expire_ns: None,
                reduce_only: true,
            },
            decided_ns,
            decided_ns,
            false,
        )?;
        self.last_exit_ts = decided_ns;
        self.working_orders += 1;
        Ok(true)
    }

    fn submit_entry(
        &mut self,
        i: usize,
        reference: f64,
        units: f64,
        policies_after_fill: bool,
    ) -> PyResult<bool> {
        let decided_ns = self.bars.ts[i];
        if let Some(gate) = self.gate {
            if !gate[i] {
                return Ok(false);
            }
        }
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        let market = self.venue.entry_market;
        let limit_price = self.snap(reference, true)?;
        let order_price =
            if market { self.bars.close[i] * self.venue.market_markup } else { limit_price };
        if !self.affordable(units, order_price)? {
            return Ok(false);
        }
        let tif = if market { "gtc" } else { "gtd" };
        let expire_ns = (!market).then(|| decided_ns + 12 * self.venue.bar_ns);
        self.submit(
            Submit {
                client_id: String::new(),
                ledger_index: 0,
                buy: true,
                market,
                units,
                limit_price: (!market).then_some(limit_price),
                tif,
                expire_ns,
                reduce_only: false,
            },
            decided_ns,
            decided_ns,
            policies_after_fill,
        )?;
        self.working_entry = true;
        self.working_orders += 1;
        Ok(true)
    }

    /// `Strategy.submit_order` and the `_OrderTerms` recorded beside it.
    fn submit(
        &mut self,
        mut order: Submit,
        decided_ns: i64,
        arrived_on_ns: i64,
        policies_after_fill: bool,
    ) -> PyResult<()> {
        if let Some(price) = order.limit_price {
            if price <= 0.0 {
                // `Limit.__post_init__` raises, which the strategy should
                // raise itself.
                return Err(declined("a limit price that is not positive"));
            }
        }
        order.client_id = format!("{}-{}", self.venue.order_tag, self.order_seq);
        self.order_seq += 1;
        let time_in_force = match order.tif {
            "gtc" => "GTC",
            "gtd" => "GTD",
            "ioc" => "IOC",
            _ => "DAY",
        };
        let index = self.ledger.submit(OrderRecord {
            client_id: order.client_id.clone(),
            buy: order.buy,
            market: order.market,
            time_in_force,
            units: order.units,
            limit_price: order.limit_price,
            expire_ns: order.expire_ns,
            reduce_only: order.reduce_only,
            decided_ns,
            arrived_on_ns,
            policies_after_fill,
        });
        order.ledger_index = index;
        self.commands.push(order);
        Ok(())
    }

    fn retarget(&mut self, target: Option<f64>) {
        self.pending_target = target;
    }

    fn drop_target(&mut self) {
        self.retarget(None);
        self.target_bars = 0;
        self.target_bar_ts = -1;
    }

    fn continue_target(&mut self, i: usize, reference: f64) -> PyResult<bool> {
        if self.cash_halted {
            return Ok(false);
        }
        let Some(target) = self.pending_target else {
            return Ok(false);
        };
        let (has_position, net_position) = held_position(self.runner);
        let held = if has_position { net_position } else { 0.0 };
        if held == target {
            self.drop_target();
            return Ok(false);
        }
        if held != 0.0 && (target == 0.0 || (held > 0.0) != (target > 0.0)) {
            let submitted = self.submit_exit(i, reference)?;
            if submitted {
                self.pending_reference = reference;
            }
            return Ok(submitted);
        }
        // A long-only replay's only target is flat, which the two branches
        // above settle whatever the book holds.
        Err(declined("a target other than flat"))
    }

    fn reconcile_target(&mut self, i: usize) -> PyResult<bool> {
        if self.pending_target == Some(0.0) {
            let (has_position, net_position) = held_position(self.runner);
            if has_position && self.is_dust(net_position, self.bars.close[i]) {
                self.drop_target();
                return Ok(false);
            }
        }
        let now = self.bars.ts[i];
        if now != self.target_bar_ts {
            self.target_bar_ts = now;
            self.target_bars += 1;
        }
        if now == self.last_exit_ts {
            return Ok(false);
        }
        self.continue_target(i, self.bars.close[i])
    }

    #[allow(clippy::too_many_arguments)]
    fn on_order_filled(
        &mut self,
        at: usize,
        order_id: u64,
        price: f64,
        size: f64,
        commission: f64,
        leaves: f64,
        gross_realized: f64,
    ) -> PyResult<()> {
        self.working_entry = false;
        let Some(order) = self.ledger_index(order_id) else {
            return Err(declined("a fill of an order this replay did not send"));
        };
        let now = self.bars.ts[at];
        self.ledger.fill(order, now, order_id, size, price, leaves, gross_realized, commission);
        // `runtime.on_fill`: the account refreshed, the next primary bar a
        // visit. With no policies and no follow-up, nothing else of the fill
        // reaches the runtime.
        self.refresh()?;
        self.rules.on_fill();
        if leaves <= 0.0 {
            self.working_orders = (self.working_orders - 1).max(0);
            self.continue_target(at, self.pending_reference)?;
        }
        self.bar_callback_required = true;
        Ok(())
    }

    /// `on_order_canceled` / `on_order_expired`.
    fn end_order(&mut self, at: usize, order_id: u64, status: EndStatus) {
        if let Some(order) = self.ledger_index(order_id) {
            self.ledger.end(order, status, self.bars.ts[at]);
        }
        self.working_entry = false;
        self.working_orders = (self.working_orders - 1).max(0);
        self.bar_callback_required = true;
    }

    /// Where the ledger keeps the terms of the order the kernel calls
    /// `order_id`, if this replay sent it.
    fn ledger_index(&self, order_id: u64) -> Option<u32> {
        let at = self.orders.partition_point(|&(id, _)| id < order_id);
        self.orders.get(at).filter(|&&(id, _)| id == order_id).map(|&(_, index)| index)
    }

    fn on_order_rejected(&mut self) {
        self.working_entry = false;
        self.working_orders = (self.working_orders - 1).max(0);
        self.bar_callback_required = true;
    }

    // -- the strategy's account reads ---------------------------------------

    /// `_flat_on_dust`.
    fn flat_on_dust(&self, has_position: bool, net_position: f64, price: f64) -> bool {
        self.pending_target != Some(0.0) && has_position && self.is_dust(net_position, price)
    }

    /// `_holds_flat_dust`.
    fn holds_flat_dust(&self, i: usize, has_position: bool, net_position: f64) -> bool {
        let held = self.account_quantity;
        held != 0.0
            && self.is_dust(held, self.venue.lowest_close)
            && self.flat_on_dust(has_position, net_position, self.bars.close[i])
    }

    /// `NativeExecutionSession.reads_dust_as_flat`.
    fn reads_dust_as_flat(&self, quantity: f64, close: f64) -> bool {
        quantity != 0.0 && self.pending_target != Some(0.0) && self.is_dust(quantity, close)
    }

    /// `RustRuleSession.highest_close_before_dispatch`.
    fn highest_close_before_dispatch(&self, i: usize) -> Option<f64> {
        let start = i + 1;
        let stop = self.rules.next_dispatch_position();
        if stop <= start {
            return None;
        }
        // numpy's max: NaN as soon as any element is NaN.
        let mut highest = f64::NEG_INFINITY;
        for idx in start..stop {
            let value = self.rules.primary_close(idx);
            if value.is_nan() {
                return Some(f64::NAN);
            }
            if value > highest {
                highest = value;
            }
        }
        Some(highest)
    }

    /// `flat_target_rests_before_dispatch`, for the pending target.
    fn flat_target_rests_before_dispatch(
        &self,
        i: usize,
        has_position: bool,
        net_position: f64,
    ) -> bool {
        if self.pending_target != Some(0.0) || !has_position {
            return false;
        }
        let units = net_position.abs();
        if units == 0.0 {
            return false;
        }
        match self.highest_close_before_dispatch(i) {
            None => true,
            Some(highest) => !self.tradeable(units, highest),
        }
    }

    /// `flat_cash_holds_before_dispatch`.
    fn flat_cash_holds_before_dispatch(&self, i: usize) -> PyResult<bool> {
        let Some(highest) = self.highest_close_before_dispatch(i) else {
            return Ok(true);
        };
        if !(0.0 < highest && highest < f64::INFINITY) {
            return Ok(false);
        }
        let cash = self.strategy_cash()?;
        Ok(!self.exhausted(cash, highest)?)
    }

    /// `InstrumentContract.is_dust`.
    fn is_dust(&self, quantity: f64, price: f64) -> bool {
        let units = quantity.abs();
        units < self.venue.size_increment
            || (self.venue.minimum_notional > 0.0 && units * price < self.venue.minimum_notional)
    }

    /// `tradeable`: `meets_instrument_minimums`.
    fn tradeable(&self, units: f64, reference: f64) -> bool {
        !((self.venue.minimum_quantity > 0.0 && units < self.venue.minimum_quantity)
            || (self.venue.minimum_notional > 0.0
                && units * reference < self.venue.minimum_notional))
    }

    /// `strategy_cash`; Python raises on a balance that is not a number.
    fn strategy_cash(&self) -> PyResult<f64> {
        money_as_double(self.runner.cash(), self.quiet.currency_precision)
            .ok_or_else(|| declined("a cash balance that is not a number"))
    }

    // -- the caller's money -------------------------------------------------

    fn quantity(&self, i: usize, reference: f64) -> PyResult<f64> {
        let cash = self.strategy_cash()?;
        let close = self.bars.close[i];
        if let Some(units) = self.money.model.and_then(|m| m.sized(cash, close, reference)) {
            return Ok(units);
        }
        self.money.units.call1((cash, close, reference))?.extract()
    }

    fn snap(&self, reference: f64, buy: bool) -> PyResult<f64> {
        if let Some(price) = self.money.model.and_then(|m| m.snap(reference, buy)) {
            return Ok(price);
        }
        self.money.snap.call1((reference, buy))?.extract()
    }

    fn affordable(&self, units: f64, price: f64) -> PyResult<bool> {
        let cash = self.strategy_cash()?;
        if let Some(fits) = self.money.model.and_then(|m| m.affordable(cash, units, price)) {
            return Ok(fits);
        }
        self.money.affordable.call1((cash, units, price))?.extract()
    }

    fn exhausted(&self, cash: f64, price: f64) -> PyResult<bool> {
        if let Some(halts) = self.money.model.and_then(|m| m.exhausted(cash, price)) {
            return Ok(halts);
        }
        self.money.exhausted.call1((cash, price))?.extract()
    }
}
