//! A crossover replay's whole bar loop, run without Python.
//!
//! The parity runner's account-coupled native lane executes a decision cursor
//! (`CrossoverRuleSession`) through a `SignalReplay` strategy, which
//! `run_strategy_backtest` drives bar by bar. The kernel already takes the
//! strategy's quiet bars itself (`KernelSession.step_rules_until`); every
//! other bar -- a composite that decides, a target being chased, a fill, a
//! flat account near its cash halt -- still crosses into Python and back.
//! This is that driver loop and that strategy, for the shape the rule session
//! covers: a replay whose targets are flat or long -- or, for a short node,
//! flat or short ([`Shorts`]) -- with no window callbacks. Its risk policies
//! are the rule session's: their closes come back as its targets, sent
//! outside the bar's decision (`risk_exit`) in their own order style.
//! A short's borrow is the caller's too: its entry gate and its loan clock
//! are Python functions, called where the strategy calls them, and what the
//! loan costs is charged afterwards from the fills, as for any run.
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

use crate::execution::orders::{OrderKind, QtySpec, TimeInForce};
use crate::portfolio::kernel::EngineEvent;
use crate::portfolio::runner::SingleRunner;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::fill_ledger::{money, or_zero, EndStatus, Ledger, OrderRecord};
use super::money::PyMoneyModel;
use super::rule_session::{money_as_double, OrderStyle, QuietParams, RuleStepper, Target};
use super::strategy_bindings::{
    held_position, policy_account, step_rules_until, step_until_event, submit_order_on, BarArrays,
};
use crate::stops::policies::Account;

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

/// A short node's terms, beyond [`Venue`]'s: `SignalReplay` under
/// `short_entries`, whose entries sell to open through a durable short
/// target and whose exits buy to cover.
///
/// A short entry is always a limit: the caller hands over no short node
/// whose entry `reference_order_type` sends at market, which `SignalReplay`
/// refuses itself.
pub(crate) struct Shorts<'a> {
    /// Whether `reference_order_type` sends a buy close -- a cover -- at
    /// market.
    pub cover_market: bool,
    /// `_MAX_SIGNED_SHORT_ATTEMPTS`: the short orders one target may send.
    pub max_attempts: u32,
    /// `_MAX_SIGNED_TARGET_STALE_BARS`: the bars a target may go unfilled
    /// before its expansion is given up.
    pub max_stale_bars: i64,
    /// `(decided_ns, units, position, opened_ns) -> bool`: whether
    /// `short_gate.refusal` refuses a short of `units` added to `position`,
    /// `opened_ns` being when the short held was opened, or None. None where
    /// the run has no gate.
    pub refusal: Option<&'a PyAny>,
    /// `(filled_ns) -> (int, int)`: `loan_clock.cover`, the bar a short
    /// opened at `filled_ns` is first due its forced cover on, and the bar a
    /// composite contests it on, each -1 for none. None where the
    /// instrument's loans carry no term.
    pub cover: Option<&'a PyAny>,
    /// `loan_admission`: the bars `handle_bar` admits, None for every bar.
    pub admission: Option<&'a [bool]>,
}

/// `StrategyBase._check_stale_day_orders`' terms, which `SignalReplay`
/// follows for one shape of exit left resting: a long close in full, met
/// again on the very next primary bar. Drifted past `stale_close_pct` of
/// that bar's close, with the bar's range short of it, it is canceled there
/// and, once the venue reports the cancel, re-sent whole at the bar's close
/// (`_retry_close_order`), at most `max_close_retries` times before a fill
/// leaves the book flat. Within the drift it is left working, bar after bar,
/// until a bar's range fills it (from `REPLAY_VERSION` 7, the lean replay's
/// 13). Without them every exit left resting declines.
#[derive(Clone, Copy)]
pub(crate) struct Resting {
    /// `_STALE_CLOSE_PCT`.
    pub stale_close_pct: f64,
    /// `_MAX_IOC_CLOSE_RETRIES`.
    pub max_close_retries: u32,
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
    shorts: Option<Shorts<'a>>,
    resting: Option<Resting>,

    // `SignalReplay`'s own state, under its own names.
    working_entry: bool,
    /// `_orders`: the kernel ids of the orders sent, rising, each with where
    /// the ledger keeps its terms and whether it buys.
    orders: Vec<(u64, u32, bool)>,
    pending_target: Option<f64>,
    /// `_pending_order_style`: the shape every order chasing the pending
    /// target takes.
    pending_order_style: OrderStyle,
    target_bars: i64,
    target_bar_ts: i64,
    pending_reference: f64,
    last_exit_ts: i64,
    /// `_last_submit_ts`: the bar any order of the replay's last reached the
    /// venue on. What a bar's step sent is still unresolved when Nautilus
    /// ends the step, however it fared here.
    last_submit_ts: i64,
    /// `_target_step_due`: set by a bar's reconcile, spent at the end of its
    /// step ([`Self::finish_target_step`]).
    target_step_due: bool,
    cash_halted: bool,
    working_orders: i64,
    bar_callback_required: bool,
    /// `runtime.account.position_quantity`: the position as the runtime's
    /// last refresh read it, which only a processed callback and a fill do.
    account_quantity: f64,
    /// The last [`Self::reads_held_from`] asked: `(held, from, until)`.
    held_reads: (f64, usize, usize),
    /// `_short_attempts`: short orders the pending target has sent.
    short_attempts: u32,
    /// `_exit_gated`: whether the last close met a venue shut to orders
    /// after it had passed instrument preflight.
    exit_gated: bool,
    /// `_short_opened_ns`, `_loan_cover_bar` and `_loan_contest_bar`: when
    /// the short held was opened, and the bars its loan's forced cover is
    /// due on and a composite contests it on, -1 for none. Kept only where a
    /// loan has a term.
    short_opened_ns: Option<i64>,
    loan_cover_bar: i64,
    loan_contest_bar: i64,
    /// `_close_retries`: the stale closes re-sent since the book was last
    /// left flat by a fill.
    close_retries: u32,
    /// `_stale_cancels`: the kernel id of the exit canceled for drift, whose
    /// cancel re-sends it.
    stale_cancel: Option<u64>,
    /// `_stale_ts`: the bar that exit was canceled on, -1 for none.
    stale_ts: i64,
    /// An exit it waits on: the last bar it was worked on, the last it
    /// waited on, and the last whose range could fill it, -1 for none.
    rest_seen_ts: i64,
    rest_wait_ts: i64,
    rest_fill_ts: i64,

    // The driver's: `Strategy._order_seq`, the queued commands and the
    // per-timestamp action budget.
    order_seq: u64,
    commands: Vec<Submit>,
    /// The cancels queued beside them, by kernel id.
    cancels: Vec<u64>,
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
    shorts: Option<Shorts<'a>>,
    resting: Option<Resting>,
    ledger: Ledger,
) -> PyResult<Ledger> {
    let order_seq = venue.first_order_seq;
    if shorts.as_ref().and_then(|s| s.admission).is_some_and(|a| a.len() != bars.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("invalid replay array lengths"));
    }
    let mut replay = Replay {
        runner,
        bars,
        mask,
        gate,
        rules,
        quiet,
        venue,
        money,
        shorts,
        resting,
        working_entry: false,
        orders: Vec::new(),
        pending_target: None,
        pending_order_style: OrderStyle::Default,
        target_bars: 0,
        target_bar_ts: -1,
        pending_reference: 0.0,
        last_exit_ts: -1,
        last_submit_ts: -1,
        target_step_due: false,
        cash_halted: false,
        working_orders: 0,
        // The first callback checks cash and establishes whether quiet bars
        // can be deferred.
        bar_callback_required: true,
        account_quantity: 0.0,
        held_reads: (0.0, 0, 0),
        short_attempts: 0,
        exit_gated: false,
        short_opened_ns: None,
        loan_cover_bar: -1,
        loan_contest_bar: -1,
        close_retries: 0,
        stale_cancel: None,
        stale_ts: -1,
        rest_seen_ts: -1,
        rest_wait_ts: -1,
        rest_fill_ts: -1,
        order_seq,
        commands: Vec::new(),
        cancels: Vec::new(),
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
    ///
    /// A cancel is queued only by a bar's stale check, ahead of anything else
    /// the bar could queue, so the cancels go first.
    fn apply_commands(&mut self, i: usize) -> PyResult<bool> {
        let commands = std::mem::take(&mut self.commands);
        let cancels = std::mem::take(&mut self.cancels);
        let now = self.bars.ts[i];
        if self.budget_ts != Some(now) {
            self.budget_ts = Some(now);
            self.budget_actions = 0;
        }
        self.budget_actions += commands.len() + cancels.len();
        if self.budget_actions > self.venue.max_actions_per_timestamp {
            return Err(declined("execution did not quiesce within its action budget"));
        }
        for &order_id in &cancels {
            self.runner.kernel_mut().cancel_order(i, order_id);
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
                Some(&(last, ..)) if last >= order_id => {
                    self.orders.partition_point(|&(id, ..)| id < order_id)
                }
                _ => self.orders.len(),
            };
            self.orders.insert(at, (order_id, command.ledger_index, command.buy));
        }
        Ok(!commands.is_empty() || !cancels.is_empty())
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
                    if self.stale_cancel == Some(order_id) {
                        self.stale_cancel = None;
                        self.retry_stale_exit(at)?;
                    }
                }
                EngineEvent::OrderExpired { order_id, .. } => {
                    if self.runner.kernel().order(order_id).is_some_and(|order| order.reduce_only) {
                        // `on_order_expired` retries a close at once
                        // (`_handle_ioc_close_expiry`); this would re-offer
                        // it from the next bar instead.
                        return Err(declined(
                            "a working exit expired, which StrategyBase retries at once and \
                             the native lane does not model",
                        ));
                    }
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
        if self.pending_target.is_some()
            || self.working_orders != 0
            || self.loan_cover_bar >= 0
            || self.loan_contest_bar >= 0
        {
            // A capped short's cover and contested bars have to reach the
            // strategy, and the kernel's quiet stepping would take them as
            // any other.
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
        // `_work_resting_exit`: asked first, ahead of anything that could
        // send beside it.
        self.work_resting_exit(i)?;
        // `_refuse_contested_loan`: a composite's check of the loan held
        // comes ahead of anything this bar does.
        if usize::try_from(self.loan_contest_bar).is_ok_and(|at| at <= i) {
            return Err(declined("a composite bar asks a held loan its age first"));
        }
        if self.cash_halted {
            self.advance(i, false)?;
            // A halted replay acts on nothing, so the kernel takes its bars
            // wherever it steps them at all.
            self.bar_callback_required = false;
            return Ok(());
        }
        let reconciled =
            if self.pending_target.is_some() { self.reconcile_target(i)? } else { false };
        let mut target = self.advance(i, true)?;
        if self.stale_ts == self.bars.ts[i] || self.rest_wait_ts == self.bars.ts[i] {
            target = self.absorb_close_on_cancel(i, target)?;
        }
        // Reconciling stands in for the bar's own decision, not for its risk
        // checks: `handle_bar` runs those whatever the reconcile chased.
        let policy_acted = target.is_some_and(|t| t.policy.is_some());
        if !reconciled || policy_acted {
            let preceding = if reconciled { None } else { target.and_then(|t| t.preceding) };
            self.submit_native_intent(i, target, preceding, policy_acted)?;
        } else if self.shorts.is_some() && target.is_some_and(|t| t.side == 0) {
            // Composable buys to cover without asking `has_pending_exit()`,
            // which only its long exit asks: a close this bar's reconcile
            // already sent does not hold back the rules' own.
            self.submit_native_intent(i, target, None, false)?;
        }
        // `_check_risk_exits` asks a capped loan its age after the bar's
        // decision, and ahead of the cash halt -- which a cover that filled
        // at once leaves flat.
        if self.loan_due(i) {
            self.check_short_loan(i)?;
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
        // Behind the halt, as there: the book it leaves is the one the bars
        // after this are asked about.
        let (has_position, net_position, flat, cash_flat) =
            if self.target_step_due && self.finish_target_step(i)? {
                let (has_position, net_position) = held_position(self.runner);
                let flat = !has_position;
                let cash_flat = flat || self.holds_flat_dust(i, has_position, net_position);
                (has_position, net_position, flat, cash_flat)
            } else {
                (has_position, net_position, flat, cash_flat)
            };
        // What the bars a sparse callback steps past before the next
        // composite can do is asked only of a run that defers them: one with
        // risk checks has every bar called back, each asked on its own.
        let defers = !self.rules.has_policies();
        self.bar_callback_required = (self.pending_target.is_some()
            && !(defers && self.flat_target_rests_before_dispatch(i, has_position, net_position)))
            || self.working_orders != 0
            || (cash_flat
                && !self.cash_halted
                && !flat_cash_safe
                && !(defers && self.flat_cash_holds_before_dispatch(i)?))
            || self.rules.visit_primary_after_fill()
            || (defers
                && !flat
                && self.rules.held_reopens(i, has_position, net_position, self.pending_target))
            || self.loan_due(i);
        Ok(())
    }

    /// `_work_resting_exit`: `StrategyBase._check_stale_day_orders` on a
    /// primary bar, where the replay follows it (see [`Resting`]);
    /// [`ReplayDeclined`] where not.
    ///
    /// Asked first in every callback. There `_reconcile_exit` runs first
    /// and, finding the close still pending, sends nothing but leaves the
    /// pending close latched through the bar (`note_close`); the stale check
    /// then cancels the exit, and the venue's report of the cancel re-sends
    /// it ([`Self::retry_stale_exit`]). Within the drift the check waits on,
    /// it leaves the exit working instead, and the bar's decision finds the
    /// close pending ([`Self::absorb_close_on_cancel`]).
    ///
    /// Nautilus offers a resting order the bar's range before the strategy
    /// hears of the bar, and this kernel after its callback: so a bar whose
    /// range reaches the exit and whose close has drifted is declined, being
    /// one on which Nautilus fills it rather than cancels it. Within the
    /// drift neither cancels, and the fill lands either side of a callback
    /// that must then act on nothing: no decision lands on the bar, the bar
    /// decides nothing, and the flat book's cash check -- Nautilus's, on that
    /// bar, after the fill -- cannot halt ([`Self::on_order_filled`]).
    fn work_resting_exit(&mut self, i: usize) -> PyResult<()> {
        let now = self.bars.ts[i];
        if !self.runner.kernel().resting_exit(now) {
            return Ok(());
        }
        let Some(terms) = self.resting else {
            return Err(declined("an exit outlived the bar it met"));
        };
        let decline = |why: &str| {
            Err(declined(&format!(
                "an exit outlived the bar it met: {why}; the native lane follows only a long \
                 close it waits on or cancels for drift and re-sends once"
            )))
        };
        if self.shorts.is_some() {
            return decline("not on a long-only native runtime");
        }
        let kernel = self.runner.kernel();
        let mut exits = kernel.open_orders().into_iter().filter(|order| order.reduce_only);
        let (Some(order), None) = (exits.next(), exits.next()) else {
            return decline("another order is working beside it");
        };
        if self.working_orders != 1 {
            return decline("another order is working beside it");
        }
        let (OrderKind::Limit { price }, TimeInForce::Day) = (order.kind, order.tif) else {
            return decline("it is not a DAY limit");
        };
        let (order_id, arrived, qty) = (order.id, order.submitted_ts, order.qty);
        if self.pending_target != Some(0.0) || self.pending_order_style != OrderStyle::Default {
            // The retry is `create_order`'s, in the config's order type: a
            // target carrying a shape of its own is not.
            return decline("it closes to no flat target in the default order type");
        }
        if now - arrived.max(self.rest_seen_ts) > self.venue.bar_ns {
            return decline("it is met again only after a gap");
        }
        let (has_position, net_position) = held_position(self.runner);
        let held = if has_position { net_position } else { 0.0 };
        let QtySpec::Units(units) = qty else {
            return decline("it is partly filled, or not a long close");
        };
        if held <= 0.0 || held != units {
            return decline("it is partly filled, or not a long close");
        }
        let reached = self.bars.high[i] >= price;
        let current = self.bars.close[i];
        if !((price - current).abs() / current > terms.stale_close_pct) {
            // Within the drift: the check leaves it working, and the bar's
            // reconcile found the close pending -- it sends nothing more on
            // this bar, and the latch holds through it.
            if reached {
                if self.mask[i] {
                    return decline(
                        "the venue may fill it on a bar the rules are called on, ahead of \
                         their decision",
                    );
                }
                self.rest_fill_ts = now;
            }
            self.rest_seen_ts = now;
            self.rest_wait_ts = now;
            self.last_exit_ts = now;
            self.rules.note_close(i);
            return Ok(());
        }
        if reached {
            return decline("the bar's range reaches it");
        }
        if self.close_retries >= terms.max_close_retries {
            return decline("its close retries are spent");
        }
        if self.gate.is_some_and(|gate| !gate[i]) {
            return decline("the venue is shut to its retry");
        }
        self.stale_cancel = Some(order_id);
        self.cancels.push(order_id);
        self.stale_ts = now;
        // The bar's reconcile found the close pending: it sends nothing more
        // on this bar, and the latch holds through it.
        self.last_exit_ts = now;
        self.rules.note_close(i);
        Ok(())
    }

    /// `_absorb_close_on_cancel`: the bar's own decision, on a bar a resting
    /// exit is canceled or waited on -- `None` where `StrategyBase` sends
    /// nothing for it.
    ///
    /// The rules' close finds the close pending (Composable sells only if
    /// not `has_pending_exit()`, which the `_pending_flat` latch answers),
    /// and an entry is refused as `PENDING_CLOSE` (`note_close`). A risk
    /// exit sends a close of its own, and anything sent ahead of the decision
    /// would stand beside the retry: both decline. So does any decision on a
    /// bar whose range may fill the exit, which Nautilus decides on the book
    /// that fill leaves.
    fn absorb_close_on_cancel(&self, i: usize, target: Option<Target>) -> PyResult<Option<Target>> {
        if self.last_submit_ts == self.bars.ts[i] {
            return Err(declined(
                "an order is sent on the bar a resting exit is cancelled on, which the native \
                 lane does not model",
            ));
        }
        if target.is_some() && self.rest_fill_ts == self.bars.ts[i] {
            return Err(declined(
                "the bar whose range may fill a resting exit decides, which Nautilus does after \
                 that fill",
            ));
        }
        match target {
            Some(target) if target.policy.is_some() || target.side != 0 => Err(declined(
                "the bar a resting exit is cancelled on decides more than a close, which the \
                 native lane does not model",
            )),
            _ => Ok(None),
        }
    }

    /// `_retry_stale_exit`: `_retry_close_order(kind="DAY")`, from the
    /// cancel's report -- the whole position, at the bar's close, priced as
    /// every close is.
    fn retry_stale_exit(&mut self, at: usize) -> PyResult<()> {
        self.close_retries += 1;
        let reference = self.bars.close[at];
        if !self.submit_exit(at, reference, None, OrderStyle::Default)? {
            return Err(declined("a resting exit's retry could not be sent"));
        }
        self.pending_reference = reference;
        Ok(())
    }

    /// `0 <= _loan_cover_bar <= idx`: a loan past its cover bar, still held.
    fn loan_due(&self, i: usize) -> bool {
        usize::try_from(self.loan_cover_bar).is_ok_and(|at| at <= i)
    }

    /// `RustRuleSession.advance`; `apply_policies` false for a halted run.
    fn advance(&mut self, i: usize, apply_policies: bool) -> PyResult<Option<Target>> {
        let (held, net) = held_position(self.runner);
        // The account the risk checks read (`_position`).
        let account =
            if self.rules.has_policies() { policy_account(self.runner) } else { Account::FLAT };
        let (mut has_position, mut quantity) = (held, net);
        // Level rules read dust themselves, on the venue's own lot.
        if !self.rules.level()
            && has_position
            && self.reads_dust_as_flat(quantity, self.rules.primary_close(i))
        {
            has_position = false;
            quantity = 0.0;
        }
        let (processed, target) = self.rules.advance(
            self.bars.ts[i],
            i,
            has_position,
            quantity,
            self.pending_target,
            account,
            apply_policies,
        )?;
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

    /// `_submit_native_intent`: `preceding` is the model's own close to send
    /// ahead of a risk close (`preceding_exit`), and `risk_exit` marks a
    /// target a risk policy decided, outside the bar's decision.
    fn submit_native_intent(
        &mut self,
        i: usize,
        target: Option<Target>,
        preceding: Option<f64>,
        risk_exit: bool,
    ) -> PyResult<()> {
        let Some(target) = target else {
            return Ok(());
        };
        let (target_side, style) = (target.side, target.style);
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
            if self.pending_target.is_some_and(|t| t != 0.0 && (t > 0.0) != (held_side > 0)) {
                // A close part way through a reversal, which moves the book
                // toward the far side and stops short of it: the target
                // stands (`_supersede_pending_signed_target`). No replay here
                // reverses, so this is the original's shape kept whole. Every
                // other close supersedes what is out, below.
                if self.submit_exit(i, reference, None, style)? {
                    self.pending_reference = reference;
                }
                return Ok(());
            }
            let previous = (
                self.pending_target,
                self.pending_reference,
                self.pending_order_style,
                self.target_bars,
                self.target_bar_ts,
                self.short_attempts,
            );
            self.pending_reference = reference;
            self.pending_order_style = style;
            self.target_bars = 0;
            self.target_bar_ts = -1;
            self.retarget(Some(0.0));
            self.exit_gated = false;
            let preceded = match preceding {
                Some(at) => self.submit_exit(i, at, None, OrderStyle::Default)?,
                None => false,
            };
            let submitted = self.submit_exit(i, reference, None, style)? || preceded;
            if submitted {
                return Ok(());
            }
            if risk_exit && self.exit_gated {
                // `run_decision` refuses the bar's own orders on a shut venue
                // before `create_order` latches anything, but a risk check
                // runs outside it: its close supersedes any target and
                // latches the pending flat, then `submit_order` meets the
                // bell. The next primary bar's `_reconcile_exit` resends it
                // in the configured close type.
                self.pending_order_style = OrderStyle::Default;
            } else if previous.0.is_some() && (risk_exit || self.gate.is_none_or(|g| g[i])) {
                // A close that reached `create_order` and stopped at
                // instrument preflight: it superseded the target there all
                // the same, and latched nothing in its place.
                self.drop_target();
            } else {
                // A rule session's close is one attempt (`create_order`; its
                // intents do not `persist_if_unsubmitted`): one the venue's
                // hours refuse before `create_order` is dropped, and whatever
                // target was in flight before it stands --
                // `_submit_native_intent`.
                (
                    self.pending_target,
                    self.pending_reference,
                    self.pending_order_style,
                    self.target_bars,
                    self.target_bar_ts,
                    self.short_attempts,
                ) = previous;
            }
            return Ok(());
        }
        if target_side == 0 || self.working_entry || self.cash_halted {
            return Ok(());
        }
        if !risk_exit && self.gate.is_some_and(|gate| !gate[i]) {
            // The bar's own entry on a shut venue: `run_decision` refuses it
            // before `create_order` -- or the sell-to-open's
            // `set_target_position` -- latches anything, so no target is left
            // to sell at the next open, and one already out stands as it was.
            return Ok(());
        }
        if units <= 0.0 {
            return Ok(());
        }
        if self.shorts.is_some() {
            // `create_order(SELL)` from flat, which the sell-to-open adapter
            // turns into `set_target_position(current - qty)`: the target is
            // the lot held -- dust `is_flat()` reads as flat, or nothing --
            // less the entry. The same target again keeps its bar and attempt
            // budgets; a new one starts both.
            let target = held - units;
            if Some(target) != self.pending_target {
                self.target_bars = 0;
                self.target_bar_ts = -1;
            }
            self.retarget(Some(target));
            self.pending_reference = reference;
            self.pending_order_style = style;
            self.submit_entry(i, reference, units, false)?;
            return Ok(());
        }
        if target_side < 0 {
            return Err(declined("a short entry"));
        }
        self.submit_entry(i, reference, units, true)?;
        Ok(())
    }

    /// `_submit_exit`: close the position, or `units` of it, at `reference`,
    /// in `style`: the venue's configured close for `Default`.
    fn submit_exit(
        &mut self,
        i: usize,
        reference: f64,
        units: Option<f64>,
        style: OrderStyle,
    ) -> PyResult<bool> {
        let decided_ns = self.bars.ts[i];
        let (_, net_position) = held_position(self.runner);
        let buy = net_position <= 0.0;
        let limit_price = self.snap(reference, net_position < 0.0)?;
        let held_units = net_position.abs();
        // Python's `min`: the first argument unless the second is less.
        let units = match units {
            Some(asked) if held_units < asked => held_units,
            Some(asked) => asked,
            None => held_units,
        };
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        if let Some(gate) = self.gate {
            if !gate[i] {
                // Past instrument preflight, which is as far as a close gets
                // in `create_order` before `submit_order` meets the shut
                // venue. Whether it outlives its order is the caller's to say.
                self.exit_gated = true;
                return Ok(false);
            }
        }
        // A buy close takes `reference_order_type`'s buy side: a cover's.
        let cover_market = match &self.shorts {
            Some(shorts) => shorts.cover_market,
            None if buy => return Err(declined("a short position")),
            None => false,
        };
        let market = match style {
            OrderStyle::Default if buy => cover_market,
            OrderStyle::Default => self.venue.exit_market,
            OrderStyle::Market => true,
            OrderStyle::Limit => false,
        };
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
        )?;
        self.last_exit_ts = decided_ns;
        self.last_submit_ts = decided_ns;
        self.working_orders += 1;
        Ok(true)
    }

    /// `_submit_entry`: open `units` on `long`'s side, in that side's shape.
    ///
    /// A long entry is the configured entry order, a limit resting twelve
    /// bars; a short entry is `_create_short_entry_ioc`'s, a limit that
    /// lives on arrival only -- IOC, or DAY where the venue refuses IOC.
    fn submit_entry(&mut self, i: usize, reference: f64, units: f64, long: bool) -> PyResult<bool> {
        let decided_ns = self.bars.ts[i];
        if let Some(gate) = self.gate {
            if !gate[i] {
                return Ok(false);
            }
        }
        // A short against a durable target spends one of its attempts, and
        // the target is given up when they are gone, ahead of the preflight.
        let durable_short = !long && self.pending_target.is_some_and(|t| t < 0.0);
        if !long {
            let Some(shorts) = &self.shorts else {
                return Err(declined("a short entry"));
            };
            if durable_short && self.short_attempts >= shorts.max_attempts {
                self.drop_target();
                return Ok(false);
            }
        }
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        if !long && self.short_refused(decided_ns, units)? {
            // `StrategyBase` retains a signed target after a local short
            // preflight refusal and retries it on later bars.
            return Ok(false);
        }
        let market = long && self.venue.entry_market;
        let limit_price = self.snap(reference, long)?;
        if long {
            // `_buy_cash_shortfall`: the order costed at the price it
            // carries, fee included. A short entry checks no cash.
            let order_price =
                if market { self.bars.close[i] * self.venue.market_markup } else { limit_price };
            if !self.affordable(units, order_price)? {
                return Ok(false);
            }
        }
        let tif = if market {
            "gtc"
        } else if long {
            "gtd"
        } else if self.venue.venue_ioc {
            "ioc"
        } else {
            "day"
        };
        let expire_ns = (tif == "gtd").then(|| decided_ns + 12 * self.venue.bar_ns);
        self.submit(
            Submit {
                client_id: String::new(),
                ledger_index: 0,
                buy: long,
                market,
                units,
                limit_price: (!market).then_some(limit_price),
                tif,
                expire_ns,
                reduce_only: false,
            },
            decided_ns,
            decided_ns,
        )?;
        if durable_short {
            self.short_attempts += 1;
        }
        self.working_entry = true;
        self.last_submit_ts = decided_ns;
        self.working_orders += 1;
        Ok(true)
    }

    /// Whether the run's short gate refuses a short of `units` decided at
    /// `decided_ns`, added to the position held: `short_gate.refusal`.
    fn short_refused(&self, decided_ns: i64, units: f64) -> PyResult<bool> {
        let Some(refusal) = self.shorts.as_ref().and_then(|s| s.refusal) else {
            return Ok(false);
        };
        let (has_position, net_position) = held_position(self.runner);
        let position = if has_position { net_position } else { 0.0 };
        refusal.call1((decided_ns, units, position, self.short_opened_ns))?.is_true()
    }

    /// `Strategy.submit_order` and the `_OrderTerms` recorded beside it.
    fn submit(&mut self, mut order: Submit, decided_ns: i64, arrived_on_ns: i64) -> PyResult<()> {
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
            // No target asks the risk policies after a fill any more (one step
            // per instant): every order's terms say so.
            policies_after_fill: false,
        });
        order.ledger_index = index;
        self.commands.push(order);
        Ok(())
    }

    /// `_retarget`: a different target starts a fresh short-attempt budget,
    /// and clearing resets it too.
    fn retarget(&mut self, target: Option<f64>) {
        if target != self.pending_target {
            self.short_attempts = 0;
        }
        self.pending_target = target;
    }

    fn drop_target(&mut self) {
        self.retarget(None);
        self.pending_order_style = OrderStyle::Default;
        self.target_bars = 0;
        self.target_bar_ts = -1;
    }

    /// `_continue_target`: one move toward the pending target -- reductions
    /// unless `reduce` is false, expansions unless `expand` is.
    fn continue_target(
        &mut self,
        i: usize,
        reference: f64,
        reduce: bool,
        expand: bool,
    ) -> PyResult<bool> {
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
            // Wrong side of the target, or anything held when it is flat:
            // reduce first, and to flat.
            if !reduce {
                return Ok(false);
            }
            let submitted = self.submit_exit(i, reference, None, self.pending_order_style)?;
            if submitted {
                self.pending_reference = reference;
            }
            return Ok(submitted);
        }
        if held != 0.0 && target.abs() < held.abs() {
            if !reduce {
                return Ok(false);
            }
            let submitted = self.submit_exit(
                i,
                reference,
                Some(held.abs() - target.abs()),
                self.pending_order_style,
            )?;
            if submitted {
                self.pending_reference = reference;
            }
            return Ok(submitted);
        }
        if !expand {
            return Ok(false);
        }
        // A long-only replay's only target is flat, which the branches above
        // settle whatever the book holds; a short node's is short of it.
        let Some(max_stale_bars) = self.shorts.as_ref().map(|s| s.max_stale_bars) else {
            return Err(declined("a target other than flat"));
        };
        // Same side as the target and short of it: the residual, on the
        // target's own side -- not while its own order is still working.
        if self.working_entry {
            return Ok(false);
        }
        if self.target_bars > max_stale_bars {
            self.drop_target();
            return Ok(false);
        }
        let delta = target - held;
        if delta > 0.0 && target > 0.0 {
            // `_fund_long_target`, which no short node's target reaches.
            return Err(declined("a long target"));
        }
        if self.pending_order_style != OrderStyle::Default {
            // Only a risk close carries a style, and it targets flat.
            return Err(declined("an entry in a close's order style"));
        }
        self.submit_entry(i, reference, delta.abs(), delta > 0.0)
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
        let closing = self.pending_target == Some(0.0);
        // Only what reduces the book, ahead of the decision; what adds to it
        // waits for the end of the step.
        self.target_step_due = true;
        let submitted = self.continue_target(i, self.bars.close[i], true, false)?;
        if submitted && closing {
            // `_reconcile_exit` resends it under the strategy's pending
            // close, which stands until the next bar's reconcile finds the
            // book flat.
            self.rules.note_close(i);
        }
        Ok(submitted)
    }

    /// `_finish_target_step`: what the target still adds, offered once the
    /// bar has decided and its risk checks have run -- unless an order went
    /// out this bar, which Nautilus still holds unresolved at the end of the
    /// step. Returns whether an order reached the venue.
    fn finish_target_step(&mut self, i: usize) -> PyResult<bool> {
        self.target_step_due = false;
        if self.last_submit_ts == self.bars.ts[i] {
            return Ok(false);
        }
        self.continue_target(i, self.bars.close[i], false, true)
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
        let Some((order, buy)) = self.ledger_index(order_id) else {
            return Err(declined("a fill of an order this replay did not send"));
        };
        let now = self.bars.ts[at];
        self.ledger.fill(order, now, order_id, size, price, leaves, gross_realized, commission);
        // Every order here is decided on the bar it arrives on, so the fill
        // is stamped by the clock (`_OrderTerms.stamp`).
        self.track_short_loan(now)?;
        // `runtime.on_fill`: the account refreshed, the next primary bar a
        // visit, and the fill booked by the risk policies -- at what it cost
        // as the strategy reads it (`_record_fill`). With no follow-up,
        // nothing else of the fill reaches the runtime.
        self.refresh()?;
        if self.rules.has_policies() {
            let cost = money(or_zero(commission), self.quiet.currency_precision)
                .ok_or_else(|| declined("a commission that is not a number"))?;
            let flat = side_of(self.account_quantity) == 0;
            // A run that borrows charges the loan on the lot the fill left.
            let fill = Some((at, self.account_quantity));
            self.rules.on_fill(if buy { 1.0 } else { -1.0 }, or_zero(size), cost, flat, fill)?;
        } else {
            self.rules.on_fill(0.0, 0.0, 0.0, true, None)?;
        }
        if self.account_quantity == 0.0 {
            // A fill that leaves the book flat restores the close retries.
            self.close_retries = 0;
            if self.rest_fill_ts == now {
                // A resting exit this bar's range filled, after its callback
                // here and before it in Nautilus -- whose cash check then
                // reads this flat book, at this close.
                let price = self.bars.close[at];
                let cash = self.strategy_cash()?;
                let q = self.quiet;
                let mut funds = cash * q.fraction;
                if funds > q.cap {
                    funds = q.cap;
                }
                let need = q.units * (price * q.markup + q.slack) + q.fee;
                if funds < need && self.exhausted(cash, price)? {
                    return Err(declined(
                        "a resting exit filled from a bar's range leaves a cash check on that \
                         bar that could halt",
                    ));
                }
            }
        }
        if leaves <= 0.0 {
            self.working_orders = (self.working_orders - 1).max(0);
            self.continue_target(at, self.pending_reference, true, true)?;
        }
        self.bar_callback_required = true;
        Ok(())
    }

    /// `_track_short_loan`, after every fill where a loan has a term: a loan
    /// is dated from the fill that left the book short, and ends with the
    /// fill that leaves it flat or long.
    fn track_short_loan(&mut self, filled_ns: i64) -> PyResult<()> {
        let Some(cover) = self.shorts.as_ref().and_then(|s| s.cover) else {
            return Ok(());
        };
        let (has_position, net_position) = held_position(self.runner);
        if has_position && net_position < 0.0 {
            if self.short_opened_ns.is_none() {
                self.short_opened_ns = Some(filled_ns);
                (self.loan_cover_bar, self.loan_contest_bar) =
                    cover.call1((filled_ns,))?.extract()?;
            }
        } else {
            self.short_opened_ns = None;
            self.loan_cover_bar = -1;
            self.loan_contest_bar = -1;
        }
        Ok(())
    }

    /// `_check_short_loan`: `_check_short_loan_age`'s forced cover, on a bar
    /// it is due on.
    ///
    /// Asked on every bar `handle_bar` admits from the cover bar on, while
    /// nothing of the strategy's own is working. The cover is the whole
    /// short at this bar's close, latched as the pending close unless a
    /// target already owns the retries. A cover the instrument cannot trade
    /// latches nothing and is asked again on the next bar; one a shut venue
    /// refused keeps its latch for the next bar's reconcile to resend.
    fn check_short_loan(&mut self, i: usize) -> PyResult<()> {
        if self.working_orders != 0 {
            return Ok(());
        }
        if let Some(admission) = self.shorts.as_ref().and_then(|s| s.admission) {
            if !admission[i] {
                return Ok(());
            }
        }
        let (has_position, net_position) = held_position(self.runner);
        if !has_position || net_position >= 0.0 {
            return Ok(());
        }
        let reference = self.bars.close[i];
        let previous = (
            self.pending_target,
            self.pending_reference,
            self.pending_order_style,
            self.target_bars,
            self.target_bar_ts,
            self.short_attempts,
        );
        // Short here, so a short target is one the cover trades against, or
        // past: it supersedes that, and latches the close in its place.
        let supersedes = self.pending_target.is_some_and(|t| t < 0.0);
        let latches = self.pending_target.is_none() || supersedes;
        if latches {
            self.retarget(Some(0.0));
            self.pending_reference = reference;
            self.pending_order_style = OrderStyle::Default;
            self.target_bars = 0;
            self.target_bar_ts = -1;
        }
        self.exit_gated = false;
        if self.submit_exit(i, reference, None, OrderStyle::Default)? || !latches || self.exit_gated
        {
            return Ok(());
        }
        if supersedes {
            // Refused at instrument preflight, past the point where
            // `create_order` dropped the target.
            self.drop_target();
            return Ok(());
        }
        (
            self.pending_target,
            self.pending_reference,
            self.pending_order_style,
            self.target_bars,
            self.target_bar_ts,
            self.short_attempts,
        ) = previous;
        Ok(())
    }

    /// `on_order_canceled` / `on_order_expired`.
    fn end_order(&mut self, at: usize, order_id: u64, status: EndStatus) {
        if let Some((order, _)) = self.ledger_index(order_id) {
            self.ledger.end(order, status, self.bars.ts[at]);
        }
        self.working_entry = false;
        self.working_orders = (self.working_orders - 1).max(0);
        self.bar_callback_required = true;
    }

    /// Where the ledger keeps the terms of the order the kernel calls
    /// `order_id`, and whether it buys, if this replay sent it.
    fn ledger_index(&self, order_id: u64) -> Option<(u32, bool)> {
        let at = self.orders.partition_point(|&(id, ..)| id < order_id);
        self.orders.get(at).filter(|&&(id, ..)| id == order_id).map(|&(_, index, buy)| (index, buy))
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
