//! The native portfolio lane's run of a long-only cash book, in Rust, for
//! many cells on one book at once.
//!
//! `parity/native_portfolio.py::native_portfolio_run` drives a sealed
//! `PortfolioSession` with the strategy's own Python policy: the feed
//! (`portfolio_feed.py`) wakes the loop at every member's signal bar and at a
//! window a timer closes, the instant clock (`base/portfolio/instants.py`)
//! decides on complete instants, and the rebalance (`base/portfolio/
//! rebalance.py`) holds each order for its own name's bar and funds the
//! entries from one budget fixed at the decision. This module is that loop,
//! statement for statement, for an equity-percent book on a cash account:
//! every float is formed in the order Python forms it -- the builtin `sum`
//! as `py_sum`, `min` and `max` keeping the first of equals, Nautilus's
//! fixed point and `Decimal` as `money` spells them -- so a cell lands on the
//! same orders, fills and result to the bit.
//!
//! What the book decides is not computed here. A [`PyPortfolioReplay`] first
//! *captures* the instants the feed forms (each member's bar at a label, or
//! its last close carried), which no order can move; the caller decides
//! each cell's weights on them (`lean_decide_series` / `lean_decide_rank`)
//! and *runs* the cells, each against a fresh session built from the
//! book's recipe. A run checks every instant against the capture, and a
//! cell whose path the loop does not transcribe declines with a reason
//! rather than guess: the caller runs it on the native lane.

// Python's `not (x <= 0)` holds for NaN where `x > 0` does not: every
// negated comparison here is one the transcribed code spells that way.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::cmp::Ordering;
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::bindings::PyPortfolioResult;
use super::lean_bindings::{in_pool, threads_pool, LeanDeclined};
use super::money::{money_as_double, round_to, units_as_double, Cost, Dec, Grid, PyMoneyModel};
use super::session_bindings::{portfolio_result, PyPortfolioSession, SessionRecipe};
use super::strategy_bindings::submit_order_on;
use crate::core::decimals::{pow10_f64, round_half_away};
use crate::execution::orders::WalkPhase;
use crate::metrics::exact::py_sum;
use crate::portfolio::kernel::{EngineEvent, StepInput};
use crate::portfolio::session::{EventSession, ScheduleData, SessionOutcome};

/// A step of the run, or why the cell leaves the path this module follows.
type Step<T> = Result<T, String>;

fn refuse<T>(why: &str) -> Step<T> {
    Err(why.to_string())
}

/// `x or why`, for a value the transcription cannot form.
fn need<T>(value: Option<T>, why: &str) -> Step<T> {
    value.ok_or_else(|| why.to_string())
}

/// Python's `max(a, b)`: `b` only where it is greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `b` only where it is less.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `np.searchsorted(xs, v, side="left")`.
fn left<T: PartialOrd>(xs: &[T], v: T) -> usize {
    xs.partition_point(|x| *x < v)
}

/// `np.searchsorted(xs, v, side="right")`.
fn right<T: PartialOrd>(xs: &[T], v: T) -> usize {
    xs.partition_point(|x| *x <= v)
}

/// `_MIN_TRADE`: the smallest notional a rebalance trades.
const MIN_TRADE: f64 = 1.0;

/// No index: `_next_admitted` finding none, or no admitted bar before.
const NONE: usize = usize::MAX;

/// A member's signal leg, as `PortfolioPeriodFeed._signal_leg` keeps it.
struct Leg {
    labels: Vec<i64>,
    /// The primary bar each window is delivered with (`searchsorted` left
    /// of its dispatch), the bar count where none follows.
    index: Vec<usize>,
    delivered: Vec<i64>,
    timer_fired: Vec<bool>,
    clocked: Vec<bool>,
    /// `instant_ohlc`: each window's prices on the price grid.
    ohlc: Vec<[f64; 4]>,
}

/// One member, in the session's order.
struct Member {
    ts: Vec<i64>,
    admitted: Vec<bool>,
    /// The first admitted bar at or after each index (the bar count where
    /// none), one past the end included.
    next_admitted: Vec<usize>,
    /// The last admitted bar before each index, [`NONE`] where none.
    last_admitted: Vec<usize>,
    /// `make_price(close)` of each primary bar: the execution mark.
    exec_mark: Vec<f64>,
    /// Whether an order may be submitted at each of the plan's instants;
    /// `None`: always.
    gate: Option<Vec<bool>>,
    leg: Leg,
    cost: Cost,
    grid: Grid,
    price_precision: u32,
    size_increment_precision: u32,
    size_precision: u32,
    /// `size_increment.as_decimal()`.
    step: Dec,
    /// `make_qty`'s floor: a positive quantity rounding below this raises.
    min_quantity: f64,
    min_notional: Option<f64>,
    base_precision: Option<u32>,
    /// `_urgent_exit_tif(iid).name.lower()`.
    urgent_tif: String,
    /// `ENTRY_LIMIT_TTL_BARS` of the execution bar, in nanoseconds.
    ttl_ns: i64,
    /// The member's rank by `str(instrument_id)`.
    str_rank: usize,
    /// The member's rank by its signal timer's name.
    name_rank: usize,
}

impl Member {
    /// `_next_admitted`: the stamp of the first admitted bar from `index`.
    fn next_admitted_ts(&self, index: usize) -> Option<i64> {
        if index >= self.ts.len() {
            return None;
        }
        let at = self.next_admitted[index];
        (at < self.ts.len()).then(|| self.ts[at])
    }

    /// `make_qty(x)`'s units, `None` where it raises `ValueError` -- a
    /// positive quantity rounding to nothing, or a negative one.
    fn make_qty(&self, x: f64) -> Step<Option<i64>> {
        let rounded = need(round_to(x, self.size_increment_precision), "unroundable quantity")?;
        if x > 0.0 && rounded < self.min_quantity {
            return Ok(None);
        }
        let units = round_half_away(rounded * pow10_f64(self.size_precision));
        if !(units.abs() < 1e17) {
            return refuse("a quantity past the fixed point");
        }
        Ok((units >= 0.0).then_some(units as i64))
    }

    /// `make_qty` of a decimal floored onto the size step
    /// (`_floor_quantity_to_increment`).
    fn floor_qty(&self, value: Dec) -> Step<Option<i64>> {
        if value.c <= 0 {
            return self.make_qty(0.0);
        }
        let units = need(value.floor_div(self.step), "a quantity past the decimal digits")?;
        let floored = need(
            Dec::checked(units, 0).and_then(|u| u.mul(self.step)),
            "a quantity past the decimal digits",
        )?;
        self.make_qty(floored.to_f64())
    }

    fn quantity(&self, units: i64) -> Dec {
        Dec { c: i128::from(units), e: -(self.size_precision as i32) }
    }
}

/// One stamp of the run's clock a window is delivered at with no bar of its
/// member to carry it (`_ClockEvents`): data first, then timers by name.
struct ClockEvent {
    ts: i64,
    timer: bool,
    instrument: usize,
    row: usize,
}

/// A book: its members, its clock and the session it runs on.
struct Plan {
    members: Vec<Member>,
    /// Session index per place in the policy's instruments (`_iids`).
    policy: Vec<usize>,
    /// Every stamp an order may be submitted at, ascending.
    union: Vec<i64>,
    clock: Vec<ClockEvent>,
    starting_cash: f64,
    currency_precision: u32,
    /// `sizer_percent if sizer_percent > 0 else 100.0`.
    percent: f64,
    recipe: SessionRecipe,
}

#[derive(Clone, Copy)]
enum Held {
    Exit { decided: i64, buy: bool, quantity: f64 },
    Entry { decided: i64, buy: bool, quantity: f64, price: f64 },
}

impl Held {
    fn decided(&self) -> i64 {
        match *self {
            Held::Exit { decided, .. } | Held::Entry { decided, .. } => decided,
        }
    }
}

/// `_NativeOrder`.
struct Order {
    seq: u64,
    member: usize,
    buy: bool,
    quantity: f64,
    reduce_only: bool,
    engine_id: u64,
    limit: f64,
    pending_cancel: bool,
}

/// An order as submitted: `(seq, member, buy, quantity, price, reduce_only,
/// ts_init)` -- its ticket's time in force and expiry follow from these.
type Submit = (u64, usize, bool, f64, f64, bool, i64);

/// What the venue did, in order: `(seq, ts, kind, order_id, size, price,
/// leaves, realized, commission)`, `kind` 0 a fill, 1 canceled, 2 expired.
type Record = (u64, i64, u8, u64, f64, f64, f64, f64, f64);

/// One instant the clock decided on: its label, each member's bar in the
/// policy's order, and which were carried.
struct Instant {
    label: i64,
    bars: Vec<[f64; 4]>,
    carried: Vec<bool>,
}

enum Mode<'a> {
    Capture(Vec<Instant>),
    /// The captured instants, and the cell's decisions: `(ordinal, weights)`
    /// ascending, the weights in the policy's order.
    Run {
        captured: &'a [Instant],
        decisions: &'a [(usize, Vec<f64>)],
        next: usize,
    },
}

enum Fire {
    Signal(usize, usize),
    Clocked(usize, usize),
}

/// A cell's run.
struct Book<'a> {
    plan: &'a Plan,
    session: EventSession,
    mode: Mode<'a>,
    clock: i64,
    current: i64,
    cursor: Vec<usize>,
    orders: Vec<Order>,
    seq: u64,
    routed: bool,
    arrivals: Vec<usize>,
    cash: f64,
    pos: Vec<f64>,
    entry: Vec<Option<f64>>,
    exec_price: Vec<Option<f64>>,
    last_close: Vec<Option<f64>>,
    held: Vec<Vec<Held>>,
    pending_buys: Vec<(usize, f64)>,
    pending_ts: i64,
    budget: Option<f64>,
    scale: Option<f64>,
    /// `_current_decision["targets"]`, by session index.
    targets: Option<Vec<f64>>,
    /// `_deferring_ts` while `_orders_on_their_own_bars` holds orders.
    deferring: Option<i64>,
    proceeds: f64,
    /// `_decision_marks`, by session index.
    marks: Option<Vec<f64>>,
    label: Option<i64>,
    bars: Vec<Option<[f64; 4]>>,
    bar_count: usize,
    carry: Vec<Option<[f64; 4]>>,
    decided: Option<i64>,
    delivered_through: Vec<Option<i64>>,
    ordinal: usize,
    applied: Vec<usize>,
    signal_applied: Vec<usize>,
    clock_next: usize,
    fills: usize,
    submits: Vec<Submit>,
    records: Vec<Record>,
}

impl<'a> Book<'a> {
    fn new(plan: &'a Plan, mode: Mode<'a>) -> Book<'a> {
        let n = plan.members.len();
        Book {
            plan,
            session: plan.recipe.build(),
            mode,
            clock: 0,
            current: -1,
            cursor: vec![0; n],
            orders: Vec::new(),
            seq: 0,
            routed: false,
            arrivals: Vec::new(),
            cash: plan.starting_cash,
            pos: vec![0.0; n],
            entry: vec![None; n],
            exec_price: vec![None; n],
            last_close: vec![None; n],
            held: vec![Vec::new(); n],
            pending_buys: Vec::new(),
            pending_ts: 0,
            budget: None,
            scale: None,
            targets: None,
            deferring: None,
            proceeds: 0.0,
            marks: None,
            label: None,
            bars: vec![None; n],
            bar_count: 0,
            carry: vec![None; n],
            decided: None,
            delivered_through: vec![None; n],
            ordinal: 0,
            applied: vec![0; n],
            signal_applied: vec![0; n],
            clock_next: 0,
            fills: 0,
            submits: Vec::new(),
            records: Vec::new(),
        }
    }

    // ---- the loop (`native_portfolio_run`) ----

    /// `(instrument, local_idx, ts)` of the session's pending bar.
    fn current_bar(&self) -> Option<(usize, usize, i64)> {
        self.session.current().and_then(|e| match e.data {
            ScheduleData::Bar(bar) => Some((e.instrument, e.local_idx, bar.timestamp)),
            _ => None,
        })
    }

    fn run(&mut self) -> Step<()> {
        while self.current_bar().is_some() {
            let wake = self.next_wake();
            self.idle_until(wake.unwrap_or(i64::MAX))?;
            let Some(wake) = wake else { break };
            let Some((_, _, ts)) = self.sync_to_cursor()? else { break };
            if ts != wake {
                if self.next_clock_instant() != Some(wake) {
                    return refuse("compiled portfolio feed diverged from the Raptor event cursor");
                }
                self.deliver_clocked_data(wake, None)?;
                self.fire_timers(wake, Vec::new())?;
                continue;
            }
            let mut timers = Vec::new();
            while let Some((instrument, local, ts)) = self.current_bar() {
                if ts != wake {
                    break;
                }
                self.deliver_clocked_data(ts, Some(instrument))?;
                let admitted = self.apply_current(instrument, local, ts)?;
                if admitted {
                    self.release_held(instrument, ts)?;
                    if self.pending_buys.iter().any(|&(m, _)| m == instrument)
                        && ts > self.pending_ts
                    {
                        self.flush_pending_buys(instrument)?;
                    }
                }
                if self.deliver_signal(instrument, local, true)? {
                    let rank = self.plan.members[instrument].name_rank;
                    timers.push((rank, Fire::Signal(instrument, local)));
                }
                let routed = self.routed;
                self.settle(ts, routed, Some(instrument))?;
                let events = self.session.apply_current(StepInput::default());
                self.report(events, ts)?;
                self.settle(ts, routed, None)?;
                self.cursor[instrument] = local + 1;
            }
            self.deliver_clocked_data(wake, None)?;
            self.fire_timers(wake, timers)?;
        }
        Ok(())
    }

    fn idle_until(&mut self, until: i64) -> Step<()> {
        loop {
            match self.current_bar() {
                Some((_, _, ts)) if ts < until => {}
                _ => return Ok(()),
            }
            let (events, event_ts) = self.session.run_until(until);
            let Some(event_ts) = event_ts else { return Ok(()) };
            if self.sync_to_cursor()?.is_none() {
                self.sync_through(event_ts)?;
            }
            self.clock = event_ts;
            self.report(events, event_ts)?;
        }
    }

    fn sync_to_cursor(&mut self) -> Step<Option<(usize, usize, i64)>> {
        let current = self.current_bar();
        if let Some((instrument, local, ts)) = current {
            self.sync_before(instrument, local, ts)?;
        }
        Ok(current)
    }

    fn deliver_clocked_data(&mut self, ts: i64, before: Option<usize>) -> Step<()> {
        for (instrument, row) in self.clocked_due(ts, false, before)? {
            self.deliver_clocked(instrument, row)?;
            self.settle(ts, false, None)?;
        }
        Ok(())
    }

    fn fire_timers(&mut self, ts: i64, mut timers: Vec<(usize, Fire)>) -> Step<()> {
        for (instrument, row) in self.clocked_due(ts, true, None)? {
            let rank = self.plan.members[instrument].name_rank;
            timers.push((rank, Fire::Clocked(instrument, row)));
        }
        if timers.is_empty() {
            return Ok(());
        }
        self.current = -1;
        timers.sort_by_key(|&(rank, _)| rank);
        for (_, fire) in timers {
            match fire {
                Fire::Signal(instrument, local) => {
                    self.deliver_signal(instrument, local, false)?;
                }
                Fire::Clocked(instrument, row) => self.deliver_clocked(instrument, row)?,
            }
        }
        self.settle(ts, false, None)
    }

    // ---- the venue (`_settle_books`, `settle_venue`, `_report_events`) ----

    fn take_routed(&mut self) -> bool {
        std::mem::replace(&mut self.routed, false)
    }

    fn settle(&mut self, ts: i64, routed: bool, skip: Option<usize>) -> Step<()> {
        if !(routed || self.routed) {
            return Ok(());
        }
        let mut routed = routed;
        loop {
            routed = self.take_routed() || routed;
            if !routed {
                return Ok(());
            }
            while routed {
                for instrument in std::mem::take(&mut self.arrivals) {
                    if Some(instrument) != skip {
                        self.walk(instrument, ts, WalkPhase::Drain)?;
                    }
                }
                routed = self.take_routed();
            }
            if skip.is_some() {
                return Ok(());
            }
            for instrument in 0..self.plan.members.len() {
                self.walk(instrument, ts, WalkPhase::Walk)?;
            }
            routed = false;
        }
    }

    fn walk(&mut self, instrument: usize, ts: i64, phase: WalkPhase) -> Step<()> {
        let events = self.session.walk_book(instrument, ts, phase);
        self.report(events, ts)
    }

    fn order_at(&self, seq: u64) -> Option<usize> {
        self.orders.binary_search_by_key(&seq, |order| order.seq).ok()
    }

    fn report(&mut self, events: Vec<EngineEvent>, ts: i64) -> Step<()> {
        for event in events {
            let (idx, client) = match &event {
                EngineEvent::OrderAccepted { idx, client_id, .. }
                | EngineEvent::OrderTriggered { idx, client_id, .. }
                | EngineEvent::OrderFilled { idx, client_id, .. }
                | EngineEvent::OrderCanceled { idx, client_id, .. }
                | EngineEvent::OrderExpired { idx, client_id, .. }
                | EngineEvent::OrderRejected { idx, client_id, .. }
                | EngineEvent::AlgoStarted { idx, client_id, .. }
                | EngineEvent::AlgoCompleted { idx, client_id, .. } => (*idx, client_id.as_str()),
                _ => continue,
            };
            let Some(seq) = client.strip_prefix("NP-").and_then(|s| s.parse::<u64>().ok()) else {
                continue;
            };
            let Some(at) = self.order_at(seq) else { continue };
            let member = self.orders[at].member;
            self.clock = ts;
            self.current = member as i64;
            self.cursor[member] = idx;
            match event {
                EngineEvent::OrderFilled {
                    order_id,
                    price,
                    size,
                    commission,
                    leaves,
                    gross_realized,
                    ..
                } => {
                    self.fills += 1;
                    self.apply_fill(at, size, price, commission)?;
                    self.records.push((
                        seq,
                        ts,
                        0,
                        order_id,
                        size,
                        price,
                        leaves,
                        gross_realized,
                        commission,
                    ));
                    self.orders[at].quantity = leaves;
                    if leaves <= 0.0 {
                        self.orders.remove(at);
                    }
                }
                EngineEvent::OrderCanceled { order_id, .. } => {
                    self.records.push((seq, ts, 1, order_id, 0.0, 0.0, 0.0, 0.0, 0.0));
                    self.orders.remove(at);
                }
                EngineEvent::OrderExpired { order_id, .. } => {
                    self.records.push((seq, ts, 2, order_id, 0.0, 0.0, 0.0, 0.0, 0.0));
                    self.orders.remove(at);
                }
                EngineEvent::OrderRejected { .. } => {
                    self.orders.remove(at);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// `_apply_policy_fill` on a cash account.
    fn apply_fill(&mut self, at: usize, size: f64, price: f64, commission: f64) -> Step<()> {
        let plan = self.plan;
        let (m, buy) = (self.orders[at].member, self.orders[at].buy);
        let member = &plan.members[m];
        let Some(units) = member.make_qty(size)? else {
            return refuse("a fill whose size rounds to no quantity");
        };
        let size = member.quantity(units).to_f64();
        if size <= 0.0 {
            return Ok(());
        }
        let fill = need(
            member
                .grid
                .price_units(price)
                .and_then(|units| units_as_double(units, member.price_precision)),
            "a fill price past the fixed point",
        )?;
        let precision = plan.currency_precision;
        let money =
            |value: f64| need(money_as_double(value, precision), "money past the fixed point");
        let commission = money(commission)?;
        let old = self.pos[m];
        let old_entry = self.entry[m];
        let delta = if buy { size } else { -size };
        let new = need(
            Dec::of(old).zip(Dec::of(delta)).and_then(|(a, b)| a.add(b)),
            "a position past the decimal digits",
        )?
        .to_f64();
        let leg = -money(delta * fill)?;
        self.cash = money(self.cash + leg - commission)?;
        if new.abs() <= 1e-12 {
            self.pos[m] = 0.0;
            self.entry[m] = None;
            return Ok(());
        }
        let new_entry = match old_entry {
            Some(entry) if old != 0.0 && !(old * new < 0.0) => {
                if old * delta > 0.0 {
                    (old.abs() * entry + delta.abs() * fill) / new.abs()
                } else {
                    entry
                }
            }
            _ => fill,
        };
        self.pos[m] = new;
        self.entry[m] = Some(new_entry);
        Ok(())
    }

    // ---- the feed (`PortfolioPeriodFeed`, instants) ----

    fn next_clock_instant(&self) -> Option<i64> {
        self.plan.clock.get(self.clock_next).map(|event| event.ts)
    }

    fn sync_before(&mut self, instrument: usize, local: usize, ts: i64) -> Step<()> {
        let stamps = &self.plan.members[instrument].ts;
        if local >= stamps.len() || stamps[local] != ts {
            return refuse("Raptor cursor is absent from the portfolio feed");
        }
        self.sync_at(ts, Some(instrument))
    }

    fn sync_at(&mut self, ts: i64, before: Option<usize>) -> Step<()> {
        for j in 0..self.plan.members.len() {
            let stamps = &self.plan.members[j].ts;
            let desired = match before {
                Some(before) if j >= before => left(stamps, ts),
                _ => right(stamps, ts),
            };
            let start = self.applied[j];
            if desired < start {
                return refuse("portfolio feed moved beyond the Raptor cursor");
            }
            self.apply_range(j, start, desired)?;
        }
        Ok(())
    }

    fn sync_through(&mut self, ts: i64) -> Step<()> {
        for j in 0..self.plan.members.len() {
            let desired = right(&self.plan.members[j].ts, ts);
            self.apply_range(j, self.applied[j], desired)?;
        }
        Ok(())
    }

    /// `_apply_range` of a member with a signal leg, on instants: its bars
    /// before *end* move its execution mark, and may fold no window.
    fn apply_range(&mut self, j: usize, start: usize, end: usize) -> Step<()> {
        if end <= start {
            return Ok(());
        }
        let member = &self.plan.members[j];
        if left(&member.leg.index, end) > self.signal_applied[j] {
            return refuse("a book decided on instants had a signal bar folded without its wake");
        }
        let last = member.last_admitted[end];
        if last != NONE && last >= start {
            self.exec_price[j] = Some(member.exec_mark[last]);
        }
        self.cursor[j] = end;
        self.applied[j] = end;
        Ok(())
    }

    fn apply_index(&mut self, i: usize, index: usize) -> Step<bool> {
        let member = &self.plan.members[i];
        let ts = member.ts[index];
        self.clock = ts;
        self.current = i as i64;
        self.cursor[i] = index;
        if !member.admitted[index] {
            return Ok(false);
        }
        self.instant_observe(i, ts)?;
        self.exec_price[i] = Some(member.exec_mark[index]);
        Ok(true)
    }

    fn apply_current(&mut self, i: usize, local: usize, ts: i64) -> Step<bool> {
        self.sync_before(i, local, ts)?;
        let admitted = self.apply_index(i, local)?;
        self.applied[i] = local + 1;
        Ok(admitted)
    }

    fn deliver_signal(&mut self, i: usize, local: usize, hold_timer: bool) -> Step<bool> {
        self.drain_signal(i, local + 1, hold_timer, None)
    }

    fn deliver_clocked(&mut self, i: usize, row: usize) -> Step<()> {
        let leg = &self.plan.members[i].leg;
        let ts = leg.delivered[row];
        let before = if leg.timer_fired[row] { None } else { Some(i) };
        self.sync_at(ts, before)?;
        self.clock = ts;
        self.current = -1;
        self.drain_signal(i, 0, false, Some(row))?;
        Ok(())
    }

    fn drain_signal(
        &mut self,
        i: usize,
        limit: usize,
        hold_timer: bool,
        through_row: Option<usize>,
    ) -> Step<bool> {
        let leg = &self.plan.members[i].leg;
        let cursor = self.signal_applied[i];
        let mut stop = match through_row {
            Some(row) => {
                if cursor != row || !leg.clocked[row] {
                    return refuse("a window the run's clock delivers was reached out of turn");
                }
                row + 1
            }
            None => {
                let stop = left(&leg.index, limit);
                if stop > cursor && leg.clocked[cursor..stop].iter().any(|&c| c) {
                    return refuse(
                        "a window the run's clock delivers was carried by its member's next bar",
                    );
                }
                stop
            }
        };
        let held = hold_timer && stop > cursor && leg.timer_fired[stop - 1];
        if held {
            stop -= 1;
        }
        for index in cursor..stop {
            let label = leg.labels[index];
            let ohlc = leg.ohlc[index];
            self.instant_observe(i, label)?;
            self.last_close[i] = Some(ohlc[3]);
            self.signal_applied[i] = index + 1;
            self.instant_take(i, label, ohlc)?;
            self.release_held(i, label)?;
        }
        Ok(held)
    }

    fn clocked_due(
        &mut self,
        ts: i64,
        timers: bool,
        before: Option<usize>,
    ) -> Step<Vec<(usize, usize)>> {
        let events = &self.plan.clock;
        let mut cursor = self.clock_next;
        if cursor < events.len() && events[cursor].ts < ts {
            return refuse("the run's clock passed a window it delivers without its wake");
        }
        let mut due = Vec::new();
        while cursor < events.len() && events[cursor].ts == ts && (!events[cursor].timer || timers)
        {
            let event = &events[cursor];
            if event.timer != timers {
                return refuse("an instant's timers were due with its data still to deliver");
            }
            if before.is_some_and(|before| event.instrument >= before) {
                break;
            }
            due.push((event.instrument, event.row));
            cursor += 1;
        }
        self.clock_next = cursor;
        Ok(due)
    }

    fn next_wake(&self) -> Option<i64> {
        let mut best: Option<i64> = None;
        let mut offer = |ts: Option<i64>| {
            if let Some(ts) = ts {
                best = Some(best.map_or(ts, |b| b.min(ts)));
            }
        };
        offer(self.next_clock_instant());
        let pending = self.label;
        for (j, member) in self.plan.members.iter().enumerate() {
            let row = self.signal_applied[j];
            if row < member.leg.index.len() {
                offer(Some(member.leg.delivered[row]));
            }
            if let Some(pending) = pending {
                let index = self.applied[j].max(right(&member.ts, pending));
                offer(member.next_admitted_ts(index));
            }
        }
        for (j, queue) in self.held.iter().enumerate() {
            let Some(decided) = queue.iter().map(Held::decided).min() else { continue };
            let member = &self.plan.members[j];
            let position = left(&member.leg.labels, decided);
            if position < member.leg.index.len() {
                offer(member.next_admitted_ts(self.cursor[j].max(member.leg.index[position])));
            }
            offer(member.next_admitted_ts(self.cursor[j].max(left(&member.ts, decided))));
        }
        for &(j, _) in &self.pending_buys {
            let member = &self.plan.members[j];
            offer(member.next_admitted_ts(self.cursor[j].max(right(&member.ts, self.pending_ts))));
        }
        best
    }

    // ---- the instant clock (`instants.py`) ----

    fn instant_observe(&mut self, m: usize, closes: i64) -> Step<()> {
        self.delivered_through[m] = Some(self.clock);
        match self.label {
            Some(pending) if closes > pending => self.decide_pending(),
            _ => Ok(()),
        }
    }

    fn instant_take(&mut self, m: usize, label: i64, ohlc: [f64; 4]) -> Step<()> {
        self.carry[m] = Some(ohlc);
        if self.decided.is_some_and(|d| label <= d) || self.label.is_some_and(|p| label < p) {
            return Ok(());
        }
        if self.label.is_none() {
            self.label = Some(label);
        }
        if self.bars[m].is_none() {
            self.bar_count += 1;
        }
        self.bars[m] = Some(ohlc);
        if self.bar_count == self.plan.members.len() {
            self.decide_pending()?;
        }
        Ok(())
    }

    fn decide_pending(&mut self) -> Step<()> {
        let n = self.plan.members.len();
        let label = self.label.take();
        let bars = std::mem::replace(&mut self.bars, vec![None; n]);
        self.bar_count = 0;
        let Some(label) = label else { return Ok(()) };
        self.decided = Some(label);
        let mut members = Vec::with_capacity(n);
        let mut carried = vec![false; n];
        for (place, &m) in self.plan.policy.iter().enumerate() {
            match (bars[m], self.carry[m]) {
                (Some(ohlc), _) => members.push(ohlc),
                (None, Some(last)) => {
                    members.push([last[3]; 4]);
                    carried[place] = true;
                }
                (None, None) => return Ok(()),
            }
        }
        let ordinal = self.ordinal;
        self.ordinal += 1;
        let weights = match &mut self.mode {
            Mode::Capture(instants) => {
                instants.push(Instant { label, bars: members, carried });
                return Ok(());
            }
            Mode::Run { captured, decisions, next } => {
                let decisions: &'a [(usize, Vec<f64>)] = decisions;
                // The cell's decisions were taken on the captured instant:
                // the run must meet it as it was, bar for bar.
                let same = captured.get(ordinal).is_some_and(|at| {
                    at.label == label && at.bars == members && at.carried == carried
                });
                if !same {
                    return refuse("the run's instants diverged from the capture");
                }
                match decisions.get(*next) {
                    Some((at, weights)) if *at == ordinal => {
                        *next += 1;
                        weights
                    }
                    _ => return Ok(()),
                }
            }
        };
        let now = self.clock;
        let mut marks = vec![0.0; n];
        for (place, &m) in self.plan.policy.iter().enumerate() {
            marks[m] = members[place][3];
        }
        self.deferring = Some(now);
        self.proceeds = 0.0;
        self.marks = Some(marks);
        let done = self.instant_rebalance(now, weights);
        self.marks = None;
        self.deferring = None;
        done?;
        for place in 0..n {
            let m = self.plan.policy[place];
            if self.delivered_through[m].unwrap_or(-1) >= now {
                self.release_held(m, now)?;
            }
        }
        Ok(())
    }

    // ---- the rebalance (`rebalance.py`, `planning.py`) ----

    /// `_mark_price`.
    fn mark(&self, m: usize) -> f64 {
        if let Some(marks) = &self.marks {
            return marks[m];
        }
        match self.exec_price[m] {
            Some(price) if price != 0.0 => price,
            _ => self.last_close[m].unwrap_or(0.0),
        }
    }

    fn net_liquidation(&self) -> f64 {
        let held: Vec<f64> = self.plan.policy.iter().map(|&m| self.pos[m] * self.mark(m)).collect();
        self.cash + py_sum(&held)
    }

    fn instant_rebalance(&mut self, now: i64, weights: &[f64]) -> Step<()> {
        self.pending_buys.clear();
        self.do_rebalance(weights)?;
        self.pending_ts = now;
        Ok(())
    }

    fn do_rebalance(&mut self, weights: &[f64]) -> Step<()> {
        self.cancel_resting_entries();
        let plan = self.plan;
        let mut clean = vec![0.0; plan.members.len()];
        let mut kept = Vec::new();
        for (place, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() || weight < 0.0 {
                return refuse("invalid target weights");
            }
            if weight > 0.0 {
                clean[plan.policy[place]] = weight;
                kept.push(weight);
            }
        }
        if py_sum(&kept) > 1.0 + 1e-9 {
            return refuse("invalid target weights");
        }
        let net_liq = self.net_liquidation();
        if net_liq <= 0.0 {
            return Ok(());
        }
        let mut targets = vec![0.0; plan.members.len()];
        for &m in &plan.policy {
            let weight = clean[m];
            if weight != 0.0 {
                targets[m] = weight * (plan.percent / 100.0) * net_liq;
            }
        }
        self.supersede_exits();
        self.rebalance_to_target(&targets)?;
        self.targets = Some(targets);
        Ok(())
    }

    fn cancel(&mut self, entries: bool) {
        for order in &mut self.orders {
            if order.reduce_only == entries || order.pending_cancel {
                continue;
            }
            let idx = self.cursor[order.member];
            self.session.kernel_mut(order.member).cancel_order(idx, order.engine_id);
            order.pending_cancel = true;
            self.routed = true;
        }
    }

    /// `_cancel_resting_entries`, with `_drop_held_entries`.
    fn cancel_resting_entries(&mut self) {
        for queue in &mut self.held {
            queue.retain(|held| matches!(held, Held::Exit { .. }));
        }
        self.cancel(true);
    }

    /// `_supersede_exit_execution`.
    fn supersede_exits(&mut self) {
        self.cancel(false);
    }

    fn rebalance_to_target(&mut self, targets: &[f64]) -> Step<()> {
        let plan = self.plan;
        let marks: Vec<f64> = (0..plan.members.len()).map(|m| self.mark(m)).collect();
        let positions = self.pos.clone();
        let mut reductions = Vec::new();
        for &m in &plan.policy {
            let mark = marks[m];
            if mark == 0.0 {
                continue;
            }
            let (position, target) = (positions[m], targets[m]);
            if position > 0.0 {
                let excess = position * mark - target;
                if excess > py_max(0.05 * target, MIN_TRADE) {
                    reductions.push((m, false, py_min(excess / mark, position)));
                }
            } else if position < 0.0 && target <= 0.0 {
                reductions.push((m, true, position.abs()));
            }
        }
        let mut entries = Vec::new();
        for &m in &plan.policy {
            let (target, mark) = (targets[m], marks[m]);
            if target <= 0.0 || mark == 0.0 {
                continue;
            }
            let need = target - positions[m] * mark;
            if need > py_max(0.05 * target, MIN_TRADE) {
                entries.push((m, need));
            }
        }
        for (m, buy, quantity) in reductions {
            self.submit(m, buy, quantity, true, None, None)?;
        }
        self.pending_buys = entries;
        self.decide_entry_budget()
    }

    fn decide_entry_budget(&mut self) -> Step<()> {
        if self.pending_buys.is_empty() {
            self.budget = None;
            self.scale = None;
            return Ok(());
        }
        let available = py_max(0.0, self.cash + self.proceeds);
        let mut fees = Vec::with_capacity(self.pending_buys.len());
        for &(m, need) in &self.pending_buys {
            fees.push(self.commission_estimate(m, need)?);
        }
        let room = py_max(0.0, available - py_sum(&fees));
        let needs: Vec<f64> = self.pending_buys.iter().map(|&(_, need)| need).collect();
        let wanted = py_sum(&needs);
        self.budget = Some(available);
        self.scale = (wanted > 0.0).then(|| py_min(1.0, room / wanted));
        Ok(())
    }

    /// `_order_commission_estimate`.
    fn commission_estimate(&self, m: usize, notional: f64) -> Step<f64> {
        let price = self.mark(m);
        if notional <= 0.0 || price == 0.0 || price <= 0.0 {
            return Ok(0.0);
        }
        let fee = Dec::of(notional / price)
            .zip(Dec::of(price))
            .and_then(|(q, p)| self.plan.members[m].cost.fee(q, p));
        Ok(need(fee, "a fee past the decimal digits")?.to_f64())
    }

    fn release_held(&mut self, m: usize, ts: i64) -> Step<()> {
        if !self.held[m].iter().any(|held| held.decided() <= ts) {
            return Ok(());
        }
        let (ready, remaining): (Vec<Held>, Vec<Held>) =
            std::mem::take(&mut self.held[m]).into_iter().partition(|held| held.decided() <= ts);
        self.held[m] = remaining;
        for held in ready {
            match held {
                Held::Exit { buy, quantity, .. } => {
                    self.submit(m, buy, quantity, true, None, None)?;
                }
                Held::Entry { buy, quantity, price, .. } => {
                    self.submit(m, buy, quantity, false, None, Some(price))?;
                }
            }
        }
        Ok(())
    }

    /// `_open_entry_notional`.
    fn open_entry_notional(&self, m: usize) -> f64 {
        let open: Vec<f64> = self
            .orders
            .iter()
            .filter(|order| order.member == m && !order.reduce_only)
            .map(|order| py_max(order.quantity, 0.0) * order.limit)
            .collect();
        py_sum(&open)
    }

    /// `_get_pending_buy_cost`.
    fn pending_buy_cost(&self) -> Step<f64> {
        let mut costs = Vec::new();
        for order in self.orders.iter().filter(|order| order.buy) {
            let cost = Dec::of(order.quantity)
                .zip(Dec::of(order.limit))
                .and_then(|(q, p)| self.plan.members[order.member].cost.buy_cost(q, p));
            costs.push(need(cost, "a cost past the decimal digits")?.to_f64());
        }
        Ok(py_sum(&costs))
    }

    /// `_entry_notional_is_tradeable`.
    fn tradeable(&self, m: usize, notional: f64) -> Step<bool> {
        let mark = self.mark(m);
        if mark <= 0.0 || notional <= 0.0 {
            return Ok(false);
        }
        let member = &self.plan.members[m];
        let raw = need(
            Dec::of(notional).zip(Dec::of(mark)).and_then(|(n, p)| n.div28(p)),
            "a quantity past the decimal digits",
        )?;
        Ok(member.floor_qty(raw)?.is_some_and(|units| units > 0))
    }

    fn flush_pending_buys(&mut self, m: usize) -> Step<()> {
        let mut consumed = Vec::new();
        let mut buys = Vec::new();
        let mark = self.mark(m);
        let target = self.targets.as_ref().map_or(0.0, |targets| targets[m]);
        if !(mark <= 0.0) && !(target <= 0.0) {
            consumed.push(m);
            let gap = target - self.pos[m] * mark;
            let tolerance = py_max(0.05 * target, MIN_TRADE);
            if gap > tolerance {
                let residual = gap - self.open_entry_notional(m);
                let tradeable = !(residual > tolerance) || self.tradeable(m, residual)?;
                if tradeable && residual > tolerance {
                    buys.push((m, residual));
                }
            }
        }
        if !buys.is_empty() {
            let outstanding: Vec<(usize, f64)> =
                self.pending_buys.iter().copied().filter(|(j, _)| !consumed.contains(j)).collect();
            let budget = self.budget;
            let committed = self.submit_budgeted_buys(buys, &outstanding, budget, self.scale)?;
            if let Some(budget) = budget {
                self.budget = Some(py_max(0.0, budget - py_sum(&committed)));
            }
        }
        self.pending_buys.retain(|(j, _)| !consumed.contains(j));
        Ok(())
    }

    fn submit_budgeted_buys(
        &mut self,
        buys: Vec<(usize, f64)>,
        outstanding: &[(usize, f64)],
        budget: Option<f64>,
        scale: Option<f64>,
    ) -> Step<Vec<f64>> {
        let plan = self.plan;
        let mut requested: Vec<(usize, f64)> = buys
            .into_iter()
            .filter(|&(m, value)| value > MIN_TRADE && self.mark(m) != 0.0)
            .collect();
        requested.sort_by(|a, b| {
            (-a.1)
                .partial_cmp(&-b.1)
                .unwrap_or(Ordering::Equal)
                .then(plan.members[a.0].str_rank.cmp(&plan.members[b.0].str_rank))
        });
        let deferred: Vec<(usize, f64)> = outstanding
            .iter()
            .copied()
            .filter(|&(m, value)| {
                !requested.iter().any(|&(r, _)| r == m) && value > MIN_TRADE && self.mark(m) != 0.0
            })
            .collect();
        let values = |list: &[(usize, f64)]| list.iter().map(|&(_, v)| v).collect::<Vec<f64>>();
        let total = py_sum(&values(&requested)) + py_sum(&values(&deferred));
        let pending_cost = if requested.is_empty() { 0.0 } else { self.pending_buy_cost()? };
        let mut available = py_max(0.0, self.cash - pending_cost);
        let mut remaining = match budget {
            None => available,
            Some(budget) => py_max(0.0, budget),
        };
        let mut fees = Vec::with_capacity(requested.len() + deferred.len());
        for &(m, value) in requested.iter().chain(&deferred) {
            fees.push(self.commission_estimate(m, value)?);
        }
        let scale = match scale {
            Some(scale) => scale,
            None => {
                let room = remaining - py_sum(&fees);
                if total > 0.0 && room > 0.0 {
                    py_min(1.0, room / total)
                } else {
                    0.0
                }
            }
        };
        let mut committed = Vec::new();
        if total <= 0.0 || remaining <= MIN_TRADE || scale <= 0.0 {
            return Ok(committed);
        }
        for (place, &(m, value)) in requested.iter().enumerate() {
            if remaining <= MIN_TRADE {
                break;
            }
            let price = self.mark(m);
            let desired = value * scale;
            let mut cap = if budget.is_none() { remaining } else { py_min(remaining, available) };
            if budget.is_some() {
                cap = py_min(cap, desired + fees[place]);
            }
            let spent =
                self.submit(m, true, py_min(desired, cap) / price, false, Some(cap), None)?;
            if !(spent <= 0.0) {
                committed.push(spent);
            }
            remaining -= spent;
            available -= spent;
        }
        Ok(committed)
    }

    /// `_cap_to_free_base_for`.
    fn cap_to_free_base(&self, m: usize, quantity: f64) -> Step<f64> {
        let member = &self.plan.members[m];
        let Some(precision) = member.base_precision else { return Ok(quantity) };
        let free = need(
            money_as_double(py_max(0.0, self.pos[m]), precision),
            "money past the fixed point",
        )?;
        let (asked, free) = need(
            Dec::of(py_max(0.0, quantity)).zip(Dec::of(free)),
            "a quantity past the decimal digits",
        )?;
        let capped = match free.compare(asked) {
            Some(Ordering::Less) => free,
            Some(_) => asked,
            None => return refuse("a quantity past the decimal digits"),
        };
        let units = need(capped.floor_div(member.step), "a quantity past the decimal digits")?;
        if units <= 0 {
            return Ok(0.0);
        }
        Ok(need(
            Dec::checked(units, 0).and_then(|u| u.mul(member.step)),
            "a quantity past the decimal digits",
        )?
        .to_f64())
    }

    fn accepts(&self, m: usize, ts: i64) -> Step<bool> {
        let Some(gate) = &self.plan.members[m].gate else { return Ok(true) };
        let union = &self.plan.union;
        let at = left(union, ts);
        if at < union.len() && union[at] == ts {
            Ok(gate[at])
        } else {
            refuse("an order at an instant the gate was not compiled for")
        }
    }

    /// `NativePortfolioView._submit`, long only: what a buy commits.
    fn submit(
        &mut self,
        m: usize,
        buy: bool,
        quantity: f64,
        reduce_only: bool,
        cash_cap: Option<f64>,
        limit_price: Option<f64>,
    ) -> Step<f64> {
        if reduce_only {
            if let Some(decided) = self.deferring {
                let price = self.mark(m);
                if !(price <= 0.0) {
                    let notional = quantity.abs() * price;
                    self.proceeds += if buy { -notional } else { notional };
                }
                self.held[m].push(Held::Exit { decided, buy, quantity });
                return Ok(0.0);
            }
        }
        if !self.accepts(m, self.clock)? {
            return Ok(0.0);
        }
        let plan = self.plan;
        let member = &plan.members[m];
        let mark = self.mark(m);
        if mark <= 0.0 || quantity <= 0.0 {
            return Ok(0.0);
        }
        let price_units = match limit_price {
            Some(limit) => member.grid.price_units(limit),
            None => member.grid.snap(mark, buy).and_then(|price| member.grid.price_units(price)),
        };
        let price_units = need(price_units, "an order price the grid does not snap")?;
        let price_value = need(
            units_as_double(price_units, member.price_precision),
            "a price past the fixed point",
        )?;
        let price = Dec { c: i128::from(price_units), e: -(member.price_precision as i32) };
        let cost = &member.cost;
        let mut quantity = quantity;
        if let (Some(cap), true) = (cash_cap, buy) {
            let affordable = need(
                cost.affordable_quantity_at(py_max(0.0, cap), price),
                "a cost past the decimal digits",
            )?;
            quantity = py_min(quantity, affordable.to_f64());
        }
        if reduce_only && !buy {
            quantity = self.cap_to_free_base(m, quantity)?;
        }
        if quantity <= 0.0 {
            return Ok(0.0);
        }
        let units = if !reduce_only || cash_cap.is_some() {
            member.floor_qty(need(Dec::of(quantity), "a quantity past the decimal digits")?)?
        } else {
            member.make_qty(quantity)?
        };
        let Some(units) = units else { return Ok(0.0) };
        let quantity_value =
            need(units_as_double(units, member.size_precision), "a quantity past the fixed point")?;
        if quantity_value <= 0.0 {
            return Ok(0.0);
        }
        let units_dec = member.quantity(units);
        let committed =
            need(cost.buy_cost(units_dec, price), "a cost past the decimal digits")?.to_f64();
        if let (Some(cap), true) = (cash_cap, buy) {
            if !need(cost.can_afford(cap, units_dec, price), "a cost past the decimal digits")? {
                return Ok(0.0);
            }
        }
        if member.min_notional.is_some_and(|minimum| quantity_value * mark < minimum) {
            return Ok(0.0);
        }
        let spent = if buy { committed } else { 0.0 };
        if !reduce_only {
            if let Some(decided) = self.deferring {
                self.held[m].push(Held::Entry {
                    decided,
                    buy,
                    quantity: quantity_value,
                    price: price_value,
                });
                return Ok(spent);
            }
        }
        if buy && !reduce_only {
            let available = self.cash - self.pending_buy_cost()?;
            if !need(
                cost.can_afford(available, units_dec, price),
                "a cost past the decimal digits",
            )? {
                return Ok(0.0);
            }
        }
        self.seq += 1;
        let seq = self.seq;
        let client_id = format!("NP-{seq}");
        let (tif, expire_ns) = if reduce_only {
            (member.urgent_tif.as_str(), None)
        } else {
            ("gtd", Some(self.clock + member.ttl_ns))
        };
        let submitted_idx = self.session.submission_idx(m, self.cursor[m]);
        let engine_id = submit_order_on(
            self.session.kernel_mut(m),
            if buy { "buy" } else { "sell" },
            "limit",
            submitted_idx,
            self.clock,
            &client_id,
            Some(quantity_value),
            None,
            Some(price_value),
            None,
            tif,
            expire_ns,
            None,
            None,
            None,
            "price",
            0.0,
            false,
            reduce_only,
            m as i64 != self.current,
            None,
        )
        .map_err(|_| "the kernel refused an order".to_string())?;
        self.orders.push(Order {
            seq,
            member: m,
            buy,
            quantity: quantity_value,
            reduce_only,
            engine_id,
            limit: price_value,
            pending_cancel: false,
        });
        self.submits.push((seq, m, buy, quantity_value, price_value, reduce_only, self.clock));
        self.routed = true;
        if !self.arrivals.contains(&m) {
            self.arrivals.push(m);
        }
        Ok(spent)
    }
}

/// A cell's run, as the caller books it.
struct Ran {
    outcome: SessionOutcome,
    fills: usize,
    submits: Vec<Submit>,
    records: Vec<Record>,
}

fn run_cell(plan: &Plan, captured: &[Instant], decisions: &[(usize, Vec<f64>)]) -> Step<Ran> {
    let mut book = Book::new(plan, Mode::Run { captured, decisions, next: 0 });
    book.run()?;
    if let Mode::Run { next, .. } = book.mode {
        if next != decisions.len() {
            return refuse("a decision the run never reached");
        }
    }
    let Book { session, fills, submits, records, .. } = book;
    Ok(Ran { outcome: session.finish(), fills, submits, records })
}

/// A long-only cash book of members trading on complete instants, replayed
/// for many cells at once: built from the book's sealed `PortfolioSession`
/// (left untouched: each cell runs on a fresh copy of its recipe), one
/// [`add_member`](PyPortfolioReplay::add_member) per member in the
/// session's order, then [`seal`](PyPortfolioReplay::seal).
#[pyclass(name = "PortfolioReplay", module = "raptorbt")]
pub struct PyPortfolioReplay {
    building: Option<Plan>,
    plan: Option<Arc<Plan>>,
    captured: Option<Arc<Vec<Instant>>>,
}

impl PyPortfolioReplay {
    fn plan(&self) -> PyResult<&Arc<Plan>> {
        self.plan.as_ref().ok_or_else(|| PyValueError::new_err("seal the replay first"))
    }
}

fn slice<T: Copy + numpy::Element>(array: &PyReadonlyArray1<'_, T>) -> PyResult<Vec<T>> {
    Ok(array.as_slice()?.to_vec())
}

#[pymethods]
impl PyPortfolioReplay {
    /// *union*: every stamp an order may be submitted at, ascending -- the
    /// members' bars and the windows the clock delivers. *policy_order*:
    /// each of the policy's instruments' session index, in its order.
    #[new]
    #[pyo3(signature = (
        session, union, *, starting_capital, currency_precision, sizer_percent, policy_order
    ))]
    fn new(
        session: PyRef<'_, PyPortfolioSession>,
        union: PyReadonlyArray1<'_, i64>,
        starting_capital: f64,
        currency_precision: u32,
        sizer_percent: f64,
        policy_order: Vec<usize>,
    ) -> PyResult<Self> {
        let recipe = session.recipe().cloned().ok_or_else(|| {
            PyValueError::new_err("the session was touched beyond its setup; build a new one")
        })?;
        let union = slice(&union)?;
        if union.windows(2).any(|w| w[0] >= w[1]) {
            return Err(PyValueError::new_err("union must ascend strictly"));
        }
        let mut seen = vec![false; recipe.len()];
        for &m in &policy_order {
            if m >= seen.len() || std::mem::replace(&mut seen[m], true) {
                return Err(PyValueError::new_err("policy_order must name each member once"));
            }
        }
        if seen.iter().any(|&s| !s) {
            return Err(PyValueError::new_err("policy_order must name each member once"));
        }
        let starting_cash = money_as_double(starting_capital, currency_precision)
            .ok_or_else(|| PyValueError::new_err("starting capital past the fixed point"))?;
        Ok(Self {
            building: Some(Plan {
                members: Vec::new(),
                policy: policy_order,
                union,
                clock: Vec::new(),
                starting_cash,
                currency_precision,
                percent: if sizer_percent > 0.0 { sizer_percent } else { 100.0 },
                recipe,
            }),
            plan: None,
            captured: None,
        })
    }

    /// The next member, in the session's order: its primary bars' stamps,
    /// admission (None: every bar) and execution marks; *gate* over the
    /// replay's union (None: always open); its signal leg's rows; its money
    /// model's cost and grid; and the order terms `_submit` reads.
    #[pyo3(signature = (
        timestamps, admitted, exec_mark, gate, labels, index, delivered, timer_fired, clocked,
        ohlc, money, *, price_precision, size_increment_precision, size_precision,
        min_notional, base_precision, urgent_tif, ttl_ns, str_rank, name_rank
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_member(
        &mut self,
        timestamps: PyReadonlyArray1<'_, i64>,
        admitted: Option<PyReadonlyArray1<'_, bool>>,
        exec_mark: PyReadonlyArray1<'_, f64>,
        gate: Option<PyReadonlyArray1<'_, bool>>,
        labels: PyReadonlyArray1<'_, i64>,
        index: PyReadonlyArray1<'_, i64>,
        delivered: PyReadonlyArray1<'_, i64>,
        timer_fired: PyReadonlyArray1<'_, bool>,
        clocked: PyReadonlyArray1<'_, bool>,
        ohlc: PyReadonlyArray2<'_, f64>,
        money: PyRef<'_, PyMoneyModel>,
        price_precision: u32,
        size_increment_precision: u32,
        size_precision: u32,
        min_notional: Option<f64>,
        base_precision: Option<u32>,
        urgent_tif: String,
        ttl_ns: i64,
        str_rank: usize,
        name_rank: usize,
    ) -> PyResult<()> {
        let plan =
            self.building.as_mut().ok_or_else(|| PyValueError::new_err("the replay is sealed"))?;
        if plan.members.len() >= plan.recipe.len() {
            return Err(PyValueError::new_err("more members than the session has"));
        }
        let bad = |why: &str| PyValueError::new_err(why.to_string());
        let ts = slice(&timestamps)?;
        let count = ts.len();
        let admitted = match admitted {
            Some(mask) => slice(&mask)?,
            None => vec![true; count],
        };
        let exec_mark = slice(&exec_mark)?;
        if admitted.len() != count || exec_mark.len() != count {
            return Err(bad("admitted and exec_mark must match timestamps"));
        }
        let gate = gate.map(|g| slice(&g)).transpose()?;
        if gate.as_ref().is_some_and(|g| g.len() != plan.union.len()) {
            return Err(bad("gate must match the union"));
        }
        let mut next_admitted = vec![count; count + 1];
        for k in (0..count).rev() {
            next_admitted[k] = if admitted[k] { k } else { next_admitted[k + 1] };
        }
        let mut last_admitted = vec![NONE; count + 1];
        for k in 0..count {
            last_admitted[k + 1] = if admitted[k] { k } else { last_admitted[k] };
        }
        let labels = slice(&labels)?;
        let rows = labels.len();
        let index: Vec<usize> = slice(&index)?
            .into_iter()
            .map(|i| usize::try_from(i).ok().filter(|&i| i <= count))
            .collect::<Option<_>>()
            .ok_or_else(|| bad("a signal row's index outside its bars"))?;
        let ohlc = ohlc.as_array();
        if ohlc.shape() != [rows, 4] {
            return Err(bad("ohlc must be (rows, 4)"));
        }
        let ohlc: Vec<[f64; 4]> =
            ohlc.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect();
        let leg = Leg {
            labels,
            index,
            delivered: slice(&delivered)?,
            timer_fired: slice(&timer_fired)?,
            clocked: slice(&clocked)?,
            ohlc,
        };
        if [leg.index.len(), leg.delivered.len(), leg.timer_fired.len(), leg.clocked.len()]
            .iter()
            .any(|&len| len != rows)
        {
            return Err(bad("a signal leg's columns must match"));
        }
        if size_precision > 16 || price_precision > 16 {
            return Err(bad("a precision past the fixed point"));
        }
        if money.grid().price_precision() != Some(price_precision) {
            return Err(bad("price_precision must be the money model's"));
        }
        plan.members.push(Member {
            ts,
            admitted,
            next_admitted,
            last_admitted,
            exec_mark,
            gate,
            leg,
            cost: money.cost().clone(),
            grid: money.grid().clone(),
            price_precision,
            size_increment_precision,
            size_precision,
            step: Dec { c: 1, e: -(size_precision as i32) },
            min_quantity: 10_f64.powf(-f64::from(size_increment_precision)) * 0.1,
            min_notional,
            base_precision,
            urgent_tif,
            ttl_ns,
            str_rank,
            name_rank,
        });
        Ok(())
    }

    /// Compile the clock and freeze the book.
    fn seal(&mut self) -> PyResult<()> {
        let mut plan =
            self.building.take().ok_or_else(|| PyValueError::new_err("the replay is sealed"))?;
        if plan.members.len() != plan.recipe.len() {
            self.building = Some(plan);
            return Err(PyValueError::new_err("a member short of the session"));
        }
        let mut events = Vec::new();
        for (instrument, member) in plan.members.iter().enumerate() {
            for row in 0..member.leg.labels.len() {
                if member.leg.clocked[row] {
                    events.push(ClockEvent {
                        ts: member.leg.delivered[row],
                        timer: member.leg.timer_fired[row],
                        instrument,
                        row,
                    });
                }
            }
        }
        let rank = |event: &ClockEvent| {
            if event.timer {
                plan.members[event.instrument].name_rank
            } else {
                event.instrument
            }
        };
        // `np.lexsort((rank, phase, ts))` is stable.
        events.sort_by_key(|event| (event.ts, event.timer, rank(event)));
        plan.clock = events;
        self.plan = Some(Arc::new(plan));
        Ok(())
    }

    /// The instants the book decides on, before any order: `(labels, bars,
    /// carried)` -- each instant's label, every member's `(open, high, low,
    /// close)` there in the policy's order (flat, `instants * members * 4`),
    /// and which were carried (flat, `instants * members`).
    #[allow(clippy::type_complexity)]
    fn capture(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(Py<numpy::PyArray1<i64>>, Py<numpy::PyArray1<f64>>, Py<numpy::PyArray1<bool>>)>
    {
        let plan = Arc::clone(self.plan()?);
        let instants = py
            .allow_threads(|| {
                let mut book = Book::new(&plan, Mode::Capture(Vec::new()));
                book.run()?;
                match book.mode {
                    Mode::Capture(instants) => Ok(instants),
                    Mode::Run { .. } => refuse("capture ran in run mode"),
                }
            })
            .map_err(LeanDeclined::new_err)?;
        let labels: Vec<i64> = instants.iter().map(|i| i.label).collect();
        let bars: Vec<f64> =
            instants.iter().flat_map(|i| i.bars.iter().flatten().copied()).collect();
        let carried: Vec<bool> = instants.iter().flat_map(|i| i.carried.iter().copied()).collect();
        self.captured = Some(Arc::new(instants));
        Ok((
            labels.into_pyarray(py).into(),
            bars.into_pyarray(py).into(),
            carried.into_pyarray(py).into(),
        ))
    }

    /// Run *cells*, each `(ordinals, weights)`: the captured instants it
    /// decides at, ascending, and its weights there, a row each in the
    /// policy's order. On *threads* threads (1: this one; 0: rayon's pool)
    /// with the GIL released. Returns per cell `(result, fills, submits,
    /// records)` -- the session's `PortfolioResult`, its fill count, each
    /// order as submitted `(seq, member, buy, quantity, price, reduce_only,
    /// ts_init)` and what the venue did with them, in order, `(seq, ts,
    /// kind, order_id, size, price, leaves, realized, commission)` with
    /// `kind` 0 a fill, 1 canceled, 2 expired -- or a `str` naming why the
    /// cell leaves the path this replay follows.
    #[pyo3(signature = (cells, *, threads = 0))]
    fn run(
        &self,
        py: Python<'_>,
        cells: Vec<(PyReadonlyArray1<'_, i64>, PyReadonlyArray2<'_, f64>)>,
        threads: usize,
    ) -> PyResult<Vec<PyObject>> {
        let plan = Arc::clone(self.plan()?);
        let captured = Arc::clone(
            self.captured
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("capture the book first"))?,
        );
        let width = plan.members.len();
        let mut tables = Vec::with_capacity(cells.len());
        for (ordinals, weights) in &cells {
            let ordinals = ordinals.as_slice()?;
            let weights = weights.as_array();
            if weights.shape() != [ordinals.len(), width] {
                return Err(PyValueError::new_err("weights must be (ordinals, members)"));
            }
            let mut table = Vec::with_capacity(ordinals.len());
            for (row, &ordinal) in weights.rows().into_iter().zip(ordinals) {
                let ordinal = usize::try_from(ordinal)
                    .map_err(|_| PyValueError::new_err("a negative ordinal"))?;
                if table.last().is_some_and(|&(last, _): &(usize, Vec<f64>)| last >= ordinal) {
                    return Err(PyValueError::new_err("ordinals must ascend strictly"));
                }
                table.push((ordinal, row.to_vec()));
            }
            tables.push(table);
        }
        let pool = if threads > 1 { Some(threads_pool(threads)?) } else { None };
        let ran: Vec<Step<Ran>> = py.allow_threads(|| {
            let each = |table: &Vec<(usize, Vec<f64>)>| run_cell(&plan, &captured, table);
            if threads == 1 {
                tables.iter().map(each).collect()
            } else {
                in_pool(&pool, || tables.par_iter().map(each).collect())
            }
        });
        ran.into_iter()
            .map(|ran| match ran {
                Err(why) => Ok(why.into_py(py)),
                Ok(Ran { outcome, fills, submits, records }) => {
                    let result: PyPortfolioResult = portfolio_result(outcome);
                    Ok((Py::new(py, result)?, fills, submits, records).into_py(py))
                }
            })
            .collect()
    }
}
