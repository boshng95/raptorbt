//! A library cell's replay: `python::replay_session`'s driver loop and its
//! `SignalReplay`, over the lean kernel and rule cursor.
//!
//! Branch for branch the same as that module, which is itself a
//! transcription of `raptorbt/strategy/runner.py` and the parity runner's
//! `SignalReplay`; the doc comments there say which line each branch is.
//! What differs is underneath: the kernel is [`super::kernel`], the rules
//! [`super::rules`], and the loop jumps the stretches that cannot act -- an
//! idle kernel and no callback owed until the next due bar, and from a flat
//! or held book that nothing can disturb, until the next resample bar that
//! enters or exits. Each jump leaves the state stepping every bar would, and
//! `Cell::jumps` turns them off to diff against that loop. The money
//! questions are the caller's `MoneyModel`; one it leaves to Python declines
//! the cell.

use super::kernel::{Event, Kernel, Terms, Tif, Trade};
use super::market::Market;
use std::cell::RefCell;

use super::rules::{segment_highs, side_of, Cursor, Dispatch, Highs, Stream, Target};
use super::Declined;
use crate::python::fill_ledger::{BookTerms, EndStatus, Ledger, OrderRecord};
use crate::python::money::PyMoneyModel;
use crate::python::replay_session::Venue;
use crate::python::rule_session::{money_as_double, QuietParams};

/// What the cells on one resampled stream of a market share, checked once
/// against it: the bars `on_bar` is due on and the stream's dispatches.
pub struct Frame {
    /// The due bars' positions, rising.
    pub due: Vec<usize>,
    pub dispatch: Dispatch,
    /// Whether each dispatch is due on the bar stamped with it, so the bars
    /// of a stretch the cursor passes are the ones its visits were due on.
    pub dispatches_due: bool,
    /// Per bar, the highest close from it to the first dispatch after it:
    /// what a dispatch window asks [`Highs`] for, read once for every cell.
    pub highs: Vec<f64>,
}

impl Frame {
    pub fn new(m: &Market, due: Vec<usize>, dispatch: Dispatch) -> Result<Self, Declined> {
        let n = m.len();
        if !rising_below(&due, n) {
            return Err(Declined("due bars that do not rise within the bars"));
        }
        if dispatch.pos.iter().any(|&at| at >= n) {
            return Err(Declined("a dispatch past the last bar"));
        }
        let mut bits = vec![false; n];
        for &bar in &due {
            bits[bar] = true;
        }
        let dispatches_due =
            dispatch.pos.iter().zip(&dispatch.ts).all(|(&at, &ts)| bits[at] && m.ts[at] == ts);
        let highs = segment_highs(&m.close, &dispatch.pos);
        Ok(Self { due, dispatch, dispatches_due, highs })
    }
}

/// What a cell runs with besides its market, terms and rules.
pub(crate) struct Cell<'a> {
    /// The frame the cell's stream is on, of the cell's market.
    pub frame: &'a Frame,
    /// The bars an order may be sent on, where the venue keeps hours.
    pub gate: Option<&'a [bool]>,
    pub quiet: QuietParams,
    pub venue: &'a Venue,
    pub money: &'a PyMoneyModel,
    /// The bars the daily curve keeps: [`Market::day_ends`] under the
    /// config's `performance_tz_transitions`.
    pub day_ends: &'a [usize],
    /// Whether to jump the bars that cannot act.
    pub jumps: bool,
}

/// A replay's records: the ledger it booked, the kernel's round trips, the
/// daily curve and the run's end value.
pub(crate) struct Outcome {
    pub ledger: Ledger,
    pub trades: Vec<Trade>,
    pub day_ns: Vec<i64>,
    pub day_equity: Vec<f64>,
    pub end_value: f64,
    /// How the loop went: bars stepped one by one, jumps taken, callbacks.
    pub stepped: usize,
    pub jumped: usize,
    pub callbacks: usize,
}

/// One order `SignalReplay` queued, as `apply_commands` sends it.
struct Submit {
    ledger_index: u32,
    buy: bool,
    units: f64,
    /// None for a market order.
    limit: Option<f64>,
    tif: Tif,
    reduce_only: bool,
}

struct Replay<'a> {
    m: &'a Market,
    k: Kernel<'a>,
    s: &'a Stream<'a>,
    cursor: Cursor,
    mask_at: &'a [usize],
    /// The frame's [`Frame::highs`].
    window_highs: &'a [f64],
    day_ends: &'a [usize],
    gate: Option<&'a [bool]>,
    quiet: QuietParams,
    venue: &'a Venue,
    money: &'a PyMoneyModel,

    // `SignalReplay`'s own state, under its own names.
    working_entry: bool,
    /// The ledger index of each order the kernel holds, by kernel id.
    orders: Vec<u32>,
    pending_target: Option<f64>,
    target_bars: i64,
    target_bar_ts: i64,
    pending_reference: f64,
    last_exit_ts: i64,
    cash_halted: bool,
    working_orders: i64,
    bar_callback_required: bool,
    account_quantity: f64,

    // The driver's.
    order_seq: u64,
    commands: Vec<Submit>,
    budget_ts: Option<i64>,
    budget_actions: usize,

    // The jumps'.
    jumps: bool,
    stepped: usize,
    jumped: usize,
    callbacks: usize,
    /// Whether every dispatch is due on its own bar, so the cursor's visits
    /// can be passed in a jump.
    far: bool,
    /// The flat cash check's need at the highest close, where it bounds
    /// the need at every close.
    need_top: Option<f64>,
    /// The highest close before the next dispatch, which every bar of a
    /// dispatch window asks for again.
    highs: RefCell<Highs>,

    ledger: Ledger,
}

/// Replay `s` on `m` from a fresh account on `terms`, booking into a ledger
/// opened on `book` that keeps the log where `log`. Without it the ledger is
/// a tally, orders go unnamed, and the kernel's finished orders are
/// reclaimed between submissions.
pub(crate) fn replay(
    m: &Market,
    terms: &Terms,
    s: &Stream<'_>,
    cell: &Cell<'_>,
    book: Option<BookTerms>,
    log: bool,
) -> Result<Outcome, Declined> {
    if m.is_empty() {
        return Err(Declined("no bars"));
    }
    if !m.rising {
        return Err(Declined("bar timestamps that do not rise"));
    }
    let n = m.len();
    if !std::ptr::eq(s.d, &cell.frame.dispatch) {
        return Err(Declined("a stream on another frame"));
    }
    if !rising_below(cell.day_ends, n) {
        return Err(Declined("day ends that do not rise within the bars"));
    }
    if cell.gate.is_some_and(|gate| gate.len() != n)
        || s.primary_admits.is_some_and(|admits| admits.len() != n)
    {
        return Err(Declined("a gate or primary admits of another length"));
    }
    let q = cell.quiet;
    let far = cell.jumps && cell.frame.dispatches_due;
    let need_top = (q.units >= 0.0 && q.markup >= 0.0)
        .then(|| q.units * (m.top_close * q.markup + q.slack) + q.fee);
    let mut replay = Replay {
        m,
        k: Kernel::new(m, terms),
        s,
        cursor: Cursor::of(s),
        mask_at: &cell.frame.due,
        window_highs: &cell.frame.highs,
        day_ends: cell.day_ends,
        gate: cell.gate,
        quiet: cell.quiet,
        venue: cell.venue,
        money: cell.money,
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
        order_seq: cell.venue.first_order_seq,
        commands: Vec::new(),
        budget_ts: None,
        budget_actions: 0,
        jumps: cell.jumps,
        stepped: 0,
        jumped: 0,
        callbacks: 0,
        far,
        need_top,
        highs: RefCell::default(),
        ledger: match book {
            Some(terms) if !log => Ledger::tally(terms),
            book => Ledger::new(book),
        },
    };
    let (day_ns, day_equity, end_value) = replay.run()?;
    let Replay { k, ledger, stepped, jumped, callbacks, .. } = replay;
    let mut k = k;
    k.finalize(n - 1);
    Ok(Outcome {
        ledger,
        trades: std::mem::take(&mut k.trades),
        day_ns,
        day_equity,
        end_value,
        stepped,
        jumped,
        callbacks,
    })
}

/// How many of the rising `positions` lie below `to`: `partition_point`,
/// searched outward from the front, since a jump passes few.
fn count_below(positions: &[usize], to: usize) -> usize {
    // `positions[..below]` all lie below `to`.
    let (mut below, mut step) = (0, 1);
    while step <= positions.len() && positions[step - 1] < to {
        below = step;
        step *= 2;
    }
    below + positions[below..step.min(positions.len())].partition_point(|&bar| bar < to)
}

/// `format!("{tag}-{seq}")`, into a string allocated once at its length.
fn client_id(tag: &str, seq: u64) -> String {
    let mut digits = [0_u8; 20];
    let mut at = digits.len();
    let mut rest = seq;
    loop {
        at -= 1;
        digits[at] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    let mut id = String::with_capacity(tag.len() + 1 + digits.len() - at);
    id.push_str(tag);
    id.push('-');
    id.extend(digits[at..].iter().map(|&digit| digit as char));
    id
}

fn rising_below(positions: &[usize], n: usize) -> bool {
    positions.windows(2).all(|pair| pair[0] < pair[1]) && positions.last().is_none_or(|&at| at < n)
}

impl Replay<'_> {
    // -- the driver: `run_strategy_backtest` -------------------------------

    /// Every bar, then the daily curve and the end value `finish` reads off
    /// the last bar, before the kernel settles what is still open.
    fn run(&mut self) -> Result<(Vec<i64>, Vec<f64>, f64), Declined> {
        let m = self.m;
        let n = m.len();
        let mut day_ns = Vec::with_capacity(self.day_ends.len());
        let mut day_equity = Vec::with_capacity(self.day_ends.len());
        let mut events = Vec::new();
        let (mut next_due, mut next_day) = (0, 0);
        let mut i = 0;
        while i < n {
            if self.jumps
                && !self.bar_callback_required
                && self.commands.is_empty()
                && self.k.idle()
            {
                let due = self.mask_at.get(next_due).copied().unwrap_or(n);
                let to = self.quiet_until(i, due);
                if to > i {
                    // The bars between only mark the curve, on an account
                    // nothing touches.
                    while let Some(&day) = self.day_ends.get(next_day).filter(|&&day| day < to) {
                        day_ns.push(m.ts[day]);
                        day_equity.push(self.k.equity(m.close[day]));
                        next_day += 1;
                    }
                    next_due += count_below(&self.mask_at[next_due..], to);
                    self.k.pass(to - 1);
                    self.jumped += 1;
                    i = to;
                    continue;
                }
            }
            let due = self.mask_at.get(next_due) == Some(&i);
            if due {
                next_due += 1;
            }
            if due || self.bar_callback_required {
                self.callbacks += 1;
                self.on_bar(i)?;
            }
            self.stepped += 1;
            self.apply_commands(i)?;
            events.clear();
            self.k.step(i, &mut events);
            self.dispatch(i, &events)?;
            self.settle(i, &mut events)?;
            // The curve's instant closes before the next bar steps, on the
            // account every fill of this one left.
            if self.day_ends.get(next_day) == Some(&i) {
                day_ns.push(m.ts[i]);
                day_equity.push(self.k.equity(m.close[i]));
                next_day += 1;
            }
            i += 1;
        }
        let end_value = self.k.equity(m.close[n - 1]);
        Ok((day_ns, day_equity, end_value))
    }

    /// With no callback owed, nothing queued and nothing working at bar
    /// `i`: the first bar from `i` that can act, `due` the next due bar.
    /// The cursor takes the visits of the bars before it.
    fn quiet_until(&mut self, i: usize, due: usize) -> usize {
        if self.far {
            if let Some((r, held)) = self.quiet_visits() {
                let to = self.s.d.pos.get(r).copied().unwrap_or(self.m.len());
                if to > i {
                    self.cursor.pass(self.s, r, held);
                    return to;
                }
            }
        }
        due
    }

    /// The first resample bar whose visit can act, and whether the book is
    /// held through the ones before it; None where a visit before it
    /// might act or owe a callback.
    ///
    /// Flat, the visits see no position, step no exit clock and decide
    /// nothing until one enters, and the cash check passes at every close
    /// on a balance nothing moves. Held, a lot no close reads as dust
    /// keeps every visit's position the book's, so each steps the exit
    /// clock and none acts until one exits.
    fn quiet_visits(&self) -> Option<(usize, bool)> {
        // Level rules step their clocks on every resample bar.
        if self.s.tree.is_some()
            || self.pending_target.is_some()
            || self.working_orders != 0
            || self.working_entry
            || self.cash_halted
            || self.cursor.visit_primary_after_fill
        {
            return None;
        }
        let (has_position, net_position) = self.k.net_position();
        if !has_position {
            let need_top = self.need_top?;
            if self.account_quantity != 0.0 {
                return None;
            }
            let q = self.quiet;
            let mut funds = self.strategy_cash().ok()? * q.fraction;
            if funds > q.cap {
                funds = q.cap;
            }
            return (funds >= need_top && funds >= q.need_at_highest)
                .then(|| (self.cursor.quiet_flat(self.s), false));
        }
        let s = self.s;
        let held = net_position;
        if !(held > 0.0)
            || self.account_quantity != held
            || self.m.low_close.is_nan()
            || self.is_dust(held, self.venue.lowest_close)
            || self.is_dust(held, self.m.low_close)
            || s.d.lowest_close.is_nan()
            || held < s.size_increment
            || (s.minimum_notional > 0.0 && held * s.d.lowest_close < s.minimum_notional)
        {
            return None;
        }
        self.cursor.quiet_held(s).map(|r| (r, true))
    }

    /// `apply_commands`: send what the strategy queued, and say whether it
    /// queued anything.
    fn apply_commands(&mut self, i: usize) -> Result<bool, Declined> {
        let mut commands = std::mem::take(&mut self.commands);
        let now = self.m.ts[i];
        if self.budget_ts != Some(now) {
            self.budget_ts = Some(now);
            self.budget_actions = 0;
        }
        self.budget_actions += commands.len();
        if self.budget_actions > self.venue.max_actions_per_timestamp {
            return Err(Declined("execution did not quiesce within its action budget"));
        }
        // A tally names no order by its kernel id: once none is live, the
        // kernel's orders and the map to them start over, so a replay that
        // trades every minute holds only the orders still working.
        if !commands.is_empty() && !self.ledger.logs() && self.k.reclaim() {
            self.orders.clear();
        }
        for command in &commands {
            let id = self.k.submit(
                i,
                now,
                command.buy,
                command.units,
                command.limit,
                command.tif,
                command.reduce_only,
            );
            debug_assert_eq!(id, self.orders.len());
            self.orders.push(command.ledger_index);
        }
        let queued = !commands.is_empty();
        // Nothing queues while the kernel takes them; the buffer goes back
        // empty, its capacity kept.
        commands.clear();
        self.commands = commands;
        Ok(queued)
    }

    /// `settle`: walk the book for orders the bar's own fills provoked.
    fn settle(&mut self, i: usize, events: &mut Vec<Event>) -> Result<(), Declined> {
        let now = self.m.ts[i];
        while self.apply_commands(i)? {
            events.clear();
            self.k.walk_book(now, events);
            self.dispatch(i, events)?;
        }
        Ok(())
    }

    /// `dispatch_events`, into the hooks `SignalReplay` defines.
    fn dispatch(&mut self, at: usize, events: &[Event]) -> Result<(), Declined> {
        let mut order_fill_preceded = false;
        for &event in events {
            match event {
                Event::Entered | Event::Exited => {
                    if !order_fill_preceded {
                        return Err(Declined("a position event with no order behind it"));
                    }
                    order_fill_preceded = false;
                }
                Event::Accepted => order_fill_preceded = false,
                Event::Filled { id, price, size, commission, leaves, gross } => {
                    self.on_order_filled(at, id, price, size, commission, leaves, gross)?;
                    order_fill_preceded = true;
                }
                Event::Canceled(id) => {
                    self.end_order(at, id, EndStatus::Canceled);
                    order_fill_preceded = false;
                }
                Event::Expired(id) => {
                    self.end_order(at, id, EndStatus::Expired);
                    order_fill_preceded = false;
                }
                Event::Rejected => {
                    self.on_order_rejected();
                    order_fill_preceded = false;
                }
            }
        }
        Ok(())
    }

    // -- the strategy: `SignalReplay` ---------------------------------------

    fn on_bar(&mut self, i: usize) -> Result<(), Declined> {
        if self.cash_halted {
            self.advance(i)?;
            self.bar_callback_required = false;
            return Ok(());
        }
        let reconciled =
            if self.pending_target.is_some() { self.reconcile_target(i)? } else { false };
        let target = self.advance(i)?;
        if !reconciled {
            self.submit_native_intent(i, target)?;
        }
        let (has_position, net_position) = self.k.net_position();
        let flat = !has_position;
        let cash_flat = flat || self.holds_flat_dust(i, has_position, net_position);
        let mut flat_cash_safe = false;
        if !self.cash_halted && cash_flat {
            let price = self.m.close[i];
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
            || self.cursor.visit_primary_after_fill
            || (!flat
                && self.cursor.held_reopens(
                    self.s,
                    &self.m.close,
                    i,
                    has_position,
                    net_position,
                    self.pending_target,
                ));
        Ok(())
    }

    /// `RustRuleSession.advance`.
    fn advance(&mut self, i: usize) -> Result<Option<Target>, Declined> {
        let (mut has_position, mut quantity) = self.k.net_position();
        // Level rules read dust themselves, on the venue's own lot.
        if self.s.tree.is_none()
            && has_position
            && self.reads_dust_as_flat(quantity, self.m.close[i])
        {
            has_position = false;
            quantity = 0.0;
        }
        let (processed, target) = self.cursor.advance(
            self.s,
            &self.m.close,
            self.m.ts[i],
            i,
            has_position,
            quantity,
            self.pending_target,
        )?;
        if processed {
            self.refresh()?;
        }
        Ok(target)
    }

    /// `NativeExecutionSession._refresh`, as far as anything reads it.
    fn refresh(&mut self) -> Result<(), Declined> {
        let (has_position, net_position) = self.k.net_position();
        self.account_quantity = if has_position { net_position } else { 0.0 };
        self.strategy_cash().map(|_| ())
    }

    fn submit_native_intent(&mut self, i: usize, target: Option<Target>) -> Result<(), Declined> {
        let Some(target) = target else {
            return Ok(());
        };
        let target_side = target.side;
        let (has_position, net_position) = self.k.net_position();
        let held = if has_position { net_position } else { 0.0 };
        let mut held_side = side_of(held);
        if held_side != 0 && self.flat_on_dust(has_position, net_position, self.m.close[i]) {
            held_side = 0;
        }
        if target_side == held_side {
            return Ok(());
        }
        let reference = target.reference.unwrap_or(self.m.close[i]);
        let closing = held_side != 0 && target_side != held_side;
        let units = if target_side != 0 { self.quantity(i, reference)? } else { 0.0 };
        if closing {
            if target_side != 0 {
                return Err(Declined("a reversal"));
            }
            self.pending_reference = reference;
            self.target_bars = 0;
            self.target_bar_ts = -1;
            self.pending_target = Some(0.0);
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
            return Err(Declined("a short entry"));
        }
        self.submit_entry(i, reference, units, target.policies_after_fill)?;
        Ok(())
    }

    fn submit_exit(&mut self, i: usize, reference: f64) -> Result<bool, Declined> {
        let decided_ns = self.m.ts[i];
        if self.gate.is_some_and(|gate| !gate[i]) {
            return Ok(false);
        }
        let (_, net_position) = self.k.net_position();
        if net_position < 0.0 {
            return Err(Declined("a short position"));
        }
        let buy = net_position <= 0.0;
        let limit_price = self.snap(reference, net_position < 0.0)?;
        let units = net_position.abs();
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        let market = self.venue.exit_market;
        let tif = if market {
            Tif::Gtc
        } else if self.venue.venue_ioc {
            Tif::Ioc
        } else {
            Tif::Day
        };
        self.submit(buy, units, (!market).then_some(limit_price), tif, true, decided_ns, false)?;
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
    ) -> Result<bool, Declined> {
        let decided_ns = self.m.ts[i];
        if self.gate.is_some_and(|gate| !gate[i]) {
            return Ok(false);
        }
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        let market = self.venue.entry_market;
        let limit_price = self.snap(reference, true)?;
        let order_price =
            if market { self.m.close[i] * self.venue.market_markup } else { limit_price };
        if !self.affordable(units, order_price)? {
            return Ok(false);
        }
        let tif = if market { Tif::Gtc } else { Tif::Gtd(decided_ns + 12 * self.venue.bar_ns) };
        self.submit(
            true,
            units,
            (!market).then_some(limit_price),
            tif,
            false,
            decided_ns,
            policies_after_fill,
        )?;
        self.working_entry = true;
        self.working_orders += 1;
        Ok(true)
    }

    /// `Strategy.submit_order` and the `_OrderTerms` recorded beside it.
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        buy: bool,
        units: f64,
        limit: Option<f64>,
        tif: Tif,
        reduce_only: bool,
        decided_ns: i64,
        policies_after_fill: bool,
    ) -> Result<(), Declined> {
        if limit.is_some_and(|price| price <= 0.0) {
            return Err(Declined("a limit price that is not positive"));
        }
        let client_id = if self.ledger.logs() {
            client_id(&self.venue.order_tag, self.order_seq)
        } else {
            String::new()
        };
        self.order_seq += 1;
        let ledger_index = self.ledger.submit(OrderRecord {
            client_id,
            buy,
            market: limit.is_none(),
            time_in_force: tif.name(),
            units,
            limit_price: limit,
            expire_ns: match tif {
                Tif::Gtd(at) => Some(at),
                _ => None,
            },
            reduce_only,
            decided_ns,
            arrived_on_ns: decided_ns,
            policies_after_fill,
        });
        self.commands.push(Submit { ledger_index, buy, units, limit, tif, reduce_only });
        Ok(())
    }

    fn drop_target(&mut self) {
        self.pending_target = None;
        self.target_bars = 0;
        self.target_bar_ts = -1;
    }

    fn continue_target(&mut self, i: usize, reference: f64) -> Result<bool, Declined> {
        if self.cash_halted {
            return Ok(false);
        }
        let Some(target) = self.pending_target else {
            return Ok(false);
        };
        let (has_position, net_position) = self.k.net_position();
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
        Err(Declined("a target other than flat"))
    }

    fn reconcile_target(&mut self, i: usize) -> Result<bool, Declined> {
        if self.pending_target == Some(0.0) {
            let (has_position, net_position) = self.k.net_position();
            if has_position && self.is_dust(net_position, self.m.close[i]) {
                self.drop_target();
                return Ok(false);
            }
        }
        let now = self.m.ts[i];
        if now != self.target_bar_ts {
            self.target_bar_ts = now;
            self.target_bars += 1;
        }
        if now == self.last_exit_ts {
            return Ok(false);
        }
        self.continue_target(i, self.m.close[i])
    }

    #[allow(clippy::too_many_arguments)]
    fn on_order_filled(
        &mut self,
        at: usize,
        id: usize,
        price: f64,
        size: f64,
        commission: f64,
        leaves: f64,
        gross_realized: f64,
    ) -> Result<(), Declined> {
        self.working_entry = false;
        let Some(&order) = self.orders.get(id) else {
            return Err(Declined("a fill of an order this replay did not send"));
        };
        let now = self.m.ts[at];
        self.ledger.fill(order, now, id as u64, size, price, leaves, gross_realized, commission);
        self.refresh()?;
        self.cursor.visit_primary_after_fill = true;
        if leaves <= 0.0 {
            self.working_orders = (self.working_orders - 1).max(0);
            self.continue_target(at, self.pending_reference)?;
        }
        self.bar_callback_required = true;
        Ok(())
    }

    /// `on_order_canceled` / `on_order_expired`.
    fn end_order(&mut self, at: usize, id: usize, status: EndStatus) {
        if let Some(&order) = self.orders.get(id) {
            self.ledger.end(order, status, self.m.ts[at]);
        }
        self.working_entry = false;
        self.working_orders = (self.working_orders - 1).max(0);
        self.bar_callback_required = true;
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
            && self.flat_on_dust(has_position, net_position, self.m.close[i])
    }

    /// `NativeExecutionSession.reads_dust_as_flat`.
    fn reads_dust_as_flat(&self, quantity: f64, close: f64) -> bool {
        quantity != 0.0 && self.pending_target != Some(0.0) && self.is_dust(quantity, close)
    }

    /// `RustRuleSession.highest_close_before_dispatch`: the frame's window
    /// high where no dispatch lies between `start` and the cursor's next.
    fn highest_close_before_dispatch(&self, i: usize) -> Option<f64> {
        let start = i + 1;
        let stop = self.cursor.next_dispatch_position(self.s, self.m.len());
        if stop <= start {
            return None;
        }
        let r = self.cursor.next_dispatch;
        if r == 0 || self.s.d.pos[r - 1] <= start {
            return Some(self.window_highs[start]);
        }
        Some(self.highs.borrow_mut().get(&self.m.close, start, stop))
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
    fn flat_cash_holds_before_dispatch(&self, i: usize) -> Result<bool, Declined> {
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
    fn strategy_cash(&self) -> Result<f64, Declined> {
        money_as_double(self.k.cash(), self.quiet.currency_precision)
            .ok_or(Declined("a cash balance that is not a number"))
    }

    // -- the caller's money -------------------------------------------------

    fn quantity(&self, i: usize, reference: f64) -> Result<f64, Declined> {
        let cash = self.strategy_cash()?;
        self.money
            .sized(cash, self.m.close[i], reference)
            .ok_or(Declined("a size the money model leaves to Python"))
    }

    fn snap(&self, reference: f64, buy: bool) -> Result<f64, Declined> {
        self.money.snap(reference, buy).ok_or(Declined("a price the money model leaves to Python"))
    }

    fn affordable(&self, units: f64, price: f64) -> Result<bool, Declined> {
        let cash = self.strategy_cash()?;
        self.money
            .affordable(cash, units, price)
            .ok_or(Declined("a cost the money model leaves to Python"))
    }

    fn exhausted(&self, cash: f64, price: f64) -> Result<bool, Declined> {
        self.money
            .exhausted(cash, price)
            .ok_or(Declined("a cash halt the money model leaves to Python"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_below_is_the_partition_point() {
        let positions: Vec<usize> = (0..300).map(|at| at * 9 + at % 7).collect();
        for to in 0..2_800 {
            for from in [0, 1, 17, 150, 299, 300] {
                let rest = &positions[from..];
                assert_eq!(count_below(rest, to), rest.partition_point(|&bar| bar < to));
            }
        }
    }

    #[test]
    fn client_ids_spell_as_format_does() {
        for seq in [0, 1, 9, 10, 99, 100, 12_345, u64::MAX / 3, u64::MAX] {
            for tag in ["", "O", "AAPL-trend"] {
                assert_eq!(client_id(tag, seq), format!("{tag}-{seq}"));
            }
        }
    }
}
