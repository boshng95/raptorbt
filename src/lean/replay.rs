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
//!
//! A cell with risk exits ([`super::rules::Policies`]) is called back on
//! every bar, as on the full lane, and defers nothing: its jumps run each
//! bar's advance on the idle account instead of passing the cursor's visits
//! unread.

use super::kernel::{Event, Kernel, Terms, Tif, Trade};
use super::market::Market;
use std::cell::RefCell;

use super::rules::{segment_highs, side_of, Cursor, Dispatch, Highs, Stream, Target};
use super::Declined;
use crate::python::fill_ledger::{money, or_zero, BookTerms, EndStatus, Ledger, OrderRecord};
use crate::python::money::PyMoneyModel;
use crate::python::replay_session::Venue;
use crate::python::rule_session::{money_as_double, QuietParams};
use crate::stops::policies::{Account, OrderStyle};

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
    /// A short node's terms; None for a long one, whose stream enters long.
    pub shorts: Option<&'a Shorts<'a>>,
}

/// A short node's terms, beyond [`Venue`]'s: `python::replay_session`'s
/// `Shorts`, with what it asks Python bar by bar read off arrays the caller
/// worked out once per market.
///
/// `ShortEntryGate.refusal` depends on the instant only through what the
/// catalog had published by it, its market dates and Rule 201, and on the
/// book only through the position and a capped loan's age; `admitted` is
/// its answer from flat, the rest is asked here. A loan's forced cover
/// depends only on the market date it opened on.
pub(crate) struct Shorts<'a> {
    /// Whether a buy close -- a cover -- goes at market.
    pub cover_market: bool,
    /// `_MAX_SIGNED_SHORT_ATTEMPTS`.
    pub max_attempts: u32,
    /// `_MAX_SIGNED_TARGET_STALE_BARS`.
    pub max_stale_bars: i64,
    /// Per bar, whether the short gate admits a short opened from flat on
    /// it; None where the run has no gate, which refuses nothing.
    pub admitted: Option<&'a [bool]>,
    /// The loan clock, where the instrument's loans carry a term.
    pub loan: Option<Loan<'a>>,
    /// `loan_admission`: the bars `handle_bar` admits, None for every bar.
    pub admission: Option<&'a [bool]>,
}

/// A capped loan's clock (`parity.market.ShortLoanClock`), per bar.
pub(crate) struct Loan<'a> {
    /// Each bar's market date in the listing's civil days, as an ordinal.
    pub days: &'a [i32],
    /// The age in those days from which a loan takes no additions.
    pub block_age: i32,
    /// The bar a short opened on each bar is first due its forced cover
    /// on, -1 for none.
    pub cover_bars: &'a [i64],
    /// The bar a composite asks a short opened on each bar its age on
    /// ahead of every primary bar (`ShortLoanClock.cover`), -1 for none:
    /// one the replay declines on while the short is held.
    pub contest_bars: &'a [i64],
}

/// A replay's records: the ledger it booked, the kernel's round trips, the
/// daily curve and the run's end value.
pub(crate) struct Outcome {
    pub ledger: Ledger,
    pub trades: Vec<Trade>,
    pub day_ns: Vec<i64>,
    pub day_equity: Vec<f64>,
    pub end_value: f64,
    /// A short node's fills, as borrow is charged from them: each change a
    /// fill made to the shares held short, by the bar it filled on.
    pub short_deltas: Vec<(usize, i64)>,
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
    shorts: Option<&'a Shorts<'a>>,

    // `SignalReplay`'s own state, under its own names.
    working_entry: bool,
    /// The ledger index of each order the kernel holds, by kernel id, and
    /// whether it buys.
    orders: Vec<(u32, bool)>,
    pending_target: Option<f64>,
    target_bars: i64,
    target_bar_ts: i64,
    pending_reference: f64,
    /// `_pending_order_style`: how the pending close is ordered.
    pending_order_style: OrderStyle,
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
    account_quantity: f64,
    /// `_short_attempts`.
    short_attempts: u32,
    /// `_exit_gated`.
    exit_gated: bool,
    /// The bar the short held was opened on, and the bars its loan's forced
    /// cover is due on and a composite contests it on, -1 for none:
    /// `_short_opened_ns`, `_loan_cover_bar` and `_loan_contest_bar`. Kept
    /// only where a loan has a term.
    short_opened: Option<usize>,
    loan_cover_bar: i64,
    loan_contest_bar: i64,
    /// The position the fills have left, signed, as the financing replays
    /// them, and what each changed of the shares held short.
    filled_position: f64,
    short_deltas: Vec<(usize, i64)>,

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
    /// Whether, under risk policies, the bars between two dispatches can be
    /// taken in one step ([`Replay::inert_run`]): every dispatch due on its
    /// own bar.
    inert: bool,
    /// The flat cash check's need at the highest close, where it bounds
    /// the need at every close.
    need_top: Option<f64>,
    /// The advance a scan ran on the due bar it stopped at, which that
    /// bar's callback takes instead of running it again.
    advanced: Option<(usize, Option<Target>)>,
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
    if s.entry_side != if cell.shorts.is_some() { -1 } else { 1 } {
        return Err(Declined("a stream entering on the other side"));
    }
    if let Some(shorts) = cell.shorts {
        let long = |bars: Option<usize>| bars.is_some_and(|len| len != n);
        if long(shorts.admitted.map(<[bool]>::len))
            || long(shorts.admission.map(<[bool]>::len))
            || long(shorts.loan.as_ref().map(|loan| loan.days.len()))
            || long(shorts.loan.as_ref().map(|loan| loan.cover_bars.len()))
            || long(shorts.loan.as_ref().map(|loan| loan.contest_bars.len()))
        {
            return Err(Declined("short terms of another length"));
        }
    }
    let q = cell.quiet;
    // Passing visits unread skips the risk checks' stream counts.
    let far = cell.jumps && cell.frame.dispatches_due && s.policies.is_none();
    let inert = cell.jumps && cell.frame.dispatches_due && s.policies.is_some();
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
        shorts: cell.shorts,
        working_entry: false,
        orders: Vec::new(),
        pending_target: None,
        target_bars: 0,
        target_bar_ts: -1,
        pending_reference: 0.0,
        pending_order_style: OrderStyle::Default,
        last_exit_ts: -1,
        last_submit_ts: -1,
        target_step_due: false,
        cash_halted: false,
        working_orders: 0,
        // The first callback checks cash and establishes whether quiet bars
        // can be deferred.
        bar_callback_required: true,
        account_quantity: 0.0,
        short_attempts: 0,
        exit_gated: false,
        short_opened: None,
        loan_cover_bar: -1,
        loan_contest_bar: -1,
        filled_position: 0.0,
        short_deltas: Vec::new(),
        order_seq: cell.venue.first_order_seq,
        commands: Vec::new(),
        budget_ts: None,
        budget_actions: 0,
        jumps: cell.jumps,
        stepped: 0,
        jumped: 0,
        callbacks: 0,
        far,
        inert,
        need_top,
        advanced: None,
        highs: RefCell::default(),
        ledger: match book {
            Some(terms) if !log => Ledger::tally(terms),
            book => Ledger::new(book),
        },
    };
    let (day_ns, day_equity, end_value) = replay.run()?;
    let Replay { k, ledger, stepped, jumped, callbacks, short_deltas, .. } = replay;
    let mut k = k;
    k.finalize(n - 1);
    Ok(Outcome {
        ledger,
        trades: std::mem::take(&mut k.trades),
        day_ns,
        day_equity,
        end_value,
        short_deltas,
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

/// The shares a double counts whole: 2**53.
pub(crate) const WHOLE_SHARES: f64 = 9_007_199_254_740_992.0;

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
                && self.advanced.is_none()
                && self.commands.is_empty()
                && self.k.idle()
            {
                let to = self.quiet_until(i, next_due)?;
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
    /// `i`: the first bar from `i` that can act, `mask_at[next_due]` the
    /// next due bar. The cursor takes the visits of the bars before it.
    fn quiet_until(&mut self, i: usize, next_due: usize) -> Result<usize, Declined> {
        if self.far {
            if let Some((r, held)) = self.quiet_visits() {
                let n = self.m.len();
                let to = self.s.d.pos.get(r).copied().unwrap_or(n);
                if to <= i {
                    // The visit that acts is this bar's own.
                    return Ok(self.mask_at.get(next_due).copied().unwrap_or(n));
                }
                if self.loan_holds_before(to) {
                    self.cursor.pass(self.s, r, held);
                    return Ok(to);
                }
            }
        }
        self.quiet_scan(next_due)
    }

    /// Whether no forced cover falls due, nor a composite contests the loan,
    /// before bar `to`: either bar is one the strategy has to see.
    fn loan_holds_before(&self, to: usize) -> bool {
        self.loan_watch().is_none_or(|at| at >= to)
    }

    /// The first bar the loan held has to be seen on: its cover bar, or the
    /// one a composite contests it on where that comes first.
    fn loan_watch(&self) -> Option<usize> {
        let cover = usize::try_from(self.loan_cover_bar).ok();
        let contest = usize::try_from(self.loan_contest_bar).ok();
        match (cover, contest) {
            (Some(cover), Some(contest)) => Some(cover.min(contest)),
            (cover, contest) => cover.or(contest),
        }
    }

    /// Where the visits cannot be passed unread: the callbacks of the due
    /// bars from `mask_at[next_due]`, run as `on_bar` runs them on an
    /// account nothing touches, up to the first that decides something or
    /// owes the next bar a callback. That bar's own callback takes the
    /// advance run on it here.
    ///
    /// Flat, the cash check passes at every close on a balance nothing
    /// moves. Held, a bar's callback is quiet where its close reads no dust
    /// and its exits stay shut. Under risk policies, the bars between two
    /// dispatches are taken by [`Self::inert_run`] instead.
    #[inline(never)]
    fn quiet_scan(&mut self, mut next_due: usize) -> Result<usize, Declined> {
        let n = self.m.len();
        let due = self.mask_at.get(next_due).copied().unwrap_or(n);
        if self.pending_target.is_some()
            || self.working_orders != 0
            || self.working_entry
            || self.cash_halted
        {
            return Ok(due);
        }
        let (has_position, net_position) = self.k.net_position();
        if !has_position {
            let Some(need_top) = self.need_top else { return Ok(due) };
            if self.account_quantity != 0.0 {
                return Ok(due);
            }
            let q = self.quiet;
            let Ok(cash) = self.strategy_cash() else { return Ok(due) };
            let mut funds = cash * q.fraction;
            if funds > q.cap {
                funds = q.cap;
            }
            if !(funds >= need_top && funds >= q.need_at_highest) {
                return Ok(due);
            }
        }
        // Whether the last bar's visit was one later bars repeat.
        let mut repeats = false;
        while let Some(&at) = self.mask_at.get(next_due) {
            if self.inert {
                let taken = self.inert_run(next_due, has_position, net_position, repeats);
                if taken > 0 {
                    self.callbacks += taken;
                    next_due += taken;
                    repeats = false;
                    continue;
                }
            }
            // Level rules' visit of an admitted primary bar no dispatch
            // reaches: one later bars repeat where it leaves the clocks as it
            // finds them.
            let noted = self.inert
                && self.s.tree.is_some()
                && at < self.cursor.next_dispatch_position(self.s, n)
                && self.s.primary_admits.is_none_or(|admits| admits[at]);
            if noted {
                self.cursor.note_clocks();
            }
            let target = self.advance(at, true)?;
            let quiet = target.is_none()
                && !self.cursor.visit_primary_after_fill
                && !self.loan_due(at)
                && (!has_position
                    || !(self.holds_flat_dust(at, has_position, net_position)
                        || self.cursor.held_reopens(
                            self.s,
                            &self.m.close,
                            at,
                            has_position,
                            net_position,
                            self.pending_target,
                        )));
            if !quiet {
                self.advanced = Some((at, target));
                return Ok(at);
            }
            self.callbacks += 1;
            next_due += 1;
            repeats = noted && self.cursor.clocks_repeat();
            // What the next bar would try first: passing the visits unread.
            if self.far {
                if let Some((r, held)) = self.quiet_visits() {
                    let to = self.s.d.pos.get(r).copied().unwrap_or(n);
                    if to > at + 1 && self.loan_holds_before(to) {
                        self.cursor.pass(self.s, r, held);
                        return Ok(to);
                    }
                }
            }
        }
        Ok(n)
    }

    /// Of the due bars from `mask_at[next_due]` before the next dispatch and
    /// the loan's cover bar, how many a cell under risk policies takes as
    /// quiet in one step ([`Cursor::inert_run`]), on the account the scan
    /// found -- `has_position` and `net_position` -- which nothing touches.
    ///
    /// Each such bar's advance reads the cash, which must convert, and a lot
    /// no close reads as dust keeps every bar's position the venue's.
    /// `repeats`: whether the last bar's visit is one level rules repeat.
    fn inert_run(
        &mut self,
        next_due: usize,
        has_position: bool,
        net_position: f64,
        repeats: bool,
    ) -> usize {
        let (m, s, mask) = (self.m, self.s, self.mask_at);
        let mut stop = self.cursor.next_dispatch_position(s, m.len());
        if let Some(watch) = self.loan_watch() {
            stop = stop.min(watch);
        }
        let due = &mask[next_due..];
        let due = &due[..count_below(due, stop)];
        if due.is_empty()
            || self.strategy_cash().is_err()
            || (has_position && (m.low_close.is_nan() || self.is_dust(net_position, m.low_close)))
        {
            return 0;
        }
        let side = if has_position { side_of(net_position) } else { 0 };
        let account = self.policy_account();
        self.cursor.inert_run(s, &m.ts, &m.close, m.low_close, due, account, side, repeats)
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
        let units = held.abs();
        if side_of(held) != s.entry_side
            || self.account_quantity != held
            || self.m.low_close.is_nan()
            || self.is_dust(held, self.venue.lowest_close)
            || self.is_dust(held, self.m.low_close)
            || s.d.lowest_close.is_nan()
            || units < s.size_increment
            || (s.minimum_notional > 0.0 && units * s.d.lowest_close < s.minimum_notional)
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
            self.orders.push((command.ledger_index, command.buy));
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
        // `_refuse_resting_exit`: asked first, ahead of anything that could
        // send beside it.
        if self.k.resting_exit(self.m.ts[i]) {
            return Err(Declined("an exit outlived the bar it met"));
        }
        // `_refuse_contested_loan`: a composite's check of the loan held
        // comes ahead of anything this bar does.
        if usize::try_from(self.loan_contest_bar).is_ok_and(|at| at <= i) {
            return Err(Declined("a composite bar asks a held loan its age first"));
        }
        // A scan stops with nothing pending: nothing reconciles before the
        // advance it ran.
        let advanced = match self.advanced.take() {
            Some((at, target)) if at == i => Some(target),
            Some(_) => return Err(Declined("an advance run for another bar")),
            None => None,
        };
        if self.cash_halted {
            if advanced.is_none() {
                self.advance(i, false)?;
            }
            self.bar_callback_required = false;
            return Ok(());
        }
        let reconciled =
            if self.pending_target.is_some() { self.reconcile_target(i)? } else { false };
        let target = match advanced {
            Some(target) => target,
            None => self.advance(i, true)?,
        };
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
        // Behind the halt, as there: the book it leaves is the one the bars
        // after this are asked about.
        let (has_position, net_position, flat, cash_flat) =
            if self.target_step_due && self.finish_target_step(i)? {
                let (has_position, net_position) = self.k.net_position();
                let flat = !has_position;
                let cash_flat = flat || self.holds_flat_dust(i, has_position, net_position);
                (has_position, net_position, flat, cash_flat)
            } else {
                (has_position, net_position, flat, cash_flat)
            };
        // What the bars a sparse callback steps past before the next
        // composite can do is asked only of a run that defers them: one with
        // risk checks has every bar called back, each asked on its own.
        let defers = self.s.policies.is_none();
        self.bar_callback_required = (self.pending_target.is_some()
            && !(defers && self.flat_target_rests_before_dispatch(i, has_position, net_position)))
            || self.working_orders != 0
            || (cash_flat
                && !self.cash_halted
                && !flat_cash_safe
                && !(defers && self.flat_cash_holds_before_dispatch(i)?))
            || self.cursor.visit_primary_after_fill
            || (defers
                && !flat
                && self.cursor.held_reopens(
                    self.s,
                    &self.m.close,
                    i,
                    has_position,
                    net_position,
                    self.pending_target,
                ))
            || self.loan_due(i);
        Ok(())
    }

    /// `0 <= _loan_cover_bar <= idx`: a loan past its cover bar, still held
    /// -- or past the bar a composite contests it on, which `on_bar`
    /// declines.
    fn loan_due(&self, i: usize) -> bool {
        self.loan_watch().is_some_and(|at| at <= i)
    }

    /// `RustRuleSession.advance`; `apply_policies` false for a halted run.
    fn advance(&mut self, i: usize, apply_policies: bool) -> Result<Option<Target>, Declined> {
        // The account the risk checks read (`_position`).
        let account = self.policy_account();
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
            account,
            apply_policies,
        )?;
        if processed {
            self.refresh()?;
        }
        Ok(target)
    }

    /// The account a cell's risk policies read in front of a bar
    /// (`NativeExecutionSession._position`): the venue's signed lot -- dust
    /// and all -- and its average entry, each `or 0.0`; flat where nothing
    /// is held, and for a cell without policies, which reads none.
    fn policy_account(&self) -> Account {
        if self.s.policies.is_none() {
            return Account::FLAT;
        }
        let (held, net) = self.k.net_position();
        if !held {
            return Account::FLAT;
        }
        let entry = self.k.entry_price().unwrap_or(0.0);
        Account { quantity: or_zero(net), entry: or_zero(entry) }
    }

    /// `NativeExecutionSession._refresh`, as far as anything reads it.
    fn refresh(&mut self) -> Result<(), Declined> {
        let (has_position, net_position) = self.k.net_position();
        self.account_quantity = if has_position { net_position } else { 0.0 };
        self.strategy_cash().map(|_| ())
    }

    /// `_submit_native_intent`: `preceding` is the rules' own close to send
    /// ahead of a risk close (`preceding_exit`), and `risk_exit` marks a
    /// target a risk policy decided, outside the bar's decision.
    fn submit_native_intent(
        &mut self,
        i: usize,
        target: Option<Target>,
        preceding: Option<f64>,
        risk_exit: bool,
    ) -> Result<(), Declined> {
        let Some(target) = target else {
            return Ok(());
        };
        let (target_side, style) = (target.side, target.style);
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
            return Err(Declined("a short entry"));
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
    ) -> Result<bool, Declined> {
        let decided_ns = self.m.ts[i];
        let (_, net_position) = self.k.net_position();
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
        if self.gate.is_some_and(|gate| !gate[i]) {
            // Past instrument preflight, which is as far as a close gets in
            // `create_order` before `submit_order` meets the shut venue.
            // Whether it outlives its order is the caller's to say.
            self.exit_gated = true;
            return Ok(false);
        }
        // A buy close takes `reference_order_type`'s buy side: a cover's.
        let cover_market = match self.shorts {
            Some(shorts) => shorts.cover_market,
            None if buy => return Err(Declined("a short position")),
            None => false,
        };
        let market = match style {
            OrderStyle::Default if buy => cover_market,
            OrderStyle::Default => self.venue.exit_market,
            OrderStyle::Market => true,
            OrderStyle::Limit => false,
        };
        let tif = if market {
            Tif::Gtc
        } else if self.venue.venue_ioc {
            Tif::Ioc
        } else {
            Tif::Day
        };
        self.submit(buy, units, (!market).then_some(limit_price), tif, true, decided_ns)?;
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
    fn submit_entry(
        &mut self,
        i: usize,
        reference: f64,
        units: f64,
        long: bool,
    ) -> Result<bool, Declined> {
        let decided_ns = self.m.ts[i];
        if self.gate.is_some_and(|gate| !gate[i]) {
            return Ok(false);
        }
        // A short against a durable target spends one of its attempts, and
        // the target is given up when they are gone, ahead of the preflight.
        let durable_short = !long && self.pending_target.is_some_and(|t| t < 0.0);
        if !long {
            let Some(shorts) = self.shorts else {
                return Err(Declined("a short entry"));
            };
            if durable_short && self.short_attempts >= shorts.max_attempts {
                self.drop_target();
                return Ok(false);
            }
        }
        if units <= 0.0 || !self.tradeable(units, reference) {
            return Ok(false);
        }
        if !long && self.short_refused(i) {
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
                if market { self.m.close[i] * self.venue.market_markup } else { limit_price };
            if !self.affordable(units, order_price)? {
                return Ok(false);
            }
        }
        let tif = if market {
            Tif::Gtc
        } else if long {
            Tif::Gtd(decided_ns + 12 * self.venue.bar_ns)
        } else if self.venue.venue_ioc {
            Tif::Ioc
        } else {
            Tif::Day
        };
        self.submit(long, units, (!market).then_some(limit_price), tif, false, decided_ns)?;
        if durable_short {
            self.short_attempts += 1;
        }
        self.working_entry = true;
        self.last_submit_ts = decided_ns;
        self.working_orders += 1;
        Ok(true)
    }

    /// Whether the run's short gate refuses a short decided on bar `i`,
    /// added to the position held: `short_gate.refusal`.
    ///
    /// A reversal must be flat first, and a capped loan takes no additions
    /// once it is `block_age` market days old -- or at all where its
    /// inception is unknown; past those, the bar's answer from flat.
    fn short_refused(&self, i: usize) -> bool {
        let Some(shorts) = self.shorts else {
            return false;
        };
        let Some(admitted) = shorts.admitted else {
            return false;
        };
        let (has_position, net_position) = self.k.net_position();
        let position = if has_position { net_position } else { 0.0 };
        if position > 0.0 {
            return true;
        }
        if let (Some(loan), true) = (&shorts.loan, position < 0.0) {
            match self.short_opened {
                None => return true,
                Some(opened) if loan.days[i] - loan.days[opened] >= loan.block_age => {
                    return true;
                }
                Some(_) => {}
            }
        }
        !admitted[i]
    }

    /// `Strategy.submit_order` and the `_OrderTerms` recorded beside it.
    fn submit(
        &mut self,
        buy: bool,
        units: f64,
        limit: Option<f64>,
        tif: Tif,
        reduce_only: bool,
        decided_ns: i64,
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
            // No target asks the risk policies after a fill any more (one step
            // per instant): every order's terms say so.
            policies_after_fill: false,
        });
        self.commands.push(Submit { ledger_index, buy, units, limit, tif, reduce_only });
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
    ) -> Result<bool, Declined> {
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
        let Some(max_stale_bars) = self.shorts.map(|s| s.max_stale_bars) else {
            return Err(Declined("a target other than flat"));
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
            return Err(Declined("a long target"));
        }
        if self.pending_order_style != OrderStyle::Default {
            // Only a risk close carries a style, and it targets flat.
            return Err(Declined("an entry in a close's order style"));
        }
        self.submit_entry(i, reference, delta.abs(), delta > 0.0)
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
        let closing = self.pending_target == Some(0.0);
        // Only what reduces the book, ahead of the decision; what adds to it
        // waits for the end of the step.
        self.target_step_due = true;
        let submitted = self.continue_target(i, self.m.close[i], true, false)?;
        if submitted && closing {
            // `_reconcile_exit` resends it under the strategy's pending
            // close, which stands until the next bar's reconcile finds the
            // book flat.
            self.cursor.note_close(i);
        }
        Ok(submitted)
    }

    /// `_finish_target_step`: what the target still adds, offered once the
    /// bar has decided and its risk checks have run -- unless an order went
    /// out this bar, which Nautilus still holds unresolved at the end of the
    /// step. Returns whether an order reached the venue.
    fn finish_target_step(&mut self, i: usize) -> Result<bool, Declined> {
        self.target_step_due = false;
        if self.last_submit_ts == self.m.ts[i] {
            return Ok(false);
        }
        self.continue_target(i, self.m.close[i], false, true)
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
        let Some(&(order, buy)) = self.orders.get(id) else {
            return Err(Declined("a fill of an order this replay did not send"));
        };
        let now = self.m.ts[at];
        self.ledger.fill(order, now, id as u64, size, price, leaves, gross_realized, commission);
        if self.shorts.is_some() {
            self.book_short_fill(at, buy, size)?;
            self.track_short_loan(at);
        }
        // `runtime.on_fill`: the account refreshed, the next primary bar a
        // visit, and the fill booked by the risk policies -- at what it cost
        // as the strategy reads it (`_record_fill`).
        self.refresh()?;
        if self.s.policies.is_some() {
            let cost = money(or_zero(commission), self.quiet.currency_precision)
                .ok_or(Declined("a commission that is not a number"))?;
            let flat = side_of(self.account_quantity) == 0;
            // A cell that borrows charges the loan on the lot the fill left.
            let side = if buy { 1.0 } else { -1.0 };
            let position = self.account_quantity;
            self.cursor.policy_fill(self.s, side, or_zero(size), cost, flat, at, position)?;
        }
        self.cursor.visit_primary_after_fill = true;
        if leaves <= 0.0 {
            self.working_orders = (self.working_orders - 1).max(0);
            self.continue_target(at, self.pending_reference, true, true)?;
        }
        self.bar_callback_required = true;
        Ok(())
    }

    /// What a fill of `size` on bar `at` changed of the shares held short,
    /// as `build_backtest_financing_report` replays the fills report: one
    /// row per execution, in fill order, each moving a signed position.
    /// Its arithmetic is decimal; on whole shares a double's is the same.
    fn book_short_fill(&mut self, at: usize, buy: bool, size: f64) -> Result<(), Declined> {
        // A row of no quantity is skipped.
        if !(size > 0.0) {
            return Ok(());
        }
        if size.fract() != 0.0 || size > WHOLE_SHARES {
            return Err(Declined("a short financed on shares that are not whole"));
        }
        let before = self.filled_position;
        let after = if buy { before + size } else { before - size };
        self.filled_position = after;
        let delta = (-after).max(0.0) - (-before).max(0.0);
        if delta != 0.0 {
            self.short_deltas.push((at, delta as i64));
        }
        Ok(())
    }

    /// `_track_short_loan`, after every fill where a loan has a term: a loan
    /// is dated from the fill that left the book short, and ends with the
    /// fill that leaves it flat or long.
    fn track_short_loan(&mut self, at: usize) {
        let Some(loan) = self.shorts.and_then(|s| s.loan.as_ref()) else {
            return;
        };
        let (has_position, net_position) = self.k.net_position();
        if has_position && net_position < 0.0 {
            if self.short_opened.is_none() {
                self.short_opened = Some(at);
                self.loan_cover_bar = loan.cover_bars[at];
                self.loan_contest_bar = loan.contest_bars[at];
            }
        } else {
            self.short_opened = None;
            self.loan_cover_bar = -1;
            self.loan_contest_bar = -1;
        }
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
    fn check_short_loan(&mut self, i: usize) -> Result<(), Declined> {
        if self.working_orders != 0 {
            return Ok(());
        }
        if self.shorts.and_then(|s| s.admission).is_some_and(|admission| !admission[i]) {
            return Ok(());
        }
        let (has_position, net_position) = self.k.net_position();
        if !has_position || net_position >= 0.0 {
            return Ok(());
        }
        let reference = self.m.close[i];
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
    fn end_order(&mut self, at: usize, id: usize, status: EndStatus) {
        if let Some(&(order, _)) = self.orders.get(id) {
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
