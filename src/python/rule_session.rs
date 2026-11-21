//! A rule set's decision cursor, run beside the kernel.
//!
//! The account-coupled native lane of the parity runner decides on composite
//! (resampled) bars by walking a merged schedule of its symbol's streams and
//! evaluating rules on each event, exactly as Nautilus's `ComposableStrategy`
//! does. This is that cursor, rule program and decision node, moved out of
//! Python for the two shapes such runs have: a legacy spec of direct
//! crossovers, whose primary bars repeat the last operands, and a legacy spec
//! of level rules -- thresholds, zones and crossovers combined -- over the
//! same frozen operands, whose primary bars repeat the last answer until a
//! clock fires or the side the rules see moves ([`Level`]).
//!
//! It is a transcription, not a model. Every branch mirrors a line of
//! `NativeExecutionSession.advance`, `NativeModelRunner.visit`,
//! `_DeclarativeNode.visit` and `DecisionProgram.evaluate`, and the Python
//! session stays the reference: the caller runs the two in lockstep in its
//! tests and compares every answer and every piece of state.
//!
//! [`PyCrossoverRuleSession::quiet_bar`] is what the kernel's fast-forward
//! uses: it takes one callback bar on the strategy's behalf and keeps it only
//! when the strategy would have done nothing with it, restoring the state it
//! started from otherwise so that Python can take the same bar itself. The
//! kernel's own replay, which takes that bar too, has nothing to restore: its
//! stepper keeps the bar's `advance` for it instead ([`RuleStepper::handing_back`]).

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const PRIMARY: usize = 0;

/// The phases of one instant's delivery, in the order they are decided:
/// `ScheduledStream.order` and `DISPATCH_PHASE_*`
/// (`src.models.raptor_capability`). A leg the strategy builds hands over
/// the windows a gap closed first, then its pre-close windows, both ahead of
/// the primary bar's own decision, then the windows that bar completed.
pub(crate) const PHASE_CLOSED_BY_GAP: i64 = -1;
pub(crate) const PHASE_PRECLOSE: i64 = 0;
pub(crate) const PHASE_PRIMARY: i64 = 1;
pub(crate) const PHASE_COMPLETED: i64 = 2;
/// `10**p` for the precisions `money_as_double` accepts, as exact doubles.
const POW10: [f64; 17] =
    [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16];
const FIXED_PRECISION: u32 = 16;
const FIXED_SCALAR: f64 = 1e16;
/// `2**126`: a product of two exact integers below it in magnitude is one a
/// 128-bit integer holds.
const I128_SAFE: f64 = 85_070_591_730_234_615_865_843_651_857_942_052_864.0;

/// `money_as_double`: round to the currency's minor unit, scale to Nautilus's
/// fixed precision as an integer, and convert that integer back to a double.
///
/// `None` where the Python original would raise or a 128-bit integer cannot
/// hold the raw value; the caller hands such a bar back to Python.
pub(crate) fn money_as_double(value: f64, currency_precision: u32) -> Option<f64> {
    if currency_precision > FIXED_PRECISION || !value.is_finite() {
        return None;
    }
    let minor = (value * POW10[currency_precision as usize]).round_ties_even();
    if minor.abs() >= 1e37 {
        return None;
    }
    // `minor` and the power of ten are exact integers, so the product is the
    // float nearest the raw integer, which is what converting it gives. Well
    // inside the 128-bit range the integer cannot overflow, and the product
    // is the answer without it -- but for the sign of a zero, which an
    // integer does not have.
    let scale = POW10[(FIXED_PRECISION - currency_precision) as usize];
    let raw = minor * scale;
    if raw.abs() < I128_SAFE {
        return Some(if raw == 0.0 { 0.0 } else { raw / FIXED_SCALAR });
    }
    let raw = (minor as i128).checked_mul(10i128.pow(FIXED_PRECISION - currency_precision))?;
    Some(raw as f64 / FIXED_SCALAR)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    Key(usize),
    Const(f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Crossing {
    clock: usize,
    fast: Operand,
    slow: Operand,
    above: bool,
}

/// `ThresholdCondition.op`.
#[derive(Clone, Copy, Debug)]
enum Compare {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl Compare {
    fn holds(self, a: f64, b: f64) -> bool {
        match self {
            Self::Gt => a > b,
            Self::Ge => a >= b,
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Eq => a == b,
        }
    }
}

/// One node of a compiled condition, `DecisionProgram._compile`'s closure.
/// Children are earlier nodes of the same arena.
#[derive(Clone, Debug)]
enum Condition {
    All(Vec<usize>),
    Any(Vec<usize>),
    Not(usize),
    Zone { value: Operand, lower: f64, upper: f64 },
    Threshold { left: Operand, right: Operand, compare: Compare },
    Cross(Crossing),
}

/// What a level rule set reads beyond a crossover one: StrategyBase's dust.
///
/// A level rule answers the same on a repeat only while the side it sees
/// stands still, and `is_flat()` reads a lot too small to close as flat. The
/// terms are the instrument contract's (`InstrumentContract.is_dust`), and
/// `two_sided` is the executor's own: a two-sided run's lot stays held.
#[derive(Clone, Copy, Debug)]
struct Level {
    two_sided: bool,
    size_increment: f64,
    minimum_notional: f64,
}

impl Level {
    /// `InstrumentContract.is_dust`.
    fn is_dust(&self, quantity: f64, price: f64) -> bool {
        let units = quantity.abs();
        units < self.size_increment
            || (self.minimum_notional > 0.0 && units * price < self.minimum_notional)
    }

    /// `NativeExecutionSession.reads_dust_as_flat`.
    fn reads_dust_as_flat(&self, quantity: f64, pending: Option<f64>, close: f64) -> bool {
        quantity != 0.0
            && pending.is_none_or(|p| p != 0.0)
            && !self.two_sided
            && self.is_dust(quantity, close)
    }
}

/// `NativeExecutionSession.advance`'s pending-close latch over one call:
/// StrategyBase's ``_pending_flat``, which a long-only close sets and only the
/// next primary bar's reconcile clears, once the book reads flat. While it
/// stands every entry is refused (``PENDING_CLOSE``).
///
/// `close_bar` is the last primary bar the latch holds through, or -1; a
/// close a resample delivered for the bar before has filled by the time this
/// bar's reconcile runs, which is what `provisional` remembers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Latch {
    pub close_bar: i64,
    pub provisional: bool,
}

impl Latch {
    pub(crate) fn new(close_bar: i64) -> Self {
        Self { close_bar, provisional: false }
    }

    /// The latch as an event on primary bar `bar` finds it, `early` for a
    /// pre-close window, from side `start_side`: a later bar's reconcile has
    /// run, and cleared the latch if the close filled.
    #[inline]
    pub(crate) fn step(&mut self, bar: i64, early: bool, start_side: i32) {
        if self.close_bar >= 0
            && (bar > self.close_bar + 1 || (bar == self.close_bar + 1 && !early))
        {
            self.close_bar =
                if self.provisional || start_side == 0 { -1 } else { bar - i64::from(early) };
            self.provisional = false;
        }
    }

    /// `_close_latch`: a close decided by an event on primary bar `bar` of
    /// the callback on primary bar `idx`.
    #[inline]
    pub(crate) fn close(&mut self, bar: i64, early: bool, idx: i64) {
        self.close_bar = self.close_bar.max(bar);
        self.provisional = !early && bar < idx;
    }

    /// Whether an entry is refused.
    #[inline]
    pub(crate) fn holds(&self) -> bool {
        self.close_bar >= 0
    }
}

/// The target a visit moves one callback's to, or `None` where it moves
/// nothing: `target_for_intent` under the rules of one delivery
/// (`NativeExecutionSession.advance`).
///
/// Every visit at one instant is decided inside the primary bar's callback,
/// and the venue settles none of their orders until it returns. So a visit
/// asking for the side already decided places nothing again, and the order
/// decided keeps its terms; a close finds nothing to sell of an entry this
/// instant placed (`entering`); and while a close is working an entry is
/// refused -- under the long-only latch as ``PENDING_CLOSE``, and otherwise
/// when it wants the side being closed, which reads as held, so the close
/// stands (`set_target_position`).
///
/// An entry moves a flat target to `entry_side`: 1 for rules that buy, -1
/// for a short node's, whose entries sell to open (`_DeclarativeNode.visit`
/// says its entries as that side, its exits as flat). Nothing else here
/// reads a side's sign.
#[inline]
pub(crate) fn delivery_step(
    wants_entry: bool,
    wants_exit: bool,
    entry_side: i32,
    target: i32,
    start_side: i32,
    entering: bool,
    latch: &Latch,
) -> Option<i32> {
    let mut moved = target;
    if wants_exit && moved != 0 {
        moved = 0;
    }
    if wants_entry && moved == 0 {
        moved = entry_side;
    }
    let refused = if moved == target {
        true
    } else if moved == 0 {
        entering
    } else {
        target == 0 && (latch.holds() || moved == start_side)
    };
    (!refused).then_some(moved)
}

/// Which rule a decision came from: `(is_exit, index)` within its list.
type RuleId = (bool, usize);

#[derive(Clone, Copy, Debug, Default)]
struct RuleMatch {
    evaluated: bool,
    ready: bool,
    exit_rule: Option<usize>,
    entry_rule: Option<usize>,
    reached: bool,
}

#[derive(Clone, Copy, Debug)]
struct NodeIntent {
    wants_entry: bool,
    wants_exit: bool,
    reference: f64,
    reason: Option<RuleId>,
}

/// What one `advance` asked for; `None` from `advance` is `IDLE`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    pub side: i32,
    pub reference: Option<f64>,
    reason: Option<RuleId>,
    /// The risk policy that decided the target, where one did: a close
    /// `handle_bar` placed outside the bar's decision (`policy_acted`).
    pub policy: Option<PolicyExit>,
    /// The model's own close of the same callback, to send ahead of the
    /// policy's at this reference, in the venue's close
    /// (`preceding_exit`).
    pub preceding: Option<f64>,
    /// How the target's order is placed (`NativeIntent.order_style`).
    pub style: OrderStyle,
}

impl Target {
    /// The rules' own target: no risk exit, in the venue's default order.
    fn model(side: i32, reference: Option<f64>, reason: Option<RuleId>) -> Self {
        Self { side, reference, reason, policy: None, preceding: None, style: OrderStyle::Default }
    }
}

/// `NativeOrderStyle`: how an exit is ordered. `Default` is the venue's
/// configured close.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderStyle {
    Default,
    Market,
    Limit,
}

impl OrderStyle {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Market => "market",
            Self::Limit => "limit",
        }
    }
}

/// The risk policy that closed a position, by its reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolicyExit {
    TradeStopLoss,
    TradeTakeProfit,
    AtrStop,
    AtrTakeProfit,
    MaxHold,
}

impl PolicyExit {
    fn reason(self) -> &'static str {
        match self {
            Self::TradeStopLoss => "trade_stop_loss",
            Self::TradeTakeProfit => "trade_take_profit",
            Self::AtrStop => "atr_stop",
            Self::AtrTakeProfit => "atr_take_profit",
            Self::MaxHold => "max_hold",
        }
    }
}

/// The account the risk policies read: the venue's signed lot -- dust and
/// all -- and the open position's average entry (`NativeAccountView`'s
/// `position_quantity` and `average_entry_price`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Account {
    pub quantity: f64,
    pub entry: f64,
}

impl Account {
    pub(crate) const FLAT: Self = Self { quantity: 0.0, entry: 0.0 };
}

/// The side a signed quantity holds.
fn side_of(quantity: f64) -> i32 {
    if quantity > 0.0 {
        1
    } else if quantity < 0.0 {
        -1
    } else {
        0
    }
}

/// `NativeTradeExitPolicy`: the percentage stop and take-profit on the
/// position's net return.
#[derive(Clone, Copy, Debug)]
struct TradeExit {
    stop: f64,
    take_profit: f64,
}

/// `NativeAtrExitPolicy`'s terms. Its ATR is indexed in its own stream's
/// bars: the executing primary bar's for `slot` [`PRIMARY`], else the last
/// bar of that resample to close.
#[derive(Clone, Copy, Debug)]
struct AtrExit {
    slot: usize,
    stop: f64,
    take_profit: f64,
    trailing: bool,
}

/// `NativeHoldExitPolicy`: a time exit counted in bars of resample `slot`.
#[derive(Clone, Copy, Debug)]
struct HoldExit {
    slot: usize,
    bars: i64,
}

/// An ATR exit with the ATR's values and initialized flags.
type AtrSeries = (AtrExit, Py<PyArray1<f64>>, Py<PyArray1<bool>>);

/// StrategyBase's risk checks, as `_POLICY_COMPILERS` builds them: each
/// optional, checked in this order, the last to act deciding.
struct Policies {
    trade: Option<TradeExit>,
    atr: Option<AtrSeries>,
    hold: Option<HoldExit>,
    /// The contract's taker fee where commissions are on, else 0.
    fee_rate: f64,
    /// `_close_style`.
    style: OrderStyle,
}

/// `_PositionCharges`: the commission the open position's fills were
/// charged, and the signed quantity those fills add up to.
#[derive(Clone, Copy, Debug)]
struct Charges {
    charged: f64,
    held: f64,
}

impl Charges {
    /// `round_trip`.
    fn round_trip(&self, fee_rate: f64, units: f64, entry: f64, current: f64) -> f64 {
        if self.charged > 0.0 {
            return self.charged * 2.0;
        }
        fee_rate * units * (entry + current)
    }

    /// `on_fill`: book a fill of `quantity` on `side` (1 buys) charged
    /// `commission`, which left the account `flat` -- and say so.
    fn on_fill(&mut self, side: f64, quantity: f64, commission: f64, flat: bool) -> bool {
        let held = self.held;
        self.held = held + side * quantity;
        if flat {
            self.charged = 0.0;
            self.held = 0.0;
            return true;
        }
        let commission = commission.abs();
        if held == 0.0 {
            self.charged = commission;
        } else if side * held < 0.0 && quantity >= held.abs() * (1.0 - 1e-9) {
            // The close leg of a replacement.
            self.charged = 0.0;
            self.held = 0.0;
        } else {
            self.charged += commission;
        }
        false
    }
}

/// `_NativeBorrowTracker` on the lane that observes every primary bar
/// (`open_short_borrow`): per primary bar its market date and the date a
/// short opened on it settles, both in days from the epoch, and one share's
/// fee for each day after `first_day` -- the first bar's date -- summed:
/// `charged[k]` through `first_day + k`, in `10**-BORROW_FEE_DIGITS` of the
/// currency.
struct Borrow {
    days: Py<PyArray1<i32>>,
    settles: Py<PyArray1<i32>>,
    first_day: i32,
    charged: Py<PyArray1<i64>>,
}

/// [`Borrow`]'s arrays, for the length of one call.
struct BorrowView<'a> {
    days: &'a [i32],
    settles: &'a [i32],
    first_day: i32,
    charged: &'a [i64],
}

/// `BORROW_FEE_DIGITS`: a fee is a whole number of `10**-10` of the currency.
const BORROW_FEE_SCALE: f64 = 1e10;

/// `float(Decimal(units).scaleb(-BORROW_FEE_DIGITS))`: the double nearest the
/// fee. Below 2**53 both operands are exact and the division rounds once;
/// past it the decimal spelling is parsed, which rounds once too.
fn borrow_amount(units: i128) -> f64 {
    if units.unsigned_abs() < 1 << 53 {
        units as f64 / BORROW_FEE_SCALE
    } else {
        format!("{units}e-10").parse().expect("an integer's decimal spelling parses")
    }
}

impl BorrowView<'_> {
    /// `_NativeBorrowTracker.observe` on primary bar `idx`, the venue holding
    /// signed `quantity`: anything but a short ends the loan; a short with
    /// none opens one, settling as a short opened on the bar does; and each
    /// date the bar reaches past the one the loan is charged through is
    /// charged on the shares held now, at one share's fee for those days.
    fn observe(&self, p: &mut PolicyState, idx: usize, quantity: f64) -> PyResult<()> {
        if quantity >= 0.0 {
            p.loan = None;
            p.borrow_units = 0;
            p.borrow = 0.0;
            return Ok(());
        }
        let marked = *p.loan.get_or_insert(self.settles[idx] - 1);
        let day = self.days[idx];
        if day <= marked {
            return Ok(());
        }
        let shares = -quantity;
        if shares.fract() != 0.0 || shares >= (1u64 << 53) as f64 {
            return Err(PyValueError::new_err("a short's borrow is charged on whole shares"));
        }
        // A loan is charged from no earlier than the day after the first
        // bar's date, and through no later than the last bar's.
        let since = (marked - self.first_day) as usize;
        let through = (day - self.first_day) as usize;
        let fees = i128::from(self.charged[through] - self.charged[since]);
        p.borrow_units += shares as i128 * fees;
        p.loan = Some(day);
        p.borrow = borrow_amount(p.borrow_units);
        Ok(())
    }
}

/// What the risk policies keep between checks.
#[derive(Clone, Copy, Debug)]
struct PolicyState {
    /// `NativeTradeExitPolicy._charges`.
    trade_charges: Charges,
    /// `NativeAtrExitPolicy`'s `_charges`, `_atr_index`, `_entry_price` and
    /// `_extreme`.
    atr_charges: Charges,
    atr_index: i64,
    atr_entry: f64,
    atr_extreme: f64,
    /// `NativeHoldExitPolicy`'s `_counted` and `_mark`.
    hold_counted: i64,
    hold_mark: i64,
    /// `_held_policy_streams` over the call in progress, as releasing it
    /// moves the policies: whether it holds any close, the last of the
    /// ATR's resample among them (-1 for none), and how many of the time
    /// exit's.
    held: bool,
    held_atr: i64,
    held_counts: i64,
    /// `policy_resampled`.
    resampled: bool,
    /// `_NativeBorrowTracker._accrual`: the date the open short's loan is
    /// charged through, `None` with no loan; what it has charged, in
    /// `10**-BORROW_FEE_DIGITS` of the currency; and that as the account's
    /// `accrued_borrow`. Untouched on a run that borrows nothing.
    loan: Option<i32>,
    borrow_units: i128,
    borrow: f64,
}

impl PolicyState {
    const NEW: Self = Self {
        trade_charges: Charges { charged: 0.0, held: 0.0 },
        atr_charges: Charges { charged: 0.0, held: 0.0 },
        atr_index: -1,
        atr_entry: 0.0,
        atr_extreme: 0.0,
        hold_counted: 0,
        hold_mark: -1,
        held: false,
        held_atr: -1,
        held_counts: 0,
        resampled: false,
        loan: None,
        borrow_units: 0,
        borrow: 0.0,
    };
}

/// `trade_roi_pct`.
fn trade_roi_pct(entry: f64, current: f64, position: f64, costs: f64) -> f64 {
    let quantity = position.abs();
    let notional = entry * quantity;
    if notional <= 0.0 {
        return 0.0;
    }
    let gross =
        if position > 0.0 { (current - entry) * quantity } else { (entry - current) * quantity };
    ((gross - costs) / notional) * 100.0
}

#[derive(Debug)]
struct Cursor {
    indexes: Vec<usize>,
    current: Option<usize>,
    timestamp: i64,
    label: i64,
    priority: i64,
    slot: usize,
    bar: usize,
}

/// Everything a bar can change. Copied before a fast-forward bar so that a
/// bar the strategy would act on can be handed back untouched.
#[derive(Debug)]
struct State {
    cursor: Cursor,
    next_dispatch: usize,
    visit_primary_after_fill: bool,
    /// `_DeclarativeNode._updates`, per key: the last decoded value.
    updates: Vec<Option<f64>>,
    /// `DecisionProgram.values` and `_unready`, per key.
    values: Vec<Option<f64>>,
    unready: Vec<bool>,
    clocks: Vec<Option<(f64, f64)>>,
    last_labels: Vec<Option<i64>>,
    last_sort_key: Option<(i64, i64)>,
    last_position: i32,
    last_reached: bool,
    last_ready: bool,
    /// Whether a crossover clock changed sides in the current visit:
    /// `_clock_flipped` over the clocks before and after it.
    flipped: bool,
    /// Each clock's answer on the bar being evaluated, when its crossings
    /// are worked out each bar (`Data::each_bar`); written before any rule
    /// reads it.
    fired: Vec<bool>,
    /// `NativeExecutionSession._close_bar`: the last primary bar a long-only
    /// close is still pending through, or -1.
    close_bar: i64,
    /// Whether the last `advance` entered from flat, which a close noted
    /// ahead of it would have refused.
    entered_from_flat: bool,
    /// The risk policies' own state; untouched where there are none.
    policy: PolicyState,
}

// `Clone` by hand so that `clone_from` keeps the copy's buffers: a run of
// quiet bars copies the state once per bar. Every field is named, so a new
// one cannot be left out.
impl Clone for Cursor {
    fn clone(&self) -> Self {
        let Self { indexes, current, timestamp, label, priority, slot, bar } = self;
        Self {
            indexes: indexes.clone(),
            current: *current,
            timestamp: *timestamp,
            label: *label,
            priority: *priority,
            slot: *slot,
            bar: *bar,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self { indexes, current, timestamp, label, priority, slot, bar } = self;
        indexes.clone_from(&source.indexes);
        *current = source.current;
        *timestamp = source.timestamp;
        *label = source.label;
        *priority = source.priority;
        *slot = source.slot;
        *bar = source.bar;
    }
}

impl Clone for State {
    fn clone(&self) -> Self {
        let Self {
            cursor,
            next_dispatch,
            visit_primary_after_fill,
            updates,
            values,
            unready,
            clocks,
            last_labels,
            last_sort_key,
            last_position,
            last_reached,
            last_ready,
            flipped,
            fired,
            close_bar,
            entered_from_flat,
            policy,
        } = self;
        Self {
            cursor: cursor.clone(),
            next_dispatch: *next_dispatch,
            visit_primary_after_fill: *visit_primary_after_fill,
            updates: updates.clone(),
            values: values.clone(),
            unready: unready.clone(),
            clocks: clocks.clone(),
            last_labels: last_labels.clone(),
            last_sort_key: *last_sort_key,
            last_position: *last_position,
            last_reached: *last_reached,
            last_ready: *last_ready,
            flipped: *flipped,
            fired: fired.clone(),
            close_bar: *close_bar,
            entered_from_flat: *entered_from_flat,
            policy: *policy,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self {
            cursor,
            next_dispatch,
            visit_primary_after_fill,
            updates,
            values,
            unready,
            clocks,
            last_labels,
            last_sort_key,
            last_position,
            last_reached,
            last_ready,
            flipped,
            fired,
            close_bar,
            entered_from_flat,
            policy,
        } = self;
        cursor.clone_from(&source.cursor);
        *next_dispatch = source.next_dispatch;
        *visit_primary_after_fill = source.visit_primary_after_fill;
        updates.clone_from(&source.updates);
        values.clone_from(&source.values);
        unready.clone_from(&source.unready);
        clocks.clone_from(&source.clocks);
        last_labels.clone_from(&source.last_labels);
        *last_sort_key = source.last_sort_key;
        *last_position = source.last_position;
        *last_reached = source.last_reached;
        *last_ready = source.last_ready;
        *flipped = source.flipped;
        fired.clone_from(&source.fired);
        *close_bar = source.close_bar;
        *entered_from_flat = source.entered_from_flat;
        *policy = source.policy;
    }
}

struct Registration {
    slot: usize,
    stream_index: i64,
    rank: i64,
    /// `ScheduledStream.built_by_strategy`: whether the strategy builds this
    /// leg from the primary bars itself, which hands a window a gap closed
    /// over ahead of the primary bar that revealed it.
    built: bool,
    timestamps: Py<PyArray1<i64>>,
    labels: Option<Py<PyArray1<i64>>>,
}

/// Borrowed views of every array, for the length of one call.
struct View<'a> {
    reg_ts: Vec<&'a [i64]>,
    reg_labels: Vec<Option<&'a [i64]>>,
    slot_close: Vec<&'a [f64]>,
    slot_admits: Vec<Option<&'a [bool]>>,
    key_values: Vec<&'a [f64]>,
    key_init: Vec<&'a [bool]>,
    dispatch_ts: &'a [i64],
    dispatch_pos: &'a [i64],
    primary_close: &'a [f64],
    /// The ATR exit's values and initialized flags, in its stream's bars.
    atr: Option<(&'a [f64], &'a [bool])>,
    borrow: Option<BorrowView<'a>>,
}

struct Data {
    regs: Vec<Registration>,
    slot_close: Vec<Py<PyArray1<f64>>>,
    slot_admits: Vec<Option<Py<PyArray1<bool>>>>,
    slot_decision: Vec<bool>,
    slot_keys: Vec<Vec<usize>>,
    key_values: Vec<Py<PyArray1<f64>>>,
    key_init: Vec<Py<PyArray1<bool>>>,
    conditions: Vec<Condition>,
    /// The root condition of each exit and entry rule, in rule order.
    exits: Vec<usize>,
    entries: Vec<usize>,
    level: Option<Level>,
    legacy: bool,
    /// `StrategySpec.crossings` ``each_bar``: each clock's crossing, in
    /// clock order, worked out once per evaluated bar ahead of every rule
    /// (`DecisionProgram._step_crossings`), which then only reads it. `None`
    /// is ``each_check``: a crossing is worked out where a rule checks it.
    each_bar: Option<Vec<Crossing>>,
    dispatch_ts: Py<PyArray1<i64>>,
    dispatch_pos: Py<PyArray1<i64>>,
    primary_close: Py<PyArray1<f64>>,
    dust: bool,
    position_required: bool,
    has_instrument: bool,
    size_increment: f64,
    minimum_notional: f64,
    /// Whether a long-only close latches entries off until it settles: every
    /// run but a two-sided one (`NativeExecutionSession._two_sided`).
    close_latch: bool,
    /// The side an entry opens: 1, or -1 for a short node (see
    /// [`delivery_step`]).
    entry_side: i32,
    /// StrategyBase's risk checks, where the plan has any. They put the
    /// session on the lane that visits every primary bar in full
    /// (`_defer_quiet_primary` off): every bar is a callback, nothing is
    /// deferred, and a quiet bar is never taken.
    policies: Option<Policies>,
    /// The open short's borrow, which the trade and ATR exits charge: on a
    /// run that may short, under one of them.
    borrow: Option<Borrow>,
}

fn slice<'a, T: numpy::Element>(array: &'a Py<PyArray1<T>>, py: Python<'a>) -> PyResult<&'a [T]> {
    // SAFETY: every array is a prepared, immutable input of one backtest run,
    // kept alive by the `Py` handle for as long as the session; nothing writes
    // to it while a run is in progress.
    unsafe { array.as_ref(py).as_slice() }
        .map_err(|_| PyValueError::new_err("rule session arrays must be contiguous"))
}

/// `stamps.partition_point(|&t| t < timestamp)` over rising `stamps`, where
/// `stamps[from] < timestamp`: `hint` when it is that point, else galloping
/// out from `from`, which a cursor a few bars behind reaches in a handful
/// of reads.
fn first_at_or_after(stamps: &[i64], from: usize, timestamp: i64, hint: usize) -> usize {
    if from < hint
        && hint < stamps.len()
        && stamps[hint - 1] < timestamp
        && stamps[hint] >= timestamp
    {
        return hint;
    }
    let (mut below, mut step) = (from, 1);
    loop {
        let probe = below + step;
        if probe >= stamps.len() || stamps[probe] >= timestamp {
            let bound = probe.min(stamps.len());
            return below + 1 + stamps[below + 1..bound].partition_point(|&t| t < timestamp);
        }
        below = probe;
        step *= 2;
    }
}

impl Data {
    fn view<'a>(&'a self, py: Python<'a>) -> PyResult<View<'a>> {
        let mut reg_ts = Vec::with_capacity(self.regs.len());
        let mut reg_labels = Vec::with_capacity(self.regs.len());
        for reg in &self.regs {
            reg_ts.push(slice(&reg.timestamps, py)?);
            reg_labels.push(match &reg.labels {
                Some(labels) => Some(slice(labels, py)?),
                None => None,
            });
        }
        Ok(View {
            reg_ts,
            reg_labels,
            slot_close: self.slot_close.iter().map(|a| slice(a, py)).collect::<PyResult<_>>()?,
            slot_admits: self
                .slot_admits
                .iter()
                .map(|a| a.as_ref().map(|a| slice(a, py)).transpose())
                .collect::<PyResult<_>>()?,
            key_values: self.key_values.iter().map(|a| slice(a, py)).collect::<PyResult<_>>()?,
            key_init: self.key_init.iter().map(|a| slice(a, py)).collect::<PyResult<_>>()?,
            dispatch_ts: slice(&self.dispatch_ts, py)?,
            dispatch_pos: slice(&self.dispatch_pos, py)?,
            primary_close: slice(&self.primary_close, py)?,
            atr: match self.policies.as_ref().and_then(|p| p.atr.as_ref()) {
                Some((_, values, init)) => Some((slice(values, py)?, slice(init, py)?)),
                None => None,
            },
            borrow: match &self.borrow {
                Some(borrow) => Some(BorrowView {
                    days: slice(&borrow.days, py)?,
                    settles: slice(&borrow.settles, py)?,
                    first_day: borrow.first_day,
                    charged: slice(&borrow.charged, py)?,
                }),
                None => None,
            },
        })
    }

    /// `ScheduledStream.order`: bar `index`'s phase, and its place in it.
    fn order(&self, view: &View, reg: usize, index: usize) -> (i64, i64) {
        let registration = &self.regs[reg];
        if registration.slot == PRIMARY {
            return (PHASE_PRIMARY, 0);
        }
        if let Some(labels) = view.reg_labels[reg] {
            let (label, dispatched) = (labels[index], view.reg_ts[reg][index]);
            if label > dispatched {
                return (PHASE_PRECLOSE, registration.stream_index);
            }
            if label < dispatched && registration.built {
                return (PHASE_CLOSED_BY_GAP, registration.rank);
            }
        }
        (PHASE_COMPLETED, registration.rank)
    }

    /// `NativeScheduleCursor._select`.
    fn select(&self, view: &View, cursor: &mut Cursor) {
        let mut selected: Option<(usize, i64, i64, i64, usize)> = None;
        for reg in 0..self.regs.len() {
            let index = cursor.indexes[reg];
            if index >= view.reg_ts[reg].len() {
                continue;
            }
            let timestamp = view.reg_ts[reg][index];
            let (priority, rank) = self.order(view, reg, index);
            let better = match selected {
                None => true,
                Some((_, ts, pri, rk, _)) => (timestamp, priority, rank) < (ts, pri, rk),
            };
            if better {
                selected = Some((reg, timestamp, priority, rank, index));
            }
        }
        cursor.current = selected.map(|(reg, ..)| reg);
        if let Some((reg, timestamp, priority, _, index)) = selected {
            cursor.timestamp = timestamp;
            cursor.label = match view.reg_labels[reg] {
                Some(labels) => labels[index],
                None => view.reg_ts[reg][index],
            };
            cursor.priority = priority;
            cursor.slot = self.regs[reg].slot;
            cursor.bar = index;
        }
    }

    fn cursor_advance(&self, view: &View, cursor: &mut Cursor) {
        if let Some(reg) = cursor.current {
            cursor.indexes[reg] += 1;
            self.select(view, cursor);
        }
    }

    /// `NativeScheduleCursor.skip_primary_before`; `hint` is where the first
    /// primary bar at or after `timestamp` likely stands.
    fn skip_primary_before(&self, view: &View, cursor: &mut Cursor, timestamp: i64, hint: usize) {
        let mut changed = false;
        for reg in 0..self.regs.len() {
            if self.regs[reg].slot != PRIMARY {
                continue;
            }
            // The cursor only moves forward, so the first primary bar at or
            // after `timestamp` is searched for from where it stands.
            let (stamps, from) = (&view.reg_ts[reg], cursor.indexes[reg]);
            if from < stamps.len() && stamps[from] < timestamp {
                cursor.indexes[reg] = first_at_or_after(stamps, from, timestamp, hint);
                changed = true;
            }
        }
        if changed {
            self.select(view, cursor);
        }
    }

    /// `_DeclarativeNode.decision_position`.
    fn decision_position(&self, side: i32, quantity: f64, pending: Option<f64>, close: f64) -> i32 {
        if side != 0
            && self.has_instrument
            && self.position_required
            && pending.is_none_or(|p| p != 0.0)
        {
            let quantity = quantity.abs();
            if quantity < self.size_increment
                || (self.minimum_notional > 0.0 && quantity * close < self.minimum_notional)
            {
                return 0;
            }
        }
        side
    }

    /// `NativeExecutionSession._reopens_exits`.
    #[allow(clippy::too_many_arguments)]
    fn reopens_exits(
        &self,
        view: &View,
        state: &State,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        start: usize,
        stop: usize,
    ) -> bool {
        if !self.dust || side == 0 || stop <= start {
            return false;
        }
        if let Some(level) = &self.level {
            return self.level_side_moves(view, state, level, side, quantity, pending, start, stop);
        }
        if state.last_position != 0 {
            return false;
        }
        let close = if stop == start + 1 {
            view.primary_close[start]
        } else {
            // numpy's max: NaN as soon as any element is NaN.
            let mut highest = f64::NEG_INFINITY;
            for &value in &view.primary_close[start..stop] {
                if value.is_nan() {
                    highest = f64::NAN;
                    break;
                }
                if value > highest {
                    highest = value;
                }
            }
            highest
        };
        self.decision_position(side, quantity, pending, close) != 0
    }

    /// `NativeExecutionSession._level_side_moves`: whether a quiet primary
    /// bar in `[start, stop)` sees another side than the last evaluation, in
    /// the rules' reading or the executor's.
    #[allow(clippy::too_many_arguments)]
    fn level_side_moves(
        &self,
        view: &View,
        state: &State,
        level: &Level,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        start: usize,
        stop: usize,
    ) -> bool {
        let closes = &view.primary_close[start..stop];
        let mut probes = [closes[0], f64::NAN, f64::NAN];
        let count = if stop == start + 1 {
            1
        } else {
            // numpy's fmin and fmax: NaN only when every close is NaN.
            probes[0] = closes.iter().fold(f64::NAN, |low, &close| low.min(close));
            probes[1] = closes.iter().fold(f64::NAN, |high, &close| high.max(close));
            if closes.iter().any(|close| close.is_nan()) {
                3
            } else {
                2
            }
        };
        let seen = state.last_position;
        for &close in &probes[..count] {
            if self.decision_position(side, quantity, pending, close) != seen {
                return true;
            }
            if level.reads_dust_as_flat(quantity, pending, close) {
                if seen != 0 {
                    return true;
                }
            } else if seen != side {
                return true;
            }
        }
        false
    }

    fn next_dispatch_position(&self, view: &View, state: &State) -> usize {
        if state.next_dispatch < view.dispatch_pos.len() {
            view.dispatch_pos[state.next_dispatch] as usize
        } else {
            view.primary_close.len()
        }
    }

    /// Whether an event on `slot` leaves the next primary bar a visit owed.
    ///
    /// A legacy program evaluates on every primary bar, and one between two
    /// of its own composites repeats the last evaluation -- which is what
    /// lets the session skip it. A slot the program reads and never decides
    /// on (another symbol's reference leg) moves the operands and evaluates
    /// nothing, so the first primary bar after it is the first to evaluate
    /// them: `NativeExecutionSession._owes_primary`.
    #[inline]
    fn owes_primary(&self, slot: usize) -> bool {
        self.legacy && slot != PRIMARY && !self.slot_decision[slot]
    }

    fn reached(&self, state: &State, slot: usize) -> bool {
        if slot == PRIMARY {
            return !state.unready.iter().any(|&u| u);
        }
        let keys = &self.slot_keys[slot];
        if keys.is_empty() {
            return !self.legacy && self.slot_decision[slot];
        }
        !keys.iter().any(|&k| state.unready[k])
    }

    fn ready(&self, state: &State, slot: usize) -> bool {
        if !self.legacy {
            return !state.unready.iter().any(|&u| u);
        }
        self.reached(state, slot)
    }

    fn operand(state: &State, operand: Operand) -> Option<f64> {
        match operand {
            Operand::Key(key) => state.values[key],
            Operand::Const(value) => Some(value),
        }
    }

    /// One compiled `CrossoverCondition`.
    fn crossing(state: &mut State, rule: &Crossing) -> bool {
        let (Some(a), Some(b)) = (Self::operand(state, rule.fast), Self::operand(state, rule.slow))
        else {
            return false;
        };
        let previous = state.clocks[rule.clock];
        state.clocks[rule.clock] = Some((a, b));
        match previous {
            None => false,
            Some((p0, p1)) => {
                if (p0 > p1) != (a > b) || (p0 < p1) != (a < b) {
                    state.flipped = true;
                }
                if rule.above {
                    p0 <= p1 && a > b
                } else {
                    p0 >= p1 && a < b
                }
            }
        }
    }

    /// One compiled condition, short-circuiting as `all` and `any` do: which
    /// clocks a bar steps depends on it, unless the bar stepped them all
    /// first (`Data::each_bar`).
    fn holds(&self, state: &mut State, node: usize) -> bool {
        match &self.conditions[node] {
            Condition::All(children) => children.iter().all(|&child| self.holds(state, child)),
            Condition::Any(children) => children.iter().any(|&child| self.holds(state, child)),
            Condition::Not(child) => !self.holds(state, *child),
            Condition::Zone { value, lower, upper } => {
                Self::operand(state, *value).is_some_and(|v| *lower <= v && v <= *upper)
            }
            Condition::Threshold { left, right, compare } => {
                match (Self::operand(state, *left), Self::operand(state, *right)) {
                    (Some(a), Some(b)) => compare.holds(a, b),
                    _ => false,
                }
            }
            Condition::Cross(rule) => {
                if self.each_bar.is_some() {
                    state.fired[rule.clock]
                } else {
                    Self::crossing(state, rule)
                }
            }
        }
    }

    /// `DecisionProgram.evaluate` over the node's current updates, for an
    /// event of dispatch phase `phase` -- the schedule's, as the event
    /// carries it (`DecisionEvent.dispatch_phase`).
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        state: &mut State,
        slot: usize,
        label: i64,
        dispatch: i64,
        phase: i64,
        position: i32,
    ) -> PyResult<RuleMatch> {
        let previous = state.last_labels[slot];
        if previous == Some(label) {
            return Ok(RuleMatch { ready: true, ..RuleMatch::default() });
        }
        let order = (dispatch, phase);
        if previous.is_some_and(|p| p > label) || state.last_sort_key.is_some_and(|k| order < k) {
            return Err(PyValueError::new_err("Events must follow configured dispatch order"));
        }
        state.last_labels[slot] = Some(label);
        state.last_sort_key = Some(order);
        for &key in &self.slot_keys[slot] {
            let value = state.updates[key];
            state.values[key] = value;
            state.unready[key] = value.is_none();
        }
        let reached = self.reached(state, slot);
        if !self.slot_decision[slot] {
            return Ok(RuleMatch { ready: true, reached, ..RuleMatch::default() });
        }
        if !self.ready(state, slot) {
            return Ok(RuleMatch { ready: false, reached, ..RuleMatch::default() });
        }
        // Every crossing, once, before any rule reads one -- flat or not, and
        // whichever rules match first.
        if let Some(crossings) = &self.each_bar {
            for rule in crossings {
                state.fired[rule.clock] = Self::crossing(state, rule);
            }
        }
        let exit_rule = if position != 0 || !self.legacy {
            self.exits.iter().position(|&rule| self.holds(state, rule))
        } else {
            None
        };
        let entry_rule = self.entries.iter().position(|&rule| self.holds(state, rule));
        Ok(RuleMatch { evaluated: true, ready: true, exit_rule, entry_rule, reached: true })
    }

    /// `NativeModelRunner.visit` and `_DeclarativeNode.visit`.
    ///
    /// `resting` is an entry an earlier visit at this instant decided from
    /// flat, still resting (`_DeclarativeNode.resting_side`): the rules read
    /// it as held where the account reads flat, or dust.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        view: &View,
        state: &mut State,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
        resting: i32,
    ) -> PyResult<Option<NodeIntent>> {
        let (slot, bar, dispatch, label, phase) = (
            state.cursor.slot,
            state.cursor.bar,
            state.cursor.timestamp,
            state.cursor.label,
            state.cursor.priority,
        );
        if let Some(admits) = view.slot_admits[slot] {
            if !admits[bar] {
                state.last_reached = false;
                return Ok(None);
            }
        }
        if let Some(policies) = &self.policies {
            if slot != PRIMARY {
                Self::policy_stream(policies, &mut state.policy, slot, bar, phase < PHASE_PRIMARY);
            }
        }
        // A composite dispatched early first lands its primary bar, which
        // updates the primary stream's indicators: none, in this lane.
        for &key in &self.slot_keys[slot] {
            state.updates[key] =
                if view.key_init[key][bar] { Some(view.key_values[key][bar]) } else { None };
        }
        let close = view.slot_close[slot][bar];
        let mut position = side;
        if position != 0 && self.position_required {
            position = self.decision_position(position, quantity, pending, close);
        }
        if position == 0 {
            position = resting;
        }
        let matched = self.evaluate(state, slot, label, dispatch, phase, position)?;
        state.last_reached = matched.reached;
        state.last_ready = matched.ready;
        if !matched.evaluated {
            return Ok(None);
        }
        state.last_position = position;
        let exiting = matched.exit_rule.is_some() && position != 0;
        Ok(Some(NodeIntent {
            wants_entry: matched.entry_rule.is_some() && !exiting,
            wants_exit: exiting,
            reference: close,
            reason: if exiting {
                matched.exit_rule.map(|i| (true, i))
            } else {
                matched.entry_rule.map(|i| (false, i))
            },
        }))
    }

    /// Whether the event `cursor` stands on is decided ahead of the primary
    /// bar it is delivered with -- a window a gap closed, or a pre-close
    /// window (`ScheduledBar.ahead_of_primary`) -- and the primary bar the
    /// latch reads it on in the callback on primary bar `idx` at `now`: a
    /// primary event's own, a resample's `idx` on the callback's clock and
    /// the bar before off it.
    #[inline]
    fn event_bar(cursor: &Cursor, now: i64, idx: usize) -> (bool, i64) {
        if cursor.slot == PRIMARY {
            (false, cursor.bar as i64)
        } else {
            (
                cursor.priority < PHASE_PRIMARY,
                if cursor.timestamp == now { idx as i64 } else { idx as i64 - 1 },
            )
        }
    }

    /// `NativeExecutionSession.advance` for a repeat-safe plan with no
    /// borrow and signal entries: deferred where it has no risk policies,
    /// and visiting every primary bar in full where it has.
    ///
    /// `has_position` and `net_position` are the lot as the rules read it
    /// (dust as flat); `account` is the venue's, which the risk checks read.
    /// `apply_policies` false is a halted run's callback, which only moves
    /// the policies' stream counts.
    ///
    /// The flag says whether the call got past the deferral, which is where
    /// the Python session refreshes its account view; `None` is `IDLE`.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        view: &View,
        state: &mut State,
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> PyResult<(bool, Option<Target>)> {
        state.entered_from_flat = false;
        let held_side = if !has_position {
            0
        } else if net_position > 0.0 {
            1
        } else if net_position < 0.0 {
            -1
        } else {
            0
        };
        let policies = self.policies.as_ref();
        if let Some(borrow) = &view.borrow {
            // `advance`'s `self._borrow.observe(ctx)`, ahead of the events: a
            // run that borrows has risk exits, so nothing is deferred.
            borrow.observe(&mut state.policy, idx, account.quantity)?;
        }
        if policies.is_none() {
            let due = state.next_dispatch < view.dispatch_ts.len();
            let next_dispatch = if due { view.dispatch_ts[state.next_dispatch] } else { now };
            if !state.visit_primary_after_fill
                && (!due || now < next_dispatch)
                && !(has_position
                    && self.reopens_exits(
                        view,
                        state,
                        held_side,
                        net_position,
                        pending,
                        idx,
                        idx + 1,
                    ))
            {
                return Ok((false, None));
            }
            // The bar stepping now is the primary bar at `now`.
            self.skip_primary_before(view, &mut state.cursor, now.min(next_dispatch), idx);
            while state.next_dispatch < view.dispatch_ts.len()
                && view.dispatch_ts[state.next_dispatch] <= now
            {
                state.next_dispatch += 1;
            }
        }
        if let Some(level) = &self.level {
            let quantity = if has_position { net_position } else { 0.0 };
            let target = self.advance_level(
                view,
                state,
                level,
                now,
                idx,
                held_side,
                quantity,
                pending,
                account,
                apply_policies,
            )?;
            return Ok((true, target));
        }
        // `_refresh`: the account the rules see for the rest of the call.
        let quantity = if has_position { net_position } else { 0.0 };
        let start_side = held_side;
        let mut target = start_side;
        let mut reference: Option<f64> = None;
        let mut reason: Option<RuleId> = None;
        let mut entry_timestamp: Option<i64> = None;
        let mut latch = Latch::new(state.close_bar);
        // The model's close this callback, and whether it was decided ahead
        // of both risk checks: the primary bar's decision, a window a gap
        // closed or a pre-close window.
        let mut model_exit: Option<(f64, bool)> = None;
        while state.cursor.current.is_some() && state.cursor.timestamp <= now {
            let timestamp = state.cursor.timestamp;
            let (early, event_bar) = Self::event_bar(&state.cursor, now, idx);
            latch.step(event_bar, early, start_side);
            if state.cursor.slot == PRIMARY
                && !state.visit_primary_after_fill
                && entry_timestamp != Some(timestamp)
                && (!self.dust
                    || !self.reopens_exits(
                        view,
                        state,
                        start_side,
                        quantity,
                        pending,
                        state.cursor.bar,
                        state.cursor.bar + 1,
                    ))
            {
                self.cursor_advance(view, &mut state.cursor);
                continue;
            }
            let slot = state.cursor.slot;
            // An entry an earlier visit at this instant decided from flat is
            // still resting: the rules run their exit clocks over it, and
            // whatever they decide finds nothing to act on -- an exit has no
            // position to sell, an entry the side already decided. Only the
            // lane that visits every bar has the visit: on the deferred one
            // it moves a clock early to where the next visit takes it anyway.
            let resting = if policies.is_some()
                && start_side == 0
                && target != 0
                && entry_timestamp == Some(timestamp)
            {
                target
            } else {
                0
            };
            let intent = self.visit(view, state, start_side, quantity, pending, resting)?;
            let intent = if resting != 0 { None } else { intent };
            self.cursor_advance(view, &mut state.cursor);
            // Only a primary bar the rules ran on settles a visit owed: one
            // outside the session, or still warming up, evaluates nothing.
            if slot == PRIMARY {
                if state.last_reached {
                    state.visit_primary_after_fill = false;
                }
            } else if self.owes_primary(slot) {
                state.visit_primary_after_fill = true;
            }
            let Some(intent) = intent else { continue };
            let entering = entry_timestamp == Some(timestamp);
            let Some(moved) = delivery_step(
                intent.wants_entry,
                intent.wants_exit,
                self.entry_side,
                target,
                start_side,
                entering,
                &latch,
            ) else {
                // Moves nothing; the clocks have moved on regardless.
                continue;
            };
            let previous_target = std::mem::replace(&mut target, moved);
            if previous_target == 0 && target != 0 {
                entry_timestamp = Some(timestamp);
                state.entered_from_flat = true;
            }
            reference = Some(intent.reference);
            reason = intent.reason;
            if start_side != 0 && previous_target == start_side && target == 0 {
                model_exit = Some((intent.reference, slot == PRIMARY || early));
            }
            if self.close_latch && previous_target != 0 && target == 0 {
                latch.close(event_bar, early, idx as i64);
            }
        }
        let risk = policies.and_then(|policies| {
            self.policy_target(
                policies,
                view,
                state,
                &mut latch,
                account,
                idx,
                apply_policies,
                start_side,
                target,
                model_exit,
            )
        });
        state.close_bar = latch.close_bar;
        if risk.is_some() {
            return Ok((true, risk));
        }
        if target == start_side {
            return Ok((true, None));
        }
        Ok((true, Some(Target::model(target, reference, reason))))
    }

    /// `advance`'s risk checks, once the callback's events are decided:
    /// the close a risk exit takes over the model's `target`, if one acts.
    /// `model_exit` is the model's own close this callback, if it decided
    /// one, and whether ahead of both checks. `apply_policies` false (a
    /// halted run) only tells the policies of the closes they were held
    /// from.
    #[allow(clippy::too_many_arguments)]
    fn policy_target(
        &self,
        policies: &Policies,
        view: &View,
        state: &mut State,
        latch: &mut Latch,
        account: Account,
        idx: usize,
        apply_policies: bool,
        start_side: i32,
        target: i32,
        model_exit: Option<(f64, bool)>,
    ) -> Option<Target> {
        let exit = if apply_policies {
            let primary = view.slot_admits[PRIMARY].is_none_or(|admits| admits[idx]);
            Self::apply_policies(policies, view, &mut state.policy, account, idx, primary)
        } else {
            Self::release_policy_streams(&mut state.policy);
            None
        };
        // A close of a lot the rules read as dust: `_make_tradeable_qty`
        // skips the order, so the bar's own decision is what stands. A check
        // acts only on a lot, so the close it takes moves the target.
        let exit = exit.filter(|_| side_of(account.quantity) == start_side)?;
        let preceding = model_exit
            .filter(|&(_, before_check)| {
                target == 0 && self.close_latch && (before_check || state.policy.resampled)
            })
            .map(|(model_reference, _)| model_reference);
        if start_side != 0 && self.close_latch {
            // `handle_bar`'s risk check closes after the reconcile.
            latch.close_bar = latch.close_bar.max(idx as i64);
        }
        Some(Target {
            side: 0,
            reference: Some(view.primary_close[idx]),
            reason: None,
            policy: Some(exit),
            preceding,
            style: policies.style,
        })
    }

    /// `NativeModelRunner._advance_readiness` for the policies: a close of
    /// resample `slot`'s bar `bar`, told now where it is decided ahead of the
    /// primary bar's risk check (`ahead`), else held back until that check
    /// has run (`_held_policy_streams`).
    fn policy_stream(
        policies: &Policies,
        p: &mut PolicyState,
        slot: usize,
        bar: usize,
        ahead: bool,
    ) {
        let atr = policies.atr.as_ref().is_some_and(|(atr, ..)| atr.slot == slot);
        let hold = policies.hold.is_some_and(|hold| hold.slot == slot);
        if ahead {
            if atr {
                p.atr_index = bar as i64;
            }
            if hold {
                p.hold_counted += 1;
            }
            return;
        }
        p.held = true;
        if atr {
            p.held_atr = bar as i64;
        }
        if hold {
            p.held_counts += 1;
        }
    }

    /// `release_policy_streams`.
    fn release_policy_streams(p: &mut PolicyState) {
        if p.held_atr >= 0 {
            p.atr_index = p.held_atr;
        }
        p.hold_counted += p.held_counts;
        p.held = false;
        p.held_atr = -1;
        p.held_counts = 0;
    }

    /// `NativeModelRunner.apply_policies` for the executing primary bar
    /// `idx`: checked on the bar -- unless the session refuses it
    /// (`primary`) -- and again once the resample closes it held back have
    /// reached the policies.
    fn apply_policies(
        policies: &Policies,
        view: &View,
        p: &mut PolicyState,
        account: Account,
        idx: usize,
        primary: bool,
    ) -> Option<PolicyExit> {
        let mut selected =
            if primary { Self::policy_intent(policies, view, p, account, idx) } else { None };
        p.resampled = false;
        let resampled = p.held;
        Self::release_policy_streams(p);
        if selected.is_none() && resampled {
            selected = Self::policy_intent(policies, view, p, account, idx);
            p.resampled = selected.is_some();
        }
        selected
    }

    /// `_policy_intent`: every policy's `on_bar`, in order, the last to act
    /// deciding.
    fn policy_intent(
        policies: &Policies,
        view: &View,
        p: &mut PolicyState,
        account: Account,
        idx: usize,
    ) -> Option<PolicyExit> {
        let current = view.primary_close[idx];
        let quantity = account.quantity;
        let mut selected = None;
        if let Some(trade) = &policies.trade {
            p.trade_charges.held = quantity;
            let entry = account.entry;
            if quantity != 0.0 && entry > 0.0 {
                let costs =
                    p.trade_charges.round_trip(policies.fee_rate, quantity.abs(), entry, current)
                        + p.borrow;
                let roi = trade_roi_pct(entry, current, quantity, costs);
                if trade.stop > 0.0 && roi <= -trade.stop {
                    selected = Some(PolicyExit::TradeStopLoss);
                } else if trade.take_profit > 0.0 && roi >= trade.take_profit {
                    selected = Some(PolicyExit::TradeTakeProfit);
                }
            }
        }
        if let (Some((atr, ..)), Some((values, initialized))) = (&policies.atr, view.atr) {
            if let Some(exit) =
                Self::atr_exit(policies, atr, values, initialized, p, account, idx, current)
            {
                selected = Some(exit);
            }
        }
        if let Some(hold) = &policies.hold {
            if quantity == 0.0 {
                p.hold_mark = -1;
            } else {
                if p.hold_mark < 0 {
                    p.hold_mark = p.hold_counted;
                }
                if p.hold_counted - p.hold_mark >= hold.bars {
                    selected = Some(PolicyExit::MaxHold);
                }
            }
        }
        selected
    }

    /// `NativeAtrExitPolicy.on_bar`.
    #[allow(clippy::too_many_arguments)]
    fn atr_exit(
        policies: &Policies,
        atr: &AtrExit,
        values: &[f64],
        initialized: &[bool],
        p: &mut PolicyState,
        account: Account,
        idx: usize,
        current: f64,
    ) -> Option<PolicyExit> {
        let quantity = account.quantity;
        p.atr_charges.held = quantity;
        let index = if atr.slot == PRIMARY { idx as i64 } else { p.atr_index };
        if index < 0 || !initialized[index as usize] {
            return None;
        }
        if quantity == 0.0 {
            p.atr_entry = 0.0;
            p.atr_charges.charged = 0.0;
            p.atr_extreme = 0.0;
            return None;
        }
        if p.atr_entry == 0.0 {
            p.atr_entry = account.entry;
            if p.atr_entry <= 0.0 {
                return None;
            }
            p.atr_extreme = p.atr_entry;
        }
        let long = quantity > 0.0;
        if atr.trailing {
            // Python's `max` and `min`: the first argument unless the second
            // compares past it.
            let extreme = p.atr_extreme;
            p.atr_extreme = if long {
                if current > extreme {
                    current
                } else {
                    extreme
                }
            } else if current < extreme {
                current
            } else {
                extreme
            };
        }
        let anchor = if atr.trailing { p.atr_extreme } else { p.atr_entry };
        let units = quantity.abs();
        let commission = p.atr_charges.round_trip(policies.fee_rate, units, p.atr_entry, current);
        let per_share = if units > 0.0 { (commission + p.borrow) / units } else { 0.0 };
        let direction = if long { 1.0 } else { -1.0 };
        let value = values[index as usize];
        if atr.stop > 0.0 {
            let stop = anchor - direction * (value * atr.stop - per_share);
            if if long { current <= stop } else { current >= stop } {
                p.atr_entry = 0.0;
                p.atr_extreme = 0.0;
                return Some(PolicyExit::AtrStop);
            }
        }
        if atr.take_profit > 0.0 {
            let take_profit = p.atr_entry + direction * (value * atr.take_profit + per_share);
            if if long { current >= take_profit } else { current <= take_profit } {
                p.atr_entry = 0.0;
                p.atr_extreme = 0.0;
                return Some(PolicyExit::AtrTakeProfit);
            }
        }
        None
    }

    /// The policies' `on_fill`, the account refreshed: a fill of `quantity`
    /// on `side` (1 buys) charged `commission`, which left it `flat` -- and
    /// the venue holding signed `position`, at primary bar `bar`, where the
    /// run borrows.
    #[allow(clippy::too_many_arguments)]
    fn policy_fill(
        &self,
        view: &View,
        state: &mut State,
        side: f64,
        quantity: f64,
        commission: f64,
        flat: bool,
        fill: Option<(usize, f64)>,
    ) -> PyResult<()> {
        let Some(policies) = &self.policies else {
            return Ok(());
        };
        let p = &mut state.policy;
        if let Some(borrow) = &view.borrow {
            // `on_fill`'s `_borrow.on_fill` and `observe`: a fill on this
            // lane is stamped by its bar's clock, so the loan it opens
            // settles as one opened on the bar.
            let Some((bar, position)) = fill else {
                return Err(PyValueError::new_err(
                    "a fill on a run that borrows names its bar and position",
                ));
            };
            borrow.observe(p, bar, position)?;
        }
        if policies.trade.is_some() {
            p.trade_charges.on_fill(side, quantity, commission, flat);
        }
        if policies.atr.is_some() && p.atr_charges.on_fill(side, quantity, commission, flat) {
            p.atr_entry = 0.0;
            p.atr_extreme = 0.0;
        }
        Ok(())
    }

    /// The events of one `advance` past its deferral, for level rules.
    ///
    /// Every primary event is visited -- a level rule's repeat can act.
    /// Deferred, the next primary bar is owed a visit after any event that
    /// flipped a clock, after a bar that acted, and after one whose rules saw
    /// another side than the one the bar moves from; only a primary bar the
    /// rules ran on settles that debt. Under risk policies nothing is
    /// deferred, and the checks run after the events as `advance`'s do.
    #[allow(clippy::too_many_arguments)]
    fn advance_level(
        &self,
        view: &View,
        state: &mut State,
        level: &Level,
        now: i64,
        idx: usize,
        held_side: i32,
        quantity: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> PyResult<Option<Target>> {
        let policies = self.policies.as_ref();
        let deferred = policies.is_none();
        // The side a decision moves from: the venue's, or flat for dust.
        let start_side = if held_side != 0
            && level.reads_dust_as_flat(quantity, pending, view.primary_close[idx])
        {
            0
        } else {
            held_side
        };
        let mut target = start_side;
        let mut reference: Option<f64> = None;
        let mut reason: Option<RuleId> = None;
        let mut entry_timestamp: Option<i64> = None;
        let mut latch = Latch::new(state.close_bar);
        let mut model_exit: Option<(f64, bool)> = None;
        while state.cursor.current.is_some() && state.cursor.timestamp <= now {
            let timestamp = state.cursor.timestamp;
            let slot = state.cursor.slot;
            let primary = slot == PRIMARY;
            let (early, event_bar) = Self::event_bar(&state.cursor, now, idx);
            latch.step(event_bar, early, start_side);
            state.flipped = false;
            // An entry decided from flat earlier at this instant, still
            // resting: see `advance`.
            let resting = if !deferred
                && start_side == 0
                && target != 0
                && entry_timestamp == Some(timestamp)
            {
                target
            } else {
                0
            };
            let intent = self.visit(view, state, held_side, quantity, pending, resting)?;
            let intent = if resting != 0 { None } else { intent };
            self.cursor_advance(view, &mut state.cursor);
            if primary {
                if state.last_reached {
                    state.visit_primary_after_fill = false;
                }
            } else if self.owes_primary(slot) {
                state.visit_primary_after_fill = true;
            }
            if deferred && state.flipped {
                state.visit_primary_after_fill = true;
            }
            let Some(intent) = intent else { continue };
            let entering = entry_timestamp == Some(timestamp);
            let Some(moved) = delivery_step(
                intent.wants_entry,
                intent.wants_exit,
                self.entry_side,
                target,
                start_side,
                entering,
                &latch,
            ) else {
                continue;
            };
            let previous_target = std::mem::replace(&mut target, moved);
            if previous_target == 0 && target != 0 {
                entry_timestamp = Some(timestamp);
                state.entered_from_flat = true;
            }
            reference = Some(intent.reference);
            reason = intent.reason;
            if start_side != 0 && previous_target == start_side && target == 0 {
                model_exit = Some((intent.reference, primary || early));
            }
            if self.close_latch && previous_target != 0 && target == 0 {
                latch.close(event_bar, early, idx as i64);
            }
        }
        let risk = policies.and_then(|policies| {
            self.policy_target(
                policies,
                view,
                state,
                &mut latch,
                account,
                idx,
                apply_policies,
                start_side,
                target,
                model_exit,
            )
        });
        state.close_bar = latch.close_bar;
        if risk.is_some() {
            return Ok(risk);
        }
        if target == start_side {
            if deferred && state.last_position != start_side {
                state.visit_primary_after_fill = true;
            }
            return Ok(None);
        }
        if deferred {
            state.visit_primary_after_fill = true;
        }
        Ok(Some(Target::model(target, reference, reason)))
    }

    /// `NativeExecutionSession.note_close`.
    fn note_close(&self, state: &mut State, bar: i64) {
        if self.close_latch {
            state.close_bar = state.close_bar.max(bar);
        }
    }

    /// `_held_reopens_exits` over the bars a sparse callback steps past.
    fn held_reopens(
        &self,
        view: &View,
        state: &State,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
    ) -> bool {
        if !has_position {
            return false;
        }
        let side = if net_position > 0.0 {
            1
        } else if net_position < 0.0 {
            -1
        } else {
            0
        };
        let stop = self.next_dispatch_position(view, state);
        self.reopens_exits(view, state, side, net_position, pending, idx + 1, stop)
    }

    /// Take callback bar `idx` for the strategy if it would do nothing on it.
    ///
    /// Mirrors `SignalReplay.on_bar` on a bar with no working order:
    /// `advance`, the flat account's cash halt, and the
    /// `bar_callback_required` that decides whether the next bar calls back.
    /// Returns `true` when the bar is quiet -- `advance` was idle, nothing
    /// halts, and nothing asks for the next bar -- and keeps what `advance`
    /// changed. Otherwise restores the state from before the bar and returns
    /// `false`, so the strategy can take the bar itself.
    ///
    /// A halted run's callbacks advance the rules' pure state and act on
    /// nothing, so they are quiet too.
    ///
    /// A run with risk checks visits every primary bar, each a callback of
    /// its own; the checks run inside `advance`, on the account the bar was
    /// read with, so a bar on which none acts is as quiet as any other.
    ///
    /// `saved` is scratch space for the copy, left holding nothing of use.
    pub(crate) fn quiet_bar(
        &self,
        view: &View,
        state: &mut State,
        saved: &mut State,
        bar: &QuietBar,
        quiet: &QuietParams,
    ) -> bool {
        saved.clone_from(state);
        let advanced = self.advance_bar(view, state, bar);
        let quiet_bar = self.settles_quiet(view, state, bar, quiet, &advanced);
        if !quiet_bar {
            std::mem::swap(state, saved);
        }
        quiet_bar
    }

    /// `advance` on callback bar `bar`, with the account it was read with.
    fn advance_bar(
        &self,
        view: &View,
        state: &mut State,
        bar: &QuietBar,
    ) -> PyResult<(bool, Option<Target>)> {
        // A halted run's callback only tells the policies of the closes
        // they were held from.
        self.advance(
            view,
            state,
            bar.now,
            bar.idx,
            bar.has_position,
            bar.net_position,
            bar.pending,
            bar.account,
            !bar.halted,
        )
    }

    /// Whether callback bar `bar`, whose `advance` answered `advanced` and
    /// left `state`, is quiet.
    fn settles_quiet(
        &self,
        view: &View,
        state: &State,
        bar: &QuietBar,
        quiet: &QuietParams,
        advanced: &PyResult<(bool, Option<Target>)>,
    ) -> bool {
        // An error is Python's to raise, on its own call.
        let &Ok((processed, target)) = advanced else {
            return false;
        };
        // `_refresh` reads the cash as `money_as_double`, which raises on a
        // value that is not a number.
        let cash = money_as_double(bar.cash, quiet.currency_precision);
        if processed && cash.is_none() {
            return false;
        }
        if bar.halted {
            return true;
        }
        if target.is_some() || bar.pending.is_some() {
            return false;
        }
        if !bar.has_position {
            let Some(cash) = cash else {
                return false;
            };
            let mut funds = cash * quiet.fraction;
            if funds > quiet.cap {
                funds = quiet.cap;
            }
            let need = quiet.units * (bar.close * quiet.markup + quiet.slack) + quiet.fee;
            // A bar that might halt, or whose next bars the halt bound cannot
            // clear, is Python's. Written as Python compares, so that a NaN
            // bound clears nothing.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let uncleared = funds < need || !(funds >= quiet.need_at_highest);
            if uncleared {
                return false;
            }
        }
        if state.visit_primary_after_fill {
            return false;
        }
        // Whether a sparse callback would step past a bar that reopens the
        // exit clocks: a run with risk checks defers nothing, and its next
        // bar is a callback of its own.
        !(self.policies.is_none()
            && bar.has_position
            && self.held_reopens(
                view,
                state,
                bar.idx,
                bar.has_position,
                bar.net_position,
                bar.pending,
            ))
    }

    /// The callback bars from `bar.idx` on that [`Self::quiet_bar`] would
    /// take as quiet, read with `bar`'s account and unable to act, taken at
    /// once: the state left as each one's `quiet_bar` leaves it, and the
    /// first bar not taken returned -- `bar.idx` itself where none is.
    ///
    /// Only the lane with risk checks has such bars: it calls back on every
    /// primary bar, and between two composites a bar's only event is its
    /// own, which the rules skip (no visit owed, no dust lot whose exits it
    /// could reopen) and no latch reads. What is left are the checks. Flat,
    /// they only clear what they keep, so once that is clear the bars up to
    /// the next event are taken in one step, each still passing the halt
    /// bound; held, they run on each bar, and the run stops before the
    /// first they act on -- or fail on -- for the bar's own call.
    ///
    /// `ts`, `close` and `mask` are the callback bars' timestamps, closes
    /// and flags, as the caller steps them; the run ends at `end`, and at
    /// the first bar that is no callback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inert_run(
        &self,
        view: &View,
        state: &mut State,
        bar: &QuietBar,
        quiet: &QuietParams,
        ts: &[i64],
        close: &[f64],
        mask: &[bool],
        end: usize,
    ) -> usize {
        let from = bar.idx;
        let Some(policies) = &self.policies else {
            return from;
        };
        let p = &state.policy;
        if from >= end
            || self.level.is_some()
            || state.visit_primary_after_fill
            || state.close_bar >= 0
            || bar.pending.is_some()
            || p.held
            || p.held_atr >= 0
            || p.held_counts != 0
        {
            return from;
        }
        // Every bar past the deferral reads the cash, which must convert.
        let Some(cash) = money_as_double(bar.cash, quiet.currency_precision) else {
            return from;
        };
        let side = if bar.has_position { side_of(bar.net_position) } else { 0 };
        if self.dust && side != 0 && state.last_position == 0 {
            // The rules might visit a bar to reopen the lot's exits.
            return from;
        }
        // The bars before the next event of another stream are their own
        // only event: a composite lands as its streams' events, and on this
        // lane the dispatch count is not kept.
        let mut horizon = i64::MAX;
        for (reg, registration) in self.regs.iter().enumerate() {
            let index = state.cursor.indexes[reg];
            if registration.slot != PRIMARY && index < view.reg_ts[reg].len() {
                horizon = horizon.min(view.reg_ts[reg][index]);
            }
        }
        let mut stop = if ts[from] >= horizon {
            from
        } else {
            first_at_or_after(ts, from, horizon, from).min(end)
        };
        stop = first_off(mask, from, stop);
        if stop == from {
            return from;
        }
        let account = bar.account;
        if side == 0 {
            // A lot the rules read as dust is one the checks still act on.
            if account.quantity != 0.0
                || !Self::flat_checks_settled(policies, view, &state.policy, account, bar.halted)
            {
                return from;
            }
            if !bar.halted && !bar.has_position {
                let mut funds = cash * quiet.fraction;
                if funds > quiet.cap {
                    funds = quiet.cap;
                }
                // As `settles_quiet` compares, so that a NaN bound clears
                // nothing and a NaN close's need halts nothing.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(funds >= quiet.need_at_highest) {
                    return from;
                }
                stop = from
                    + close[from..stop]
                        .iter()
                        .position(|&c| {
                            funds < quiet.units * (c * quiet.markup + quiet.slack) + quiet.fee
                        })
                        .unwrap_or(stop - from);
            }
        } else {
            let admits = view.slot_admits[PRIMARY];
            let mut p = state.policy;
            let mut at = from;
            while at < stop {
                let mut trial = p;
                if let Some(borrow) = &view.borrow {
                    if borrow.observe(&mut trial, at, account.quantity).is_err() {
                        break;
                    }
                }
                if !bar.halted {
                    if admits.is_none_or(|admits| admits[at])
                        && Self::policy_intent(policies, view, &mut trial, account, at).is_some()
                    {
                        break;
                    }
                    trial.resampled = false;
                }
                p = trial;
                at += 1;
            }
            state.policy = p;
            stop = at;
        }
        if stop > from {
            state.entered_from_flat = false;
            // Each bar's call skips the primary events up to its own.
            self.skip_primary_before(view, &mut state.cursor, ts[stop - 1] + 1, stop);
        }
        stop
    }

    /// Whether a flat bar's risk checks leave `p` as they find it: each
    /// clears what it keeps once nothing is held, and the borrow tracker
    /// ends the loan.
    fn flat_checks_settled(
        policies: &Policies,
        view: &View,
        p: &PolicyState,
        account: Account,
        halted: bool,
    ) -> bool {
        if view.borrow.is_some()
            && (p.loan.is_some() || p.borrow_units != 0 || p.borrow.to_bits() != 0)
        {
            return false;
        }
        if halted {
            // A halted run's checks only release the closes held from them.
            return true;
        }
        let held = account.quantity.to_bits();
        if p.resampled
            || (policies.trade.is_some() && p.trade_charges.held.to_bits() != held)
            || (policies.hold.is_some() && p.hold_mark != -1)
        {
            return false;
        }
        if policies.atr.is_some() && view.atr.is_some() {
            return p.atr_charges.held.to_bits() == held
                && p.atr_entry.to_bits() == 0
                && p.atr_charges.charged.to_bits() == 0
                && p.atr_extreme.to_bits() == 0;
        }
        true
    }
}

/// The first bar in `from..stop` that is no callback, or `stop`.
fn first_off(mask: &[bool], from: usize, stop: usize) -> usize {
    let flags = &mask[from..stop];
    // SAFETY: a `bool` is one byte holding 0 or 1, each a valid `u8`.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(flags.as_ptr().cast(), flags.len()) };
    let (chunks, remainder) = bytes.as_chunks::<64>();
    let mut offset = from;
    for chunk in chunks {
        // One vector AND across the chunk; most of a dense mask is set.
        if chunk.iter().fold(1, |all, &flag| all & flag) == 0 {
            return offset + chunk.iter().position(|&flag| flag == 0).unwrap_or(0);
        }
        offset += 64;
    }
    offset + remainder.iter().position(|&flag| flag == 0).unwrap_or(remainder.len())
}

/// The account in front of one callback bar, read before the kernel steps it.
///
/// `account` is what a run's risk checks read ([`Account::FLAT`] for a run
/// with none, or none held).
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuietBar {
    pub idx: usize,
    pub now: i64,
    pub close: f64,
    pub has_position: bool,
    pub net_position: f64,
    pub cash: f64,
    pub halted: bool,
    pub pending: Option<f64>,
    pub account: Account,
}

/// The halt bound and currency precision a flat callback bar is checked with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuietParams {
    pub currency_precision: u32,
    pub fraction: f64,
    pub cap: f64,
    pub units: f64,
    pub markup: f64,
    pub slack: f64,
    pub fee: f64,
    pub need_at_highest: f64,
}

/// `(currency_precision, fraction, cap, units, markup, slack, fee,
/// need_at_highest)`, as the Python side passes a [`QuietParams`].
pub(crate) type QuietTuple = (u32, f64, f64, f64, f64, f64, f64, f64);

impl From<QuietTuple> for QuietParams {
    fn from(q: QuietTuple) -> Self {
        Self {
            currency_precision: q.0,
            fraction: q.1,
            cap: q.2,
            units: q.3,
            markup: q.4,
            slack: q.5,
            fee: q.6,
            need_at_highest: q.7,
        }
    }
}

/// The decision cursor of one account-coupled crossover run.
#[pyclass(name = "CrossoverRuleSession")]
pub struct PyCrossoverRuleSession {
    data: Data,
    state: State,
    /// Scratch for [`Data::quiet_bar`]'s copy, kept to keep its buffers.
    saved: State,
}

fn operand(key: i64, value: f64, n_keys: usize) -> PyResult<Operand> {
    if key < 0 {
        Ok(Operand::Const(value))
    } else if (key as usize) < n_keys {
        Ok(Operand::Key(key as usize))
    } else {
        Err(PyValueError::new_err("rule operand names an unknown key"))
    }
}

fn crossings(
    rules: Vec<(usize, i64, f64, i64, f64, bool)>,
    n_keys: usize,
    n_clocks: usize,
) -> PyResult<Vec<Crossing>> {
    rules
        .into_iter()
        .map(|(clock, fast_key, fast_value, slow_key, slow_value, above)| {
            if clock >= n_clocks {
                return Err(PyValueError::new_err("rule names an unknown clock"));
            }
            Ok(Crossing {
                clock,
                fast: operand(fast_key, fast_value, n_keys)?,
                slow: operand(slow_key, slow_value, n_keys)?,
                above,
            })
        })
        .collect()
}

/// One node of a condition arena, as the Python side passes it: `(kind,
/// children, a_key, a_value, b_key, b_value, code, lower, upper)`.
///
/// `kind` is 0 for AND, 1 for OR, 2 for NOT (one child), 3 for a zone of
/// operand `a` within `[lower, upper]`, 4 for a threshold `a <op> b` with
/// `code` 0-4 for GT, GE, LT, LE, EQ, and 5 or 6 for a crossover of `a` above
/// or below `b` on clock `code`. Operands are a key, or a constant where the
/// key is negative.
type ConditionTuple = (u8, Vec<usize>, i64, f64, i64, f64, usize, f64, f64);

fn conditions(
    nodes: Vec<ConditionTuple>,
    n_keys: usize,
    n_clocks: usize,
) -> PyResult<Vec<Condition>> {
    let malformed = |what: &str| PyValueError::new_err(format!("malformed condition: {what}"));
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (kind, children, a_key, a_value, b_key, b_value, code, lower, upper))| {
            // Children first: the arena is acyclic, and evaluation ends.
            if children.iter().any(|&child| child >= index) {
                return Err(malformed("a child must precede its parent"));
            }
            Ok(match kind {
                0 => Condition::All(children),
                1 => Condition::Any(children),
                2 => match children.as_slice() {
                    [child] => Condition::Not(*child),
                    _ => return Err(malformed("NOT takes one child")),
                },
                3 => Condition::Zone { value: operand(a_key, a_value, n_keys)?, lower, upper },
                4 => Condition::Threshold {
                    left: operand(a_key, a_value, n_keys)?,
                    right: operand(b_key, b_value, n_keys)?,
                    compare: match code {
                        0 => Compare::Gt,
                        1 => Compare::Ge,
                        2 => Compare::Lt,
                        3 => Compare::Le,
                        4 => Compare::Eq,
                        _ => return Err(malformed("unknown comparison")),
                    },
                },
                5 | 6 => {
                    if code >= n_clocks {
                        return Err(PyValueError::new_err("rule names an unknown clock"));
                    }
                    Condition::Cross(Crossing {
                        clock: code,
                        fast: operand(a_key, a_value, n_keys)?,
                        slow: operand(b_key, b_value, n_keys)?,
                        above: kind == 5,
                    })
                }
                _ => return Err(malformed("unknown kind")),
            })
        })
        .collect()
}

/// The risk policies, as the Python side passes them: `(fee_rate,
/// close_limit, trade, atr, hold)`. `fee_rate` is the contract's taker fee
/// where commissions are on, else 0; `close_limit` closes at a limit, else at
/// market (`_close_style`); `trade` is `(stop, take_profit)` in per cent;
/// `atr` is `(slot, values, initialized, stop, take_profit, trailing)`, its
/// series in that slot's bars; `hold` is `(slot, bars)`, counted in a
/// resample's bars. Each is `None` where the plan has no such check.
type PolicyTuple = (
    f64,
    bool,
    Option<(f64, f64)>,
    Option<(usize, Py<PyArray1<f64>>, Py<PyArray1<bool>>, f64, f64, bool)>,
    Option<(usize, i64)>,
);

/// [`Borrow`] as the Python side passes it: `(days, settles, first_day,
/// charged)`.
type BorrowTuple = (Py<PyArray1<i32>>, Py<PyArray1<i32>>, i32, Py<PyArray1<i64>>);

impl BorrowView<'_> {
    /// Refuse arrays [`Self::observe`] could read past, over `bars` primary
    /// bars: a date and a settlement for every bar, dates that never fall,
    /// from `first_day` on, each loan settling after the date it opens, and
    /// a sum for every date from `first_day` through the last bar's.
    fn validate(&self, bars: usize) -> PyResult<()> {
        let refuse = |what: &str| Err(PyValueError::new_err(format!("borrow terms: {what}")));
        if self.days.len() != bars || self.settles.len() != bars {
            return refuse("a date and a settlement for every primary bar");
        }
        let (Some(&first), Some(&last)) = (self.days.first(), self.days.last()) else {
            return Ok(());
        };
        if first != self.first_day || self.days.windows(2).any(|w| w[1] < w[0]) {
            return refuse("dates rising from the first bar's");
        }
        if self.days.iter().zip(self.settles).any(|(&day, &settles)| settles <= day) {
            return refuse("a loan settles after the date it opens");
        }
        if self.charged.len() != (last - first) as usize + 1 || self.charged[0] != 0 {
            return refuse("a sum for every date through the last bar's");
        }
        Ok(())
    }
}

/// [`Policies`] from their tuple, over `n_slots` slots.
fn policies(tuple: PolicyTuple, n_slots: usize) -> PyResult<Policies> {
    let (fee_rate, close_limit, trade, atr, hold) = tuple;
    if trade.is_none() && atr.is_none() && hold.is_none() {
        return Err(PyValueError::new_err("risk policies name no check"));
    }
    if atr.as_ref().is_some_and(|(slot, ..)| *slot >= n_slots) {
        return Err(PyValueError::new_err("the ATR exit names an unknown stream"));
    }
    if hold.is_some_and(|(slot, _)| slot == PRIMARY || slot >= n_slots) {
        return Err(PyValueError::new_err("the time exit counts a resample stream"));
    }
    Ok(Policies {
        trade: trade.map(|(stop, take_profit)| TradeExit { stop, take_profit }),
        atr: atr.map(|(slot, values, initialized, stop, take_profit, trailing)| {
            (AtrExit { slot, stop, take_profit, trailing }, values, initialized)
        }),
        hold: hold.map(|(slot, bars)| HoldExit { slot, bars }),
        fee_rate,
        style: if close_limit { OrderStyle::Limit } else { OrderStyle::Market },
    })
}

/// Each clock's crossing, in clock order, for rules whose crossings are
/// worked out each bar: every clock named by a crossing, and by one only --
/// a clock is its operands and direction (the name `DecisionProgram` keys
/// it by), so two crossings that differ are two clocks.
fn clock_crossings(arena: &[Condition], n_clocks: usize) -> PyResult<Vec<Crossing>> {
    let mut found: Vec<Option<Crossing>> = vec![None; n_clocks];
    for node in arena {
        let Condition::Cross(rule) = node else { continue };
        match found[rule.clock] {
            None => found[rule.clock] = Some(*rule),
            Some(seen) if seen == *rule => {}
            Some(_) => return Err(PyValueError::new_err("one clock names two crossings")),
        }
    }
    found
        .into_iter()
        .map(|rule| rule.ok_or_else(|| PyValueError::new_err("a clock no crossing names")))
        .collect()
}

#[pymethods]
impl PyCrossoverRuleSession {
    /// The decision rules this build transcribes. The parity runner steps a
    /// session only on its own version (`RULE_SESSION_VERSION`), and keeps
    /// the Python session otherwise, so no build decides by other rules.
    ///
    /// 1: condition trees and level rules, and the pending-close latch
    /// (`two_sided`, `note_close`). 2: one step per instant -- a window a gap
    /// closed is decided ahead of the primary bar that revealed it, and the
    /// visits of one delivery settle by [`delivery_step`]; the registrations
    /// say which legs the strategy builds, and no target asks the risk
    /// policies after a fill. 3: `entry_side` -- a short node's rules, whose
    /// entries sell to open. 4: reference legs -- a legacy program's event on
    /// a slot it does not decide on owes the next primary bar a visit
    /// ([`Self::owes_primary`]) -- and a visit owed is settled, in both
    /// lanes, only by a primary bar the rules ran on -- `each_bar`: every
    /// crossing worked out once per evaluated bar, ahead of the rules
    /// (`StrategySpec.crossings`) -- and risk policies (`policies`): the
    /// percentage, ATR and time exits, on the lane that visits every primary
    /// bar in full, with the venue's account handed to `advance` and each
    /// fill to `on_fill`. 5: `borrow` -- the open short's borrow, which the
    /// trade and ATR exits charge, observed on every primary bar and fill.
    /// 6: a bar of a run with risk policies on which none acts is quiet
    /// ([`Data::quiet_bar`]), asked with the account they read
    /// (`probe_quiet_bar`'s `quantity` and `entry`).
    #[classattr]
    const VERSION: u32 = 6;

    /// A session over crossover rules `exits` and `entries`, or, given
    /// `conditions`, over the rules rooted at `exit_roots` and `entry_roots`
    /// of that arena (see [`ConditionTuple`]). `level`, as `(two_sided,
    /// size_increment, minimum_notional)` of the instrument contract, makes
    /// them level rules. `two_sided` is the executor's own test: a two-sided
    /// run keeps no pending-close latch. `entry_side` is the side an entry
    /// opens: 1, or -1 for a short node's rules. `each_bar` works every
    /// crossing out once per evaluated bar before any rule reads it
    /// (`StrategySpec.crossings` ``each_bar``); otherwise a rule works out
    /// the crossings it checks. `policies` (see [`PolicyTuple`]) are the
    /// plan's risk checks. `borrow` is `(days, settles, first_day, charged)`
    /// (see [`Borrow`]), for a run under them that may short.
    #[new]
    #[pyo3(signature = (
        reg_slots, reg_stream_index, reg_rank, reg_built, reg_timestamps, reg_labels, slot_close,
        slot_admits, slot_decision, key_slot, key_values, key_init, exits, entries, n_clocks,
        legacy, dispatch_timestamps, dispatch_positions, primary_close, dust,
        position_required, has_instrument, size_increment, minimum_notional, *,
        conditions=None, exit_roots=None, entry_roots=None, level=None, two_sided=false,
        entry_side=1, each_bar=false, policies=None, borrow=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        reg_slots: Vec<usize>,
        reg_stream_index: Vec<i64>,
        reg_rank: Vec<i64>,
        reg_built: Vec<bool>,
        reg_timestamps: Vec<Py<PyArray1<i64>>>,
        reg_labels: Vec<Option<Py<PyArray1<i64>>>>,
        slot_close: Vec<Py<PyArray1<f64>>>,
        slot_admits: Vec<Option<Py<PyArray1<bool>>>>,
        slot_decision: Vec<bool>,
        key_slot: Vec<usize>,
        key_values: Vec<Py<PyArray1<f64>>>,
        key_init: Vec<Py<PyArray1<bool>>>,
        exits: Vec<(usize, i64, f64, i64, f64, bool)>,
        entries: Vec<(usize, i64, f64, i64, f64, bool)>,
        n_clocks: usize,
        legacy: bool,
        dispatch_timestamps: Py<PyArray1<i64>>,
        dispatch_positions: Py<PyArray1<i64>>,
        primary_close: Py<PyArray1<f64>>,
        dust: bool,
        position_required: bool,
        has_instrument: bool,
        size_increment: f64,
        minimum_notional: f64,
        conditions: Option<Vec<ConditionTuple>>,
        exit_roots: Option<Vec<usize>>,
        entry_roots: Option<Vec<usize>>,
        level: Option<(bool, f64, f64)>,
        two_sided: bool,
        entry_side: i32,
        each_bar: bool,
        policies: Option<PolicyTuple>,
        borrow: Option<BorrowTuple>,
    ) -> PyResult<Self> {
        if entry_side != 1 && entry_side != -1 {
            return Err(PyValueError::new_err("an entry opens side 1 or -1"));
        }
        let n_regs = reg_slots.len();
        let n_slots = slot_close.len();
        let n_keys = key_slot.len();
        if [
            reg_stream_index.len(),
            reg_rank.len(),
            reg_built.len(),
            reg_timestamps.len(),
            reg_labels.len(),
        ]
        .iter()
        .any(|&n| n != n_regs)
            || [slot_admits.len(), slot_decision.len()].iter().any(|&n| n != n_slots)
            || key_values.len() != n_keys
            || key_init.len() != n_keys
            || n_slots == 0
            || reg_slots.iter().any(|&s| s >= n_slots)
            || key_slot.iter().any(|&s| s >= n_slots)
        {
            return Err(PyValueError::new_err("inconsistent rule session layout"));
        }
        let mut slot_keys = vec![Vec::new(); n_slots];
        for (key, &slot) in key_slot.iter().enumerate() {
            slot_keys[slot].push(key);
        }
        if !slot_keys[PRIMARY].is_empty() {
            return Err(PyValueError::new_err(
                "the rule session reads no primary-stream indicator",
            ));
        }
        let regs: Vec<Registration> = (0..n_regs)
            .map(|i| Registration {
                slot: reg_slots[i],
                stream_index: reg_stream_index[i],
                rank: reg_rank[i],
                built: reg_built[i],
                timestamps: reg_timestamps[i].clone_ref(py),
                labels: reg_labels[i].as_ref().map(|a| a.clone_ref(py)),
            })
            .collect();
        let (arena, exit_roots, entry_roots) = match (conditions, exit_roots, entry_roots) {
            (None, None, None) => {
                // Crossover rules: one node per rule, each its own root.
                let exits = crossings(exits, n_keys, n_clocks)?;
                let entries = crossings(entries, n_keys, n_clocks)?;
                let exit_roots = (0..exits.len()).collect();
                let entry_roots = (exits.len()..exits.len() + entries.len()).collect();
                let arena = exits.into_iter().chain(entries).map(Condition::Cross).collect();
                (arena, exit_roots, entry_roots)
            }
            (Some(nodes), Some(exit_roots), Some(entry_roots))
                if exits.is_empty() && entries.is_empty() =>
            {
                let arena = self::conditions(nodes, n_keys, n_clocks)?;
                let roots: &[usize] = &[exit_roots.as_slice(), entry_roots.as_slice()].concat();
                if roots.iter().any(|&root| root >= arena.len()) {
                    return Err(PyValueError::new_err("rule root names an unknown condition"));
                }
                (arena, exit_roots, entry_roots)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "pass crossover rules, or conditions with both root lists",
                ))
            }
        };
        let each_bar = if each_bar { Some(clock_crossings(&arena, n_clocks)?) } else { None };
        let policies = policies.map(|tuple| self::policies(tuple, n_slots)).transpose()?;
        if borrow.is_some() && policies.is_none() {
            return Err(PyValueError::new_err("only the risk exits read the borrow"));
        }
        let borrow = borrow.map(|(days, settles, first_day, charged)| Borrow {
            days,
            settles,
            first_day,
            charged,
        });
        let data = Data {
            regs,
            slot_close,
            slot_admits,
            slot_decision,
            slot_keys,
            key_values,
            key_init,
            conditions: arena,
            exits: exit_roots,
            entries: entry_roots,
            level: level.map(|(two_sided, size_increment, minimum_notional)| Level {
                two_sided,
                size_increment,
                minimum_notional,
            }),
            legacy,
            each_bar,
            dispatch_ts: dispatch_timestamps,
            dispatch_pos: dispatch_positions,
            primary_close,
            dust,
            position_required,
            has_instrument,
            size_increment,
            minimum_notional,
            // A level rule set's contract says so too.
            close_latch: !(two_sided || level.is_some_and(|(two_sided, ..)| two_sided)),
            entry_side,
            policies,
            borrow,
        };
        let state = {
            let view = data.view(py)?;
            // Every array a slot or key indexes must cover that slot's bars.
            for (key, &slot) in key_slot.iter().enumerate() {
                let bars = view.slot_close[slot].len();
                if view.key_values[key].len() != bars || view.key_init[key].len() != bars {
                    return Err(PyValueError::new_err(
                        "indicator series do not match their stream",
                    ));
                }
            }
            for slot in 0..n_slots {
                let bars = view.slot_close[slot].len();
                if view.slot_admits[slot].is_some_and(|m| m.len() != bars) {
                    return Err(PyValueError::new_err("stream arrays do not match their bars"));
                }
            }
            for (reg, registration) in data.regs.iter().enumerate() {
                let bars = view.slot_close[registration.slot].len();
                if view.reg_ts[reg].len() != bars
                    || view.reg_labels[reg].is_some_and(|l| l.len() != bars)
                {
                    return Err(PyValueError::new_err("schedule does not match its stream"));
                }
            }
            if view.dispatch_pos.len() != view.dispatch_ts.len() {
                return Err(PyValueError::new_err("dispatch positions do not match"));
            }
            if let (Some((atr, ..)), Some((values, initialized))) =
                (data.policies.as_ref().and_then(|p| p.atr.as_ref()), view.atr)
            {
                let bars = view.slot_close[atr.slot].len();
                if values.len() != bars || initialized.len() != bars {
                    return Err(PyValueError::new_err("the ATR series does not match its stream"));
                }
            }
            if view.dispatch_pos.iter().any(|&p| p < 0 || p as usize > view.primary_close.len()) {
                return Err(PyValueError::new_err("dispatch position out of range"));
            }
            if let Some(borrow) = &view.borrow {
                borrow.validate(view.primary_close.len())?;
            }
            let mut cursor = Cursor {
                indexes: vec![0; n_regs],
                current: None,
                timestamp: 0,
                label: 0,
                priority: 0,
                slot: PRIMARY,
                bar: 0,
            };
            data.select(&view, &mut cursor);
            State {
                cursor,
                next_dispatch: 0,
                visit_primary_after_fill: false,
                updates: vec![None; n_keys],
                values: vec![None; n_keys],
                unready: vec![true; n_keys],
                clocks: vec![None; n_clocks],
                last_labels: vec![None; n_slots],
                last_sort_key: None,
                last_position: 0,
                last_reached: false,
                last_ready: true,
                flipped: false,
                fired: vec![false; n_clocks],
                close_bar: -1,
                entered_from_flat: false,
                policy: PolicyState::NEW,
            }
        };
        let saved = state.clone();
        Ok(Self { data, state, saved })
    }

    /// One `advance` callback: `(processed, target)`, where `processed` says
    /// the call got past the deferral and `target` is `None` for `IDLE`, else
    /// `(target_side, reference_price, (is_exit, rule_index) | None,
    /// policy_reason | None, preceding_reference | None, order_style)`.
    ///
    /// `has_position` and `net_position` are the lot as the rules read it;
    /// `quantity` and `entry` are the venue's signed lot and its average
    /// entry, which the risk policies read. `apply_policies` false is a
    /// halted run's callback.
    #[pyo3(signature = (
        now, idx, has_position, net_position, pending_target=None, *, quantity=0.0, entry=0.0,
        apply_policies=true,
    ))]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn advance(
        &mut self,
        py: Python<'_>,
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending_target: Option<f64>,
        quantity: f64,
        entry: f64,
        apply_policies: bool,
    ) -> PyResult<(
        bool,
        Option<(
            i32,
            Option<f64>,
            Option<(bool, usize)>,
            Option<&'static str>,
            Option<f64>,
            &'static str,
        )>,
    )> {
        let Self { data, state, .. } = self;
        let view = data.view(py)?;
        let (processed, target) = data.advance(
            &view,
            state,
            now,
            idx,
            has_position,
            net_position,
            pending_target,
            Account { quantity, entry },
            apply_policies,
        )?;
        Ok((
            processed,
            target.map(|t| {
                (
                    t.side,
                    t.reference,
                    t.reason,
                    t.policy.map(PolicyExit::reason),
                    t.preceding,
                    t.style.name(),
                )
            }),
        ))
    }

    /// `reopens_exits_before_dispatch`: whether a primary bar between this
    /// callback and the next composite can restart the exit clocks.
    #[pyo3(signature = (idx, has_position, net_position, pending_target=None))]
    fn reopens_exits_before_dispatch(
        &self,
        py: Python<'_>,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending_target: Option<f64>,
    ) -> PyResult<bool> {
        let view = self.data.view(py)?;
        Ok(self.data.held_reopens(
            &view,
            &self.state,
            idx,
            has_position,
            net_position,
            pending_target,
        ))
    }

    /// Whether the kernel's fast-forward would take callback bar `idx` for
    /// the strategy, asked of a copy: the session itself does not move.
    /// `quantity` and `entry` are the account the risk policies read, as
    /// [`Self::advance`] takes them.
    #[pyo3(signature = (
        idx, now, close, has_position, net_position, cash, halted, pending_target, quiet, *,
        quantity=0.0, entry=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn probe_quiet_bar(
        &self,
        py: Python<'_>,
        idx: usize,
        now: i64,
        close: f64,
        has_position: bool,
        net_position: f64,
        cash: f64,
        halted: bool,
        pending_target: Option<f64>,
        quiet: QuietTuple,
        quantity: f64,
        entry: f64,
    ) -> PyResult<bool> {
        let view = self.data.view(py)?;
        let mut state = self.state.clone();
        let bar = QuietBar {
            idx,
            now,
            close,
            has_position,
            net_position,
            cash,
            halted,
            pending: pending_target,
            account: Account { quantity, entry },
        };
        let mut saved = state.clone();
        Ok(self.data.quiet_bar(&view, &mut state, &mut saved, &bar, &QuietParams::from(quiet)))
    }

    /// A fill landed, which makes the next primary bar a visit: of
    /// `quantity` on `side` (1 buys, -1 sells) charged `commission`, which
    /// left the account `flat` -- read after the fill -- for the risk
    /// policies to book. A run that borrows is told the primary `bar` the
    /// fill landed on and the venue's signed `position` after it.
    #[pyo3(signature = (side=0.0, quantity=0.0, commission=0.0, flat=true, *, bar=None, position=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn on_fill(
        &mut self,
        py: Python<'_>,
        side: f64,
        quantity: f64,
        commission: f64,
        flat: bool,
        bar: Option<usize>,
        position: f64,
    ) -> PyResult<()> {
        let Self { data, state, .. } = self;
        let view = data.view(py)?;
        let fill = bar.map(|bar| (bar, position));
        data.policy_fill(&view, state, side, quantity, commission, flat, fill)?;
        state.visit_primary_after_fill = true;
        Ok(())
    }

    /// `NativeExecutionSession.note_close`: a close the executor resent on
    /// primary bar `bar_index`, under a pending close that stands through it.
    fn note_close(&mut self, bar_index: i64) {
        self.data.note_close(&mut self.state, bar_index);
    }

    /// `NativeExecutionSession._close_bar`.
    #[getter]
    fn close_bar(&self) -> i64 {
        self.state.close_bar
    }

    #[getter]
    fn visit_primary_after_fill(&self) -> bool {
        self.state.visit_primary_after_fill
    }

    #[getter]
    fn next_dispatch_index(&self) -> usize {
        self.state.next_dispatch
    }

    #[getter]
    fn last_position(&self) -> i32 {
        self.state.last_position
    }

    #[getter]
    fn last_reached(&self) -> bool {
        self.state.last_reached
    }

    #[getter]
    fn last_ready(&self) -> bool {
        self.state.last_ready
    }

    /// Whether the session runs risk policies.
    #[getter]
    fn has_policies(&self) -> bool {
        self.data.policies.is_some()
    }

    /// Whether the session charges the open short's borrow.
    #[getter]
    fn has_borrow(&self) -> bool {
        self.data.borrow.is_some()
    }

    /// `NativeAccountView.accrued_borrow`: what the open short's loan has
    /// been charged, as of the last `advance` or `on_fill`.
    #[getter]
    fn accrued_borrow(&self) -> f64 {
        self.state.policy.borrow
    }

    /// `NativeModelRunner.policy_resampled`: whether the last check to act
    /// was the one after the held resample closes reached the policies.
    #[getter]
    fn policy_resampled(&self) -> bool {
        self.state.policy.resampled
    }

    /// Every piece of mutable state, for lockstep comparison with Python, as
    /// `((cursor_indexes, cursor_active, next_dispatch_index,
    /// visit_primary_after_fill), (updates, values, unready, clocks,
    /// last_labels, last_sort_key), (last_position, last_reached,
    /// last_ready), (trade_charged, trade_held, atr_charged, atr_held,
    /// atr_index, atr_entry, atr_extreme, hold_counted, hold_mark,
    /// accrued_borrow))`.
    #[allow(clippy::type_complexity)]
    fn snapshot(
        &self,
    ) -> (
        (Vec<usize>, bool, usize, bool),
        (
            Vec<Option<f64>>,
            Vec<Option<f64>>,
            Vec<bool>,
            Vec<Option<(f64, f64)>>,
            Vec<Option<i64>>,
            Option<(i64, i64)>,
        ),
        (i32, bool, bool),
        (f64, f64, f64, f64, i64, f64, f64, i64, i64, f64),
    ) {
        let s = &self.state;
        let p = &s.policy;
        (
            (
                s.cursor.indexes.clone(),
                s.cursor.current.is_some(),
                s.next_dispatch,
                s.visit_primary_after_fill,
            ),
            (
                s.updates.clone(),
                s.values.clone(),
                s.unready.clone(),
                s.clocks.clone(),
                s.last_labels.clone(),
                s.last_sort_key,
            ),
            (s.last_position, s.last_reached, s.last_ready),
            (
                p.trade_charges.charged,
                p.trade_charges.held,
                p.atr_charges.charged,
                p.atr_charges.held,
                p.atr_index,
                p.atr_entry,
                p.atr_extreme,
                p.hold_counted,
                p.hold_mark,
                p.borrow,
            ),
        )
    }
}

impl PyCrossoverRuleSession {
    /// Step callback bars on the strategy's behalf; see
    /// `PyKernelSession::step_rules_until`.
    pub(crate) fn split<'a>(&'a mut self, py: Python<'a>) -> PyResult<RuleStepper<'a>> {
        let view = self.data.view(py)?;
        Ok(RuleStepper {
            data: &self.data,
            view,
            state: &mut self.state,
            saved: &mut self.saved,
            hands_back: false,
            taken: None,
        })
    }
}

/// One borrow of a rule session's arrays, for a run of quiet bars.
pub(crate) struct RuleStepper<'a> {
    data: &'a Data,
    view: View<'a>,
    state: &'a mut State,
    saved: &'a mut State,
    /// Whether a callback bar [`Self::quiet_bar`] finds would act is left
    /// advanced, for the caller's own `advance` on it, rather than restored.
    hands_back: bool,
    /// That bar, while the caller has yet to take it.
    taken: Option<Taken>,
}

/// The `advance` of a callback bar a quiet scan found would act, kept for
/// the caller's own call on the bar.
///
/// The rules stand past the bar, as `advance` left them; the caller's call
/// answers with what that `advance` did, read with the same account. A call
/// with another account, or anything else that touches the rules first,
/// finds the rules where no run of the caller's would have them, and the
/// replay declines ([`Taken::spoil`]).
struct Taken {
    idx: usize,
    now: i64,
    has_position: bool,
    net_position: f64,
    pending: Option<f64>,
    account: Account,
    apply_policies: bool,
    advanced: PyResult<(bool, Option<Target>)>,
}

impl Taken {
    /// Whether the caller's `advance` reads the bar as the scan did.
    #[allow(clippy::too_many_arguments)]
    fn answers(
        &self,
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> bool {
        self.idx == idx
            && self.now == now
            && self.has_position == has_position
            && self.net_position.to_bits() == net_position.to_bits()
            && self.pending.map(f64::to_bits) == pending.map(f64::to_bits)
            && self.account.quantity.to_bits() == account.quantity.to_bits()
            && self.account.entry.to_bits() == account.entry.to_bits()
            && self.apply_policies == apply_policies
    }

    fn spoil(&mut self) {
        self.advanced =
            Err(super::replay_session::declined("the rules moved past a callback bar handed back"));
    }
}

impl RuleStepper<'_> {
    /// This stepper, keeping a callback bar that would act advanced for the
    /// caller's own `advance` on it: a caller that takes every bar handed
    /// back, and touches the rules no other way first, never needs the state
    /// from before the bar, and [`Self::quiet_bar`] then copies nothing.
    pub(crate) fn handing_back(mut self) -> Self {
        self.hands_back = true;
        self
    }

    pub(crate) fn quiet_bar(&mut self, bar: &QuietBar, quiet: &QuietParams) -> bool {
        if !self.hands_back {
            return self.data.quiet_bar(&self.view, self.state, self.saved, bar, quiet);
        }
        if let Some(taken) = &mut self.taken {
            // A scan past a bar handed back is a scan over rules standing
            // past it.
            taken.spoil();
            return false;
        }
        let advanced = self.data.advance_bar(&self.view, self.state, bar);
        if self.data.settles_quiet(&self.view, self.state, bar, quiet, &advanced) {
            return true;
        }
        self.taken = Some(Taken {
            idx: bar.idx,
            now: bar.now,
            has_position: bar.has_position,
            net_position: bar.net_position,
            pending: bar.pending,
            account: bar.account,
            apply_policies: !bar.halted,
            advanced,
        });
        false
    }

    /// [`Data::inert_run`]: the callback bars from `bar.idx` that are quiet
    /// for want of anything able to act, taken at once; the first bar not
    /// taken.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inert_run(
        &mut self,
        bar: &QuietBar,
        quiet: &QuietParams,
        ts: &[i64],
        close: &[f64],
        mask: &[bool],
        end: usize,
    ) -> usize {
        if self.taken.is_some() {
            return bar.idx;
        }
        self.data.inert_run(&self.view, self.state, bar, quiet, ts, close, mask, end)
    }

    /// [`PyCrossoverRuleSession::advance`], answering with the target itself.
    ///
    /// A bar handed back answers with the `advance` its quiet scan ran, when
    /// asked with the account and the halt it was read with.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance(
        &mut self,
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
        account: Account,
        apply_policies: bool,
    ) -> PyResult<(bool, Option<Target>)> {
        if let Some(mut taken) = self.taken.take() {
            if !taken.answers(
                now,
                idx,
                has_position,
                net_position,
                pending,
                account,
                apply_policies,
            ) {
                taken.spoil();
            }
            return taken.advanced;
        }
        self.data.advance(
            &self.view,
            self.state,
            now,
            idx,
            has_position,
            net_position,
            pending,
            account,
            apply_policies,
        )
    }

    /// [`PyCrossoverRuleSession::reopens_exits_before_dispatch`].
    pub(crate) fn held_reopens(
        &self,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending: Option<f64>,
    ) -> bool {
        debug_assert!(self.taken.is_none(), "rules read past a bar handed back");
        self.data.held_reopens(&self.view, self.state, idx, has_position, net_position, pending)
    }

    /// The primary bar the next composite reaches, or the bar count past the
    /// last one.
    pub(crate) fn next_dispatch_position(&self) -> usize {
        debug_assert!(self.taken.is_none(), "rules read past a bar handed back");
        self.data.next_dispatch_position(&self.view, self.state)
    }

    /// Whether the rules are level rules, which read dust themselves.
    pub(crate) fn level(&self) -> bool {
        self.data.level.is_some()
    }

    /// [`PyCrossoverRuleSession::on_fill`]; `fill` is the bar and position
    /// a run that borrows is told.
    pub(crate) fn on_fill(
        &mut self,
        side: f64,
        quantity: f64,
        commission: f64,
        flat: bool,
        fill: Option<(usize, f64)>,
    ) -> PyResult<()> {
        if let Some(taken) = &mut self.taken {
            taken.spoil();
        }
        self.data.policy_fill(&self.view, self.state, side, quantity, commission, flat, fill)?;
        self.state.visit_primary_after_fill = true;
        Ok(())
    }

    /// Whether the session runs risk policies, which put every bar in front
    /// of the strategy.
    pub(crate) fn has_policies(&self) -> bool {
        self.data.policies.is_some()
    }

    /// [`PyCrossoverRuleSession::note_close`], for the reconcile ahead of
    /// callback bar `bar`'s `advance`.
    ///
    /// A bar handed back has had that `advance` run already, ahead of the
    /// note. It stands unless it entered from flat, which the note would
    /// have refused: a bar that resends a close submits nothing else, so its
    /// target is never read, and it leaves the latch at `bar` either way --
    /// its events are at most `bar`, which the note raises the latch to and
    /// nothing in the bar lowers it from.
    pub(crate) fn note_close(&mut self, bar: usize) {
        if let Some(taken) = &mut self.taken {
            if self.state.entered_from_flat {
                taken.spoil();
            }
        }
        self.data.note_close(self.state, bar as i64);
    }

    pub(crate) fn visit_primary_after_fill(&self) -> bool {
        debug_assert!(self.taken.is_none(), "rules read past a bar handed back");
        self.state.visit_primary_after_fill
    }

    /// Whether a bar handed back has yet to be taken.
    pub(crate) fn holds_taken(&self) -> bool {
        self.taken.is_some()
    }

    /// The session's primary close at bar `idx`.
    pub(crate) fn primary_close(&self, idx: usize) -> f64 {
        self.view.primary_close[idx]
    }

    /// The session's primary closes.
    pub(crate) fn primary_closes(&self) -> &[f64] {
        self.view.primary_close
    }
}

#[cfg(test)]
mod tests {
    use super::{
        borrow_amount, clock_crossings, conditions, delivery_step, first_at_or_after,
        money_as_double, trade_roi_pct, Account, AtrExit, BorrowView, Charges, Compare, Condition,
        ConditionTuple, Crossing, Data, Latch, Level, Operand, OrderStyle, Policies, PolicyExit,
        PolicyState, PRIMARY,
    };

    #[test]
    fn one_delivery_settles_its_visits_before_the_venue_does() {
        // The same steps for rules that buy and for a short node's, its
        // sides the long ones negated.
        for side in [1, -1] {
            let open = Latch::new(-1);
            // Nothing asked moves nothing.
            assert_eq!(delivery_step(false, false, side, 0, 0, false, &open), None);
            // From flat an entry moves the target; asked again it moves
            // nothing, so the entry keeps the terms it was decided on.
            assert_eq!(delivery_step(true, false, side, 0, 0, false, &open), Some(side));
            assert_eq!(delivery_step(true, false, side, side, 0, true, &open), None);
            // A close finds nothing to sell of an entry still resting.
            assert_eq!(delivery_step(false, true, side, side, 0, true, &open), None);
            // Held, a close moves it, and an entry back to the side being
            // closed leaves the close standing, latch or no latch.
            assert_eq!(delivery_step(false, true, side, side, side, false, &open), Some(0));
            assert_eq!(delivery_step(true, false, side, 0, side, false, &open), None);
            // Held, an entry asks for the side held: nothing moves.
            assert_eq!(delivery_step(true, false, side, side, side, false, &open), None);
            // The long-only latch refuses an entry from flat.
            let mut latched = Latch::new(-1);
            latched.close(7, false, 7);
            assert_eq!(delivery_step(true, false, side, 0, 0, false, &latched), None);
        }
    }

    #[test]
    fn a_close_refuses_entries_through_its_own_bar() {
        // Closed on primary bar 7 by an event on its clock.
        let mut latch = Latch::new(-1);
        latch.close(7, false, 7);
        assert!(latch.holds() && !latch.provisional);
        latch.step(7, false, 1);
        assert!(latch.holds(), "the same bar's later events");
        // The next bar's reconcile finds the book flat, or still held.
        let mut flat = latch;
        flat.step(8, false, 0);
        assert!(!flat.holds());
        let mut held = latch;
        held.step(8, false, 1);
        assert_eq!(held.close_bar, 8, "a close the venue has not taken carries");
    }

    #[test]
    fn a_pre_close_window_runs_ahead_of_its_bar_reconcile() {
        let mut latch = Latch::new(-1);
        latch.close(7, false, 7);
        latch.step(8, true, 0);
        assert!(latch.holds(), "the window on bar 8 sees the close pending");
        latch.step(8, false, 0);
        assert!(!latch.holds());
        // Held, the window carries it to the bar before its own.
        let mut held = Latch::new(5);
        held.step(9, true, 1);
        assert_eq!(held.close_bar, 8);
    }

    #[test]
    fn a_close_for_the_bar_before_is_filled_by_this_bar() {
        // A resample delivered late on bar 7 closes for bar 6: provisional.
        let mut latch = Latch::new(-1);
        latch.close(6, false, 7);
        assert!(latch.provisional && latch.holds());
        latch.step(7, false, 1);
        assert!(!latch.holds(), "bar 7's reconcile ran after that close filled");
        // A pre-close window's close is never provisional.
        let mut early = Latch::new(-1);
        early.close(6, true, 7);
        assert!(!early.provisional);
    }

    #[test]
    fn a_resent_close_keeps_the_latch_through_its_bar() {
        let mut latch = Latch::new(-1);
        latch.close(6, false, 7);
        // `note_close` on bar 7, ahead of its events.
        latch.close_bar = latch.close_bar.max(7);
        latch.step(7, false, 1);
        assert!(latch.holds());
    }

    #[test]
    fn the_first_bar_at_or_after_is_found_whatever_the_hint() {
        let stamps: Vec<i64> = (0..200).map(|i| i * 60 + (i / 7) * 13).collect();
        for from in 0..stamps.len() {
            for timestamp in stamps[from] + 1..stamps[from] + 400 {
                let expected = stamps.partition_point(|&t| t < timestamp);
                for hint in
                    [0, from, from + 1, expected, expected + 1, expected.saturating_sub(1), 999]
                {
                    assert_eq!(first_at_or_after(&stamps, from, timestamp, hint), expected);
                }
            }
        }
    }

    #[test]
    fn money_rounds_half_even_at_the_minor_unit() {
        assert_eq!(money_as_double(0.125, 2), Some(0.12));
        assert_eq!(money_as_double(0.375, 2), Some(0.38));
        assert_eq!(money_as_double(10_000.0, 2), Some(10_000.0));
        assert_eq!(money_as_double(f64::NAN, 2), None);
        assert_eq!(money_as_double(1.0, 17), None);
    }

    #[test]
    fn dust_is_under_a_lot_or_under_the_minimum_notional() {
        let level = Level { two_sided: false, size_increment: 0.001, minimum_notional: 5.0 };
        assert!(level.is_dust(0.0005, 100_000.0));
        assert!(level.is_dust(-0.002, 1_000.0));
        assert!(!level.is_dust(0.002, 10_000.0));
        // A NaN close proves nothing about the notional.
        assert!(!level.is_dust(0.002, f64::NAN));
    }

    #[test]
    fn dust_reads_as_flat_only_one_sided_and_with_no_close_pending() {
        let level = Level { two_sided: false, size_increment: 1.0, minimum_notional: 0.0 };
        assert!(level.reads_dust_as_flat(0.5, None, 10.0));
        assert!(level.reads_dust_as_flat(0.5, Some(1.0), 10.0));
        assert!(!level.reads_dust_as_flat(0.5, Some(0.0), 10.0));
        assert!(!level.reads_dust_as_flat(0.0, None, 10.0));
        assert!(!level.reads_dust_as_flat(2.0, None, 10.0));
        let two_sided = Level { two_sided: true, ..level };
        assert!(!two_sided.reads_dust_as_flat(0.5, None, 10.0));
    }

    #[test]
    fn comparisons_answer_as_python_floats_do() {
        for compare in [Compare::Gt, Compare::Ge, Compare::Lt, Compare::Le, Compare::Eq] {
            assert!(!compare.holds(f64::NAN, 1.0));
            assert!(!compare.holds(1.0, f64::NAN));
        }
        assert!(Compare::Ge.holds(1.0, 1.0) && !Compare::Gt.holds(1.0, 1.0));
        assert!(Compare::Le.holds(1.0, 1.0) && !Compare::Lt.holds(1.0, 1.0));
        assert!(Compare::Eq.holds(0.0, -0.0));
    }

    fn node(kind: u8, children: Vec<usize>, key: i64, code: usize) -> ConditionTuple {
        (kind, children, key, 1.0, -1, 2.0, code, 0.0, 1.0)
    }

    #[test]
    fn an_arena_parses_children_before_parents() {
        let nodes = vec![
            node(3, vec![], 0, 0),
            node(4, vec![], -1, 3),
            node(2, vec![1], -1, 0),
            node(0, vec![0, 2], -1, 0),
            node(6, vec![], 0, 0),
        ];
        let Ok(arena) = conditions(nodes, 1, 1) else { panic!("a well-formed arena refused") };
        assert!(matches!(arena[0], Condition::Zone { value: Operand::Key(0), .. }));
        assert!(matches!(
            arena[1],
            Condition::Threshold { left: Operand::Const(_), compare: Compare::Le, .. }
        ));
        assert!(matches!(arena[2], Condition::Not(1)));
        assert!(matches!(&arena[3], Condition::All(children) if children == &[0, 2]));
        assert!(matches!(arena[4], Condition::Cross(Crossing { clock: 0, above: false, .. })));
    }

    #[test]
    fn a_malformed_arena_is_refused() {
        for (nodes, n_clocks) in [
            (vec![node(0, vec![0], -1, 0)], 0),
            (vec![node(2, vec![], -1, 0)], 0),
            (vec![node(4, vec![], -1, 5)], 0),
            (vec![node(5, vec![], -1, 1)], 1),
            (vec![node(3, vec![], 1, 0)], 0),
            (vec![node(7, vec![], -1, 0)], 0),
        ] {
            assert!(conditions(nodes, 1, n_clocks).is_err());
        }
    }

    #[test]
    fn each_bar_steps_one_crossing_a_clock() {
        // `each_bar`'s list: each clock's crossing once, in clock order, however
        // many rules check it; the clock's name holds its operands and side, so
        // one clock naming two crossings, or none, is a malformed tree.
        let up = |clock| Crossing {
            clock,
            fast: Operand::Key(0),
            slow: Operand::Const(1.0),
            above: true,
        };
        let down =
            Crossing { clock: 1, fast: Operand::Key(0), slow: Operand::Const(1.0), above: false };
        let arena = vec![
            Condition::Cross(down),
            Condition::Cross(up(0)),
            Condition::All(vec![0, 1]),
            Condition::Cross(up(0)),
        ];
        let Ok(found) = clock_crossings(&arena, 2) else { panic!("a well-formed tree refused") };
        assert_eq!(found, vec![up(0), down]);
        let twice =
            vec![Condition::Cross(up(0)), Condition::Cross(Crossing { above: false, ..up(0) })];
        assert!(clock_crossings(&twice, 1).is_err());
        assert!(clock_crossings(&[Condition::Cross(up(0))], 2).is_err());
    }

    #[test]
    fn charges_book_what_the_open_position_paid() {
        // `_PositionCharges`: the opening fill's commission, plus every fill
        // after it that reduced the position; flat starts over, and so does
        // the close leg of a replacement.
        let mut charges = Charges { charged: 0.0, held: 0.0 };
        assert!(!charges.on_fill(1.0, 10.0, -1.0, false));
        assert_eq!((charges.charged, charges.held), (1.0, 10.0));
        assert!(!charges.on_fill(-1.0, 4.0, 0.4, false));
        assert_eq!((charges.charged, charges.held), (1.4, 6.0));
        assert_eq!(charges.round_trip(0.01, 6.0, 100.0, 110.0), 2.8);
        assert!(charges.on_fill(-1.0, 6.0, 0.6, true));
        assert_eq!((charges.charged, charges.held), (0.0, 0.0));
        // Uncharged, the fee rate across both notionals.
        assert_eq!(charges.round_trip(0.5, 2.0, 100.0, 110.0), 210.0);
        charges.on_fill(1.0, 10.0, 1.0, false);
        assert!(!charges.on_fill(-1.0, 10.0, 1.0, false));
        assert_eq!((charges.charged, charges.held), (0.0, 0.0));
    }

    #[test]
    fn a_trade_returns_net_of_costs_on_either_side() {
        assert_eq!(trade_roi_pct(100.0, 110.0, 2.0, 2.0), 9.0);
        assert_eq!(trade_roi_pct(100.0, 90.0, -2.0, 2.0), 9.0);
        assert_eq!(trade_roi_pct(100.0, 90.0, 2.0, 0.0), -10.0);
        assert_eq!(trade_roi_pct(0.0, 90.0, 2.0, 0.0), 0.0);
    }

    #[test]
    fn a_trailing_atr_stop_hangs_off_the_best_close() {
        // Long one unit from 100, ATR 2, a 2-ATR trailing stop: 96 at entry,
        // 101 once 105 prints, hit at 101. The hit clears the entry.
        let policies = Policies {
            trade: None,
            atr: None,
            hold: None,
            fee_rate: 0.0,
            style: OrderStyle::Market,
        };
        let atr = AtrExit { slot: PRIMARY, stop: 2.0, take_profit: 0.0, trailing: true };
        let (values, initialized) = ([2.0; 4], [false, true, true, true]);
        let account = Account { quantity: 1.0, entry: 100.0 };
        let mut p = PolicyState::NEW;
        let mut step = |idx, close| {
            Data::atr_exit(&policies, &atr, &values, &initialized, &mut p, account, idx, close)
        };
        // Not ready yet: nothing anchors.
        assert_eq!(step(0, 100.0), None);
        assert_eq!(step(1, 100.0), None);
        assert_eq!(step(2, 105.0), None);
        assert_eq!(step(3, 101.0), Some(PolicyExit::AtrStop));
        assert_eq!((p.atr_entry, p.atr_extreme), (0.0, 0.0));
        // A short's stop sits above, and trails the lowest close.
        let short = Account { quantity: -1.0, entry: 100.0 };
        let mut p = PolicyState::NEW;
        let fixed = AtrExit { trailing: false, take_profit: 1.0, ..atr };
        let mut step = |idx, close| {
            Data::atr_exit(&policies, &fixed, &values, &initialized, &mut p, short, idx, close)
        };
        assert_eq!(step(1, 100.0), None);
        assert_eq!(step(2, 98.0), Some(PolicyExit::AtrTakeProfit));
    }

    #[test]
    fn a_fee_is_the_double_nearest_its_decimal() {
        assert_eq!(borrow_amount(0), 0.0);
        assert_eq!(borrow_amount(12_345), 12_345e-10);
        assert_eq!(borrow_amount(-7), -7e-10);
        // Past 2**53 the integer is no longer exact as a double: the
        // decimal spelling still rounds once.
        let big: i128 = (1 << 60) + 1;
        assert_eq!(borrow_amount(big), "1152921504606846977e-10".parse::<f64>().unwrap());
        assert_eq!(borrow_amount(1 << 53), (1u64 << 53) as f64 / 1e10);
    }

    #[test]
    fn a_loan_is_charged_from_settlement_on_the_shares_held() {
        // Five bars over dates 10, 10, 11, 14, 15; a short opened on date 10
        // settles on 12, one opened on 14 on 16. One share costs 3, 5, 7, 11
        // and 13 units on dates 11..=15.
        let days = [10, 10, 11, 14, 15];
        let settles = [12, 12, 13, 16, 17];
        let charged = [0, 3, 8, 15, 26, 39];
        let borrow =
            BorrowView { days: &days, settles: &settles, first_day: 10, charged: &charged };
        let mut p = PolicyState::NEW;
        // Long or flat: no loan.
        borrow.observe(&mut p, 0, 4.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (None, 0));
        // Opened on bar 1: charged through date 11, the day before it settles.
        borrow.observe(&mut p, 1, -2.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(11), 0));
        borrow.observe(&mut p, 2, -2.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(11), 0));
        // Date 14 charges 12, 13 and 14 -- 5 + 7 + 11 units a share -- on
        // the three shares held when it gets there.
        borrow.observe(&mut p, 3, -3.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(14), 3 * 23));
        assert_eq!(p.borrow, 69.0 / 1e10);
        borrow.observe(&mut p, 4, -1.0).unwrap();
        assert_eq!((p.loan, p.borrow_units), (Some(15), 69 + 13));
        // Covered: the loan ends, and a new one starts from its own bar.
        borrow.observe(&mut p, 4, 0.0).unwrap();
        assert_eq!((p.loan, p.borrow_units, p.borrow), (None, 0, 0.0));
        borrow.observe(&mut p, 3, -1.0).unwrap();
        assert_eq!(p.loan, Some(15));
        // Fractional shares are not charged here.
        let mut p = PolicyState::NEW;
        borrow.observe(&mut p, 1, -0.5).unwrap();
        assert!(borrow.observe(&mut p, 3, -0.5).is_err());
    }
}
