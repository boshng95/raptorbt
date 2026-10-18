//! A crossover rule set's decision cursor, run beside the kernel.
//!
//! The account-coupled native lane of the parity runner decides on composite
//! (resampled) bars by walking a merged schedule of its symbol's streams and
//! evaluating crossover rules on each event, exactly as Nautilus's
//! `ComposableStrategy` does. This is that cursor, rule program and decision
//! node, moved out of Python for the one shape every such run has: a legacy
//! spec of direct crossovers, whose primary bars repeat the last operands.
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
//! started from otherwise so that Python can take the same bar itself.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const PRIMARY: usize = 0;

/// `10**p` for the precisions `money_as_double` accepts, as exact doubles.
const POW10: [f64; 17] =
    [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16];
const FIXED_PRECISION: u32 = 16;
const FIXED_SCALAR: f64 = 1e16;

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
    let raw = (minor as i128).checked_mul(10i128.pow(FIXED_PRECISION - currency_precision))?;
    Some(raw as f64 / FIXED_SCALAR)
}

#[derive(Clone, Copy, Debug)]
enum Operand {
    Key(usize),
    Const(f64),
}

#[derive(Clone, Copy, Debug)]
struct Crossing {
    clock: usize,
    fast: Operand,
    slow: Operand,
    above: bool,
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
    pub policies_after_fill: bool,
}

#[derive(Clone, Debug)]
struct Cursor {
    indexes: Vec<usize>,
    current: Option<usize>,
    timestamp: i64,
    label: i64,
    priority: i64,
    slot: usize,
    bar: usize,
}

/// Everything a bar can change. Cloned before a fast-forward bar so that a
/// bar the strategy would act on can be handed back untouched.
#[derive(Clone, Debug)]
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
}

struct Registration {
    slot: usize,
    stream_index: i64,
    rank: i64,
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
}

struct Data {
    regs: Vec<Registration>,
    slot_close: Vec<Py<PyArray1<f64>>>,
    slot_admits: Vec<Option<Py<PyArray1<bool>>>>,
    slot_decision: Vec<bool>,
    slot_keys: Vec<Vec<usize>>,
    key_values: Vec<Py<PyArray1<f64>>>,
    key_init: Vec<Py<PyArray1<bool>>>,
    exits: Vec<Crossing>,
    entries: Vec<Crossing>,
    legacy: bool,
    dispatch_ts: Py<PyArray1<i64>>,
    dispatch_pos: Py<PyArray1<i64>>,
    primary_close: Py<PyArray1<f64>>,
    dust: bool,
    position_required: bool,
    has_instrument: bool,
    size_increment: f64,
    minimum_notional: f64,
}

fn slice<'a, T: numpy::Element>(array: &'a Py<PyArray1<T>>, py: Python<'a>) -> PyResult<&'a [T]> {
    // SAFETY: every array is a prepared, immutable input of one backtest run,
    // kept alive by the `Py` handle for as long as the session; nothing writes
    // to it while a run is in progress.
    unsafe { array.as_ref(py).as_slice() }
        .map_err(|_| PyValueError::new_err("rule session arrays must be contiguous"))
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
        })
    }

    /// `ScheduledStream.order`.
    fn order(&self, view: &View, reg: usize, index: usize) -> (i64, i64) {
        let registration = &self.regs[reg];
        if registration.slot == PRIMARY {
            return (1, 0);
        }
        if let Some(labels) = view.reg_labels[reg] {
            if labels[index] > view.reg_ts[reg][index] {
                return (0, registration.stream_index);
            }
        }
        (2, registration.rank)
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

    /// `NativeScheduleCursor.skip_primary_before`.
    fn skip_primary_before(&self, view: &View, cursor: &mut Cursor, timestamp: i64) {
        let mut changed = false;
        for reg in 0..self.regs.len() {
            if self.regs[reg].slot != PRIMARY {
                continue;
            }
            let index = view.reg_ts[reg].partition_point(|&t| t < timestamp);
            if index > cursor.indexes[reg] {
                cursor.indexes[reg] = index;
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
        if !self.dust || state.last_position != 0 || side == 0 || stop <= start {
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

    fn next_dispatch_position(&self, view: &View, state: &State) -> usize {
        if state.next_dispatch < view.dispatch_pos.len() {
            view.dispatch_pos[state.next_dispatch] as usize
        } else {
            view.primary_close.len()
        }
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
                if rule.above {
                    p0 <= p1 && a > b
                } else {
                    p0 >= p1 && a < b
                }
            }
        }
    }

    /// `DecisionProgram.evaluate` over the node's current updates.
    fn evaluate(
        &self,
        state: &mut State,
        slot: usize,
        label: i64,
        dispatch: i64,
        position: i32,
    ) -> PyResult<RuleMatch> {
        let previous = state.last_labels[slot];
        if previous == Some(label) {
            return Ok(RuleMatch { ready: true, ..RuleMatch::default() });
        }
        let phase = if slot == PRIMARY {
            1
        } else if label > dispatch {
            0
        } else {
            2
        };
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
        let exit_rule = if position != 0 || !self.legacy {
            self.exits.iter().position(|rule| Self::crossing(state, rule))
        } else {
            None
        };
        let entry_rule = self.entries.iter().position(|rule| Self::crossing(state, rule));
        Ok(RuleMatch { evaluated: true, ready: true, exit_rule, entry_rule, reached: true })
    }

    /// `NativeModelRunner.visit` and `_DeclarativeNode.visit`.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        view: &View,
        state: &mut State,
        side: i32,
        quantity: f64,
        pending: Option<f64>,
    ) -> PyResult<Option<NodeIntent>> {
        let (slot, bar, dispatch, label) =
            (state.cursor.slot, state.cursor.bar, state.cursor.timestamp, state.cursor.label);
        if let Some(admits) = view.slot_admits[slot] {
            if !admits[bar] {
                state.last_reached = false;
                return Ok(None);
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
        let matched = self.evaluate(state, slot, label, dispatch, position)?;
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

    /// `NativeExecutionSession.advance` for a deferred, repeat-safe plan with
    /// no policies, no borrow and signal entries.
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
    ) -> PyResult<(bool, Option<Target>)> {
        let held_side = if !has_position {
            0
        } else if net_position > 0.0 {
            1
        } else if net_position < 0.0 {
            -1
        } else {
            0
        };
        let due = state.next_dispatch < view.dispatch_ts.len();
        let next_dispatch = if due { view.dispatch_ts[state.next_dispatch] } else { now };
        if !state.visit_primary_after_fill
            && (!due || now < next_dispatch)
            && !(has_position
                && self.reopens_exits(view, state, held_side, net_position, pending, idx, idx + 1))
        {
            return Ok((false, None));
        }
        self.skip_primary_before(view, &mut state.cursor, now.min(next_dispatch));
        while state.next_dispatch < view.dispatch_ts.len()
            && view.dispatch_ts[state.next_dispatch] <= now
        {
            state.next_dispatch += 1;
        }
        // `_refresh`: the account the rules see for the rest of the call.
        let quantity = if has_position { net_position } else { 0.0 };
        let start_side = held_side;
        let mut target = start_side;
        let mut reference: Option<f64> = None;
        let mut reason: Option<RuleId> = None;
        let mut entry_timestamp: Option<i64> = None;
        let mut policies_after_fill = false;
        while state.cursor.current.is_some() && state.cursor.timestamp <= now {
            let timestamp = state.cursor.timestamp;
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
            if state.cursor.slot == PRIMARY {
                state.visit_primary_after_fill = false;
            }
            if entry_timestamp == Some(timestamp) {
                policies_after_fill = true;
            }
            let intent = self.visit(view, state, start_side, quantity, pending)?;
            self.cursor_advance(view, &mut state.cursor);
            let previous_target = target;
            let Some(intent) = intent else { continue };
            if !(intent.wants_entry || intent.wants_exit) {
                continue;
            }
            // `target_for_intent`.
            if intent.wants_exit && target != 0 {
                target = 0;
            }
            if intent.wants_entry && target == 0 {
                target = 1;
            }
            if previous_target == 0 && target != 0 {
                entry_timestamp = Some(timestamp);
            }
            reference = Some(intent.reference);
            reason = intent.reason;
        }
        if target == start_side {
            return Ok((true, None));
        }
        Ok((true, Some(Target { side: target, reference, reason, policies_after_fill })))
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
    pub(crate) fn quiet_bar(
        &self,
        view: &View,
        state: &mut State,
        bar: &QuietBar,
        quiet: &QuietParams,
    ) -> bool {
        let saved = state.clone();
        let quiet_bar = self.try_quiet_bar(view, state, bar, quiet);
        if !quiet_bar {
            *state = saved;
        }
        quiet_bar
    }

    fn try_quiet_bar(
        &self,
        view: &View,
        state: &mut State,
        bar: &QuietBar,
        quiet: &QuietParams,
    ) -> bool {
        // An error is Python's to raise, on its own call.
        let Ok((processed, target)) = self.advance(
            view,
            state,
            bar.now,
            bar.idx,
            bar.has_position,
            bar.net_position,
            bar.pending,
        ) else {
            return false;
        };
        // `_refresh` reads the cash as `money_as_double`, which raises on a
        // value that is not a number.
        if processed && money_as_double(bar.cash, quiet.currency_precision).is_none() {
            return false;
        }
        if bar.halted {
            return true;
        }
        if target.is_some() || bar.pending.is_some() {
            return false;
        }
        if !bar.has_position {
            let Some(cash) = money_as_double(bar.cash, quiet.currency_precision) else {
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
        !(bar.has_position
            && self.held_reopens(
                view,
                state,
                bar.idx,
                bar.has_position,
                bar.net_position,
                bar.pending,
            ))
    }
}

/// The account in front of one callback bar, read before the kernel steps it.
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

#[pymethods]
impl PyCrossoverRuleSession {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        reg_slots: Vec<usize>,
        reg_stream_index: Vec<i64>,
        reg_rank: Vec<i64>,
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
    ) -> PyResult<Self> {
        let n_regs = reg_slots.len();
        let n_slots = slot_close.len();
        let n_keys = key_slot.len();
        if [reg_stream_index.len(), reg_rank.len(), reg_timestamps.len(), reg_labels.len()]
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
                timestamps: reg_timestamps[i].clone_ref(py),
                labels: reg_labels[i].as_ref().map(|a| a.clone_ref(py)),
            })
            .collect();
        let data = Data {
            regs,
            slot_close,
            slot_admits,
            slot_decision,
            slot_keys,
            key_values,
            key_init,
            exits: crossings(exits, n_keys, n_clocks)?,
            entries: crossings(entries, n_keys, n_clocks)?,
            legacy,
            dispatch_ts: dispatch_timestamps,
            dispatch_pos: dispatch_positions,
            primary_close,
            dust,
            position_required,
            has_instrument,
            size_increment,
            minimum_notional,
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
            if view.dispatch_pos.iter().any(|&p| p < 0 || p as usize > view.primary_close.len()) {
                return Err(PyValueError::new_err("dispatch position out of range"));
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
            }
        };
        Ok(Self { data, state })
    }

    /// One `advance` callback: `(processed, target)`, where `processed` says
    /// the call got past the deferral and `target` is `None` for `IDLE`, else
    /// `(target_side, reference_price, (is_exit, rule_index) | None,
    /// policies_after_fill)`.
    #[pyo3(signature = (now, idx, has_position, net_position, pending_target=None))]
    #[allow(clippy::type_complexity)]
    fn advance(
        &mut self,
        py: Python<'_>,
        now: i64,
        idx: usize,
        has_position: bool,
        net_position: f64,
        pending_target: Option<f64>,
    ) -> PyResult<(bool, Option<(i32, Option<f64>, Option<(bool, usize)>, bool)>)> {
        let Self { data, state } = self;
        let view = data.view(py)?;
        let (processed, target) =
            data.advance(&view, state, now, idx, has_position, net_position, pending_target)?;
        Ok((processed, target.map(|t| (t.side, t.reference, t.reason, t.policies_after_fill))))
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
    #[pyo3(signature = (
        idx, now, close, has_position, net_position, cash, halted, pending_target, quiet,
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
        };
        Ok(self.data.quiet_bar(&view, &mut state, &bar, &QuietParams::from(quiet)))
    }

    /// Mark that a fill landed, which makes the next primary bar a visit.
    fn on_fill(&mut self) {
        self.state.visit_primary_after_fill = true;
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

    /// Every piece of mutable state, for lockstep comparison with Python, as
    /// `((cursor_indexes, cursor_active, next_dispatch_index,
    /// visit_primary_after_fill), (updates, values, unready, clocks,
    /// last_labels, last_sort_key), (last_position, last_reached,
    /// last_ready))`.
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
    ) {
        let s = &self.state;
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
        )
    }
}

impl PyCrossoverRuleSession {
    /// Step callback bars on the strategy's behalf; see
    /// `PyKernelSession::step_rules_until`.
    pub(crate) fn split<'a>(&'a mut self, py: Python<'a>) -> PyResult<RuleStepper<'a>> {
        let view = self.data.view(py)?;
        Ok(RuleStepper { data: &self.data, view, state: &mut self.state })
    }
}

/// One borrow of a rule session's arrays, for a run of quiet bars.
pub(crate) struct RuleStepper<'a> {
    data: &'a Data,
    view: View<'a>,
    state: &'a mut State,
}

impl RuleStepper<'_> {
    pub(crate) fn quiet_bar(&mut self, bar: &QuietBar, quiet: &QuietParams) -> bool {
        self.data.quiet_bar(&self.view, self.state, bar, quiet)
    }
}

#[cfg(test)]
mod tests {
    use super::money_as_double;

    #[test]
    fn money_rounds_half_even_at_the_minor_unit() {
        assert_eq!(money_as_double(0.125, 2), Some(0.12));
        assert_eq!(money_as_double(0.375, 2), Some(0.38));
        assert_eq!(money_as_double(10_000.0, 2), Some(10_000.0));
        assert_eq!(money_as_double(f64::NAN, 2), None);
        assert_eq!(money_as_double(1.0, 17), None);
    }
}
