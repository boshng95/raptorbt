//! The lean lane's door: a symbol's market built once, the frame each of its
//! resampled streams puts the cells on, and library cells replayed on a
//! frame into what the account-coupled native lane gives -- its replay log,
//! daily curve, end value and canonical report.
//!
//! The arguments are the ones that lane's `KernelSession.replay_rules` and
//! `CrossoverRuleSession` take, already read down to the one shape the lean
//! lane runs (`crate::lean`). Where a cell leaves that shape the call raises
//! [`LeanDeclined`] (or, in a batch, names why) and the caller runs the cell
//! on the full lane.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;

use super::fill_ledger::{BookTerms, Ledger};
use super::money::PyMoneyModel;
use super::replay_session::Venue;
use super::report_bindings::{to_python, trip_rows, Trip};
use super::rule_session::{QuietParams, QuietTuple};
use crate::lean::kernel::{Terms, Trade};
use crate::lean::market::Market;
use crate::lean::replay::{replay, Cell, Frame, Outcome};
use crate::lean::rules::{
    Compare, Condition, Cross, Dispatch, Level, Operand, Signals, Stream, Tree,
};
use crate::lean::Declined;
use crate::metrics::canonical::{calculate, LocalClock, ReportInputs, Value};

create_exception!(
    raptorbt,
    LeanDeclined,
    PyRuntimeError,
    "The cell takes a branch the lean lane does not transcribe; run it on the full lane."
);

fn declined(why: Declined) -> PyErr {
    LeanDeclined::new_err(why.0)
}

/// `(bar_ns, venue_ioc, entry_market, exit_market, market_markup,
/// size_increment, minimum_quantity, minimum_notional, lowest_close,
/// order_tag, first_order_seq, max_actions_per_timestamp)`, as
/// `replay_rules` takes it.
type VenueTuple = (i64, bool, bool, bool, f64, f64, f64, f64, f64, String, u64, usize);

/// `(balance, currency_precision, size_increment, size_precision,
/// price_precision, currency)`.
type BookTuple = (f64, u32, f64, usize, usize, String);

/// `(initial_capital, price_increment, max_quantity, currency_precision,
/// fees, fee_per_share, fee_minimum, fee_max_pct)`.
type TermsTuple = (f64, f64, Option<f64>, Option<u32>, f64, f64, f64, f64);

/// One crossover: `(a_key, a_value, b_key, b_value, above, clock)`, an
/// operand a key or, where the key is negative, the constant.
type CrossTuple = (i64, f64, i64, f64, bool, usize);

/// `(key_values, key_init, admits, primary_admits)`: a cell's keys on its
/// frame's resample bars, and its session filter's admits on them and on the
/// primary bars, None where it admits every bar.
type KeyArrays<'py> = (
    Vec<PyReadonlyArray1<'py, f64>>,
    Vec<PyReadonlyArray1<'py, bool>>,
    Option<PyReadonlyArray1<'py, bool>>,
    Option<PyReadonlyArray1<'py, bool>>,
);

/// `(dust, position_required, has_instrument, size_increment,
/// minimum_notional)`, the rule session's.
type RuleFlags = (bool, bool, bool, f64, f64);

/// A batch's arrays, passed once: `(key_values, key_init, admits)`. Its
/// cells name their keys and admits by place.
type KeyPool<'py> = (
    Vec<PyReadonlyArray1<'py, f64>>,
    Vec<PyReadonlyArray1<'py, bool>>,
    Vec<PyReadonlyArray1<'py, bool>>,
);

/// One cell of a batch: `(keys, admits, primary_admits, entry, exit,
/// flags)`, its keys places in the pool's values and init alike, an admits
/// a place in its admits or -1 for none.
type CellTuple = (Vec<usize>, i64, i64, CrossTuple, CrossTuple, RuleFlags);

/// One node of a condition arena, as `CrossoverRuleSession` takes it:
/// `(kind, children, a_key, a_value, b_key, b_value, code, lower, upper)`
/// (`rule_session::ConditionTuple`).
type ConditionTuple = (u8, Vec<usize>, i64, f64, i64, f64, usize, f64, f64);

/// Level rules: `(conditions, exit_roots, entry_roots, n_clocks, (two_sided,
/// size_increment, minimum_notional))`, the arena and `level` arguments
/// `CrossoverRuleSession` takes.
type TreeTuple = (Vec<ConditionTuple>, Vec<usize>, Vec<usize>, usize, (bool, f64, f64));

/// One symbol's primary bars and the book each leaves, shared by every cell
/// replayed on them.
#[pyclass(name = "LeanMarket", module = "raptorbt", frozen)]
pub struct PyLeanMarket {
    market: Arc<Market>,
    /// The bars the daily curve keeps.
    day_ends: Arc<Vec<usize>>,
}

#[pymethods]
impl PyLeanMarket {
    /// *lot* is the instrument's lot size (0 for none), *slices* the
    /// config's `bar_volume_slices`, *transitions* its
    /// `performance_tz_transitions`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        timestamps: PyReadonlyArray1<'_, i64>,
        open: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        low: PyReadonlyArray1<'_, f64>,
        close: PyReadonlyArray1<'_, f64>,
        volume: PyReadonlyArray1<'_, f64>,
        lot: f64,
        slices: f64,
        transitions: Vec<(i64, i64)>,
    ) -> PyResult<Self> {
        let n = timestamps.len();
        if [open.len(), high.len(), low.len(), close.len(), volume.len()].iter().any(|&l| l != n) {
            return Err(PyValueError::new_err("bar columns of different lengths"));
        }
        let market = Market::new(
            timestamps.as_array().to_vec(),
            open.as_array().to_vec(),
            high.as_array().to_vec(),
            low.as_array().to_vec(),
            close.as_array().to_vec(),
            volume.as_array().to_vec(),
            lot,
            slices,
        );
        let day_ends = Arc::new(market.day_ends(transitions));
        Ok(Self { market: Arc::new(market), day_ends })
    }

    fn __len__(&self) -> usize {
        self.market.len()
    }
}

/// The cells of one resampled stream on a market: the bars `on_bar` is due
/// on, as positions (`np.flatnonzero` of the replay mask), and the stream's
/// `(dispatch_timestamps, dispatch_positions, labels, close)`. Checked once
/// and shared by every cell whose keys are on the stream.
#[pyclass(name = "LeanFrame", module = "raptorbt", frozen)]
pub struct PyLeanFrame {
    market: Arc<Market>,
    day_ends: Arc<Vec<usize>>,
    frame: Frame,
}

#[pymethods]
impl PyLeanFrame {
    #[new]
    fn new(
        market: PyRef<'_, PyLeanMarket>,
        due: PyReadonlyArray1<'_, i64>,
        dispatch_timestamps: PyReadonlyArray1<'_, i64>,
        dispatch_positions: PyReadonlyArray1<'_, i64>,
        labels: PyReadonlyArray1<'_, i64>,
        close: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let positions = |array: &PyReadonlyArray1<'_, i64>, what: &str| -> PyResult<Vec<usize>> {
            let array = array.as_array();
            if array.iter().any(|&at| at < 0) {
                return Err(PyValueError::new_err(format!("{what} before the first bar")));
            }
            Ok(array.iter().map(|&at| at as usize).collect())
        };
        let dispatch = Dispatch::new(
            dispatch_timestamps.as_array().to_vec(),
            positions(&dispatch_positions, "a dispatch")?,
            &labels.as_array().to_vec(),
            close.as_array().to_vec(),
        )
        .map_err(declined)?;
        let frame = Frame::new(&market.market, positions(&due, "a due bar")?, dispatch)
            .map_err(declined)?;
        Ok(Self {
            market: Arc::clone(&market.market),
            day_ends: Arc::clone(&market.day_ends),
            frame,
        })
    }

    fn __len__(&self) -> usize {
        self.frame.dispatch.len()
    }
}

fn operand(key: i64, value: f64) -> Operand {
    if key >= 0 {
        Operand::Key(key as usize)
    } else {
        Operand::Const(value)
    }
}

fn cross(spec: CrossTuple) -> Cross {
    Cross { fast: operand(spec.0, spec.1), slow: operand(spec.2, spec.3), above: spec.4 }
}

/// A [`TreeTuple`]'s level rules; ValueError where it is malformed.
fn tree(spec: &TreeTuple) -> PyResult<Tree> {
    let (nodes, exits, entries, clocks, (two_sided, size_increment, minimum_notional)) = spec;
    let malformed = |what: &str| PyValueError::new_err(format!("malformed condition: {what}"));
    let conditions = nodes
        .iter()
        .map(|(kind, children, a_key, a_value, b_key, b_value, code, lower, upper)| {
            let (a, b) = (operand(*a_key, *a_value), operand(*b_key, *b_value));
            Ok(match kind {
                0 => Condition::All(children.clone()),
                1 => Condition::Any(children.clone()),
                2 => match children.as_slice() {
                    [child] => Condition::Not(*child),
                    _ => return Err(malformed("NOT takes one child")),
                },
                3 => Condition::Zone { value: a, lower: *lower, upper: *upper },
                4 => Condition::Threshold {
                    left: a,
                    right: b,
                    compare: match code {
                        0 => Compare::Gt,
                        1 => Compare::Ge,
                        2 => Compare::Lt,
                        3 => Compare::Le,
                        4 => Compare::Eq,
                        _ => return Err(malformed("unknown comparison")),
                    },
                },
                5 | 6 => Condition::Cross {
                    clock: *code,
                    cross: Cross { fast: a, slow: b, above: *kind == 5 },
                },
                _ => return Err(malformed("unknown kind")),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let level = Level {
        two_sided: *two_sided,
        size_increment: *size_increment,
        minimum_notional: *minimum_notional,
    };
    Tree::new(conditions, exits.clone(), entries.clone(), *clocks, level)
        .map_err(|why| malformed(why.0))
}

/// What every cell of a call runs with besides its keys and rules.
struct Shared<'a> {
    frame: &'a PyLeanFrame,
    gate: Option<&'a [bool]>,
    terms: Terms,
    quiet: QuietParams,
    venue: Venue,
    money: &'a PyMoneyModel,
    book: BookTuple,
    starting: f64,
    timezone: &'a str,
    /// None where the offsets do not make a clock, and no cell reports.
    clock: Option<LocalClock>,
    jumps: bool,
    /// What the cells share on the frame's dispatch.
    signals: Signals<'a>,
}

impl<'a> Shared<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        frame: &'a PyLeanFrame,
        gate: Option<&'a [bool]>,
        terms: TermsTuple,
        quiet: QuietTuple,
        venue: VenueTuple,
        money: &'a PyMoneyModel,
        book: BookTuple,
        starting: f64,
        timezone: &'a str,
        offsets: &[(i64, i64)],
        offsets_until_ns: i64,
        jumps: bool,
    ) -> Self {
        Self {
            frame,
            gate,
            terms: Terms {
                initial_capital: terms.0,
                price_increment: terms.1,
                max_quantity: terms.2,
                currency_precision: terms.3,
                fees: terms.4,
                fee_per_share: terms.5,
                fee_minimum: terms.6,
                fee_max_pct: terms.7,
            },
            quiet: QuietParams::from(quiet),
            venue: Venue {
                bar_ns: venue.0,
                venue_ioc: venue.1,
                entry_market: venue.2,
                exit_market: venue.3,
                market_markup: venue.4,
                size_increment: venue.5,
                minimum_quantity: venue.6,
                minimum_notional: venue.7,
                lowest_close: venue.8,
                order_tag: venue.9,
                first_order_seq: venue.10,
                max_actions_per_timestamp: venue.11,
            },
            money,
            book,
            starting,
            timezone,
            clock: LocalClock::new(offsets, offsets_until_ns),
            jumps,
            signals: Signals::new(&frame.frame.dispatch),
        }
    }

    fn book(&self) -> BookTerms {
        BookTerms {
            balance: self.book.0,
            currency_precision: self.book.1,
            size_increment: self.book.2,
            size_precision: self.book.3,
            price_precision: self.book.4,
            currency: self.book.5.clone(),
        }
    }
}

/// One cell's keys and rules, borrowed from the caller's arrays.
struct Spec<'a> {
    values: Vec<&'a [f64]>,
    init: Vec<&'a [bool]>,
    admits: Option<&'a [bool]>,
    primary_admits: Option<&'a [bool]>,
    entry: CrossTuple,
    exit: CrossTuple,
    /// Level rules, in place of `entry` and `exit`.
    tree: Option<Tree>,
    flags: RuleFlags,
}

impl<'a> Spec<'a> {
    fn of(
        keys: &'a KeyArrays<'_>,
        entry: CrossTuple,
        exit: CrossTuple,
        tree: Option<Tree>,
        flags: RuleFlags,
    ) -> PyResult<Self> {
        let (values, init, admits, primary_admits) = keys;
        Ok(Self {
            values: values.iter().map(|key| key.as_slice()).collect::<Result<_, _>>()?,
            init: init.iter().map(|flags| flags.as_slice()).collect::<Result<_, _>>()?,
            admits: admits.as_ref().map(|admits| admits.as_slice()).transpose()?,
            primary_admits: primary_admits.as_ref().map(|admits| admits.as_slice()).transpose()?,
            entry,
            exit,
            tree,
            flags,
        })
    }

    /// A batch cell's, off the pool's slices; ValueError where it names a
    /// place the pool does not have.
    fn pooled(pool: &Pool<'a>, cell: &CellTuple, tree: Option<Tree>) -> PyResult<Self> {
        let (keys, admits, primary_admits, entry, exit, flags) = cell;
        let outside =
            || PyValueError::new_err("a cell names a key or admits the pool does not have");
        let admitted = |at: i64| -> PyResult<Option<&'a [bool]>> {
            match usize::try_from(at) {
                Ok(at) => pool.admits.get(at).copied().map(Some).ok_or_else(outside),
                Err(_) if at == -1 => Ok(None),
                Err(_) => Err(outside()),
            }
        };
        Ok(Self {
            values: keys
                .iter()
                .map(|&key| pool.values.get(key).copied().ok_or_else(outside))
                .collect::<PyResult<_>>()?,
            init: keys
                .iter()
                .map(|&key| pool.init.get(key).copied().ok_or_else(outside))
                .collect::<PyResult<_>>()?,
            admits: admitted(*admits)?,
            primary_admits: admitted(*primary_admits)?,
            entry: *entry,
            exit: *exit,
            tree,
            flags: *flags,
        })
    }
}

/// A [`KeyPool`]'s arrays as slices.
struct Pool<'a> {
    values: Vec<&'a [f64]>,
    init: Vec<&'a [bool]>,
    admits: Vec<&'a [bool]>,
}

impl<'a> Pool<'a> {
    fn of(pool: &'a KeyPool<'_>) -> PyResult<Self> {
        let (values, init, admits) = pool;
        if values.len() != init.len() {
            return Err(PyValueError::new_err("key values and init of different counts"));
        }
        Ok(Self {
            values: values.iter().map(|key| key.as_slice()).collect::<Result<_, _>>()?,
            init: init.iter().map(|flags| flags.as_slice()).collect::<Result<_, _>>()?,
            admits: admits.iter().map(|admits| admits.as_slice()).collect::<Result<_, _>>()?,
        })
    }
}

/// A cell's replay, with the report built from it.
struct Ran {
    end_value: f64,
    day_ns: Vec<i64>,
    day_equity: Vec<f64>,
    /// Bars stepped one by one, jumps taken, callbacks.
    stats: (usize, usize, usize),
    /// `(report, total_trades, closed_trade_pnls)`; None where
    /// `canonical_report` declines.
    report: Option<(Value, usize, Vec<f64>)>,
    /// Seconds spent setting the stream up, replaying, and reporting.
    secs: (f64, f64, f64),
}

/// Replay one cell, keeping its log where `log`, and hand its outcome to
/// `keep` before it is reported (the single-cell call reads the log off it
/// there).
fn run(
    shared: &Shared<'_>,
    spec: &Spec<'_>,
    log: bool,
    keep: impl FnOnce(&Outcome),
) -> Result<Ran, Declined> {
    if spec.tree.is_none() && spec.entry.5 == spec.exit.5 {
        return Err(Declined("a clock both sides share"));
    }
    let frame = shared.frame;
    let entered = Instant::now();
    let stream = Stream::new(
        &shared.signals,
        spec.values.clone(),
        &spec.init,
        spec.admits,
        spec.primary_admits,
        cross(spec.entry),
        cross(spec.exit),
        spec.tree.as_ref(),
        spec.flags.0,
        spec.flags.1,
        spec.flags.2,
        spec.flags.3,
        spec.flags.4,
    )?;
    let cell = Cell {
        frame: &frame.frame,
        gate: shared.gate,
        quiet: shared.quiet,
        venue: &shared.venue,
        money: shared.money,
        day_ends: &frame.day_ends,
        jumps: shared.jumps,
    };
    let started = Instant::now();
    let outcome = replay(&frame.market, &shared.terms, &stream, &cell, Some(shared.book()), log)?;
    let reported = Instant::now();
    keep(&outcome);
    let Outcome { ledger, trades, day_ns, day_equity, end_value, stepped, jumped, callbacks } =
        outcome;
    let report = report(shared, ledger, &trades, &day_ns, &day_equity);
    let secs = (
        started.duration_since(entered).as_secs_f64(),
        reported.duration_since(started).as_secs_f64(),
        reported.elapsed().as_secs_f64(),
    );
    Ok(Ran { end_value, day_ns, day_equity, stats: (stepped, jumped, callbacks), report, secs })
}

/// `canonical_report` of a replay: the report, its order count and the
/// closed trades' P&Ls.
fn report(
    shared: &Shared<'_>,
    ledger: Ledger,
    trades: &[Trade],
    day_ns: &[i64],
    day_equity: &[f64],
) -> Option<(Value, usize, Vec<f64>)> {
    let clock = shared.clock.as_ref()?;
    let precision = ledger.currency_precision()?;
    let trips = trades.iter().map(|trade| Trip {
        entry_time: trade.entry_time,
        exit_time: trade.exit_time,
        size: trade.size,
        entry_price: trade.entry_price,
        exit_price: trade.exit_price,
        pnl: trade.pnl,
        entry_fees: trade.entry_fees,
        exit_fees: trade.exit_fees,
        short: false,
        end_of_data: trade.end_of_data,
    });
    let rows = trip_rows(trips, &ledger.reduction_times()?, precision)?;
    let fills = ledger.report_amounts()?;
    let report = calculate(&ReportInputs {
        starting: shared.starting,
        risk_free_rate: 0.0,
        timezone: shared.timezone,
        clock,
        day_ns,
        day_equity,
        trades: &rows,
        fills: &fills,
    })?;
    let closed: Vec<f64> =
        rows.iter().filter(|row| row.closed_s.is_some()).map(|row| row.realized).collect();
    Some((report, fills.price.len(), closed))
}

/// CPython 3.12's `sum()` of floats from its int start 0: the first term
/// added to 0 (a -0.0 reads 0.0), the rest by Neumaier's compensated sum,
/// whose compensation is added last where nonzero and finite.
#[derive(Default)]
struct PySum {
    terms: usize,
    total: f64,
    compensation: f64,
}

impl PySum {
    fn add(&mut self, x: f64) {
        if self.terms == 0 {
            self.total = 0.0 + x;
        } else {
            let t = self.total + x;
            if self.total.abs() >= x.abs() {
                self.compensation += (self.total - t) + x;
            } else {
                self.compensation += (x - t) + self.total;
            }
            self.total = t;
        }
        self.terms += 1;
    }

    fn value(&self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.total + self.compensation
        } else {
            self.total
        }
    }
}

/// What the sweep's legacy trade columns read of closed trades' P&Ls:
/// `(trades, wins, the wins' sum, losses, the losses' sum)`, a win strictly
/// positive and a loss at most zero (a NaN is neither), each sum Python's
/// own (`legacy_closed_trade_stats`).
fn closed_trade_tally(pnls: &[f64]) -> (usize, usize, f64, usize, f64) {
    let (mut wins, mut losses) = (PySum::default(), PySum::default());
    for &pnl in pnls {
        if pnl > 0.0 {
            wins.add(pnl);
        } else if pnl <= 0.0 {
            losses.add(pnl);
        }
    }
    (pnls.len(), wins.terms, wins.value(), losses.terms, losses.value())
}

/// A replayed cell as a dict: `report`, `total_trades`,
/// `closed_trade_pnls` and their `closed_trade_tally`, `end_value`, `secs`
/// (the replay's), `split` (the stream's setup, the report's) and `stats`
/// (bars stepped, jumps, callbacks), with its daily curve as `day_ns` and
/// `day_equity` where *curve*.
fn to_dict(py: Python<'_>, ran: &Ran, curve: bool) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    match &ran.report {
        Some((report, orders, closed)) => {
            out.set_item("report", to_python(py, report)?)?;
            out.set_item("total_trades", orders)?;
            out.set_item("closed_trade_pnls", PyList::new(py, closed))?;
            out.set_item("closed_trade_tally", closed_trade_tally(closed))?;
        }
        None => {
            out.set_item("report", py.None())?;
            out.set_item("total_trades", py.None())?;
            out.set_item("closed_trade_pnls", py.None())?;
            out.set_item("closed_trade_tally", py.None())?;
        }
    }
    out.set_item("end_value", ran.end_value)?;
    if curve {
        out.set_item("day_ns", &ran.day_ns)?;
        out.set_item("day_equity", &ran.day_equity)?;
    }
    out.set_item("secs", ran.secs.1)?;
    out.set_item("split", (ran.secs.0, ran.secs.2))?;
    out.set_item("stats", ran.stats)?;
    Ok(out.into_py(py))
}

/// The columns of a batch's screen: its reports' metric, month and year
/// names, in the order the first reported cell gives them.
struct Axes {
    metrics: Vec<String>,
    months: Vec<String>,
    years: Vec<String>,
}

fn names(map: Option<&Value>) -> Vec<String> {
    match map {
        Some(Value::Map(entries)) => entries.iter().map(|(name, _)| name.clone()).collect(),
        _ => Vec::new(),
    }
}

impl Axes {
    fn of(report: &Value) -> Self {
        Self {
            metrics: names(report.get("metrics")),
            months: names(report.get("monthly_returns_pct")),
            years: names(report.get("yearly_returns_pct")),
        }
    }
}

/// A report value as a screen number: NaN for None and for text that is
/// not one (the worst calendar year is read as its year).
fn number(value: &Value) -> f64 {
    match value {
        Value::Float(value) => *value,
        Value::Int(value) => *value as f64,
        Value::Bool(value) => f64::from(u8::from(*value)),
        Value::Text(text) => text.parse().unwrap_or(f64::NAN),
        Value::None | Value::Map(_) => f64::NAN,
    }
}

/// One cell's screen row.
struct Row {
    end_value: f64,
    /// -1 where the cell is not reported.
    total_trades: i64,
    metrics: Vec<f64>,
    months: Vec<f64>,
    years: Vec<f64>,
    /// A report name outside the axes, its value left out.
    off_axis: bool,
    stats: (usize, usize, usize),
    secs: (f64, f64, f64),
}

/// `map`'s values in the places of `axis`, NaN where it has none; false
/// where it names one `axis` does not have.
fn place(axis: &[String], map: Option<&Value>, row: &mut [f64]) -> bool {
    let Some(Value::Map(entries)) = map else {
        return true;
    };
    let mut placed = true;
    for (at, (name, value)) in entries.iter().enumerate() {
        let slot =
            if axis.get(at) == Some(name) { Some(at) } else { axis.iter().position(|n| n == name) };
        match slot {
            Some(slot) => row[slot] = number(value),
            None => placed = false,
        }
    }
    placed
}

impl Row {
    fn of(ran: Ran, axes: &Axes) -> Self {
        let mut row = Self {
            end_value: ran.end_value,
            total_trades: -1,
            metrics: vec![f64::NAN; axes.metrics.len()],
            months: vec![f64::NAN; axes.months.len()],
            years: vec![f64::NAN; axes.years.len()],
            off_axis: false,
            stats: ran.stats,
            secs: ran.secs,
        };
        if let Some((report, orders, _)) = &ran.report {
            row.total_trades = *orders as i64;
            let placed = [
                place(&axes.metrics, report.get("metrics"), &mut row.metrics),
                place(&axes.months, report.get("monthly_returns_pct"), &mut row.months),
                place(&axes.years, report.get("yearly_returns_pct"), &mut row.years),
            ];
            row.off_axis = placed.contains(&false);
        }
        row
    }
}

fn matrix<'py>(py: Python<'py>, flat: Vec<f64>, rows: usize, cols: usize) -> PyResult<&'py PyAny> {
    Array2::from_shape_vec((rows, cols), flat)
        .map(|array| array.into_pyarray(py).as_ref())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// A batch's screen as a dict of arrays, a row per cell.
fn screen_dict(py: Python<'_>, axes: &Axes, rows: &[Result<Row, Declined>]) -> PyResult<PyObject> {
    let n = rows.len();
    let (k, m, y) = (axes.metrics.len(), axes.months.len(), axes.years.len());
    let mut end_value = vec![f64::NAN; n];
    let mut total_trades = vec![-1i64; n];
    let (mut metrics, mut months, mut years) =
        (vec![f64::NAN; n * k], vec![f64::NAN; n * m], vec![f64::NAN; n * y]);
    let (mut stats, mut secs) = (vec![0i64; n * 3], vec![f64::NAN; n * 3]);
    let declined = PyDict::new(py);
    let off_axis = PyList::empty(py);
    for (at, row) in rows.iter().enumerate() {
        match row {
            Ok(row) => {
                end_value[at] = row.end_value;
                total_trades[at] = row.total_trades;
                // A cell without a report keeps NaN: the cells ahead of the
                // first reported one were reduced before the columns were known.
                if row.total_trades >= 0 {
                    metrics[at * k..(at + 1) * k].copy_from_slice(&row.metrics);
                    months[at * m..(at + 1) * m].copy_from_slice(&row.months);
                    years[at * y..(at + 1) * y].copy_from_slice(&row.years);
                }
                let (stepped, jumped, callbacks) = row.stats;
                stats[at * 3..(at + 1) * 3].copy_from_slice(&[
                    stepped as i64,
                    jumped as i64,
                    callbacks as i64,
                ]);
                secs[at * 3..(at + 1) * 3].copy_from_slice(&[row.secs.0, row.secs.1, row.secs.2]);
                if row.off_axis {
                    off_axis.append(at)?;
                }
            }
            Err(why) => declined.set_item(at, why.0)?,
        }
    }
    let out = PyDict::new(py);
    out.set_item("end_value", PyArray1::from_vec(py, end_value))?;
    out.set_item("total_trades", PyArray1::from_vec(py, total_trades))?;
    out.set_item("metrics", PyList::new(py, &axes.metrics))?;
    out.set_item("values", matrix(py, metrics, n, k)?)?;
    out.set_item("months", PyList::new(py, &axes.months))?;
    out.set_item("monthly", matrix(py, months, n, m)?)?;
    out.set_item("years", PyList::new(py, &axes.years))?;
    out.set_item("yearly", matrix(py, years, n, y)?)?;
    let stats =
        Array2::from_shape_vec((n, 3), stats).map_err(|e| PyValueError::new_err(e.to_string()))?;
    out.set_item("stats", stats.into_pyarray(py))?;
    out.set_item("secs", matrix(py, secs, n, 3)?)?;
    out.set_item("declined", declined)?;
    out.set_item("off_axis", off_axis)?;
    Ok(out.into_py(py))
}

/// Replay one cell on *frame*.
///
/// *jumps* false steps every bar, the reference the jumps are diffed
/// against. *log* false leaves the replay log out.
///
/// Returns a dict: `log` (the replay log as `replay_rules` returns it, None
/// without *log*), `report` (the canonical report as `canonical_report`
/// builds it, None where that declines), `total_trades`,
/// `closed_trade_pnls`, `end_value`, `day_ns` and `day_equity` (the daily
/// curve), `secs` (the replay's), `split` (the stream's setup and the
/// report's seconds) and `stats` (bars stepped, jumps, callbacks).
/// Raises [`LeanDeclined`] where the cell leaves the lean shape.
#[pyfunction]
#[pyo3(signature = (
    frame, gate, terms, quiet, venue, money, book, keys, entry, exit, flags,
    *, starting, timezone, offsets, offsets_until_ns, jumps = true, log = true, tree = None
))]
#[allow(clippy::too_many_arguments)]
pub fn lean_replay(
    py: Python<'_>,
    frame: PyRef<'_, PyLeanFrame>,
    gate: Option<PyReadonlyArray1<'_, bool>>,
    terms: TermsTuple,
    quiet: QuietTuple,
    venue: VenueTuple,
    money: PyRef<'_, PyMoneyModel>,
    book: BookTuple,
    keys: KeyArrays<'_>,
    entry: CrossTuple,
    exit: CrossTuple,
    flags: RuleFlags,
    starting: f64,
    timezone: &str,
    offsets: Vec<(i64, i64)>,
    offsets_until_ns: i64,
    jumps: bool,
    log: bool,
    tree: Option<TreeTuple>,
) -> PyResult<PyObject> {
    let gate = gate.as_ref().map(|gate| gate.as_slice()).transpose()?;
    let shared = Shared::new(
        &frame,
        gate,
        terms,
        quiet,
        venue,
        &money,
        book,
        starting,
        timezone,
        &offsets,
        offsets_until_ns,
        jumps,
    );
    let spec = Spec::of(&keys, entry, exit, tree.as_ref().map(self::tree).transpose()?, flags)?;
    let mut entries = None;
    let ran = run(&shared, &spec, log, |outcome| {
        if log {
            entries = Some(outcome.ledger.log(py));
        }
    })
    .map_err(declined)?;
    let out = to_dict(py, &ran, true)?;
    let dict: &PyDict = out.downcast(py)?;
    match entries {
        Some(entries) => dict.set_item("log", PyList::new(py, entries))?,
        None => dict.set_item("log", py.None())?,
    }
    Ok(out)
}

/// *work* on *pool*, or on rayon's own where None.
fn in_pool<R: Send>(pool: &Option<Arc<rayon::ThreadPool>>, work: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// The last pool a call asked for, with the process that built it.
static POOL: Mutex<Option<(u32, usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

/// A pool of *threads* threads, kept for the next call that asks for as many
/// in this process. A forked child builds its own: the pool it inherited
/// names threads that were never copied into it.
fn threads_pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    let pid = std::process::id();
    let mut kept = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((owner, size, pool)) = kept.as_ref() {
        if *owner == pid && *size == threads {
            return Ok(Arc::clone(pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    );
    *kept = Some((pid, threads, Arc::clone(&pool)));
    Ok(pool)
}

/// Replay *specs*, on this thread where *threads* is 1, else in parallel.
fn replay_all<T: Send>(
    pool: &Option<Arc<rayon::ThreadPool>>,
    threads: usize,
    specs: &[Spec<'_>],
    each: impl Fn(&Spec<'_>) -> T + Sync + Send,
) -> Vec<T> {
    if threads == 1 {
        return specs.iter().map(each).collect();
    }
    in_pool(pool, || specs.par_iter().map(each).collect())
}

/// Replay many cells on *frame*, on *threads* threads (1: this one; 0:
/// rayon's global pool) with the GIL released, under the same terms as
/// [`lean_replay`].
///
/// *keys* is the batch's `(key_values, key_init, admits)` and *cells* a
/// list of `(keys, admits, primary_admits, entry, exit, flags)` naming its
/// keys and admits by place (-1: none). *trees*, where given, holds per cell
/// None or its level rules as a [`TreeTuple`], which stand in for its
/// `entry` and `exit`.
///
/// Returns, in the cells' order, per cell the dict [`lean_replay`] returns
/// without `log` (and without the daily curve unless *curve*), or, where
/// the cell leaves the lean shape, a `str` naming why. With *screen*, a
/// dict of arrays instead, a row per cell: `end_value`, `total_trades` (-1
/// where unreported), `values` (the report's `metrics`, named by `metrics`,
/// NaN for None), `monthly` and `yearly` (its returns, named by `months`
/// and `years`), `stats`, `secs` (setup, replay, report), `declined`
/// (place: why) and `off_axis` (cells whose report names a column the
/// first reported cell's does not, that value left out).
#[pyfunction]
#[pyo3(signature = (
    frame, gate, terms, quiet, venue, money, book, keys, cells,
    *, starting, timezone, offsets, offsets_until_ns, threads = 0, screen = false,
    curve = false, timings = false, trees = None
))]
#[allow(clippy::too_many_arguments)]
pub fn lean_replay_many(
    py: Python<'_>,
    frame: PyRef<'_, PyLeanFrame>,
    gate: Option<PyReadonlyArray1<'_, bool>>,
    terms: TermsTuple,
    quiet: QuietTuple,
    venue: VenueTuple,
    money: PyRef<'_, PyMoneyModel>,
    book: BookTuple,
    keys: KeyPool<'_>,
    cells: Vec<CellTuple>,
    starting: f64,
    timezone: &str,
    offsets: Vec<(i64, i64)>,
    offsets_until_ns: i64,
    threads: usize,
    screen: bool,
    curve: bool,
    timings: bool,
    trees: Option<Vec<Option<TreeTuple>>>,
) -> PyResult<PyObject> {
    let entered = Instant::now();
    let gate = gate.as_ref().map(|gate| gate.as_slice()).transpose()?;
    let shared = Shared::new(
        &frame,
        gate,
        terms,
        quiet,
        venue,
        &money,
        book,
        starting,
        timezone,
        &offsets,
        offsets_until_ns,
        true,
    );
    let keys = Pool::of(&keys)?;
    if trees.as_ref().is_some_and(|trees| trees.len() != cells.len()) {
        return Err(PyValueError::new_err("trees and cells of different counts"));
    }
    let specs: Vec<Spec<'_>> = cells
        .iter()
        .enumerate()
        .map(|(at, cell)| {
            let tree = trees.as_ref().and_then(|trees| trees[at].as_ref()).map(tree).transpose()?;
            Spec::pooled(&keys, cell, tree)
        })
        .collect::<PyResult<_>>()?;
    let pool = (threads > 1).then(|| threads_pool(threads)).transpose()?;
    let parallel = Instant::now();
    let (out, converting): (PyObject, Instant) = if screen {
        // The first reported cell sets the columns; the rest are reduced to
        // rows on the threads that replay them.
        let (mut rows, axes, probed) = py.allow_threads(|| {
            let mut rows: Vec<Result<Row, Declined>> = Vec::with_capacity(specs.len());
            for spec in &specs {
                let ran = run(&shared, spec, false, |_| ());
                if let Ok(Ran { report: Some((report, _, _)), .. }) = &ran {
                    let axes = Axes::of(report);
                    rows.push(ran.map(|ran| Row::of(ran, &axes)));
                    let probed = rows.len();
                    return (rows, axes, probed);
                }
                rows.push(ran.map(|ran| Row::of(ran, &Axes::of(&Value::None))));
            }
            let probed = rows.len();
            (rows, Axes::of(&Value::None), probed)
        });
        rows.extend(py.allow_threads(|| {
            replay_all(&pool, threads, &specs[probed..], |spec| {
                run(&shared, spec, false, |_| ()).map(|ran| Row::of(ran, &axes))
            })
        }));
        let converting = Instant::now();
        (screen_dict(py, &axes, &rows)?, converting)
    } else {
        let results: Vec<Result<Ran, Declined>> = py.allow_threads(|| {
            replay_all(&pool, threads, &specs, |spec| run(&shared, spec, false, |_| ()))
        });
        let converting = Instant::now();
        let list = PyList::empty(py);
        for result in &results {
            match result {
                Ok(ran) => list.append(to_dict(py, ran, curve)?)?,
                Err(why) => list.append(why.0)?,
            }
        }
        (list.into_py(py), converting)
    };
    if timings {
        let secs = (
            parallel.duration_since(entered).as_secs_f64(),
            converting.duration_since(parallel).as_secs_f64(),
            converting.elapsed().as_secs_f64(),
        );
        return Ok((out, secs).into_py(py));
    }
    Ok(out.into_py(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn py_sum(terms: &[f64]) -> f64 {
        let mut sum = PySum::default();
        terms.iter().for_each(|&x| sum.add(x));
        sum.value()
    }

    #[test]
    fn sums_floats_as_python_does() {
        // CPython 3.12's own answers, where a plain left fold differs.
        assert_eq!(py_sum(&[0.1; 10]), 1.0);
        assert_eq!(py_sum(&[1e100, 1.0, -1e100, 1.0]), 2.0);
        assert_eq!(py_sum(&[-0.0]).to_bits(), 0.0f64.to_bits());
        assert_eq!(py_sum(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert!(py_sum(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }

    #[test]
    fn tallies_wins_and_the_rest_as_the_legacy_columns_do() {
        let pnls = [3.25, -1.5, 0.1, -0.0, 0.2, -2.75, 0.3, 1e-9];
        let (trades, wins, won, losses, lost) = closed_trade_tally(&pnls);
        assert_eq!((trades, wins, losses), (8, 5, 3));
        assert_eq!(won.to_bits(), 0x400e_cccc_ccef_28e5);
        assert_eq!(lost, -4.25);
        // A NaN counts as a trade in neither bucket.
        assert_eq!(closed_trade_tally(&[f64::NAN, 1.0]), (2, 1, 1.0, 0, 0.0));
    }
}
