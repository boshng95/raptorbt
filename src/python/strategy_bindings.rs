//! Python bindings for the per-bar strategy session.
//!
//! Exposes [`SingleRunner`] to Python as [`PyKernelSession`], the execution
//! core behind the class-based strategy contract: a Python driver loop feeds
//! bars and per-bar order inputs, and receives engine events to dispatch to
//! strategy hooks. Result accounting is shared with the array-based runners,
//! so both paths produce identical metrics for identical decisions.

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::types::{BacktestConfig, Direction, InstrumentConfig, StopConfig, TargetConfig};
use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce, TrailOffset};
use crate::portfolio::bar_index::{BarIndex, Extremes};
use crate::portfolio::kernel::{BarColumns, EngineEvent, KernelBar, StepInput};
use crate::portfolio::runner::SingleRunner;

use super::bar_index_cache::{self, Stream};
use super::bindings::{
    convert_result, convert_trade, PyBacktestConfig, PyBacktestResult, PyInstrumentConfig, PyTrade,
};
use super::fill_ledger::{BookTerms, Ledger, PyReplayLedger};
use super::instrument_bindings::PyInstrumentSpec;
use super::money::PyMoneyModel;
use super::replay_session::{replay, Money, Venue};
use super::rule_session::{PyCrossoverRuleSession, QuietBar, QuietParams, QuietTuple, RuleStepper};

/// `ctx.position is not None` and `ctx.net_position`, as the Python context
/// reads them: a left-to-right sum of signed sizes over the open positions.
pub(crate) fn held_position(runner: &SingleRunner) -> (bool, f64) {
    runner.kernel().net_position()
}

/// Fold the `units` / `size_frac` kwargs into a quantity spec.
///
/// Shared by the single-symbol and portfolio `modify_order` bindings so the
/// two report the same error for the same mistake.
pub(crate) fn parse_qty_spec(
    units: Option<f64>,
    size_frac: Option<f64>,
) -> PyResult<Option<QtySpec>> {
    match (units, size_frac) {
        (Some(_), Some(_)) => Err(PyValueError::new_err("pass units or size_frac, not both")),
        (Some(u), None) => Ok(Some(QtySpec::Units(u))),
        (None, Some(f)) => Ok(Some(QtySpec::CapitalFrac(f))),
        (None, None) => Ok(None),
    }
}

/// Parse the `account_type` / `leverage` kwargs shared by the single-symbol
/// and portfolio session constructors.
pub(crate) fn parse_account_mode(
    account_type: &str,
    leverage: f64,
) -> PyResult<crate::accounts::AccountMode> {
    match account_type {
        "cash" => Ok(crate::accounts::AccountMode::Cash),
        "margin" => {
            if leverage <= 0.0 {
                return Err(PyValueError::new_err("leverage must be > 0"));
            }
            Ok(crate::accounts::AccountMode::Margin { leverage })
        }
        other => Err(PyValueError::new_err(format!(
            "account_type must be 'cash' or 'margin', got {other:?}"
        ))),
    }
}

/// One observable outcome of a session step.
///
/// `kind` is `"entered"`, `"exited"`, `"entry_rejected"`, or one of the
/// order-lifecycle kinds (`"order_accepted"`, `"order_triggered"`,
/// `"order_filled"`, `"order_canceled"`, `"order_expired"`,
/// `"order_rejected"`); the optional fields are populated according to the
/// kind.
#[pyclass(name = "EngineEvent")]
#[derive(Debug, Clone)]
pub struct PyEngineEvent {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub idx: usize,
    /// Fill price, for `entered`/`order_filled` events.
    #[pyo3(get)]
    pub price: Option<f64>,
    /// Position size, for `entered`/`order_filled` events.
    #[pyo3(get)]
    pub size: Option<f64>,
    /// Trade direction (1 long, -1 short), for `entered` events.
    #[pyo3(get)]
    pub direction: Option<i32>,
    /// Completed trade, for `exited` events.
    #[pyo3(get)]
    pub trade: Option<PyTrade>,
    /// Refusal reason, for `entry_rejected`/`order_rejected` events.
    #[pyo3(get)]
    pub reject_reason: Option<String>,
    /// Engine order id, for `order_*` events.
    #[pyo3(get)]
    pub order_id: Option<u64>,
    /// Caller-supplied order identifier, for `order_*` events.
    #[pyo3(get)]
    pub client_order_id: Option<String>,
    /// Fees this fill alone paid, for `order_filled` events.
    #[pyo3(get)]
    pub commission: Option<f64>,
    /// Quantity the order still has outstanding after this fill; `0.0`
    /// means the fill completed it. For `order_filled` events.
    #[pyo3(get)]
    pub leaves: Option<f64>,
    /// PnL this fill alone realized, before its own commission; `0.0` for
    /// a fill that opened or grew a position. An account moves by
    /// `gross_realized - commission` on every fill, whichever it was.
    /// For `order_filled` events.
    #[pyo3(get)]
    pub gross_realized: Option<f64>,
}

#[pymethods]
impl PyEngineEvent {
    fn __repr__(&self) -> String {
        format!("EngineEvent(kind={}, idx={})", self.kind, self.idx)
    }
}

impl From<EngineEvent> for PyEngineEvent {
    fn from(event: EngineEvent) -> Self {
        let empty = Self {
            kind: String::new(),
            idx: 0,
            price: None,
            size: None,
            direction: None,
            trade: None,
            reject_reason: None,
            order_id: None,
            client_order_id: None,
            commission: None,
            leaves: None,
            gross_realized: None,
        };
        match event {
            EngineEvent::Entered { idx, price, size, direction } => Self {
                kind: "entered".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                direction: Some(direction as i32),
                ..empty
            },
            EngineEvent::Exited { idx, trade } => Self {
                kind: "exited".to_string(),
                idx,
                price: Some(trade.exit_price),
                size: Some(trade.size),
                direction: Some(trade.direction as i32),
                trade: Some(convert_trade(*trade)),
                ..empty
            },
            EngineEvent::EntryRejected { idx, reason } => Self {
                kind: "entry_rejected".to_string(),
                idx,
                // `as_str()`, not the Debug form. Both this event and
                // `OrderRejected` reach the same Python handler and are
                // counted in one dict keyed by this string, so a Debug
                // `"MaxPositions"` here and an `as_str()` `"max_positions"`
                // there split ONE cause into two buckets that each
                // undercount. `as_str()` is documented as the stable
                // identifier for reporting; this was the one site not using it.
                reject_reason: Some(reason.as_str().to_string()),
                ..empty
            },
            EngineEvent::OrderAccepted { idx, order_id, client_id } => Self {
                kind: "order_accepted".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderTriggered { idx, order_id, client_id } => Self {
                kind: "order_triggered".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderFilled {
                idx,
                order_id,
                client_id,
                price,
                size,
                commission,
                leaves,
                gross_realized,
            } => Self {
                kind: "order_filled".to_string(),
                idx,
                price: Some(price),
                size: Some(size),
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                commission: Some(commission),
                leaves: Some(leaves),
                gross_realized: Some(gross_realized),
                ..empty
            },
            EngineEvent::OrderCanceled { idx, order_id, client_id } => Self {
                kind: "order_canceled".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderExpired { idx, order_id, client_id } => Self {
                kind: "order_expired".to_string(),
                idx,
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::OrderRejected { idx, order_id, client_id, reason } => Self {
                kind: "order_rejected".to_string(),
                idx,
                reject_reason: Some(reason.to_string()),
                order_id: Some(order_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::AlgoStarted { idx, algo_id, client_id } => Self {
                kind: "algo_started".to_string(),
                idx,
                order_id: Some(algo_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::AlgoCompleted { idx, algo_id, client_id } => Self {
                kind: "algo_completed".to_string(),
                idx,
                order_id: Some(algo_id),
                client_order_id: Some(client_id),
                ..empty
            },
            EngineEvent::MarginCall { idx, equity, required } => Self {
                kind: "margin_call".to_string(),
                idx,
                // Reuse the price/size slots: current equity and the
                // maintenance requirement it fell below.
                price: Some(equity),
                size: Some(required),
                ..empty
            },
        }
    }
}

/// Read-only view of the session's open position.
#[pyclass(name = "PositionSnapshot")]
#[derive(Debug, Clone)]
pub struct PyPositionSnapshot {
    /// Ledger position id, unique within a session.
    #[pyo3(get)]
    pub position_id: u64,
    #[pyo3(get)]
    pub entry_idx: usize,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub size: f64,
    /// 1 for long, -1 for short.
    #[pyo3(get)]
    pub direction: i32,
    #[pyo3(get)]
    pub stop_price: Option<f64>,
    #[pyo3(get)]
    pub target_price: Option<f64>,
}

#[pymethods]
impl PyPositionSnapshot {
    fn __repr__(&self) -> String {
        format!(
            "PositionSnapshot(entry_idx={}, entry_price={:.2}, size={:.2}, direction={})",
            self.entry_idx, self.entry_price, self.size, self.direction
        )
    }
}

/// Per-bar simulation session for one instrument.
///
/// Drive it by calling [`PyKernelSession::step`] once per bar in ascending
/// order, then [`PyKernelSession::finish`] to obtain the standard backtest
/// result. Scalars cross the boundary per bar, so a Python driver loop pays
/// one FFI call per bar with no array allocation.
#[pyclass(name = "KernelSession")]
pub struct PyKernelSession {
    runner: Option<SingleRunner>,
}

#[pymethods]
impl PyKernelSession {
    #[new]
    #[pyo3(signature = (symbol="ASSET", direction=1, config=None, instrument_config=None, instrument=None,
                        oms_type="netting", account_type="cash", leverage=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbol: &str,
        direction: i32,
        config: Option<&PyBacktestConfig>,
        instrument_config: Option<&PyInstrumentConfig>,
        instrument: Option<&PyInstrumentSpec>,
        oms_type: &str,
        account_type: &str,
        leverage: f64,
    ) -> PyResult<Self> {
        let direction = match direction {
            1 => Direction::Long,
            -1 => Direction::Short,
            other => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 1 (long) or -1 (short), got {other}"
                )))
            }
        };

        let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
        let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);

        let policy = match oms_type {
            "netting" => crate::portfolio::ledger::PositionPolicy::Net,
            "netting-averaging" => crate::portfolio::ledger::PositionPolicy::NetAveraging,
            "hedging" => crate::portfolio::ledger::PositionPolicy::Independent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "oms_type must be 'netting', 'netting-averaging' or 'hedging', got {other:?}"
                )))
            }
        };
        let account = parse_account_mode(account_type, leverage)?;

        let mut runner =
            SingleRunner::from_config(rust_config, symbol.to_string(), direction, inst.as_ref());
        if let Some(spec) = instrument {
            let spec = crate::instruments::InstrumentSpec::from(spec);
            if !spec.kind.tradable() {
                return Err(PyValueError::new_err(format!(
                    "instrument {:?} is not tradable (kind={:?})",
                    spec.symbol, spec.kind
                )));
            }
            runner = runner.with_instrument(spec);
        }
        runner = runner.with_position_policy(policy).with_account_mode(account);

        Ok(Self { runner: Some(runner) })
    }

    /// Advance the session by one bar.
    ///
    /// `entry`/`exit` carry the strategy's order intents for this bar;
    /// `stop_price`/`target_price` optionally pin explicit exit levels for an
    /// entry opened on this bar, overriding the configured stop/target models.
    /// `clock_only` marks a bar the driver made up to advance the venue clock
    /// through a data gap: it matches like any bar but, with no price traded,
    /// opens no instant of the equity curve (see `SingleRunner::step_clock`).
    #[pyo3(signature = (
        idx, timestamp, open, high, low, close, volume,
        entry=false, exit=false, atr=0.0, size_mult=None,
        stop_price=None, target_price=None, clock_only=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        idx: usize,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        entry: bool,
        exit: bool,
        atr: f64,
        size_mult: Option<f64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
        clock_only: bool,
    ) -> PyResult<Vec<PyEngineEvent>> {
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))?;

        let bar = KernelBar { timestamp, open, high, low, close, volume };
        let input = StepInput {
            entry,
            exit,
            atr,
            size_mult,
            stop_price_override: stop_price,
            target_price_override: target_price,
            // The class-strategy contract opens through the order API, whose
            // own side names the direction; the signal field stays unused.
            entry_direction: None,
        };

        let events = if clock_only {
            runner.step_clock(idx, &bar, input)
        } else {
            runner.step(idx, &bar, input)
        };
        Ok(events.into_iter().map(PyEngineEvent::from).collect())
    }

    /// Step consecutive bars with no Python handlers, stopping at the first
    /// engine event so the driver can dispatch it at its original bar index.
    #[pyo3(signature = (start, end, timestamps, open, high, low, close, volume, atr=None))]
    #[allow(clippy::too_many_arguments)]
    fn step_until_event(
        &mut self,
        py: Python<'_>,
        start: usize,
        end: usize,
        timestamps: PyReadonlyArray1<'_, i64>,
        open: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        low: PyReadonlyArray1<'_, f64>,
        close: PyReadonlyArray1<'_, f64>,
        volume: PyReadonlyArray1<'_, f64>,
        atr: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<(usize, Vec<PyEngineEvent>)> {
        let bars =
            BarArrays::new(py, &timestamps, &open, &high, &low, &close, &volume, atr.as_ref())?;
        if start >= end || end > bars.len() {
            return Err(PyValueError::new_err("invalid idle bar range or array lengths"));
        }
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))?;
        if start != runner.bars_seen() {
            return Err(PyValueError::new_err("idle range must start at the next bar"));
        }
        let (last, events) = step_until_event(runner, &bars, start, end);
        Ok((last, events.into_iter().map(PyEngineEvent::from).collect()))
    }

    /// Step bars for a crossover rule set the kernel can run on its own.
    ///
    /// Like [`Self::step_until_event`], but a bar in `mask` -- one the
    /// strategy's callback runs on -- is taken by `rules` on the strategy's
    /// behalf when the strategy would have done nothing on it (see
    /// `CrossoverRuleSession`), instead of ending the range. Returns
    /// `(kind, idx, events)`:
    ///
    /// - `0`: bar `idx` was stepped and produced `events`, to dispatch now;
    /// - `1`: every bar through `idx == end - 1` was stepped, with no events;
    /// - `2`: bar `idx` is the strategy's to run; it has not been stepped and
    ///   `rules` stands as it did before that bar.
    ///
    /// `pending_target` and `halted` are the strategy's own state, which no
    /// quiet bar changes. `quiet` is the halt bound a flat bar is checked
    /// against: `(currency_precision, fraction, cap, units, markup, slack,
    /// fee, need_at_highest)`.
    #[pyo3(signature = (
        start, end, timestamps, open, high, low, close, volume, atr, mask, rules,
        halted, pending_target, quiet,
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn step_rules_until(
        &mut self,
        py: Python<'_>,
        start: usize,
        end: usize,
        timestamps: PyReadonlyArray1<'_, i64>,
        open: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        low: PyReadonlyArray1<'_, f64>,
        close: PyReadonlyArray1<'_, f64>,
        volume: PyReadonlyArray1<'_, f64>,
        atr: Option<PyReadonlyArray1<'_, f64>>,
        mask: PyReadonlyArray1<'_, bool>,
        mut rules: PyRefMut<'_, PyCrossoverRuleSession>,
        halted: bool,
        pending_target: Option<f64>,
        quiet: QuietTuple,
    ) -> PyResult<(u8, usize, Vec<PyEngineEvent>)> {
        let bars =
            BarArrays::new(py, &timestamps, &open, &high, &low, &close, &volume, atr.as_ref())?;
        let m = mask.as_slice().map_err(|_| PyValueError::new_err("mask must be contiguous"))?;
        if start >= end || end > bars.len() || m.len() != bars.len() {
            return Err(PyValueError::new_err("invalid rule bar range or array lengths"));
        }
        let quiet = QuietParams::from(quiet);
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))?;
        if start != runner.bars_seen() {
            return Err(PyValueError::new_err("rule range must start at the next bar"));
        }
        let mut stepper = rules.split(py)?;
        let (kind, last, events) = step_rules_until(
            runner,
            &bars,
            m,
            &mut stepper,
            halted,
            pending_target,
            &quiet,
            start,
            end,
        );
        Ok((kind, last, events.into_iter().map(PyEngineEvent::from).collect()))
    }

    /// Run every bar of this fresh session as a crossover replay would.
    ///
    /// `run_strategy_backtest` with a strategy whose `replay_rules` hook
    /// names its rule session, venue terms and money functions; see
    /// `replay_session`. Returns the ordered log of what the strategy booked,
    /// or raises `ReplayDeclined` where it would take a branch the replay
    /// does not transcribe, leaving the session spent. Given `book`, the
    /// terms of the strategy's own book, it returns that log booked instead,
    /// as a `ReplayLedger`.
    ///
    /// `venue` is `(bar_ns, venue_ioc, entry_market, exit_market,
    /// market_markup, size_increment, minimum_quantity, minimum_notional,
    /// lowest_close, order_tag, first_order_seq, max_actions_per_timestamp)`.
    /// `money`, a `MoneyModel` of the same functions, answers them without
    /// calling back wherever it can. `book` is `(balance,
    /// currency_precision, size_increment, size_precision, price_precision,
    /// currency)`.
    #[pyo3(signature = (
        timestamps, open, high, low, close, volume, atr, mask, rules, quiet, venue,
        units, snap, affordable, exhausted, gate=None, money=None, book=None,
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn replay_rules(
        &mut self,
        py: Python<'_>,
        timestamps: PyReadonlyArray1<'_, i64>,
        open: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        low: PyReadonlyArray1<'_, f64>,
        close: PyReadonlyArray1<'_, f64>,
        volume: PyReadonlyArray1<'_, f64>,
        atr: Option<PyReadonlyArray1<'_, f64>>,
        mask: PyReadonlyArray1<'_, bool>,
        mut rules: PyRefMut<'_, PyCrossoverRuleSession>,
        quiet: QuietTuple,
        venue: (i64, bool, bool, bool, f64, f64, f64, f64, f64, String, u64, usize),
        units: &PyAny,
        snap: &PyAny,
        affordable: &PyAny,
        exhausted: &PyAny,
        gate: Option<PyReadonlyArray1<'_, bool>>,
        money: Option<PyRef<'_, PyMoneyModel>>,
        book: Option<(f64, u32, f64, usize, usize, String)>,
    ) -> PyResult<PyObject> {
        let bars =
            BarArrays::new(py, &timestamps, &open, &high, &low, &close, &volume, atr.as_ref())?;
        let m = mask.as_slice().map_err(|_| PyValueError::new_err("mask must be contiguous"))?;
        let g = gate
            .as_ref()
            .map(|array| array.as_slice())
            .transpose()
            .map_err(|_| PyValueError::new_err("gate must be contiguous"))?;
        if bars.len() == 0 || m.len() != bars.len() || g.is_some_and(|g| g.len() != bars.len()) {
            return Err(PyValueError::new_err("invalid replay array lengths"));
        }
        let runner = self
            .runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))?;
        if runner.bars_seen() != 0 {
            return Err(PyValueError::new_err("a replay must start on a fresh session"));
        }
        let venue = Venue {
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
        };
        let money = Money { units, snap, affordable, exhausted, model: money.as_deref() };
        let stepper = rules.split(py)?.handing_back();
        let booked = book.is_some();
        let ledger = Ledger::new(book.map(
            |(
                balance,
                currency_precision,
                size_increment,
                size_precision,
                price_precision,
                currency,
            )| {
                BookTerms {
                    balance,
                    currency_precision,
                    size_increment,
                    size_precision,
                    price_precision,
                    currency,
                }
            },
        ));
        let ledger =
            replay(runner, bars, m, g, stepper, QuietParams::from(quiet), venue, money, ledger)?;
        if booked {
            Ok(Py::new(py, PyReplayLedger::new(ledger))?.into_py(py))
        } else {
            Ok(ledger.log(py).into_py(py))
        }
    }

    /// Settle resting orders off-schedule, at `ts_now`.
    ///
    /// A venue walks its book every time it drains a batch of commands, so a
    /// driver that steps once per bar under-fills an order the strategy
    /// placed on hearing that bar's own fills. Call this after routing such a
    /// batch, and again while the fills it reports keep producing commands.
    fn walk_book(&mut self, ts_now: i64) -> PyResult<Vec<PyEngineEvent>> {
        Ok(self.runner_mut()?.walk_book(ts_now).into_iter().map(PyEngineEvent::from).collect())
    }

    /// Submit an order.
    ///
    /// `side` is `"buy"`/`"sell"`; `kind` is `"market"`, `"limit"`,
    /// `"stop_market"`, or `"stop_limit"`; `tif` is `"gtc"`, `"day"`,
    /// `"gtd"` (requires `expire_ns`), `"ioc"`, or `"fok"`. Exactly one of
    /// `units`/`size_frac` sizes an opening order; omit both to close the
    /// full position with a closing-side order. Returns the engine order id.
    #[pyo3(signature = (
        side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="gtc", expire_ns=None, stop_price=None, target_price=None,
        offset=None, offset_kind="price", limit_offset=0.0,
        post_only=false, reduce_only=false, arrives_before_bar=false,
        parent_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_order(
        &mut self,
        side: &str,
        kind: &str,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
        tif: &str,
        expire_ns: Option<i64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
        offset: Option<f64>,
        offset_kind: &str,
        limit_offset: f64,
        post_only: bool,
        reduce_only: bool,
        arrives_before_bar: bool,
        parent_id: Option<u64>,
    ) -> PyResult<u64> {
        submit_order_on(
            self.runner_mut()?.kernel_mut(),
            side,
            kind,
            submitted_idx,
            submitted_ts,
            client_id,
            units,
            size_frac,
            limit_price,
            trigger_price,
            tif,
            expire_ns,
            stop_price,
            target_price,
            offset,
            offset_kind,
            limit_offset,
            post_only,
            reduce_only,
            arrives_before_bar,
            parent_id,
        )
    }

    /// Put working orders in one one-cancels-other group.
    fn link_oco(&mut self, order_ids: Vec<u64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().link_oco(&order_ids);
        Ok(())
    }

    /// Cancel a working order; `false` for unknown/finished ids.
    fn cancel_order(&mut self, idx: usize, order_id: u64) -> PyResult<bool> {
        Ok(self.runner_mut()?.kernel_mut().cancel_order(idx, order_id))
    }

    /// Cancel every working order, returning canceled ids.
    fn cancel_all_orders(&mut self, idx: usize) -> PyResult<Vec<u64>> {
        Ok(self.runner_mut()?.kernel_mut().cancel_all_orders(idx))
    }

    /// Set the underlying price used to settle options at expiry.
    ///
    /// An option's own bars carry the option's price, so intrinsic value
    /// needs the underlying from somewhere else. Without it, contracts
    /// settle at their own close.
    #[pyo3(signature = (price=None))]
    fn set_underlying_price(&mut self, price: Option<f64>) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().set_underlying_price(price);
        Ok(())
    }

    /// Register a TWAP schedule; returns its id.
    #[pyo3(signature = (units, side, slices, interval_ns, submitted_idx, submitted_ts, client_id, reduce_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn submit_twap(
        &mut self,
        units: f64,
        side: &str,
        slices: u32,
        interval_ns: i64,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        reduce_only: bool,
    ) -> PyResult<u64> {
        let side = match side {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            other => {
                return Err(PyValueError::new_err(format!(
                    "side must be 'buy' or 'sell', got {other:?}"
                )))
            }
        };
        self.runner_mut()?
            .kernel_mut()
            .submit_algo(
                side,
                QtySpec::Units(units),
                OrderKind::Market,
                TimeInForce::Gtc,
                client_id.to_string(),
                crate::execution::algos::ExecAlgorithm::Twap { slices, interval_ns },
                reduce_only,
                submitted_ts,
                submitted_idx,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Stop a schedule and cancel the slices it has working.
    fn cancel_twap(&mut self, algo_id: u64, idx: usize) -> PyResult<bool> {
        Ok(self.runner_mut()?.kernel_mut().cancel_algo(algo_id, idx))
    }

    /// Replace a working order's prices and/or quantity.
    #[pyo3(signature = (order_id, units=None, size_frac=None, limit_price=None, trigger_price=None))]
    fn modify_order(
        &mut self,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<bool> {
        let qty = parse_qty_spec(units, size_frac)?;
        Ok(self.runner_mut()?.kernel_mut().modify_order(order_id, qty, limit_price, trigger_price))
    }

    /// Ids of all non-terminal orders, in submission order.
    fn open_order_ids(&self) -> PyResult<Vec<u64>> {
        Ok(self.runner_ref()?.kernel().open_orders().iter().map(|o| o.id).collect())
    }

    /// Overwrite a position's stop price; no-op when flat.
    ///
    /// Without `position_id` targets the earliest open position.
    #[pyo3(signature = (price, position_id=None))]
    fn set_stop_price(&mut self, price: Option<f64>, position_id: Option<u64>) -> PyResult<()> {
        let kernel = self.runner_mut()?.kernel_mut();
        match position_id {
            Some(id) => {
                if !kernel.set_stop_price_for(id, price) {
                    return Err(PyValueError::new_err(format!("unknown position id {id}")));
                }
            }
            None => kernel.set_stop_price(price),
        }
        Ok(())
    }

    /// Overwrite a position's target price; no-op when flat.
    #[pyo3(signature = (price, position_id=None))]
    fn set_target_price(&mut self, price: Option<f64>, position_id: Option<u64>) -> PyResult<()> {
        let kernel = self.runner_mut()?.kernel_mut();
        match position_id {
            Some(id) => {
                if !kernel.set_target_price_for(id, price) {
                    return Err(PyValueError::new_err(format!("unknown position id {id}")));
                }
            }
            None => kernel.set_target_price(price),
        }
        Ok(())
    }

    /// Request a close of a specific position on the next step.
    fn request_close(&mut self, position_id: u64) -> PyResult<()> {
        self.runner_mut()?.kernel_mut().request_close(position_id);
        Ok(())
    }

    /// Read-only views of every open position, in opening order.
    fn positions(&self) -> PyResult<Vec<PyPositionSnapshot>> {
        Ok(self
            .runner_ref()?
            .kernel()
            .position_snapshots()
            .into_iter()
            .map(convert_snapshot)
            .collect())
    }

    /// Cash not locked as margin (margin mode); all cash otherwise.
    fn free_capital(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.kernel().free_capital())
    }

    /// Mark-to-market equity after the most recent step.
    fn equity(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.equity())
    }

    /// Current uninvested cash.
    fn cash(&self) -> PyResult<f64> {
        Ok(self.runner_ref()?.cash())
    }

    /// Whether a position is currently open.
    fn is_in_position(&self) -> PyResult<bool> {
        Ok(self.runner_ref()?.is_in_position())
    }

    /// Read-only view of the earliest open position, or `None` when flat.
    fn position(&self) -> PyResult<Option<PyPositionSnapshot>> {
        Ok(self.runner_ref()?.kernel().position_snapshot().map(convert_snapshot))
    }

    /// Force-close any open position and compute final metrics.
    ///
    /// Consumes the session; further calls raise.
    fn finish(&mut self) -> PyResult<PyBacktestResult> {
        let runner = self
            .runner
            .take()
            .ok_or_else(|| PyValueError::new_err("session is already finished"))?;
        Ok(convert_result(runner.finish()))
    }
}

pub(crate) fn convert_snapshot(
    p: crate::portfolio::kernel::PositionSnapshot,
) -> PyPositionSnapshot {
    PyPositionSnapshot {
        position_id: p.position_id,
        entry_idx: p.entry_idx,
        entry_price: p.entry_price,
        size: p.size,
        direction: p.direction as i32,
        stop_price: p.stop_price,
        target_price: p.target_price,
    }
}

impl PyKernelSession {
    fn runner_ref(&self) -> PyResult<&SingleRunner> {
        self.runner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }

    pub(crate) fn runner_mut(&mut self) -> PyResult<&mut SingleRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }
}

/// Resolve the ATR period a backtest would use for stop/target computation.
///
/// Mirrors the batch engine's resolution: per-instrument stop/target override
/// the global config; the stop's period wins over the target's; `None` when
/// neither is ATR-based. The Python strategy runner uses this to precompute
/// the same ATR series the array path would.
#[pyfunction]
#[pyo3(signature = (config=None, instrument_config=None))]
pub fn resolve_atr_period(
    config: Option<&PyBacktestConfig>,
    instrument_config: Option<&PyInstrumentConfig>,
) -> Option<usize> {
    let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
    let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);

    let effective_stop =
        inst.as_ref().and_then(|ic| ic.stop.as_ref()).copied().unwrap_or(rust_config.stop);
    let effective_target =
        inst.as_ref().and_then(|ic| ic.target.as_ref()).copied().unwrap_or(rust_config.target);

    match (effective_stop, effective_target) {
        (StopConfig::Atr { period, .. }, _) => Some(period),
        (_, TargetConfig::Atr { period, .. }) => Some(period),
        _ => None,
    }
}

/// One run's primary bars, borrowed for the length of a stepping call.
pub(crate) struct BarArrays<'a> {
    pub ts: &'a [i64],
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
    pub atr: Option<&'a [f64]>,
    /// Their summaries, shared by every call stepping these same arrays.
    pub index: Option<Arc<BarIndex>>,
}

impl<'a> BarArrays<'a> {
    /// Borrow the driver's arrays, which must be contiguous and of one length.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        py: Python<'_>,
        timestamps: &'a PyReadonlyArray1<'_, i64>,
        open: &'a PyReadonlyArray1<'_, f64>,
        high: &'a PyReadonlyArray1<'_, f64>,
        low: &'a PyReadonlyArray1<'_, f64>,
        close: &'a PyReadonlyArray1<'_, f64>,
        volume: &'a PyReadonlyArray1<'_, f64>,
        atr: Option<&'a PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let mut bars = Self {
            ts: timestamps
                .as_slice()
                .map_err(|_| PyValueError::new_err("timestamps must be contiguous"))?,
            open: open.as_slice().map_err(|_| PyValueError::new_err("open must be contiguous"))?,
            high: high.as_slice().map_err(|_| PyValueError::new_err("high must be contiguous"))?,
            low: low.as_slice().map_err(|_| PyValueError::new_err("low must be contiguous"))?,
            close: close
                .as_slice()
                .map_err(|_| PyValueError::new_err("close must be contiguous"))?,
            volume: volume
                .as_slice()
                .map_err(|_| PyValueError::new_err("volume must be contiguous"))?,
            atr: atr
                .map(|array| array.as_slice())
                .transpose()
                .map_err(|_| PyValueError::new_err("atr must be contiguous"))?,
            index: None,
        };
        let n = bars.ts.len();
        if [bars.open.len(), bars.high.len(), bars.low.len(), bars.close.len(), bars.volume.len()]
            .iter()
            .any(|&length| length != n)
            || bars.atr.is_some_and(|values| values.len() != n)
        {
            return Err(PyValueError::new_err("bar arrays must all have one length"));
        }
        let stream = Stream {
            objects: [
                timestamps.as_ref(),
                high.as_ref(),
                low.as_ref(),
                close.as_ref(),
                volume.as_ref(),
            ],
            timestamp: bars.ts,
            high: bars.high,
            low: bars.low,
            close: bars.close,
            volume: bars.volume,
        };
        bars.index = bar_index_cache::shared(py, &stream)?;
        Ok(bars)
    }

    pub(crate) fn len(&self) -> usize {
        self.ts.len()
    }

    /// The same arrays, for the kernel's own runs of bars.
    pub(crate) fn columns(&self) -> BarColumns<'_> {
        BarColumns {
            timestamp: self.ts,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            atr: self.atr,
            index: self.index.as_deref(),
        }
    }

    /// Bar `idx`, and the inputs the Python driver steps it with.
    pub(crate) fn bar(&self, idx: usize) -> (KernelBar, StepInput) {
        let bar = KernelBar {
            timestamp: self.ts[idx],
            open: self.open[idx],
            high: self.high[idx],
            low: self.low[idx],
            close: self.close[idx],
            volume: self.volume[idx],
        };
        let input =
            StepInput { atr: self.atr.map_or(0.0, |values| values[idx]), ..StepInput::default() };
        (bar, input)
    }
}

/// Step `start..end` until a bar produces events; see
/// [`PyKernelSession::step_until_event`].
pub(crate) fn step_until_event(
    runner: &mut SingleRunner,
    bars: &BarArrays<'_>,
    start: usize,
    end: usize,
) -> (usize, Vec<EngineEvent>) {
    let columns = bars.columns();
    let mut idx = start;
    while idx < end {
        idx = runner.step_idle_run(&columns, idx, end);
        if idx == end {
            break;
        }
        let (bar, input) = bars.bar(idx);
        let events = runner.step(idx, &bar, input);
        if !events.is_empty() {
            return (idx, events);
        }
        idx += 1;
    }
    (end - 1, Vec::new())
}

/// The bars ahead of a callback whose closes the margin bound is first tried
/// over, in [`step_rules_until`]: doubled each time the bound holds, halved
/// each time it does not.
const MARGIN_SPAN: usize = 4096;

/// The first callback bar in `mask[from..end]`, sixty-four flags to a read.
fn next_due(mask: &[bool], from: usize, end: usize) -> Option<usize> {
    let flags = &mask[from..end];
    // SAFETY: a `bool` is one byte holding 0 or 1, each a valid `u8`.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(flags.as_ptr().cast(), flags.len()) };
    let (chunks, remainder) = bytes.as_chunks::<64>();
    let mut offset = from;
    for chunk in chunks {
        // One vector OR across the chunk; most of a sparse mask is zero.
        if chunk.iter().fold(0, |any, &flag| any | flag) != 0 {
            return chunk.iter().position(|&flag| flag != 0).map(|at| offset + at);
        }
        offset += 64;
    }
    remainder.iter().position(|&flag| flag != 0).map(|at| offset + at)
}

/// Step `start..end`, taking quiet callback bars for the strategy; see
/// [`PyKernelSession::step_rules_until`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_rules_until(
    runner: &mut SingleRunner,
    bars: &BarArrays<'_>,
    mask: &[bool],
    stepper: &mut RuleStepper<'_>,
    halted: bool,
    pending_target: Option<f64>,
    quiet: &QuietParams,
    start: usize,
    end: usize,
) -> (u8, usize, Vec<EngineEvent>) {
    let columns = bars.columns();
    let kernel = runner.kernel();
    if start < end && kernel.idle_state() && kernel.idle_at_every_bar() {
        // Every bar from here steps idle, and an idle bar moves neither cash
        // nor the position: each callback reads the account as it stands
        // now, and its rules need no bar stepped to be asked. So the
        // callbacks are asked first, in order, up to the first that would
        // act, and the bars before it are stepped in one run -- the run
        // `step_until_event` takes across bars no callback interrupts.
        let (has_position, net_position) = held_position(runner);
        let cash = runner.cash();
        let mut stop = end;
        let mut at = start;
        while let Some(idx) = next_due(mask, at, end) {
            let bar = QuietBar {
                idx,
                now: bars.ts[idx],
                close: bars.close[idx],
                has_position,
                net_position,
                cash,
                halted,
                pending: pending_target,
            };
            if !stepper.quiet_bar(&bar, quiet) {
                stop = idx;
                break;
            }
            at = idx + 1;
        }
        let reached = runner.step_idle_run(&columns, start, stop);
        debug_assert_eq!(reached, stop, "an idle kernel stopped standing idle");
        return if stop < end { (2, stop, Vec::new()) } else { (1, end - 1, Vec::new()) };
    }
    let mut idx = start;
    if let Some(index) =
        columns.index.filter(|_| start < end && kernel.idle_state() && kernel.never_expires())
    {
        // The same, where a close could call margin: a callback is asked
        // once the bars before it are bound to have called none, the bound
        // `idle_run` steps a run on. A stretch the bound does not settle
        // ends at the callback it failed at, unasked, and the next starts
        // there; a bar the kernel then does not step idle leaves the rest
        // to the walk below.
        let (has_position, net_position) = held_position(runner);
        let cash = runner.cash();
        // Whether a close can call margin at all; when none can, the bound
        // is never read and its closes are not folded.
        let reads_closes = runner.kernel().margin_reads_closes();
        let mut from = start;
        loop {
            let mut extremes = Extremes::EMPTY;
            let mut covered = from;
            // Every callback up to this bar is bound to call no margin: the
            // closes of a run that reaches past it do not, and the bound
            // over fewer closes is the looser one.
            let mut cleared = from;
            let mut span = MARGIN_SPAN;
            let mut at = from;
            let mut stop = end;
            let mut acts = false;
            while let Some(callback) = next_due(mask, at, end) {
                if reads_closes && callback > cleared {
                    let ahead = end.min(callback.saturating_add(span));
                    let mut wider = extremes;
                    wider.merge(&index.extremes(bars.close, bars.high, bars.low, covered, ahead));
                    if runner.kernel().margin_clears(&wider) {
                        (extremes, covered, cleared) = (wider, ahead, ahead);
                        span = span.saturating_mul(2);
                    } else {
                        // Settled at this callback alone, as `idle_run` would.
                        extremes.merge(
                            &index.extremes(bars.close, bars.high, bars.low, covered, callback),
                        );
                        covered = callback;
                        span = (span / 2).max(1);
                        if !runner.kernel().margin_clears(&extremes) {
                            stop = callback;
                            break;
                        }
                    }
                }
                let bar = QuietBar {
                    idx: callback,
                    now: bars.ts[callback],
                    close: bars.close[callback],
                    has_position,
                    net_position,
                    cash,
                    halted,
                    pending: pending_target,
                };
                if !stepper.quiet_bar(&bar, quiet) {
                    stop = callback;
                    acts = true;
                    break;
                }
                at = callback + 1;
            }
            let reached = runner.step_idle_run(&columns, from, stop);
            debug_assert!(!acts || reached == stop, "the bound stepped short of a callback asked");
            if reached < stop {
                // Every callback asked sits at or before this bar.
                let (bar, input) = bars.bar(reached);
                let events = runner.step(reached, &bar, input);
                if !events.is_empty() {
                    return (0, reached, events);
                }
                idx = reached + 1;
                break;
            }
            if acts {
                return (2, stop, Vec::new());
            }
            if stop == end {
                return (1, end - 1, Vec::new());
            }
            if stop == from {
                idx = from;
                break;
            }
            from = stop;
        }
    }
    // The first callback bar after `idx`, once found.
    let mut callback = start;
    while idx < end {
        if mask[idx] {
            // The callback runs before the bar steps, against the account
            // the previous bar left.
            let (has_position, net_position) = held_position(runner);
            let bar = QuietBar {
                idx,
                now: bars.ts[idx],
                close: bars.close[idx],
                has_position,
                net_position,
                cash: runner.cash(),
                halted,
                pending: pending_target,
            };
            if !stepper.quiet_bar(&bar, quiet) {
                return (2, idx, Vec::new());
            }
        }
        // This bar and those up to the next callback are the kernel's.
        if callback <= idx {
            callback = next_due(mask, idx + 1, end).unwrap_or(end);
        }
        let reached = runner.step_idle_run(&columns, idx, callback);
        if reached == callback {
            idx = callback;
            continue;
        }
        let (bar, input) = bars.bar(reached);
        let events = runner.step(reached, &bar, input);
        if !events.is_empty() {
            return (0, reached, events);
        }
        idx = reached + 1;
    }
    (1, end - 1, Vec::new())
}

/// Parse Python order arguments and submit onto a kernel — shared by the
/// single-instrument session and the portfolio session.
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_order_on(
    kernel: &mut crate::portfolio::kernel::EngineKernel,
    side: &str,
    kind: &str,
    submitted_idx: usize,
    submitted_ts: i64,
    client_id: &str,
    units: Option<f64>,
    size_frac: Option<f64>,
    limit_price: Option<f64>,
    trigger_price: Option<f64>,
    tif: &str,
    expire_ns: Option<i64>,
    stop_price: Option<f64>,
    target_price: Option<f64>,
    offset: Option<f64>,
    offset_kind: &str,
    limit_offset: f64,
    post_only: bool,
    reduce_only: bool,
    arrives_before_bar: bool,
    parent_id: Option<u64>,
) -> PyResult<u64> {
    let side = match side {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        other => return Err(PyValueError::new_err(format!("unknown side {other:?}"))),
    };
    let qty = match (units, size_frac) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err("pass units or size_frac, not both"))
        }
        (Some(u), None) if u <= 0.0 => return Err(PyValueError::new_err("units must be > 0")),
        (Some(u), None) => QtySpec::Units(u),
        (None, Some(f)) if f <= 0.0 || f > 1.0 => {
            return Err(PyValueError::new_err("size_frac must be in (0, 1]"))
        }
        (None, Some(f)) => QtySpec::CapitalFrac(f),
        (None, None) => QtySpec::FullPosition,
    };
    let need_limit =
        || limit_price.ok_or_else(|| PyValueError::new_err(format!("{kind} needs limit_price")));
    let need_trigger = || {
        trigger_price.ok_or_else(|| PyValueError::new_err(format!("{kind} needs trigger_price")))
    };
    let trail_offset = |kernel: &crate::portfolio::kernel::EngineKernel| -> PyResult<TrailOffset> {
        let raw = offset.ok_or_else(|| PyValueError::new_err(format!("{kind} needs offset")))?;
        if raw <= 0.0 {
            return Err(PyValueError::new_err("offset must be > 0"));
        }
        match offset_kind {
            "price" => Ok(TrailOffset::Price(raw)),
            "bps" => Ok(TrailOffset::Bps(raw)),
            "ticks" => {
                let tick = kernel.price_increment();
                if tick <= 0.0 {
                    return Err(PyValueError::new_err(
                        "offset_kind='ticks' needs an instrument with price_increment",
                    ));
                }
                Ok(TrailOffset::Price(raw * tick))
            }
            other => Err(PyValueError::new_err(format!(
                "offset_kind must be 'price', 'bps', or 'ticks', got {other:?}"
            ))),
        }
    };

    let kernel_ref = &*kernel;
    let kind_parsed = match kind {
        "market" => OrderKind::Market,
        "limit" => OrderKind::Limit { price: need_limit()? },
        "stop_market" => OrderKind::StopMarket { trigger: need_trigger()? },
        "stop_limit" => OrderKind::StopLimit { trigger: need_trigger()?, price: need_limit()? },
        "market_if_touched" => OrderKind::MarketIfTouched { trigger: need_trigger()? },
        "limit_if_touched" => {
            OrderKind::LimitIfTouched { trigger: need_trigger()?, price: need_limit()? }
        }
        "market_to_limit" => OrderKind::MarketToLimit,
        "trailing_stop_market" => {
            OrderKind::TrailingStopMarket { offset: trail_offset(kernel_ref)? }
        }
        "trailing_stop_limit" => {
            OrderKind::TrailingStopLimit { offset: trail_offset(kernel_ref)?, limit_offset }
        }
        other => return Err(PyValueError::new_err(format!("unknown order kind {other:?}"))),
    };
    let tif = match tif {
        "gtc" => TimeInForce::Gtc,
        "day" => TimeInForce::Day,
        "gtd" => TimeInForce::Gtd {
            expire_ns: expire_ns
                .ok_or_else(|| PyValueError::new_err("gtd orders need expire_ns"))?,
        },
        "ioc" => TimeInForce::Ioc,
        "fok" => TimeInForce::Fok,
        "at_open" => TimeInForce::AtOpen,
        "at_close" => TimeInForce::AtClose,
        other => return Err(PyValueError::new_err(format!("unknown tif {other:?}"))),
    };
    if matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose)
        && !matches!(kind_parsed, OrderKind::Market)
    {
        return Err(PyValueError::new_err("at_open/at_close apply to market orders"));
    }
    if post_only && !matches!(kind_parsed, OrderKind::Limit { .. }) {
        return Err(PyValueError::new_err("post_only applies to limit orders"));
    }

    Ok(kernel.submit_order_full(
        side,
        qty,
        kind_parsed,
        tif,
        submitted_idx,
        submitted_ts,
        client_id.to_string(),
        stop_price,
        target_price,
        post_only,
        reduce_only,
        arrives_before_bar,
        parent_id,
    ))
}

#[cfg(test)]
mod tests {
    use super::next_due;

    #[test]
    fn the_next_due_bar_is_the_first_flag_set() {
        // Masks either side of a chunk, flags at every offset of one.
        for len in [0, 1, 7, 63, 64, 65, 127, 128, 129, 300] {
            for set in [None, Some(0), Some(1), Some(62), Some(63), Some(64), Some(65), Some(200)] {
                let mut mask = vec![false; len];
                if let Some(at) = set.filter(|&at| at < len) {
                    mask[at] = true;
                    if at + 70 < len {
                        mask[at + 70] = true;
                    }
                }
                for from in 0..=len {
                    for end in from..=len {
                        let want =
                            mask[from..end].iter().position(|&flag| flag).map(|at| from + at);
                        assert_eq!(
                            next_due(&mask, from, end),
                            want,
                            "len {len} from {from} end {end}"
                        );
                    }
                }
            }
        }
    }
}
