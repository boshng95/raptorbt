//! A replay's fills, booked in the kernel the way its caller books them.
//!
//! `SignalReplay` books every fill of a replay through `FillBook.record` and
//! reports the orders behind them through `order_fills_report`
//! (`parity/portfolio.py` in the caller): per fill, an execution, the
//! position it moved, the peak that position reached and the balance it
//! left; per order, one report row. Built in Python from the replay's log,
//! that is a handful of objects and a dozen float operations per fill, paid
//! thousands of times by a replay that trades every hour. This is that
//! booking and that report, transcribed in the order the Python performs
//! them, so every number is the same double and every spelling the same
//! text. What the caller still wants as objects -- the executions, the
//! order ends -- it builds from the columns here, if it asks.
//!
//! Booking quantizes money through `_money` and `round`, which raise on a
//! value that is not finite. The ledger then stops booking and reports
//! itself unbooked, and the caller books the replay's log (`log`) itself,
//! raising where it always did.
//!
//! Strings the caller would have shared between an execution and its report
//! row -- an order's id, side, type and time in force, a fill's venue order
//! id and trade id -- are made once here and handed out as that one object
//! wherever they appear, as the Python run shares them.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use super::money::{fixed, round_to};

/// `10**p` for the precisions `_money` is asked for, as exact doubles.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `x or 0.0`: a zero of either sign is falsy, a NaN is not.
fn or_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `max(a, b)`: `b` only when it compares greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `_money`: scale by the currency's precision, round half away from zero,
/// scale back. `None` where it raises (`math.floor` of an infinity or NaN).
fn money(value: f64, precision: u32) -> Option<f64> {
    let scale = *POW10.get(precision as usize)?;
    let scaled = value * scale;
    if !scaled.is_finite() {
        return None;
    }
    let mut whole = scaled.abs().floor();
    if scaled.abs() - whole >= 0.5 {
        whole += 1.0;
    }
    Some(whole.copysign(scaled) / scale)
}

/// `round(x, precision)` on a float: correctly rounded, ties to even on the
/// exact binary value, as Rust's fixed formatting spells it. `None` near
/// the range's end, where Python's may overflow.
fn py_round(x: f64, precision: u32) -> Option<f64> {
    if !x.is_finite() {
        return Some(x);
    }
    if x.abs() >= 1e300 {
        return None;
    }
    round_to(x, precision)
}

/// `f"{x:.{precision}f}"`.
fn spelled(x: f64, precision: usize) -> String {
    if x.is_nan() {
        // Python spells a NaN of either sign without one; Rust as `NaN`.
        return "nan".to_string();
    }
    fixed(x, precision)
}

/// The book the caller opened, and the contract terms its report spells
/// with.
pub(crate) struct BookTerms {
    pub balance: f64,
    pub currency_precision: u32,
    pub size_increment: f64,
    pub size_precision: usize,
    pub price_precision: usize,
    pub currency: String,
}

/// An order as the replay sent it: `_OrderTerms` and its `OrderTicket`.
pub(crate) struct OrderRecord {
    pub client_id: String,
    pub buy: bool,
    pub market: bool,
    pub time_in_force: &'static str,
    pub units: f64,
    pub limit_price: Option<f64>,
    pub expire_ns: Option<i64>,
    pub reduce_only: bool,
    pub decided_ns: i64,
    pub arrived_on_ns: i64,
    pub policies_after_fill: bool,
}

impl OrderRecord {
    /// `_OrderTerms.stamp`.
    fn stamp(&self, now_ns: i64) -> i64 {
        if self.arrived_on_ns == now_ns && self.decided_ns != 0 {
            self.decided_ns
        } else {
            now_ns
        }
    }
}

/// A fill as the kernel reported it.
struct FillRecord {
    order: u32,
    now_ns: i64,
    order_id: u64,
    size: f64,
    price: f64,
    leaves: f64,
    gross_realized: f64,
    commission: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndStatus {
    Canceled,
    Expired,
}

enum Entry {
    Submit(u32),
    Fill(u32),
    End { order: u32, status: EndStatus, now_ns: i64 },
}

/// What `FillBook.record` and `end_order` keep, as columns.
struct Booking {
    balance: f64,
    held: f64,
    ts: Vec<i64>,
    size: Vec<f64>,
    price: Vec<f64>,
    leaves: Vec<f64>,
    cost: Vec<f64>,
    balances: Vec<f64>,
    reduces: Vec<bool>,
    peak_ts: Vec<i64>,
    peak_size: Vec<f64>,
    /// Per order, the first end the venue reported, stamped.
    ends: Vec<Option<(EndStatus, i64)>>,
    /// Orders in the order their first end arrived.
    ended: Vec<u32>,
}

/// Everything a replay sent and the venue did, in order.
pub(crate) struct Ledger {
    terms: Option<BookTerms>,
    orders: Vec<OrderRecord>,
    fills: Vec<FillRecord>,
    entries: Vec<Entry>,
    booking: Option<Booking>,
}

impl Ledger {
    /// A ledger that books into a `FillBook` opened on `terms`, or, without
    /// them, only records the log.
    pub(crate) fn new(terms: Option<BookTerms>) -> Self {
        let booking = terms.as_ref().map(|terms| Booking {
            balance: terms.balance,
            held: 0.0,
            ts: Vec::new(),
            size: Vec::new(),
            price: Vec::new(),
            leaves: Vec::new(),
            cost: Vec::new(),
            balances: Vec::new(),
            reduces: Vec::new(),
            peak_ts: Vec::new(),
            peak_size: Vec::new(),
            ends: Vec::new(),
            ended: Vec::new(),
        });
        Self { terms, orders: Vec::new(), fills: Vec::new(), entries: Vec::new(), booking }
    }

    /// Record an order; its index names it from here on.
    pub(crate) fn submit(&mut self, order: OrderRecord) -> u32 {
        let index = self.orders.len() as u32;
        self.orders.push(order);
        self.entries.push(Entry::Submit(index));
        if let Some(booking) = self.booking.as_mut() {
            booking.ends.push(None);
        }
        index
    }

    /// `_record_fill`, then `FillBook.record`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill(
        &mut self,
        order: u32,
        now_ns: i64,
        order_id: u64,
        size: f64,
        price: f64,
        leaves: f64,
        gross_realized: f64,
        commission: f64,
    ) {
        let index = self.fills.len() as u32;
        self.fills.push(FillRecord {
            order,
            now_ns,
            order_id,
            size,
            price,
            leaves,
            gross_realized,
            commission,
        });
        self.entries.push(Entry::Fill(index));
        if !self.book(index) {
            self.booking = None;
        }
    }

    /// Book fill `index`; false where the Python booking would raise.
    fn book(&mut self, index: u32) -> bool {
        let (Some(terms), Some(booking)) = (self.terms.as_ref(), self.booking.as_mut()) else {
            return true;
        };
        let raw = &self.fills[index as usize];
        let order = &self.orders[raw.order as usize];
        let precision = terms.currency_precision;
        let ts = order.stamp(raw.now_ns);
        let left = or_zero(raw.leaves);
        let Some(cost) = money(or_zero(raw.commission), precision) else {
            return false;
        };
        let size = or_zero(raw.size);
        let price = or_zero(raw.price);
        let realized = raw.gross_realized;

        let signed = if order.buy { size } else { -size };
        let held = booking.held;
        let reduces = held != 0.0 && (held > 0.0) != (signed > 0.0);
        let mut after = held + signed;
        if after.abs() <= py_max(held.abs(), size) * 1e-12 {
            after = 0.0;
        }
        booking.held = after;

        // `_track_peak`.
        let scale = py_max(held.abs(), after.abs());
        let flat_before = held.abs() <= scale * 1e-12;
        let flat_after = after.abs() <= scale * 1e-12;
        if !flat_after {
            if flat_before || (held > 0.0) != (after > 0.0) {
                booking.peak_ts.push(ts);
                booking.peak_size.push(after.abs());
            } else if let Some(peak) = booking.peak_size.last_mut() {
                if after.abs() > *peak {
                    *peak = after.abs();
                }
            }
        }

        // `_book_fill`: every fill here reports what it realized.
        let Some(realized_money) = money(realized, precision) else {
            return false;
        };
        let Some(balance) = py_round(booking.balance + realized_money - cost, precision) else {
            return false;
        };
        booking.balance = balance;
        booking.ts.push(ts);
        booking.size.push(size);
        booking.price.push(price);
        booking.leaves.push(left);
        booking.cost.push(cost);
        booking.balances.push(balance);
        booking.reduces.push(reduces);
        true
    }

    /// `_record_end`, then `FillBook.end_order`: an order ends once.
    pub(crate) fn end(&mut self, order: u32, status: EndStatus, now_ns: i64) {
        self.entries.push(Entry::End { order, status, now_ns });
        if let Some(booking) = self.booking.as_mut() {
            let slot = &mut booking.ends[order as usize];
            if slot.is_none() {
                *slot = Some((status, self.orders[order as usize].stamp(now_ns)));
                booking.ended.push(order);
            }
        }
    }

    pub(crate) fn submitted(&self) -> usize {
        self.orders.len()
    }

    /// The replay's log, as `KernelSession.replay_rules` returns it without
    /// a book.
    pub(crate) fn log(&self, py: Python<'_>) -> Vec<PyObject> {
        self.entries
            .iter()
            .map(|entry| match *entry {
                Entry::Submit(index) => {
                    let order = &self.orders[index as usize];
                    (
                        "submit",
                        order.client_id.as_str(),
                        if order.buy { "BUY" } else { "SELL" },
                        if order.market { "MARKET" } else { "LIMIT" },
                        order.time_in_force,
                        order.units,
                        order.limit_price,
                        order.expire_ns,
                        order.reduce_only,
                        order.decided_ns,
                        order.arrived_on_ns,
                        order.policies_after_fill,
                    )
                        .into_py(py)
                }
                Entry::Fill(index) => {
                    let fill = &self.fills[index as usize];
                    (
                        "fill",
                        self.orders[fill.order as usize].client_id.as_str(),
                        fill.now_ns,
                        fill.order_id,
                        fill.size,
                        fill.price,
                        fill.leaves,
                        fill.gross_realized,
                        fill.commission,
                    )
                        .into_py(py)
                }
                Entry::End { order, status, now_ns } => (
                    "end",
                    self.orders[order as usize].client_id.as_str(),
                    status_name(py, status),
                    now_ns,
                )
                    .into_py(py),
            })
            .collect()
    }
}

/// Interned, as the Python loop's literal is: a report's status column then
/// shares one object per status, as its run does.
fn status_name(py: Python<'_>, status: EndStatus) -> &PyString {
    match status {
        EndStatus::Canceled => pyo3::intern!(py, "CANCELED"),
        EndStatus::Expired => pyo3::intern!(py, "EXPIRED"),
    }
}

/// One `order_fills_report` row, before it is spelled.
struct ReportRow {
    order: u32,
    last_fill: u32,
    filled: f64,
    avg_px: f64,
    commission: f64,
    /// 0 FILLED, 1 CANCELED, 2 EXPIRED, 3 PARTIALLY_FILLED.
    status: u8,
    ts_last: i64,
    quantity: f64,
    slippage: f64,
}

/// A replay's ledger, for the caller to adopt into its `FillBook`.
#[pyclass(name = "ReplayLedger", module = "raptorbt")]
pub struct PyReplayLedger {
    ledger: Ledger,
    /// Per order: its client id, side, type and time in force.
    order_names: Vec<Option<[Py<PyString>; 4]>>,
    /// Per fill: its venue order id and trade id.
    fill_names: Vec<Option<[Py<PyString>; 2]>>,
}

impl PyReplayLedger {
    pub(crate) fn new(ledger: Ledger) -> Self {
        let orders = ledger.orders.len();
        let fills = ledger.fills.len();
        Self { ledger, order_names: vec![None; orders], fill_names: vec![None; fills] }
    }

    fn booking(&self) -> PyResult<&Booking> {
        self.ledger.booking.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this ledger did not book its fills")
        })
    }

    fn order_names(&mut self, py: Python<'_>, index: u32) -> [Py<PyString>; 4] {
        let slot = &mut self.order_names[index as usize];
        if slot.is_none() {
            let order = &self.ledger.orders[index as usize];
            *slot = Some([
                PyString::new(py, &order.client_id).into(),
                PyString::new(py, if order.buy { "BUY" } else { "SELL" }).into(),
                PyString::new(py, if order.market { "MARKET" } else { "LIMIT" }).into(),
                PyString::new(py, order.time_in_force).into(),
            ]);
        }
        slot.as_ref().map(|names| names.clone()).unwrap_or_else(|| unreachable!())
    }

    fn fill_names(&mut self, py: Python<'_>, index: u32) -> [Py<PyString>; 2] {
        let slot = &mut self.fill_names[index as usize];
        if slot.is_none() {
            let fill = &self.ledger.fills[index as usize];
            *slot = Some([
                PyString::new(py, &format!("R-{}", fill.order_id)).into(),
                PyString::new(py, &format!("T-{}", index + 1)).into(),
            ]);
        }
        slot.as_ref().map(|names| names.clone()).unwrap_or_else(|| unreachable!())
    }

    /// `order_fills_report`'s rows: one per order that filled at all, in the
    /// order the orders first filled, then stably by when they were placed.
    fn report_rows(&self, booking: &Booking, size_increment: f64) -> Vec<ReportRow> {
        let mut rows: Vec<ReportRow> = Vec::new();
        let mut row_of: Vec<Option<usize>> = vec![None; self.ledger.orders.len()];
        for (index, fill) in self.ledger.fills.iter().enumerate() {
            let at = match row_of[fill.order as usize] {
                Some(at) => at,
                None => {
                    rows.push(ReportRow {
                        order: fill.order,
                        last_fill: 0,
                        filled: 0.0,
                        avg_px: 0.0,
                        commission: 0.0,
                        status: 0,
                        ts_last: 0,
                        quantity: 0.0,
                        slippage: 0.0,
                    });
                    row_of[fill.order as usize] = Some(rows.len() - 1);
                    rows.len() - 1
                }
            };
            // `_order_row`'s running sums, in its order of operations.
            let row = &mut rows[at];
            let (qty, px) = (booking.size[index], booking.price[index]);
            let before = row.filled;
            row.filled = before + qty;
            row.avg_px =
                if row.avg_px == 0.0 { px } else { (row.avg_px * before + px * qty) / row.filled };
            row.commission += booking.cost[index];
            row.last_fill = index as u32;
        }
        for row in &mut rows {
            let order = &self.ledger.orders[row.order as usize];
            let last = row.last_fill as usize;
            row.quantity = order.units;
            if booking.leaves[last] <= 0.0 {
                row.status = 0;
                row.ts_last = booking.ts[last];
                if row.filled < row.quantity - size_increment / 2.0 {
                    row.quantity = row.filled;
                }
            } else if let Some((status, ts)) = booking.ends[row.order as usize] {
                row.status = match status {
                    EndStatus::Canceled => 1,
                    EndStatus::Expired => 2,
                };
                row.ts_last = ts;
            } else {
                row.status = 3;
                row.ts_last = booking.ts[last];
            }
            if let Some(price) = order.limit_price {
                row.slippage = if order.buy { row.avg_px - price } else { price - row.avg_px };
            }
        }
        rows.sort_by_key(|row| self.ledger.orders[row.order as usize].decided_ns);
        rows
    }
}

#[pymethods]
impl PyReplayLedger {
    /// Orders the replay sent.
    #[getter]
    fn submitted(&self) -> usize {
        self.ledger.submitted()
    }

    /// Fills the venue reported.
    #[getter]
    fn filled(&self) -> usize {
        self.ledger.fills.len()
    }

    /// Whether every fill was booked; if not, book `log()` instead.
    #[getter]
    fn booked(&self) -> bool {
        self.ledger.booking.is_some()
    }

    /// The balance the last fill left, as `FillBook.balance`.
    #[getter]
    fn balance(&self) -> PyResult<f64> {
        Ok(self.booking()?.balance)
    }

    /// The replay's log, as `replay_rules` returns it without a book.
    fn log(&self, py: Python<'_>) -> Vec<PyObject> {
        self.ledger.log(py)
    }

    /// Per fill, in fill order: `(order, ts_event, last_qty, last_px,
    /// leaves_qty, gross_realized, commission, balance, reduces)` -- the
    /// index of its order in `orders()`, then `ExecutionRecord`'s numbers,
    /// the balance it left and whether it took exposure back.
    #[allow(clippy::type_complexity)]
    fn fills<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        &'py PyArray1<i64>,
        &'py PyArray1<i64>,
        &'py PyArray1<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<f64>,
        &'py PyArray1<bool>,
    )> {
        let booking = self.booking()?;
        let fills = &self.ledger.fills;
        Ok((
            PyArray1::from_vec(py, fills.iter().map(|fill| i64::from(fill.order)).collect()),
            PyArray1::from_slice(py, &booking.ts),
            PyArray1::from_slice(py, &booking.size),
            PyArray1::from_slice(py, &booking.price),
            PyArray1::from_slice(py, &booking.leaves),
            PyArray1::from_vec(py, fills.iter().map(|fill| fill.gross_realized).collect()),
            PyArray1::from_slice(py, &booking.cost),
            PyArray1::from_slice(py, &booking.balances),
            PyArray1::from_slice(py, &booking.reduces),
        ))
    }

    /// Per order, as its `OrderTicket` holds it: `(client_order_id, side,
    /// order_type, time_in_force, quantity, price, expire_time_ns,
    /// is_reduce_only, ts_init)`.
    #[allow(clippy::type_complexity)]
    fn orders(
        &mut self,
        py: Python<'_>,
    ) -> Vec<(
        Py<PyString>,
        Py<PyString>,
        Py<PyString>,
        Py<PyString>,
        f64,
        Option<f64>,
        Option<i64>,
        bool,
        i64,
    )> {
        (0..self.ledger.orders.len() as u32)
            .map(|index| {
                let [client_id, side, order_type, tif] = self.order_names(py, index);
                let order = &self.ledger.orders[index as usize];
                (
                    client_id,
                    side,
                    order_type,
                    tif,
                    order.units,
                    order.limit_price,
                    order.expire_ns,
                    order.reduce_only,
                    order.decided_ns,
                )
            })
            .collect()
    }

    /// Per fill: `(venue_order_id, trade_id)`.
    fn fill_ids(&mut self, py: Python<'_>) -> Vec<(Py<PyString>, Py<PyString>)> {
        (0..self.ledger.fills.len() as u32)
            .map(|index| {
                let [venue_order_id, trade_id] = self.fill_names(py, index);
                (venue_order_id, trade_id)
            })
            .collect()
    }

    /// `FillBook.peaks` for the one instrument: `(opened_ns, peak units)`.
    fn peaks<'py>(&self, py: Python<'py>) -> PyResult<(&'py PyArray1<i64>, &'py PyArray1<f64>)> {
        let booking = self.booking()?;
        Ok((
            PyArray1::from_slice(py, &booking.peak_ts),
            PyArray1::from_slice(py, &booking.peak_size),
        ))
    }

    /// `FillBook.order_ends`, in the order they arrived: `(order, status,
    /// ts_event)`.
    fn ends<'py>(&self, py: Python<'py>) -> PyResult<Vec<(u32, &'py PyString, i64)>> {
        let booking = self.booking()?;
        Ok(booking
            .ended
            .iter()
            .filter_map(|&order| {
                booking.ends[order as usize]
                    .map(|(status, ts)| (order, status_name(py, status), ts))
            })
            .collect())
    }

    /// `order_fills_report`'s columns that differ by row, by name, or
    /// `None` when nothing filled: numbers and flags as arrays, the rest as
    /// lists. `status` holds codes: 0 FILLED, 1 CANCELED, 2 EXPIRED, 3
    /// PARTIALLY_FILLED.
    fn report<'py>(&mut self, py: Python<'py>) -> PyResult<Option<&'py PyDict>> {
        let (size_increment, size_precision, price_precision, currency_precision, currency) = {
            let terms = self.ledger.terms.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("this ledger did not book its fills")
            })?;
            (
                terms.size_increment,
                terms.size_precision,
                terms.price_precision,
                terms.currency_precision as usize,
                terms.currency.clone(),
            )
        };
        let rows = self.report_rows(self.booking()?, size_increment);
        if rows.is_empty() {
            return Ok(None);
        }
        let n = rows.len();
        let mut client_id = Vec::with_capacity(n);
        let mut venue_order_id = Vec::with_capacity(n);
        let mut last_trade_id = Vec::with_capacity(n);
        let mut order_type = Vec::with_capacity(n);
        let mut side = Vec::with_capacity(n);
        let mut quantity = Vec::with_capacity(n);
        let mut price: Vec<PyObject> = Vec::with_capacity(n);
        let mut tif = Vec::with_capacity(n);
        let mut expire: Vec<Option<i64>> = Vec::with_capacity(n);
        let mut filled_qty = Vec::with_capacity(n);
        let mut avg_px: Vec<f64> = Vec::with_capacity(n);
        let mut slippage: Vec<f64> = Vec::with_capacity(n);
        let mut commissions: Vec<PyObject> = Vec::with_capacity(n);
        let mut status: Vec<i64> = Vec::with_capacity(n);
        let mut is_post_only: Vec<Option<bool>> = Vec::with_capacity(n);
        let mut is_reduce_only: Vec<bool> = Vec::with_capacity(n);
        let mut ts_init: Vec<i64> = Vec::with_capacity(n);
        let mut ts_last: Vec<i64> = Vec::with_capacity(n);
        for row in &rows {
            let [order_client_id, order_side, kind, time_in_force] =
                self.order_names(py, row.order);
            let [venue_id, trade_id] = self.fill_names(py, row.last_fill);
            let order = &self.ledger.orders[row.order as usize];
            client_id.push(order_client_id);
            venue_order_id.push(venue_id);
            last_trade_id.push(trade_id);
            order_type.push(kind);
            side.push(order_side);
            quantity.push(spelled(row.quantity, size_precision));
            price.push(match order.limit_price {
                None => py.None(),
                Some(limit) => spelled(limit, price_precision).into_py(py),
            });
            tif.push(time_in_force);
            // `order.expire_time_ns or None`.
            expire.push(order.expire_ns.filter(|&ns| ns != 0));
            filled_qty.push(spelled(row.filled, size_precision));
            avg_px.push(row.avg_px);
            slippage.push(row.slippage);
            // `_money_text`: `amount + 0.0` spells a negative zero as zero.
            let text =
                format!("{} {}", spelled(row.commission + 0.0, currency_precision), currency);
            commissions.push(PyList::new(py, [text]).into_py(py));
            status.push(i64::from(row.status));
            is_post_only.push((!order.market).then_some(false));
            is_reduce_only.push(order.reduce_only);
            ts_init.push(order.decided_ns);
            ts_last.push(row.ts_last);
        }
        let columns = PyDict::new(py);
        columns.set_item("client_order_id", client_id)?;
        columns.set_item("venue_order_id", venue_order_id)?;
        columns.set_item("last_trade_id", last_trade_id)?;
        columns.set_item("type", order_type)?;
        columns.set_item("side", side)?;
        columns.set_item("quantity", quantity)?;
        columns.set_item("price", price)?;
        columns.set_item("time_in_force", tif)?;
        columns.set_item("expire_time_ns", expire)?;
        columns.set_item("filled_qty", filled_qty)?;
        columns.set_item("avg_px", PyArray1::from_vec(py, avg_px))?;
        columns.set_item("slippage", PyArray1::from_vec(py, slippage))?;
        columns.set_item("commissions", commissions)?;
        columns.set_item("status", PyArray1::from_vec(py, status))?;
        columns.set_item("is_post_only", is_post_only)?;
        columns.set_item("is_reduce_only", PyArray1::from_vec(py, is_reduce_only))?;
        columns.set_item("ts_init", PyArray1::from_vec(py, ts_init))?;
        columns.set_item("ts_last", PyArray1::from_vec(py, ts_last))?;
        Ok(Some(columns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_rounds_half_away_from_zero() {
        // -320.315 is stored a hair above itself; `Money` books -320.32.
        assert_eq!(money(-320.315, 2), Some(-320.32));
        assert_eq!(money(2.675, 2), Some(2.68));
        assert_eq!(money(-0.0, 2).map(f64::is_sign_negative), Some(true));
        assert_eq!(money(f64::NAN, 2), None);
        assert_eq!(money(f64::INFINITY, 2), None);
    }

    #[test]
    fn rounds_as_python_round_does() {
        // Ties on the exact binary value go to even; 2.675 lies below its tie.
        assert_eq!(py_round(0.125, 2), Some(0.12));
        assert_eq!(py_round(0.375, 2), Some(0.38));
        assert_eq!(py_round(2.675, 2), Some(2.67));
        assert_eq!(py_round(-0.001, 2).map(f64::is_sign_negative), Some(true));
        assert!(py_round(f64::NAN, 2).is_some_and(f64::is_nan));
    }

    fn order(buy: bool) -> OrderRecord {
        OrderRecord {
            client_id: "O-1".into(),
            buy,
            market: false,
            time_in_force: "GTD",
            units: 2.0,
            limit_price: Some(10.0),
            expire_ns: Some(99),
            reduce_only: false,
            decided_ns: 5,
            arrived_on_ns: 7,
            policies_after_fill: false,
        }
    }

    fn terms() -> BookTerms {
        BookTerms {
            balance: 1000.0,
            currency_precision: 2,
            size_increment: 1.0,
            size_precision: 0,
            price_precision: 2,
            currency: "USD".into(),
        }
    }

    #[test]
    fn books_a_round_trip_as_fill_book_does() {
        let mut ledger = Ledger::new(Some(terms()));
        let entry = ledger.submit(order(true));
        // Filled on the bar the order beat to the venue: stamped when decided.
        ledger.fill(entry, 7, 1, 2.0, 10.0, 0.0, 0.0, 0.015);
        let exit = ledger.submit(OrderRecord { buy: false, ..order(false) });
        ledger.fill(exit, 9, 2, 2.0, 11.0, -0.0, 2.0, 0.025);
        let booking = ledger.booking.as_ref().unwrap();
        assert_eq!(booking.ts, vec![5, 9]);
        assert_eq!(booking.cost, vec![0.02, 0.03]);
        assert_eq!(booking.balances, vec![999.98, 1001.95]);
        assert_eq!(booking.reduces, vec![false, true]);
        assert_eq!(booking.peak_ts, vec![5]);
        assert_eq!(booking.peak_size, vec![2.0]);
        assert_eq!(booking.held, 0.0);
        assert!(booking.leaves[1].is_sign_positive());
    }

    #[test]
    fn stops_booking_where_python_would_raise() {
        let mut ledger = Ledger::new(Some(terms()));
        let entry = ledger.submit(order(true));
        ledger.fill(entry, 7, 1, 2.0, 10.0, 0.0, f64::NAN, 0.0);
        assert!(ledger.booking.is_none());
        assert_eq!(ledger.entries.len(), 2);
    }

    #[test]
    fn keeps_an_orders_first_end() {
        let mut ledger = Ledger::new(Some(terms()));
        let entry = ledger.submit(order(true));
        ledger.end(entry, EndStatus::Expired, 7);
        ledger.end(entry, EndStatus::Canceled, 8);
        let booking = ledger.booking.as_ref().unwrap();
        assert!(booking.ends[0] == Some((EndStatus::Expired, 5)));
        assert_eq!(booking.ended, vec![0]);
    }
}
