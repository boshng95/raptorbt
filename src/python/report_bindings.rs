//! A single-name replay's canonical report, from the kernel's own records.
//!
//! The engine turns a finished session into its result (`convert_session`)
//! and reads the canonical report off that (`PerformanceReportService`). A
//! sweep keeps the report and a few totals, and building the whole result
//! for them cost a run more than its bars did. `canonical_report` builds the
//! same report from the same records: the session's round trips rendered as
//! `round_trip_rows` renders them, in `Trade.realized_order_key` order, and
//! the ledger's order fills report read as `normalize_performance_fills`
//! reads it.
//!
//! It answers `None` wherever it does not transcribe the Python -- a round
//! trip still open, a settled position trimmed on the way, an instant that
//! is not a whole second, an amount Python would refuse -- and the caller
//! builds the result instead.

use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::bindings::{PyBacktestResult, PyTrade};
use super::fill_ledger::{money, py_round, PyReplayLedger};
use crate::metrics::canonical::{calculate, LocalClock, ReportInputs, TradeRow, Value};
use crate::metrics::exact;

const NS_PER_SECOND: i64 = 1_000_000_000;

/// A trade row and the rest of the key it is ordered by.
struct Keyed {
    row: TradeRow,
    short: bool,
    entry_price: f64,
    exit_price: f64,
    quantity: f64,
}

impl Keyed {
    /// `Trade.realized_order_key`, on one instrument.
    fn order(&self, other: &Self) -> Ordering {
        let (a, b) = (&self.row, &other.row);
        a.closed_s
            .is_none()
            .cmp(&b.closed_s.is_none())
            .then(a.closed_s.unwrap_or(0).cmp(&b.closed_s.unwrap_or(0)))
            .then(a.opened_s.cmp(&b.opened_s))
            .then(self.short.cmp(&other.short))
            .then(float_order(self.entry_price, other.entry_price))
            .then(float_order(self.exit_price, other.exit_price))
            .then(float_order(a.realized, b.realized))
            .then(float_order(self.quantity, other.quantity))
    }
}

/// Floats as Python compares them; every key here is finite.
fn float_order(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn whole_seconds(ns: i64) -> Option<i64> {
    (ns % NS_PER_SECOND == 0).then_some(ns / NS_PER_SECOND)
}

/// `convert_session`'s trade rows, sorted as `BacktestResult` sorts them.
///
/// Round trips in the order they opened; one that closed renders its one
/// row, one the data ended on its open row and its settlement. *reduction_ns*
/// are the reducing fills of the book on *symbol*, which a settled position
/// must not have taken (`trims_of`): its rows would then be split by them.
fn trade_rows(
    trades: &[PyTrade],
    symbol: &str,
    reduction_ns: &[i64],
    precision: u32,
) -> Option<Vec<TradeRow>> {
    let mut opened: Vec<&PyTrade> = trades.iter().collect();
    opened.sort_by_key(|item| item.entry_time);
    let mut rows: Vec<Keyed> = Vec::with_capacity(opened.len() + 1);
    for item in opened {
        let amounts = [
            item.size,
            item.entry_price,
            item.exit_price,
            item.pnl,
            item.entry_fees,
            item.exit_fees,
        ];
        if item.symbol != symbol
            || item.exit_time <= 0
            || amounts.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let opened_s = whole_seconds(item.entry_time)?;
        let closed_s = whole_seconds(item.exit_time)?;
        let entry_fee = money(item.entry_fees, precision)?;
        money(item.exit_fees, precision)?;
        let duration_hours = Some((closed_s - opened_s) as f64 / 3600.0);
        let short = item.direction <= 0;
        if item.exit_reason == "EndOfData" {
            let trimmed =
                reduction_ns.iter().any(|&ns| ns > item.entry_time && ns <= item.exit_time);
            if trimmed {
                return None;
            }
            let direction = if short { -1.0 } else { 1.0 };
            rows.push(Keyed {
                row: TradeRow {
                    opened_s,
                    closed_s: None,
                    realized: py_round(-entry_fee, precision)?,
                    duration_hours: None,
                },
                short,
                entry_price: item.entry_price,
                exit_price: 0.0,
                quantity: item.size,
            });
            rows.push(Keyed {
                row: TradeRow {
                    opened_s,
                    closed_s: Some(closed_s),
                    realized: direction * (item.exit_price - item.entry_price) * item.size,
                    duration_hours,
                },
                short,
                entry_price: item.entry_price,
                exit_price: item.exit_price,
                quantity: item.size,
            });
        } else {
            rows.push(Keyed {
                row: TradeRow {
                    opened_s,
                    closed_s: Some(closed_s),
                    realized: money(item.pnl, precision)?,
                    duration_hours,
                },
                short,
                entry_price: item.entry_price,
                exit_price: item.exit_price,
                quantity: 0.0,
            });
        }
    }
    if rows.iter().any(|keyed| !keyed.row.realized.is_finite()) {
        return None;
    }
    rows.sort_by(Keyed::order);
    Some(rows.into_iter().map(|keyed| keyed.row).collect())
}

fn to_python(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    Ok(match value {
        Value::None => py.None(),
        Value::Bool(flag) => flag.into_py(py),
        Value::Int(number) => number.into_py(py),
        Value::Float(number) => number.into_py(py),
        Value::Text(text) => text.into_py(py),
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, to_python(py, item)?)?;
            }
            dict.into_py(py)
        }
    })
}

/// The canonical report of a replay, as `PerformanceReportService.calculate`
/// writes it with `execution_lane` None; the number of order fills report
/// rows (`total_trades`); and the realized P&L of each closed trade row, in
/// order. None where the result must be built in Python instead.
///
/// *symbol* is the name *ledger* booked, and every round trip's. *offsets* are `timezone_transition_schedule` of *timezone* over the run's
/// window, known through *offsets_until_ns*.
#[pyfunction]
#[pyo3(signature = (
    result, ledger, *, symbol, starting, timezone, offsets, offsets_until_ns,
    risk_free_rate = 0.0
))]
#[allow(clippy::too_many_arguments)]
pub fn canonical_report(
    py: Python<'_>,
    result: PyRef<'_, PyBacktestResult>,
    ledger: PyRef<'_, PyReplayLedger>,
    symbol: &str,
    starting: f64,
    timezone: &str,
    offsets: Vec<(i64, i64)>,
    offsets_until_ns: i64,
    risk_free_rate: f64,
) -> PyResult<Option<(PyObject, usize, Py<PyList>)>> {
    let built = (|| {
        let precision = ledger.currency_precision()?;
        let rows = trade_rows(&result.trades, symbol, &ledger.reduction_times()?, precision)?;
        let fills = ledger.report_amounts()?;
        let clock = LocalClock::new(&offsets, offsets_until_ns)?;
        let report = calculate(&ReportInputs {
            starting,
            risk_free_rate,
            timezone,
            clock: &clock,
            day_ns: &result.daily_performance_timestamps,
            day_equity: &result.daily_performance_equity,
            trades: &rows,
            fills: &fills,
        })?;
        let closed: Vec<f64> =
            rows.iter().filter(|row| row.closed_s.is_some()).map(|row| row.realized).collect();
        Some((report, fills.price.len(), closed))
    })();
    match built {
        None => Ok(None),
        Some((report, orders, closed)) => {
            Ok(Some((to_python(py, &report)?, orders, PyList::new(py, closed).into())))
        }
    }
}

/// `math.fsum`; None where it raises.
#[pyfunction]
#[pyo3(name = "_exact_fsum")]
pub fn exact_fsum(values: Vec<f64>) -> Option<f64> {
    exact::fsum(values)
}

/// The builtin `sum` of floats.
#[pyfunction]
#[pyo3(name = "_exact_sum")]
pub fn exact_sum(values: Vec<f64>) -> f64 {
    exact::py_sum(&values)
}

/// `numpy.sum` of a float64 array.
#[pyfunction]
#[pyo3(name = "_exact_np_sum")]
pub fn exact_np_sum(values: Vec<f64>) -> f64 {
    exact::np_sum(&values)
}

/// `statistics.stdev` of floats; None where it is not transcribed.
#[pyfunction]
#[pyo3(name = "_exact_stdev")]
pub fn exact_stdev(values: Vec<f64>) -> Option<f64> {
    exact::stdev(&values)
}

/// `statistics.median` of floats.
#[pyfunction]
#[pyo3(name = "_exact_median")]
pub fn exact_median(values: Vec<f64>) -> Option<f64> {
    exact::median(&values)
}

/// `base ** exponent` of floats; None where Python raises or this declines.
#[pyfunction]
#[pyo3(name = "_exact_pow")]
pub fn exact_pow(base: f64, exponent: f64) -> Option<f64> {
    exact::py_pow(base, exponent).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = NS_PER_SECOND;

    fn trade(entry_s: i64, exit_s: i64, pnl: f64, exit_reason: &str) -> PyTrade {
        PyTrade {
            id: 1,
            symbol: "AAPL".into(),
            entry_idx: 0,
            exit_idx: 1,
            entry_price: 100.0,
            exit_price: 110.0,
            size: 3.0,
            direction: 1,
            pnl,
            return_pct: 0.0,
            entry_time: entry_s * S,
            exit_time: exit_s * S,
            fees: 1.5,
            entry_fees: 0.755,
            exit_fees: 0.745,
            fee_breakdown: None,
            exit_reason: exit_reason.into(),
            mae_price: None,
            mfe_price: None,
            mae_pnl: None,
            mfe_pnl: None,
        }
    }

    #[test]
    fn a_closed_round_trip_renders_one_row() {
        let rows = trade_rows(&[trade(3600, 7200, 28.505, "Signal")], "AAPL", &[], 2).unwrap();
        assert_eq!(
            rows,
            vec![TradeRow {
                opened_s: 3600,
                closed_s: Some(7200),
                realized: 28.51,
                duration_hours: Some(1.0),
            }]
        );
    }

    #[test]
    fn a_settled_position_renders_its_open_row_and_its_settlement() {
        let rows = trade_rows(&[trade(3600, 9000, 0.0, "EndOfData")], "AAPL", &[], 2).unwrap();
        // The open row carries the entry fee, as booked (`Money`), then
        // rounded again; it sorts after every closed row.
        assert_eq!(
            rows,
            vec![
                TradeRow {
                    opened_s: 3600,
                    closed_s: Some(9000),
                    realized: 30.0,
                    duration_hours: Some(1.5),
                },
                TradeRow { opened_s: 3600, closed_s: None, realized: -0.76, duration_hours: None },
            ]
        );
    }

    #[test]
    fn a_settled_position_trimmed_on_the_way_is_left_to_python() {
        let trades = [trade(3600, 9000, 0.0, "EndOfData")];
        assert!(trade_rows(&trades, "AAPL", &[5000 * S], 2).is_none());
        // A reduction at its entry belongs to an earlier position.
        assert!(trade_rows(&trades, "AAPL", &[3600 * S], 2).is_some());
    }

    #[test]
    fn rows_sort_as_the_result_sorts_its_trades() {
        let trades = [
            trade(100, 400, 1.0, "Signal"),
            trade(200, 300, 2.0, "Signal"),
            trade(50, 300, 3.0, "Signal"),
        ];
        let rows = trade_rows(&trades, "AAPL", &[], 2).unwrap();
        let keys: Vec<(Option<i64>, i64)> =
            rows.iter().map(|row| (row.closed_s, row.opened_s)).collect();
        assert_eq!(keys, vec![(Some(300), 50), (Some(300), 200), (Some(400), 100)]);
    }

    #[test]
    fn declines_what_it_does_not_transcribe() {
        let mut other = trade(3600, 7200, 1.0, "Signal");
        other.symbol = "MSFT".into();
        let mut open = trade(3600, 0, 1.0, "Signal");
        open.exit_time = 0;
        let mut fraction = trade(3600, 7200, 1.0, "Signal");
        fraction.exit_time += 1;
        let mut nan = trade(3600, 7200, 1.0, "Signal");
        nan.exit_fees = f64::NAN;
        for item in [other, open, fraction, nan] {
            assert!(trade_rows(&[item], "AAPL", &[], 2).is_none());
        }
        assert_eq!(trade_rows(&[], "AAPL", &[], 2), Some(vec![]));
    }
}
