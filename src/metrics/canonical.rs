//! The canonical daily report, built as the engine's
//! `PerformanceReportService.calculate` builds it.
//!
//! A run's report is what a sweep keeps of it, and building it in Python cost
//! a single-name run more than its bars did. This is the same report from the
//! same inputs -- the daily net-liquidation curve, the run's trade rows in
//! their realized order, and its order fills report -- in the same arithmetic
//! (`exact`), so every number is Python's to the bit and every key is where
//! Python puts it.
//!
//! Instants are whole UTC seconds: the report reads them as Python
//! `datetime`s, which keep microseconds, and only a whole second reads the
//! same in both. Anything this does not transcribe, or that Python would
//! raise on, answers `None`, and the caller builds the report in Python.

use std::cmp::Ordering;

use super::exact::{fmean, fsum, median, np_sum, py_max, py_min, py_pow, py_sum, stdev, PowError};

pub const SCHEMA: &str = "canonical_daily_mtm_v1";

const DAY_SECONDS: i64 = 86_400;
const NS_PER_SECOND: i64 = 1_000_000_000;
/// `calendar_returns.YEAR_DAYS`.
const YEAR_DAYS: f64 = 365.2425;
/// `calendar_returns._SECONDS_PER_YEAR`.
const SECONDS_PER_YEAR: f64 = 86_400.0 * YEAR_DAYS;
/// `86_400.0 * performance_report._MONTH_DAYS`.
const MONTH_SECONDS: f64 = 86_400.0 * (YEAR_DAYS / 12.0);
/// A return or P&L within this of zero is neither a win nor a loss.
const FLAT: f64 = 1e-12;
/// A drawdown within this of zero is back at its high.
const RECOVERED: f64 = -1e-15;

/// A report value: what the Python report's dicts hold.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    /// A dict, its keys in insertion order.
    Map(Vec<(String, Value)>),
}

impl Value {
    fn float(value: Option<f64>) -> Self {
        value.map_or(Value::None, Value::Float)
    }

    /// `_finite`: what the metrics dict keeps of a float.
    fn finite(value: Option<f64>) -> Self {
        Value::float(finite(value))
    }

    fn int(value: usize) -> Self {
        Value::Int(value as i64)
    }

    /// The value under `key` of a map.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

/// A trade row as the report reads it, from `round_trip_rows`.
#[derive(Clone, Debug, PartialEq)]
pub struct TradeRow {
    pub opened_s: i64,
    /// None for the open row a settled position leaves.
    pub closed_s: Option<i64>,
    pub realized: f64,
    pub duration_hours: Option<f64>,
}

/// The order fills report as `normalize_performance_fills` reads it: one
/// commission, quantity and price per order row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FillAmounts {
    pub commission: Vec<f64>,
    pub quantity: Vec<f64>,
    pub price: Vec<f64>,
}

/// A timezone as `timezone_transition_schedule` writes it down: the offset
/// in force from each UTC second on, known up to `until_s`.
#[derive(Clone, Debug)]
pub struct LocalClock {
    offsets: Vec<(i64, i64)>,
    until_s: i64,
}

impl LocalClock {
    /// From the schedule in nanoseconds; None unless every entry is a whole
    /// second and the entries rise.
    pub fn new(offsets_ns: &[(i64, i64)], until_ns: i64) -> Option<Self> {
        let mut offsets = Vec::with_capacity(offsets_ns.len());
        for &(from, offset) in offsets_ns {
            if from % NS_PER_SECOND != 0 || offset % NS_PER_SECOND != 0 {
                return None;
            }
            let entry = (from / NS_PER_SECOND, offset / NS_PER_SECOND);
            if offsets.last().is_some_and(|&(last, _): &(i64, i64)| last >= entry.0) {
                return None;
            }
            offsets.push(entry);
        }
        Some(Self { offsets, until_s: until_ns.div_euclid(NS_PER_SECOND) })
    }

    /// Local wall-clock seconds at a UTC second, inside the schedule.
    fn local(&self, utc_s: i64) -> Option<i64> {
        if utc_s > self.until_s {
            return None;
        }
        let at = self.offsets.partition_point(|&(from, _)| from <= utc_s);
        (at > 0).then(|| utc_s + self.offsets[at - 1].1)
    }

    /// `calendar_returns.close_date`: the local day an instant closes, the
    /// day before at the stroke of midnight.
    fn close_date(&self, utc_s: i64) -> Option<Date> {
        let local = self.local(utc_s)?;
        let mut days = local.div_euclid(DAY_SECONDS);
        if local.rem_euclid(DAY_SECONDS) == 0 {
            days -= 1;
        }
        Some(Date::of_days(days))
    }

    /// The local date of an instant.
    fn date(&self, utc_s: i64) -> Option<Date> {
        Some(Date::of_days(self.local(utc_s)?.div_euclid(DAY_SECONDS)))
    }
}

/// A proleptic Gregorian date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i64,
    pub month: u32,
    pub day: u32,
}

impl Date {
    /// The date `days` after 1970-01-01 (Hinnant's `civil_from_days`).
    fn of_days(days: i64) -> Self {
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        Self { year: yoe + era * 400 + i64::from(month <= 2), month, day }
    }

    /// `date.isoformat()`.
    fn iso(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// `datetime.isoformat()` of a UTC instant at a whole second.
fn utc_iso(utc_s: i64) -> String {
    let date = Date::of_days(utc_s.div_euclid(DAY_SECONDS));
    let second = utc_s.rem_euclid(DAY_SECONDS);
    format!("{}T{:02}:{:02}:{:02}+00:00", date.iso(), second / 3_600, second / 60 % 60, second % 60)
}

/// What the report is built from.
pub struct ReportInputs<'a> {
    pub starting: f64,
    pub risk_free_rate: f64,
    pub timezone: &'a str,
    pub clock: &'a LocalClock,
    /// The daily curve: each mark's instant in nanoseconds, and its value.
    pub day_ns: &'a [i64],
    pub day_equity: &'a [f64],
    /// Every trade row, in `Trade.realized_order_key` order.
    pub trades: &'a [TradeRow],
    pub fills: &'a FillAmounts,
}

/// `PerformanceReportService.calculate`, its `execution_lane` left None for
/// the caller to stamp. None where the Python would raise, sort the curve
/// first, or take the slow loop of `_Drawdown.of`.
pub fn calculate(inputs: &ReportInputs<'_>) -> Option<Value> {
    let equity = inputs.day_equity;
    let starting = inputs.starting;
    if equity.is_empty()
        || equity.len() != inputs.day_ns.len()
        || !(starting > 0.0 && starting.is_finite())
        || equity.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let mut stamps = Vec::with_capacity(inputs.day_ns.len());
    for &ns in inputs.day_ns {
        if ns % NS_PER_SECOND != 0 || stamps.last().is_some_and(|&last| last > ns / NS_PER_SECOND) {
            return None;
        }
        stamps.push(ns / NS_PER_SECOND);
    }
    let (start_s, end_s) = (stamps[0], stamps[stamps.len() - 1]);
    let elapsed = positive((end_s - start_s) as f64);
    let years = elapsed / SECONDS_PER_YEAR;
    let elapsed_months = elapsed / MONTH_SECONDS;
    let drawdown = Drawdown::of(equity, starting);
    let calendar = Calendar::of(&stamps, equity, inputs.clock, starting)?;

    let mut metrics = return_metrics(equity, starting, years, inputs.risk_free_rate, &drawdown)?;
    metrics.extend(trade_metrics(inputs.trades, elapsed_months)?);
    metrics.extend(turnover_metrics(inputs, years, start_s, end_s)?);
    metrics.extend(month_metrics(&calendar, starting)?);
    metrics.extend(year_metrics(&calendar.yearly));

    let first_active = match inputs.trades.iter().map(|trade| trade.opened_s).min() {
        Some(opened) => {
            let date = inputs.clock.date(opened)?;
            Value::Text(format!("{:04}-{:02}", date.year, date.month))
        }
        None => Value::None,
    };
    let dates = &calendar.local_dates;
    let recovery = drawdown.recovery_index;
    let report = vec![
        entry("performance_schema", Value::Text(SCHEMA.to_string())),
        entry("execution_lane", Value::None),
        entry("raptor_commit", Value::None),
        entry("raptor_source_path", Value::None),
        entry("raptor_version", Value::None),
        entry("timezone", Value::Text(inputs.timezone.to_string())),
        entry("effective_start_timestamp", Value::Text(utc_iso(start_s))),
        entry("effective_end_timestamp", Value::Text(utc_iso(end_s))),
        entry("risk_free_rate", Value::Float(inputs.risk_free_rate)),
        entry("metrics", Value::Map(metrics)),
        entry(
            "monthly_returns_pct",
            Value::Map(
                calendar
                    .monthly
                    .iter()
                    .map(|(key, value)| (month_key(*key), Value::float(*value)))
                    .collect(),
            ),
        ),
        entry(
            "yearly_returns_pct",
            Value::Map(
                calendar
                    .yearly
                    .iter()
                    .map(|&(year, value)| (year_key(year), Value::Float(value)))
                    .collect(),
            ),
        ),
        entry(
            "partial_years",
            Value::Map(
                calendar
                    .partial_years
                    .iter()
                    .map(|&(year, partial)| (year.to_string(), Value::Bool(partial)))
                    .collect(),
            ),
        ),
        entry("first_active_month", first_active),
        entry(
            "drawdown",
            Value::Map(vec![
                entry(
                    "deepest_peak_date",
                    drawdown
                        .deepest_peak_index
                        .map_or(Value::None, |index| Value::Text(dates[index].iso())),
                ),
                entry("deepest_trough_date", Value::Text(dates[drawdown.trough_index].iso())),
                entry(
                    "recovery_date",
                    recovery.map_or(Value::None, |index| Value::Text(dates[index].iso())),
                ),
                entry(
                    "recovery_duration_trading_days",
                    recovery.map_or(Value::None, |index| {
                        Value::Int(index as i64 - drawdown.trough_index as i64)
                    }),
                ),
                entry(
                    "unrecovered_at_end",
                    Value::Bool(drawdown.max_drawdown < 0.0 && recovery.is_none()),
                ),
            ]),
        ),
    ];
    Some(Value::Map(report))
}

fn entry(key: &str, value: Value) -> (String, Value) {
    (key.to_string(), value)
}

fn month_key((year, month): (i64, u32)) -> String {
    format!("{year:04}-{month:02}")
}

fn year_key(year: i64) -> String {
    format!("{year:04}")
}

/// `max(0.0, value)`.
fn positive(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// `_Drawdown._of_array`: the deepest fall from the running high, which
/// starts at the funded capital.
struct Drawdown {
    max_drawdown: f64,
    trough_index: usize,
    deepest_peak_index: Option<usize>,
    recovery_index: Option<usize>,
    underwater_days: usize,
}

impl Drawdown {
    fn of(equity: &[f64], starting: f64) -> Self {
        let mut mark = starting;
        let mut drawdowns = Vec::with_capacity(equity.len());
        let mut is_peak = Vec::with_capacity(equity.len());
        for &value in equity {
            is_peak.push(value >= mark);
            // `np.maximum`: the earlier on a tie, and a NaN carried on.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(mark >= value) {
                mark = value;
            }
            drawdowns.push(value / mark - 1.0);
        }
        // `np.argmin`: the first of the deepest.
        let mut trough_index = 0;
        for (index, &value) in drawdowns.iter().enumerate() {
            if value < drawdowns[trough_index] {
                trough_index = index;
            }
        }
        let deepest_peak_index = is_peak[..=trough_index].iter().rposition(|&peak| peak);
        let recovery_index = drawdowns[trough_index + 1..]
            .iter()
            .position(|&value| value >= RECOVERED)
            .map(|offset| trough_index + 1 + offset);
        Self {
            max_drawdown: drawdowns[trough_index],
            trough_index,
            deepest_peak_index,
            recovery_index,
            underwater_days: max_streak(drawdowns.iter().map(|&value| value < RECOVERED)),
        }
    }
}

/// `_max_streak`: the longest run of trues.
fn max_streak(flags: impl IntoIterator<Item = bool>) -> usize {
    let (mut longest, mut current) = (0, 0);
    for flag in flags {
        if flag {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

type Month = (i64, u32);

/// `_Calendar`: the curve on the local calendar.
struct Calendar {
    local_dates: Vec<Date>,
    /// `month_end_marks`, in the order each month first appears.
    month_ends: Vec<(Month, f64)>,
    monthly: Vec<(Month, Option<f64>)>,
    yearly: Vec<(i64, f64)>,
    partial_years: Vec<(i64, bool)>,
}

impl Calendar {
    fn of(stamps: &[i64], equity: &[f64], clock: &LocalClock, starting: f64) -> Option<Self> {
        let local_dates =
            stamps.iter().map(|&stamp| clock.close_date(stamp)).collect::<Option<Vec<_>>>()?;
        let mut month_ends: Vec<(Month, f64)> = Vec::new();
        for (date, &value) in local_dates.iter().zip(equity) {
            let month = (date.year, date.month);
            match month_ends.iter_mut().rev().find(|(key, _)| *key == month) {
                Some(slot) => slot.1 = value,
                None => month_ends.push((month, value)),
            }
        }
        let monthly = monthly_returns_pct(&month_ends, starting);
        let yearly = yearly_returns_pct(&monthly);
        let partial_years =
            partial_years(&monthly, local_dates[0], local_dates[local_dates.len() - 1]);
        Some(Self { local_dates, month_ends, monthly, yearly, partial_years })
    }
}

/// `month_range`: every month from `first` through `last`.
fn month_range(first: Month, last: Month) -> impl Iterator<Item = Month> {
    std::iter::successors(Some(first), |&(year, month)| {
        Some(if month == 12 { (year + 1, 1) } else { (year, month + 1) })
    })
    .take_while(move |&month| month <= last)
}

/// `calendar_returns.monthly_returns_pct`.
fn monthly_returns_pct(month_ends: &[(Month, f64)], starting: f64) -> Vec<(Month, Option<f64>)> {
    let (Some(first), Some(last)) =
        (month_ends.iter().map(|e| e.0).min(), month_ends.iter().map(|e| e.0).max())
    else {
        return Vec::new();
    };
    let mut prior = starting;
    month_range(first, last)
        .map(|month| match month_ends.iter().find(|(key, _)| *key == month) {
            None => (month, None),
            Some(&(_, value)) => {
                let change = (prior > 0.0).then(|| (value / prior - 1.0) * 100.0);
                prior = value;
                (month, change)
            }
        })
        .collect()
}

/// `calendar_returns.yearly_returns_pct`: each year's observed months
/// compounded (`math.prod`, a left fold from 1).
fn yearly_returns_pct(monthly: &[(Month, Option<f64>)]) -> Vec<(i64, f64)> {
    let mut growth: Vec<(i64, f64)> = Vec::new();
    for &((year, _), value) in monthly {
        let Some(value) = value else { continue };
        let factor = 1.0 + value / 100.0;
        match growth.last_mut() {
            Some(last) if last.0 == year => last.1 *= factor,
            _ => growth.push((year, factor)),
        }
    }
    growth.into_iter().map(|(year, product)| (year, (product - 1.0) * 100.0)).collect()
}

/// `calendar_returns.partial_years`.
fn partial_years(monthly: &[(Month, Option<f64>)], first: Date, last: Date) -> Vec<(i64, bool)> {
    (first.year..=last.year)
        .map(|year| {
            let missing = (1..=12).any(|month| {
                !monthly.iter().any(|&(key, value)| key == (year, month) && value.is_some())
            });
            let partial = missing
                || (year == first.year && (first.month, first.day) != (1, 1))
                || (year == last.year && (last.month, last.day) != (12, 31));
            (year, partial)
        })
        .collect()
}

/// `DailyRiskRatios`.
struct Ratios {
    sharpe: Option<f64>,
    sortino: Option<f64>,
    annual_volatility: Option<f64>,
    periods_per_year: Option<f64>,
}

/// `daily_risk_ratios`, on a curve whose returns are all finite.
fn daily_risk_ratios(
    equity: &[f64],
    starting: f64,
    years: f64,
    risk_free_rate: f64,
) -> Option<Ratios> {
    let periods_per_year = (years > 0.0).then(|| equity.len() as f64 / years);
    let mut returns = Vec::with_capacity(equity.len());
    let mut previous = starting;
    for &value in equity {
        returns.push(if previous != 0.0 { value / previous - 1.0 } else { 0.0 });
        previous = value;
    }
    if returns.iter().any(|value| !value.is_finite()) {
        // `_stdev` would ask `statistics.stdev` instead.
        return None;
    }
    let mean = if returns.is_empty() {
        0.0
    } else {
        fsum(returns.iter().copied())? / returns.len() as f64
    };
    let deviation = if returns.len() > 1 { Some(stdev(&returns)?) } else { None };
    let annual_volatility = match (deviation, periods_per_year) {
        (Some(deviation), Some(periods)) => Some(deviation * periods.sqrt()),
        _ => None,
    };
    let floor = match periods_per_year {
        Some(periods) if periods != 0.0 && risk_free_rate > -1.0 => {
            py_pow(1.0 + risk_free_rate, 1.0 / periods).ok()? - 1.0
        }
        _ => 0.0,
    };
    let sharpe = match (deviation, periods_per_year) {
        (Some(deviation), Some(periods)) if deviation != 0.0 && periods != 0.0 => {
            Some((mean - floor) / deviation * periods.sqrt())
        }
        _ => None,
    };
    let downside =
        if returns.is_empty() { 0.0 } else { mean_square_shortfall(&returns, floor)?.sqrt() };
    let sortino = match periods_per_year {
        Some(periods) if downside > 0.0 && periods != 0.0 => {
            Some((mean - floor) / downside * periods.sqrt())
        }
        _ => None,
    };
    Some(Ratios {
        sharpe: finite(sharpe),
        sortino: finite(sortino),
        annual_volatility: finite(annual_volatility),
        periods_per_year,
    })
}

/// `_mean_square_shortfall`: each gap not above the floor squared as Python
/// squares a float (`pow`), `fsum`med, over every day.
fn mean_square_shortfall(returns: &[f64], floor: f64) -> Option<f64> {
    let mut squares = Vec::new();
    for &value in returns {
        let gap = value - floor;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(gap > 0.0) {
            squares.push(py_pow(gap, 2.0).ok()?);
        }
    }
    Some(fsum(squares)? / returns.len() as f64)
}

/// `compound_annual_growth`; the outer None where Python's answer is not
/// transcribed.
fn compound_annual_growth(starting: f64, final_value: f64, years: f64) -> Option<Option<f64>> {
    if !(years > 0.0 && starting > 0.0 && final_value > 0.0) {
        return Some(None);
    }
    match py_pow(final_value / starting, 1.0 / years) {
        Ok(power) => {
            let growth = power - 1.0;
            Some(growth.is_finite().then_some(growth))
        }
        Err(PowError::Overflow) => Some(None),
        Err(PowError::Unsupported) => None,
    }
}

fn return_metrics(
    equity: &[f64],
    starting: f64,
    years: f64,
    risk_free_rate: f64,
    drawdown: &Drawdown,
) -> Option<Vec<(String, Value)>> {
    let final_value = equity[equity.len() - 1];
    let ratios = daily_risk_ratios(equity, starting, years, risk_free_rate)?;
    let cagr = compound_annual_growth(starting, final_value, years)?;
    let max_drawdown = drawdown.max_drawdown;
    Some(vec![
        entry("starting_equity", Value::finite(Some(starting))),
        entry("final_equity", Value::finite(Some(final_value))),
        entry(
            "total_return_pct",
            Value::finite((starting != 0.0).then(|| (final_value / starting - 1.0) * 100.0)),
        ),
        entry("cagr_pct", Value::finite(cagr.map(|growth| growth * 100.0))),
        entry(
            "annualized_volatility_pct",
            Value::finite(ratios.annual_volatility.map(|volatility| volatility * 100.0)),
        ),
        entry("sharpe_ratio", Value::finite(ratios.sharpe)),
        entry("sortino_ratio", Value::finite(ratios.sortino)),
        entry(
            "calmar_ratio",
            Value::finite(
                cagr.filter(|_| max_drawdown < 0.0).map(|growth| growth / max_drawdown.abs()),
            ),
        ),
        entry("max_drawdown_pct", Value::finite(Some(max_drawdown * 100.0))),
        entry("max_drawdown_duration_trading_days", Value::int(drawdown.underwater_days)),
        entry("observed_trading_days", Value::int(equity.len())),
        entry("observed_trading_days_per_year", Value::finite(ratios.periods_per_year)),
    ])
}

fn trade_metrics(trades: &[TradeRow], elapsed_months: f64) -> Option<Vec<(String, Value)>> {
    let pnls: Vec<f64> = trades.iter().map(|trade| trade.realized).collect();
    let wins: Vec<f64> = pnls.iter().copied().filter(|&value| value > FLAT).collect();
    let losses: Vec<f64> = pnls.iter().copied().filter(|&value| value < -FLAT).collect();
    let average_win = if wins.is_empty() { None } else { Some(fmean(&wins)?) };
    let average_loss = if losses.is_empty() { None } else { Some(fmean(&losses)?) };
    let durations: Vec<f64> = trades.iter().filter_map(|trade| trade.duration_hours).collect();
    let loss_total = py_sum(&losses);
    let count = pnls.len();
    Some(vec![
        entry("closed_trades", Value::int(count)),
        entry("winning_trades", Value::int(wins.len())),
        entry("losing_trades", Value::int(losses.len())),
        entry("breakeven_trades", Value::int(count - wins.len() - losses.len())),
        entry(
            "trades_per_elapsed_month",
            Value::finite((elapsed_months > 0.0).then(|| count as f64 / elapsed_months)),
        ),
        entry(
            "win_rate_pct",
            Value::finite(Some(if count > 0 {
                wins.len() as f64 / count as f64 * 100.0
            } else {
                0.0
            })),
        ),
        entry("average_win", Value::finite(average_win)),
        entry("average_loss", Value::finite(average_loss)),
        entry(
            "payoff_ratio",
            Value::finite(match (average_win, average_loss) {
                (Some(win), Some(loss)) if loss != 0.0 => Some(win / loss.abs()),
                _ => None,
            }),
        ),
        entry(
            "profit_factor",
            Value::finite(
                (!losses.is_empty() && loss_total != 0.0).then(|| py_sum(&wins) / loss_total.abs()),
            ),
        ),
        entry(
            "expectancy_per_closed_trade",
            Value::finite(Some(if count > 0 { fmean(&pnls)? } else { 0.0 })),
        ),
        entry("best_trade", Value::finite(py_max(&pnls))),
        entry("worst_trade", Value::finite(py_min(&pnls))),
        entry(
            "average_trade_duration_hours",
            Value::finite(if durations.is_empty() { None } else { Some(fmean(&durations)?) }),
        ),
        entry(
            "max_consecutive_losses",
            Value::int(max_streak(pnls.iter().map(|&value| value < -FLAT))),
        ),
    ])
}

fn turnover_metrics(
    inputs: &ReportInputs<'_>,
    years: f64,
    start_s: i64,
    end_s: i64,
) -> Option<Vec<(String, Value)>> {
    let fills = inputs.fills;
    if fills.quantity.len() != fills.price.len() || fills.commission.len() != fills.price.len() {
        return None;
    }
    let notionals: Vec<f64> =
        fills.quantity.iter().zip(&fills.price).map(|(q, p)| q.abs() * p.abs()).collect();
    let total_notional = np_sum(&notionals);
    let average_equity = fmean(inputs.day_equity)?;
    Some(vec![
        entry("total_fees", Value::finite(Some(np_sum(&fills.commission)))),
        entry("total_traded_notional", Value::finite(Some(total_notional))),
        entry(
            "annualized_turnover",
            Value::finite(
                (average_equity > 0.0 && years > 0.0)
                    .then(|| total_notional / average_equity / years),
            ),
        ),
        entry(
            "exposure_pct",
            Value::finite(Some(trade_exposure_pct(inputs.trades, start_s, end_s))),
        ),
        entry("average_leverage", Value::None),
        entry("maximum_leverage", Value::None),
    ])
}

/// `_trade_exposure_pct`: the share of the curve's span some trade was
/// open, overlapping trades counted once.
fn trade_exposure_pct(trades: &[TradeRow], start_s: i64, end_s: i64) -> f64 {
    let mut intervals: Vec<(i64, i64)> = trades
        .iter()
        .map(|trade| (trade.opened_s.max(start_s), trade.closed_s.unwrap_or(end_s).min(end_s)))
        .filter(|(opened, closed)| closed > opened)
        .collect();
    if intervals.is_empty() || end_s <= start_s {
        return 0.0;
    }
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(intervals.len());
    for (opened, closed) in intervals {
        match merged.last_mut() {
            Some(last) if opened <= last.1 => last.1 = last.1.max(closed),
            _ => merged.push((opened, closed)),
        }
    }
    let spans: Vec<f64> = merged.iter().map(|(opened, closed)| (closed - opened) as f64).collect();
    100.0 * py_sum(&spans) / (end_s - start_s) as f64
}

fn month_metrics(calendar: &Calendar, starting: f64) -> Option<Vec<(String, Value)>> {
    let numeric: Vec<f64> = calendar.monthly.iter().filter_map(|&(_, value)| value).collect();
    let positive: Vec<f64> = numeric.iter().copied().filter(|&value| value > FLAT).collect();
    let flat = numeric.iter().filter(|value| value.abs() <= FLAT).count();
    let share = |part: usize| {
        if numeric.is_empty() {
            0.0
        } else {
            part as f64 / numeric.len() as f64 * 100.0
        }
    };
    let positive_total = py_sum(&positive);
    let top_three = if !positive.is_empty() && positive_total != 0.0 {
        let mut ranked = positive.clone();
        // `sorted(..., reverse=True)`: largest first, ties as they came.
        ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        ranked.truncate(3);
        Some(py_sum(&ranked) / positive_total * 100.0)
    } else {
        None
    };
    Some(vec![
        entry("positive_month_count", Value::int(positive.len())),
        entry("positive_month_pct", Value::finite(Some(share(positive.len())))),
        entry("flat_month_count", Value::int(flat)),
        entry("flat_month_pct", Value::finite(Some(share(flat)))),
        entry(
            "mean_monthly_return_pct",
            Value::finite(if numeric.is_empty() { None } else { Some(fmean(&numeric)?) }),
        ),
        entry(
            "median_monthly_return_pct",
            Value::finite(if numeric.is_empty() { None } else { Some(median(&numeric)?) }),
        ),
        entry("best_month_return_pct", Value::finite(py_max(&numeric))),
        entry("worst_month_return_pct", Value::finite(py_min(&numeric))),
        entry(
            "longest_negative_month_streak",
            Value::int(max_streak(
                calendar.monthly.iter().map(|&(_, value)| value.is_some_and(|value| value < 0.0)),
            )),
        ),
        entry(
            "longest_month_end_underwater_period",
            Value::int(month_underwater(&calendar.month_ends, starting)),
        ),
        entry("top_three_positive_month_contribution_share_pct", Value::finite(top_three)),
    ])
}

/// `_month_underwater`: the longest run of month ends below their high.
fn month_underwater(month_ends: &[(Month, f64)], starting: f64) -> usize {
    let mut ordered = month_ends.to_vec();
    ordered.sort_by_key(|&(month, _)| month);
    let mut peak = starting;
    let (mut longest, mut current) = (0, 0);
    for (_, value) in ordered {
        if value >= peak {
            peak = value;
            current = 0;
        } else {
            current += 1;
            longest = longest.max(current);
        }
    }
    longest
}

fn year_metrics(yearly: &[(i64, f64)]) -> Vec<(String, Value)> {
    let values: Vec<f64> = yearly.iter().map(|&(_, value)| value).collect();
    // `min(yearly, key=...)`: the first of the worst.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let worst = yearly.iter().fold(None, |worst: Option<(i64, f64)>, &(year, value)| match worst {
        Some((_, least)) if !(value < least) => worst,
        _ => Some((year, value)),
    });
    vec![
        entry("positive_year_count", Value::int(values.iter().filter(|&&v| v > 0.0).count())),
        entry("negative_year_count", Value::int(values.iter().filter(|&&v| v < 0.0).count())),
        entry(
            "worst_calendar_year",
            worst.map_or(Value::None, |(year, _)| Value::Text(year_key(year))),
        ),
        entry("worst_calendar_year_return_pct", Value::finite(py_min(&values))),
        entry(
            "return_2022_pct",
            Value::finite(yearly.iter().find(|&&(year, _)| year == 2022).map(|&(_, v)| v)),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc() -> LocalClock {
        LocalClock::new(&[(0, 0)], i64::MAX).unwrap()
    }

    #[test]
    fn dates_are_civil() {
        assert_eq!(Date::of_days(0).iso(), "1970-01-01");
        assert_eq!(Date::of_days(19_052).iso(), "2022-03-01");
        assert_eq!(Date::of_days(-1).iso(), "1969-12-31");
        assert_eq!(utc_iso(1_640_995_200 + 3_661), "2022-01-01T01:01:01+00:00");
    }

    #[test]
    fn midnight_closes_the_day_before() {
        let clock = utc();
        assert_eq!(clock.close_date(1_640_995_200).unwrap().iso(), "2021-12-31");
        assert_eq!(clock.close_date(1_640_995_201).unwrap().iso(), "2022-01-01");
        // Sydney at +11: 13:00 UTC is local midnight.
        let sydney = LocalClock::new(&[(0, 39_600 * NS_PER_SECOND)], i64::MAX).unwrap();
        assert_eq!(sydney.close_date(1_640_955_600).unwrap().iso(), "2021-12-31");
        assert!(LocalClock::new(&[(10, 0)], 0).is_none());
    }

    #[test]
    fn drawdown_starts_at_the_funded_capital() {
        let dd = Drawdown::of(&[90.0, 110.0, 99.0, 110.0], 100.0);
        assert_eq!(dd.trough_index, 0);
        assert_eq!(dd.deepest_peak_index, None);
        assert_eq!(dd.recovery_index, Some(1));
        assert_eq!(dd.underwater_days, 1);
    }

    #[test]
    fn months_chain_through_gaps() {
        let monthly = monthly_returns_pct(&[((2022, 1), 110.0), ((2022, 3), 121.0)], 100.0);
        assert_eq!(monthly.len(), 3);
        assert_eq!(monthly[1], ((2022, 2), None));
        assert!(monthly[2].1.is_some());
        let yearly = yearly_returns_pct(&monthly);
        assert_eq!(yearly.len(), 1);
    }

    #[test]
    fn exposure_merges_overlaps() {
        let rows = [
            TradeRow { opened_s: 0, closed_s: Some(50), realized: 1.0, duration_hours: None },
            TradeRow { opened_s: 25, closed_s: Some(75), realized: 1.0, duration_hours: None },
        ];
        assert_eq!(trade_exposure_pct(&rows, 0, 100), 75.0);
    }

    #[test]
    fn a_run_reports_every_section() {
        let day = 86_400 * NS_PER_SECOND;
        let start = 1_640_995_200 * NS_PER_SECOND + day;
        let day_ns: Vec<i64> = (0..40).map(|i| start + i * day).collect();
        let day_equity: Vec<f64> = (0..40).map(|i| 100_000.0 + f64::from(i) * 10.0).collect();
        let clock = utc();
        let fills = FillAmounts::default();
        let report = calculate(&ReportInputs {
            starting: 100_000.0,
            risk_free_rate: 0.0,
            timezone: "UTC",
            clock: &clock,
            day_ns: &day_ns,
            day_equity: &day_equity,
            trades: &[],
            fills: &fills,
        })
        .unwrap();
        let metrics = report.get("metrics").unwrap();
        assert_eq!(metrics.get("observed_trading_days"), Some(&Value::Int(40)));
        assert_eq!(metrics.get("closed_trades"), Some(&Value::Int(0)));
        assert_eq!(report.get("first_active_month"), Some(&Value::None));
        assert_eq!(
            report.get("effective_start_timestamp"),
            Some(&Value::Text("2022-01-02T00:00:00+00:00".into()))
        );
    }
}
