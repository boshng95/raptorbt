//! A short node's borrow, charged from its fills: the fee
//! `build_backtest_financing_report` works out, and what
//! `apply_short_financing` does with it to the daily curve and end value.
//!
//! The fees are whole units of `BORROW_FEE_SCALE` (10**-10 of the
//! currency): one share's fee a day is the caller's schedule
//! (`BorrowFeeSchedule.units_per_share`), the shares a day are what the
//! fills left short as of their settlement, and every sum is exact. Each
//! figure Python turns into a double -- the total, and the running total at
//! each fee day -- is one correctly rounded conversion of that exact sum,
//! as `float(Decimal)` is.

/// One run's borrow terms: its instrument's schedule and the dates its
/// bars fall on, in the accounting profile's market.
pub struct Schedule<'a> {
    /// Per bar, its market date, as an ordinal.
    pub dates: &'a [i32],
    /// Per bar, the date a trade on its market date settles on
    /// (`add_business_days(trade_date, settlement_sessions_for_trade(..))`).
    pub settles: &'a [i32],
    /// Whether the instrument has a close to mark a loan at; a short it
    /// cannot mark is not charged.
    pub priced: bool,
    /// The schedule's first day and the run's end date.
    pub first_day: i32,
    pub end_day: i32,
    /// One share's fee on each day from `first_day` through `end_day`, 0
    /// where the day is not charged.
    pub units: &'a [i64],
    /// The end date's last instant in the profile's zone, in UTC
    /// nanoseconds: where a fee the curve has no point to post on lands.
    pub terminal_ns: i64,
}

impl Schedule<'_> {
    /// Whether the arrays fit `bars` bars and the schedule's own days.
    pub fn fits(&self, bars: usize) -> bool {
        let days = i64::from(self.end_day) - i64::from(self.first_day) + 1;
        self.dates.len() == bars
            && self.settles.len() == bars
            && self.units.len() as i64 == days.max(0)
    }
}

/// The double nearest `units` 10**-10ths of the currency.
fn to_double(units: i128) -> f64 {
    format!("{units}e-10").parse().unwrap_or(f64::NAN)
}

/// Each day's fee, in units, on the days that cost anything: the shares
/// short as of each settlement date, charged every calendar day from the
/// first settlement through the end date. `deltas` is what each fill on
/// its bar changed of the shares held short, in fill order.
///
/// A day the schedule leaves uncharged costs nothing, which leaves every
/// running total as a day posted at zero would.
pub fn daily_fees(schedule: &Schedule<'_>, deltas: &[(usize, i64)]) -> Vec<(i32, i128)> {
    if deltas.is_empty() || !schedule.priced {
        return Vec::new();
    }
    let mut settling: Vec<(i32, i64)> =
        deltas.iter().map(|&(bar, delta)| (schedule.settles[bar], delta)).collect();
    settling.sort_by_key(|&(day, _)| day);
    let mut fees = Vec::new();
    let (mut shares, mut next) = (0_i64, 0);
    let mut day = settling[0].0;
    while day <= schedule.end_day {
        while let Some(&(_, delta)) = settling.get(next).filter(|&&(settles, _)| settles == day) {
            shares += delta;
            next += 1;
        }
        shares = shares.max(0);
        if shares == 0 {
            // Nothing is charged until the next settlement.
            match settling.get(next) {
                Some(&(settles, _)) => {
                    day = settles;
                    continue;
                }
                None => break,
            }
        }
        if day >= schedule.first_day {
            let units = schedule.units[(day - schedule.first_day) as usize];
            if units != 0 {
                fees.push((day, i128::from(units) * i128::from(shares)));
            }
        }
        day += 1;
    }
    fees
}

/// Charge `fees` to the daily curve -- `day_bars` the bars its points were
/// marked on -- as `apply_short_financing` does, and return the borrow fee.
///
/// A fee day's amount posts at the last point of its market date, and at
/// every point of a later one. Where the last point has not taken the whole
/// fee -- a short held over days with no bar -- a terminal point carries
/// the financed end value, at the end date's last instant or just after the
/// last point, in microseconds.
pub fn charge(
    schedule: &Schedule<'_>,
    fees: &[(i32, i128)],
    day_bars: &[usize],
    day_ns: &mut Vec<i64>,
    day_equity: &mut Vec<f64>,
    end_value: f64,
) -> f64 {
    if fees.is_empty() {
        return 0.0;
    }
    let total: i128 = fees.iter().map(|&(_, fee)| fee).sum();
    let borrow_fee = to_double(total);
    let mut running = 0_i128;
    let cumulative: Vec<f64> = fees
        .iter()
        .map(|&(_, fee)| {
            running += fee;
            to_double(running)
        })
        .collect();
    let date_of = |point: usize| schedule.dates[day_bars[point]];
    let mut most = 0.0_f64;
    for point in 0..day_equity.len() {
        let date = date_of(point);
        let mut posted = fees.partition_point(|&(day, _)| day <= date);
        if posted > 0
            && fees[posted - 1].0 == date
            && point + 1 < day_equity.len()
            && date_of(point + 1) == date
        {
            // Not the date's last point: its own fee posts at the close.
            posted -= 1;
        }
        let accrued = if posted > 0 { cumulative[posted - 1] } else { 0.0 };
        day_equity[point] -= accrued;
        // Python's `max`, which keeps the first of equals.
        if accrued > most {
            most = accrued;
        }
    }
    if most + 1e-12 < borrow_fee {
        let last = day_ns.last().copied().unwrap_or(schedule.terminal_ns);
        let at = schedule.terminal_ns.max(last + 1);
        day_ns.push(at - at.rem_euclid(1_000));
        day_equity.push(end_value - borrow_fee);
    }
    borrow_fee
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_NS: i64 = 86_400 * 1_000_000_000;

    /// Days 100..=109, bars at noon, one per day; a trade settles two days on.
    struct Fixture {
        dates: Vec<i32>,
        settles: Vec<i32>,
        units: Vec<i64>,
    }

    fn fixture() -> Fixture {
        let dates: Vec<i32> = (100..110).collect();
        let settles = dates.iter().map(|day| day + 2).collect();
        // The schedule starts on day 101, the day after the first close.
        let units = (101..110).map(|day| i64::from(day) * 1_000).collect();
        Fixture { dates, settles, units }
    }

    fn schedule(f: &Fixture) -> Schedule<'_> {
        Schedule {
            dates: &f.dates,
            settles: &f.settles,
            priced: true,
            first_day: 101,
            end_day: 109,
            units: &f.units,
            terminal_ns: 110 * DAY_NS - 1,
        }
    }

    #[test]
    fn fees_charge_settled_shares_each_day() {
        let f = fixture();
        let s = schedule(&f);
        assert!(s.fits(10));
        // Short 10 on bar 1 (settles day 103), 5 more on bar 3 (day 105),
        // covered on bar 5 (day 107).
        let fees = daily_fees(&s, &[(1, 10), (3, 5), (5, -15)]);
        let expect: Vec<(i32, i128)> = vec![
            (103, 103_000 * 10),
            (104, 104_000 * 10),
            (105, 105_000 * 15),
            (106, 106_000 * 15),
        ];
        assert_eq!(fees, expect);
    }

    #[test]
    fn a_cover_settling_first_is_clamped_at_flat() {
        let f = fixture();
        let s = schedule(&f);
        // A cover whose settlement precedes its short's leaves nothing short
        // until the short settles.
        let settles: Vec<i32> = vec![100, 101, 108, 103, 104, 105, 106, 107, 108, 109];
        let s = Schedule { settles: &settles, ..s };
        let fees = daily_fees(&s, &[(2, 4), (1, -4)]);
        assert_eq!(fees, vec![(108, 108_000 * 4), (109, 109_000 * 4)]);
    }

    #[test]
    fn nothing_is_charged_before_the_schedule_or_unpriced() {
        let f = fixture();
        let s = schedule(&f);
        let settles: Vec<i32> = vec![99; 10];
        let early = Schedule { settles: &settles, ..schedule(&f) };
        let fees = daily_fees(&early, &[(0, 1)]);
        assert_eq!(fees.first(), Some(&(101, 101_000)));
        assert_eq!(fees.len(), 9);
        let unpriced = Schedule { priced: false, ..s };
        assert!(daily_fees(&unpriced, &[(0, 1)]).is_empty());
    }

    #[test]
    fn the_curve_posts_each_fee_at_its_dates_close() {
        let f = fixture();
        let s = schedule(&f);
        let fees = vec![(103, 7_000_000_000_i128), (104, 3_000_000_000)];
        // Two points on day 103 (bars 3 and 4 both dated 103), one on 104.
        let dates = vec![100, 101, 102, 103, 103, 104, 105, 106, 107, 108];
        let s = Schedule { dates: &dates, ..s };
        let bars = vec![2, 3, 4, 5];
        let mut ns: Vec<i64> = bars.iter().map(|&b| b as i64 * DAY_NS).collect();
        let mut equity = vec![100.0; 4];
        let fee = charge(&s, &fees, &bars, &mut ns, &mut equity, 100.0);
        assert_eq!(fee, 1.0);
        assert_eq!(equity, vec![100.0, 100.0, 99.3, 99.0]);
        assert_eq!(ns.len(), 4, "the last point took the whole fee");
    }

    #[test]
    fn a_fee_past_the_last_point_lands_on_a_terminal_point() {
        let f = fixture();
        let s = schedule(&f);
        let fees = vec![(103, 5_000_000_000_i128), (108, 5_000_000_000)];
        let bars = vec![3, 4];
        let mut ns: Vec<i64> = bars.iter().map(|&b| b as i64 * DAY_NS).collect();
        let mut equity = vec![100.0, 101.0];
        let fee = charge(&s, &fees, &bars, &mut ns, &mut equity, 101.0);
        assert_eq!(fee, 1.0);
        assert_eq!(equity, vec![99.5, 100.5, 100.0]);
        // The end date's last instant, to the microsecond.
        assert_eq!(ns[2], 110 * DAY_NS - 1_000);
        // Past it, just after the last point.
        let late = Schedule { terminal_ns: 0, ..schedule(&f) };
        let mut ns: Vec<i64> = bars.iter().map(|&b| b as i64 * DAY_NS).collect();
        let mut equity = vec![100.0, 101.0];
        charge(&late, &fees, &bars, &mut ns, &mut equity, 101.0);
        assert_eq!(ns[2], ns[1]);
    }

    #[test]
    fn running_totals_round_once() {
        // 0.1 + 0.2 posted as exact tenths: each running total is the double
        // nearest the exact sum, not a sum of doubles.
        let fees = vec![(1, 1_000_000_000_i128), (2, 2_000_000_000)];
        let dates = vec![1, 2];
        let s = Schedule {
            dates: &dates,
            settles: &dates,
            priced: true,
            first_day: 1,
            end_day: 2,
            units: &[0, 0],
            terminal_ns: 0,
        };
        let mut ns = vec![0, DAY_NS];
        let mut equity = vec![1.0, 1.0];
        let fee = charge(&s, &fees, &[0, 1], &mut ns, &mut equity, 1.0);
        assert_eq!(fee, 0.3);
        assert_eq!(equity[1], 1.0 - 0.3);
        assert_ne!(0.1 + 0.2, 0.3);
    }
}
