//! The equity curve a run records: one sample per instant.
//!
//! An instant is a timestamp at which market data carrying a traded price
//! reaches the venue. Everything that happens at it -- every instrument's
//! print, the fills those prints cause, the orders the strategy places on
//! hearing them -- settles before the instant is over, and none of it is
//! finished when the instant's first print arrives. So an instant is sampled
//! when it closes: just before the first priced event of a later instant is
//! applied, or when the run finishes.
//!
//! Sampling any earlier records a book that never existed. A session marking
//! after each event of a multi-instrument instant prices the first name at
//! its new close and the rest one print stale; a runner marking straight
//! after its bar misses the fills its strategy's answer to that bar produced
//! moments later. Either sample can sit below every real one, and a drawdown
//! read off it is a drawdown of the bookkeeping order, not of the account.
//!
//! The reference engine closes an instant at the same moment -- Nautilus
//! samples from a simulation module's `pre_process`, which runs as each bar
//! reaches its venue and before it is matched -- so two engines that trade
//! alike report the same curve.
//!
//! Risk controls are not fed from here. A margin check or a drawdown
//! kill-switch must answer before the next event, so the drivers keep feeding
//! them per event; this records what the account was worth, not what a
//! control saw in the middle of an instant.
//!
//! A run whose caller reads neither the series nor the metrics computed from
//! them does not store them: the curve keeps only its latest sample, and the
//! end-of-day marks a daily report needs are taken as each instant closes --
//! the same samples, in the same order, a pass over the stored series reads.

use crate::core::performance::DailyPerformanceCollector;
use crate::core::types::DailyPerformance;

/// Streaming per-instant equity, drawdown and return series.
#[derive(Debug, Clone)]
pub(crate) struct InstantCurve {
    /// The instant still settling: the latest priced event applied so far.
    open: Option<i64>,
    /// High-water mark the drawdown is measured from; starts at the capital
    /// the run was funded with, so a run that opens with a loss is under
    /// water from its first sample.
    peak: f64,
    /// Whether the series are stored; off, only `last` is kept.
    record: bool,
    /// The latest sample committed.
    last: Option<f64>,
    /// End-of-day marks, observed as each instant is sampled.
    daily: Option<DailyPerformanceCollector>,
    equity: Vec<f64>,
    drawdown: Vec<f64>,
    returns: Vec<f64>,
    timestamps: Vec<i64>,
}

/// The finished series, index-aligned.
#[derive(Debug, Default)]
pub(crate) struct CurveParts {
    pub equity: Vec<f64>,
    pub drawdown: Vec<f64>,
    pub returns: Vec<f64>,
    pub timestamps: Vec<i64>,
}

impl InstantCurve {
    pub(crate) fn new(initial_capital: f64) -> Self {
        Self {
            open: None,
            peak: initial_capital,
            record: true,
            last: None,
            daily: None,
            equity: Vec::new(),
            drawdown: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Store the series only when `record`: a run that will read neither
    /// them nor metrics computed from them keeps just the latest sample.
    pub(crate) fn recording(mut self, record: bool) -> Self {
        self.record = record;
        self
    }

    /// Keep one end-of-day mark per exchange-local day, the day boundaries
    /// given as `(effective_utc_ns, offset_ns)` transitions; `None` keeps none.
    pub(crate) fn with_daily_performance(mut self, transitions: Option<Vec<(i64, i64)>>) -> Self {
        self.daily = transitions.map(DailyPerformanceCollector::new);
        self
    }

    /// Whether a priced event at `timestamp` belongs to a later instant than
    /// the one still settling -- in which case that one has finished and is
    /// [`Self::close`]d before the event is applied.
    #[inline]
    pub(crate) fn closes(&self, timestamp: i64) -> bool {
        self.open.is_some_and(|open| timestamp > open)
    }

    /// Sample the instant still settling, now that it has finished: at the
    /// next instant's arrival, or at the end of the run with its settled
    /// value. A no-op when nothing is open.
    pub(crate) fn close(&mut self, equity: f64) {
        if let Some(open) = self.open.take() {
            self.commit(open, equity);
        }
    }

    /// A priced event at `timestamp` has been applied: its instant is open.
    #[inline]
    pub(crate) fn open_at(&mut self, timestamp: i64) {
        self.open = Some(self.open.map_or(timestamp, |open| open.max(timestamp)));
    }

    /// The instant still settling, if any.
    #[inline]
    pub(crate) fn open_instant(&self) -> Option<i64> {
        self.open
    }

    /// Whether the series are stored.
    #[inline]
    pub(crate) fn records(&self) -> bool {
        self.record
    }

    /// [`Self::close`] each of the instants `timestamps[first..last]` at
    /// `equity(idx)`, as the bar after each arrives, and leave
    /// `timestamps[last]` open.
    ///
    /// For a curve that does not record, standing open at
    /// `timestamps[first]`, over rising timestamps. Of a day's samples the
    /// daily marks keep only the last, so only the samples either side of
    /// each day boundary are taken.
    pub(crate) fn close_run(
        &mut self,
        timestamps: &[i64],
        first: usize,
        last: usize,
        equity: impl Fn(usize) -> f64,
    ) {
        debug_assert!(!self.record && self.open == Some(timestamps[first]) && first < last);
        if let Some(daily) = &mut self.daily {
            let mut idx = first;
            // Where the day being walked began, and how many bars the one
            // before it held: evenly spaced bars put the next day's first bar
            // that far on, so each day of a long run is found a probe or two
            // from there rather than by a search across the rest of the run.
            let (mut day_start, mut span) = (first, 0);
            while idx < last {
                let until = daily.same_day_until();
                // Most runs end inside the day they started, which their
                // newest bar settles without a search.
                let next = if timestamps[last - 1] < until {
                    last
                } else {
                    first_at_or_after(timestamps, idx, last, day_start + span, until)
                };
                if next > idx {
                    daily.observe(timestamps[next - 1], equity(next - 1));
                }
                if next == last {
                    break;
                }
                daily.observe(timestamps[next], equity(next));
                span = next - day_start;
                day_start = next;
                idx = next + 1;
            }
        }
        self.last = Some(equity(last - 1));
        self.open = Some(timestamps[last]);
    }

    /// Whether any priced event has been applied yet.
    #[inline]
    pub(crate) fn has_started(&self) -> bool {
        self.open.is_some() || self.last.is_some()
    }

    /// The latest sample, or `None` before the first instant has closed.
    pub(crate) fn last(&self) -> Option<f64> {
        self.last
    }

    /// The end-of-day marks observed so far, when they are being kept.
    pub(crate) fn take_daily_performance(&mut self) -> Option<DailyPerformance> {
        self.daily.take().map(DailyPerformanceCollector::finish)
    }

    /// Samples committed so far.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.equity.len()
    }

    pub(crate) fn into_parts(self) -> CurveParts {
        CurveParts {
            equity: self.equity,
            drawdown: self.drawdown,
            returns: self.returns,
            timestamps: self.timestamps,
        }
    }

    fn commit(&mut self, timestamp: i64, equity: f64) {
        if let Some(daily) = &mut self.daily {
            daily.observe(timestamp, equity);
        }
        if self.record {
            let ret = match self.last {
                Some(previous) if previous != 0.0 => (equity - previous) / previous,
                _ => 0.0,
            };
            if equity > self.peak {
                self.peak = equity;
            }
            self.equity.push(equity);
            self.drawdown.push((self.peak - equity) / self.peak * 100.0);
            self.returns.push(ret);
            self.timestamps.push(timestamp);
        }
        self.last = Some(equity);
    }
}

/// The first of `timestamps[lo..hi]` at or after `until`, `hi` when none is:
/// `lo + timestamps[lo..hi].partition_point(|&ts| ts < until)` over rising
/// timestamps, searched outward from `guess` -- galloping away from it until
/// the answer is bracketed, then bisecting the bracket. A guess a few bars
/// off costs a few probes near it; any guess gives the same answer.
fn first_at_or_after(timestamps: &[i64], lo: usize, hi: usize, guess: usize, until: i64) -> usize {
    let guess = guess.clamp(lo, hi);
    // The answer lies in `from..=to`.
    let (mut from, mut to) = (lo, hi);
    if guess < hi && timestamps[guess] < until {
        from = guess + 1;
        let mut step = 1;
        while guess + step < hi {
            let probe = guess + step;
            if timestamps[probe] >= until {
                to = probe;
                break;
            }
            from = probe + 1;
            step *= 2;
        }
    } else {
        to = guess;
        let mut step = 1;
        while step <= guess - lo {
            let probe = guess - step;
            if timestamps[probe] < until {
                from = probe + 1;
                break;
            }
            to = probe;
            step *= 2;
        }
    }
    from + timestamps[from..to].partition_point(|&ts| ts < until)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guided_search_finds_what_bisection_finds() {
        // A small xorshift: rising timestamps with repeats and gaps, every
        // bound and every guess, against the plain bisection.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..80 {
            let len = (next() % 24) as usize;
            let mut timestamps = Vec::with_capacity(len);
            let mut ts = (next() % 7) as i64 - 3;
            for _ in 0..len {
                ts += [0, 1, 1, 2, 5, 60][(next() % 6) as usize];
                timestamps.push(ts);
            }
            for lo in 0..=len {
                for hi in lo..=len {
                    for until in [i64::MIN, -5, 0, 3, 7, 20, 60, 200, i64::MAX] {
                        let want = lo + timestamps[lo..hi].partition_point(|&t| t < until);
                        for guess in 0..=len + 2 {
                            assert_eq!(
                                first_at_or_after(&timestamps, lo, hi, guess, until),
                                want,
                                "{timestamps:?} lo {lo} hi {hi} guess {guess} until {until}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Drive `curve` the way both drivers do for one priced event.
    fn priced(curve: &mut InstantCurve, timestamp: i64, equity_before: f64) {
        if curve.closes(timestamp) {
            curve.close(equity_before);
        }
        curve.open_at(timestamp);
    }

    #[test]
    fn an_instant_is_sampled_once_when_the_next_one_arrives() {
        let mut curve = InstantCurve::new(100.0);
        assert!(!curve.closes(10), "nothing is open yet");
        priced(&mut curve, 10, f64::NAN);
        // A second print of the same instant samples nothing.
        assert!(!curve.closes(10));
        priced(&mut curve, 10, f64::NAN);
        priced(&mut curve, 20, 101.0);
        curve.close(99.0);
        let parts = curve.into_parts();
        assert_eq!(parts.timestamps, vec![10, 20]);
        assert_eq!(parts.equity, vec![101.0, 99.0]);
        assert_eq!(parts.drawdown[0], 0.0);
        assert!((parts.drawdown[1] - (2.0 / 101.0 * 100.0)).abs() < 1e-12);
        assert!((parts.returns[1] - (99.0 / 101.0 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn drawdown_is_measured_from_the_funded_capital() {
        let mut curve = InstantCurve::new(100.0);
        curve.open_at(1);
        curve.close(95.0);
        assert_eq!(curve.into_parts().drawdown, vec![5.0]);
    }

    #[test]
    fn a_run_with_no_priced_event_has_no_samples() {
        let mut curve = InstantCurve::new(100.0);
        assert!(!curve.has_started());
        curve.close(100.0);
        assert_eq!(curve.len(), 0);
    }

    #[test]
    fn an_earlier_print_does_not_reopen_a_closed_instant() {
        // A live feed delivers in arrival order; a late print is folded into
        // the instant already settling rather than rewinding it.
        let mut curve = InstantCurve::new(100.0);
        priced(&mut curve, 20, f64::NAN);
        priced(&mut curve, 15, f64::NAN);
        priced(&mut curve, 30, 102.0);
        curve.close(103.0);
        assert_eq!(curve.into_parts().timestamps, vec![20, 30]);
    }

    #[test]
    fn a_curve_that_does_not_record_keeps_its_last_sample_and_its_days() {
        const DAY: i64 = 86_400_000_000_000;
        let run = |record: bool| {
            let mut curve =
                InstantCurve::new(100.0).recording(record).with_daily_performance(Some(vec![]));
            assert!(!curve.has_started());
            priced(&mut curve, 10, f64::NAN);
            priced(&mut curve, DAY + 10, 101.0);
            priced(&mut curve, DAY + 20, 99.0);
            curve.close(102.0);
            assert!(curve.has_started());
            let last = curve.last();
            let daily = curve.take_daily_performance().expect("days are kept");
            (last, daily, curve.into_parts())
        };
        let (last, daily, parts) = run(true);
        assert_eq!(last, Some(102.0));
        assert_eq!(parts.equity, vec![101.0, 99.0, 102.0]);
        assert_eq!(daily.timestamps, vec![10, DAY + 20]);
        assert_eq!(daily.equity, vec![101.0, 102.0]);

        let (lean_last, lean_daily, lean_parts) = run(false);
        assert_eq!(lean_last, last);
        assert_eq!(lean_daily.timestamps, daily.timestamps);
        assert_eq!(lean_daily.equity, daily.equity);
        assert!(lean_parts.equity.is_empty() && lean_parts.timestamps.is_empty());
        assert!(lean_parts.drawdown.is_empty() && lean_parts.returns.is_empty());
    }
}
