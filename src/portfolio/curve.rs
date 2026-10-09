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

/// Streaming per-instant equity, drawdown and return series.
#[derive(Debug, Clone)]
pub(crate) struct InstantCurve {
    /// The instant still settling: the latest priced event applied so far.
    open: Option<i64>,
    /// High-water mark the drawdown is measured from; starts at the capital
    /// the run was funded with, so a run that opens with a loss is under
    /// water from its first sample.
    peak: f64,
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
            equity: Vec::new(),
            drawdown: Vec::new(),
            returns: Vec::new(),
            timestamps: Vec::new(),
        }
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

    /// Whether any priced event has been applied yet.
    #[inline]
    pub(crate) fn has_started(&self) -> bool {
        self.open.is_some() || !self.equity.is_empty()
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
        let ret = match self.equity.last() {
            Some(&previous) if previous != 0.0 => (equity - previous) / previous,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
