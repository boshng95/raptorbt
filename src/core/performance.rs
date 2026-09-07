//! Compact exchange-local end-of-day performance sampling.
//!
//! The execution engine still uses its full per-event curves to calculate its
//! legacy metrics.  This collector is deliberately separate: when enabled it
//! keeps only the last net-liquidation value observed in each local trading
//! day, so reporting memory grows with days rather than bars.

use crate::core::types::DailyPerformance;

const DAY_NS: i64 = 86_400_000_000_000;

/// Streaming local-day compactor for monotonically increasing UTC timestamps.
#[derive(Debug, Clone)]
pub struct DailyPerformanceCollector {
    transitions: Vec<(i64, i64)>,
    transition_idx: usize,
    next_transition_utc: i64,
    offset_ns: i64,
    last_local_day: Option<i64>,
    next_local_boundary_utc: i64,
    pending_timestamp: i64,
    pending_equity: f64,
    timestamps: Vec<i64>,
    equity: Vec<f64>,
}

impl DailyPerformanceCollector {
    /// Create a collector from `(effective_utc_ns, offset_ns)` transitions.
    ///
    /// The schedule must be sorted by effective UTC timestamp.  An empty
    /// schedule means UTC.  Callers normally include an entry at or before
    /// the first data timestamp; timestamps preceding the first entry use UTC.
    pub fn new(mut transitions: Vec<(i64, i64)>) -> Self {
        transitions.sort_unstable_by_key(|item| item.0);
        transitions.dedup_by_key(|item| item.0);
        let next_transition_utc = transitions.first().map_or(i64::MAX, |item| item.0);
        Self {
            transitions,
            transition_idx: 0,
            next_transition_utc,
            offset_ns: 0,
            last_local_day: None,
            next_local_boundary_utc: i64::MIN,
            pending_timestamp: 0,
            pending_equity: 0.0,
            timestamps: Vec::new(),
            equity: Vec::new(),
        }
    }

    /// Record a mark, overwriting the current local day's prior mark.
    #[inline]
    pub fn observe(&mut self, timestamp: i64, equity: f64) {
        if self.last_local_day.is_some()
            && timestamp < self.next_local_boundary_utc
            && timestamp < self.next_transition_utc
        {
            // The usual path: another intraday bar. Avoid signed division in
            // the millions-of-bars loop and keep the live point in registers;
            // only midnight and DST append to the retained vectors.
            self.pending_timestamp = timestamp;
            self.pending_equity = equity;
            return;
        }

        while self.transition_idx < self.transitions.len()
            && self.transitions[self.transition_idx].0 <= timestamp
        {
            self.offset_ns = self.transitions[self.transition_idx].1;
            self.transition_idx += 1;
        }
        self.next_transition_utc =
            self.transitions.get(self.transition_idx).map_or(i64::MAX, |item| item.0);

        let local_day = timestamp.saturating_add(self.offset_ns).div_euclid(DAY_NS);
        self.next_local_boundary_utc =
            local_day.saturating_add(1).saturating_mul(DAY_NS).saturating_sub(self.offset_ns);
        if self.last_local_day == Some(local_day) {
            self.pending_timestamp = timestamp;
            self.pending_equity = equity;
        } else {
            if self.last_local_day.is_some() {
                self.timestamps.push(self.pending_timestamp);
                self.equity.push(self.pending_equity);
            }
            self.last_local_day = Some(local_day);
            self.pending_timestamp = timestamp;
            self.pending_equity = equity;
        }
    }

    /// Reconcile the last retained point after end-of-data settlement.
    #[inline]
    pub fn reconcile_final(&mut self, equity: f64) {
        if self.last_local_day.is_some() {
            self.pending_equity = equity;
        }
    }

    pub fn finish(mut self) -> DailyPerformance {
        if self.last_local_day.is_some() {
            self.timestamps.push(self.pending_timestamp);
            self.equity.push(self.pending_equity);
        }
        DailyPerformance { timestamps: self.timestamps, equity: self.equity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_only_last_mark_per_local_day() {
        let mut collector = DailyPerformanceCollector::new(vec![(0, 10 * 3_600_000_000_000)]);
        collector.observe(0, 100.0);
        collector.observe(1_000, 101.0);
        collector.observe(DAY_NS, 103.0);
        let daily = collector.finish();
        assert_eq!(daily.timestamps, vec![1_000, DAY_NS]);
        assert_eq!(daily.equity, vec![101.0, 103.0]);
    }

    #[test]
    fn applies_offset_transitions_without_splitting_a_local_day() {
        let hour = 3_600_000_000_000;
        let mut collector =
            DailyPerformanceCollector::new(vec![(0, 10 * hour), (16 * hour, 11 * hour)]);
        collector.observe(15 * hour, 100.0); // 01:00 local before transition
        collector.observe(16 * hour, 101.0); // 03:00 local after DST jump
        let daily = collector.finish();
        assert_eq!(daily.timestamps, vec![16 * hour]);
        assert_eq!(daily.equity, vec![101.0]);
    }

    #[test]
    fn final_reconciliation_updates_settled_value() {
        let mut collector = DailyPerformanceCollector::new(Vec::new());
        collector.observe(0, 100.0);
        collector.reconcile_final(99.5);
        assert_eq!(collector.finish().equity, vec![99.5]);
    }
}
