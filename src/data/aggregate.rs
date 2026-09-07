//! Streaming bar builders.
//!
//! A builder consumes source records one at a time — raw trades or finer
//! bars — and emits a completed bar whenever its boundary condition is met.
//! A window's bar is emitted when the first record of a *later* window
//! arrives (plus [`BarBuilder::flush`] at end of data), and is stamped with
//! the window's end timestamp.
//!
//! Which window a record falls in depends on what its own timestamp means,
//! so the builders take a [`SourceLabel`]:
//!
//! * [`SourceLabel::Open`] (default) — the timestamp opens the record's
//!   period, the raw-provider convention. A record with timestamp `t`
//!   belongs to the window `floor(t / width)`, so a bar labeled `t`
//!   contains only data stamped strictly before `t`.
//! * [`SourceLabel::Close`] — the timestamp closes it, the Nautilus
//!   convention. A record landing exactly on a boundary closes the window
//!   ending there rather than opening the next one, so a bar labeled `t`
//!   contains data stamped up to and including `t`.
//!
//! The label also decides *when* a window is known to be finished. Under
//! `Open` nothing proves a window closed until a record from the next one
//! arrives, so that is when its bar is emitted. Under `Close` the record
//! stamped at the boundary is by definition the last the window can hold,
//! so the bar is emitted there and then — one record earlier, and at the
//! instant it is stamped with, which is what a clock-driven aggregator
//! (Nautilus\' `TimeBarAggregator`) does. Waiting for the next record
//! instead would hand the strategy a bar a whole primary bar late.
//!
//! Under either label the emitted bar describes only trading that had
//! already happened by the instant it is stamped with — no look-ahead by
//! construction. Choosing the wrong one does not create look-ahead; it
//! shifts every boundary record by one window, which is silent and wrong.

use crate::core::types::{OhlcvBar, Price, Timestamp};
use crate::data::bar_spec::{AggregationUnit, BarSpec, BuilderParams, SourceLabel, SpecError};

/// One source record: a trade, or a finer bar's worth of trading.
#[derive(Debug, Clone, Copy)]
pub struct SourceRecord {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: f64,
    /// Buy-initiated minus sell-initiated volume. `0.0` means unknown; the
    /// signed-flow builders then classify by the tick rule instead.
    pub signed_volume: f64,
}

impl SourceRecord {
    /// A single trade as a degenerate bar, with no known flow direction.
    pub fn trade(timestamp: Timestamp, price: Price, size: f64) -> Self {
        Self::signed_trade(timestamp, price, size, 0.0)
    }

    /// A trade whose buy/sell split is known.
    pub fn signed_trade(timestamp: Timestamp, price: Price, size: f64, signed: f64) -> Self {
        Self {
            timestamp,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: size,
            signed_volume: signed,
        }
    }

    pub fn from_bar(bar: &OhlcvBar) -> Self {
        Self {
            timestamp: bar.timestamp,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            signed_volume: 0.0,
        }
    }
}

/// In-progress accumulation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Partial {
    open: Price,
    high: Price,
    low: Price,
    close: Price,
    volume: f64,
}

impl Partial {
    pub(crate) fn start(rec: &SourceRecord) -> Self {
        Self { open: rec.open, high: rec.high, low: rec.low, close: rec.close, volume: rec.volume }
    }

    pub(crate) fn absorb(&mut self, rec: &SourceRecord) {
        self.high = self.high.max(rec.high);
        self.low = self.low.min(rec.low);
        self.close = rec.close;
        self.volume += rec.volume;
    }

    pub(crate) fn into_bar(self, timestamp: Timestamp) -> OhlcvBar {
        OhlcvBar {
            timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        }
    }
}

/// Streaming aggregator: push records, receive completed bars.
pub trait BarBuilder: std::fmt::Debug {
    /// Consume one record; returns a bar completed *by* this record (the
    /// record itself belongs to the next bar for time windows, and to the
    /// emitted bar for count/volume/value thresholds).
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar>;

    /// Emit any in-progress bar at end of data.
    fn flush(&mut self) -> Option<OhlcvBar>;

    /// A bar already completed but held back because [`BarBuilder::push`]
    /// can return only one.
    ///
    /// Only multi-emit builders override this: one large move completes
    /// several Renko bricks at once. Callers must drain it after every
    /// push, or those bars are silently lost.
    fn next_pending(&mut self) -> Option<OhlcvBar> {
        None
    }
}

/// Build a streaming aggregator for a spec.
///
/// `align_offset_ns` shifts time-window boundaries: `0` aligns to UTC
/// (epoch), a timezone offset (e.g. [`IST_OFFSET_NS`]) aligns day/week/
/// month/year windows to that timezone's civil dates — an NSE day bar
/// covers one IST trading date rather than a UTC one.
pub fn builder_for(
    spec: BarSpec,
    align_offset_ns: i64,
) -> Result<Box<dyn BarBuilder + Send>, SpecError> {
    builder_for_with(spec, align_offset_ns, BuilderParams::default())
}

/// [`builder_for`] with the extra parameters some units need.
pub fn builder_for_with(
    spec: BarSpec,
    align_offset_ns: i64,
    params: BuilderParams,
) -> Result<Box<dyn BarBuilder + Send>, SpecError> {
    if let Some(unit_ns) = spec.unit.fixed_ns() {
        return Ok(Box::new(TimeBarBuilder::new(
            unit_ns * spec.step as i64,
            align_offset_ns,
            params.label,
        )));
    }
    match spec.unit {
        AggregationUnit::Millisecond
        | AggregationUnit::Second
        | AggregationUnit::Minute
        | AggregationUnit::Hour
        | AggregationUnit::Day
        | AggregationUnit::Week => unreachable!("fixed-duration units handled above"),
        AggregationUnit::Month | AggregationUnit::Year => {
            Ok(Box::new(CalendarBarBuilder::new(spec, align_offset_ns, params.label)))
        }
        AggregationUnit::Tick => Ok(Box::new(CountBarBuilder::ticks(spec.step as u64))),
        AggregationUnit::Volume => Ok(Box::new(CountBarBuilder::volume(spec.step as f64))),
        AggregationUnit::Value => Ok(Box::new(CountBarBuilder::value(spec.step as f64))),
        AggregationUnit::Renko => {
            Ok(Box::new(crate::data::renko::RenkoBarBuilder::new(params.resolved_brick(spec)?)))
        }
        AggregationUnit::TickImbalance
        | AggregationUnit::TickRuns
        | AggregationUnit::VolumeImbalance
        | AggregationUnit::VolumeRuns
        | AggregationUnit::ValueImbalance
        | AggregationUnit::ValueRuns => {
            let threshold = spec.step as f64;
            crate::data::flow_bars::FlowBarBuilder::for_unit(spec.unit, threshold)
                .map(|b| Box::new(b) as Box<dyn BarBuilder + Send>)
                .ok_or(SpecError::Unimplemented("signed-flow unit"))
        }
    }
}

/// India Standard Time offset (UTC+5:30) in nanoseconds, for aligning
/// day/week/calendar windows to IST trading dates.
pub const IST_OFFSET_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;

/// Fixed-width time windows, aligned to the epoch shifted by an offset.
#[derive(Debug)]
pub struct TimeBarBuilder {
    width_ns: i64,
    align_offset_ns: i64,
    label: SourceLabel,
    window: Option<i64>,
    partial: Option<Partial>,
    /// A second bar completed by one record: the window a gap closed, plus
    /// the window that record itself finished. Only `Close` can produce it.
    pending: Option<OhlcvBar>,
}

impl TimeBarBuilder {
    pub fn new(width_ns: i64, align_offset_ns: i64, label: SourceLabel) -> Self {
        Self { width_ns, align_offset_ns, label, window: None, partial: None, pending: None }
    }

    /// Whether *ts* is the last instant its window can contain.
    ///
    /// Only true under [`SourceLabel::Close`]: an open-labelled record
    /// stamped on a boundary opens the next window rather than ending one,
    /// so no record ever proves an open-labelled window finished.
    #[inline]
    fn completes_window(&self, ts: Timestamp, window: i64) -> bool {
        self.label == SourceLabel::Close && ts == self.window_end(window)
    }

    #[inline]
    fn window_of(&self, ts: Timestamp) -> i64 {
        // The bias is the whole of the labelling difference: it pulls a
        // close-labelled record sitting exactly on a boundary back into the
        // window that ends there, and moves nothing else.
        (ts + self.align_offset_ns - self.label.boundary_bias_ns()).div_euclid(self.width_ns)
    }

    /// End timestamp (exclusive boundary) of a window, in the source clock.
    #[inline]
    fn window_end(&self, window: i64) -> Timestamp {
        (window + 1) * self.width_ns - self.align_offset_ns
    }
}

impl BarBuilder for TimeBarBuilder {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        let window = self.window_of(rec.timestamp);
        let mut done = match (self.window, &mut self.partial) {
            (Some(current), Some(partial)) if window == current => {
                partial.absorb(rec);
                None
            }
            (Some(current), partial_slot) if window != current => {
                let done = partial_slot.take().map(|p| p.into_bar(self.window_end(current)));
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
                done
            }
            _ => {
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
                None
            }
        };
        if self.completes_window(rec.timestamp, window) {
            // This record ends its own window, so the bar is finished here.
            let finished = self.partial.take().map(|p| p.into_bar(self.window_end(window)));
            self.window = None;
            // One record can finish two windows: the one a gap left open,
            // and its own. They are emitted oldest first, so the earlier
            // bar is returned and the later one queued behind it.
            match done {
                Some(_) => self.pending = finished,
                None => done = finished,
            }
        }
        done
    }

    fn next_pending(&mut self) -> Option<OhlcvBar> {
        self.pending.take()
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        // A window closed on its own boundary left nothing behind, which is
        // why `window` is cleared there rather than only `partial`.
        let window = self.window.take()?;
        self.partial.take().map(|p| p.into_bar(self.window_end(window)))
    }
}

const NS_PER_DAY: i64 = 86_400_000_000_000;

/// Days since epoch → civil (year, month) — Howard Hinnant's algorithm.
fn civil_from_days(days: i64) -> (i64, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32)
}

/// Civil (year, month, day=1) → days since epoch.
fn days_from_civil(y: i64, m: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5; // day-of-year for the 1st of the month
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Calendar month/year windows (irregular durations), aligned to civil
/// dates in the clock shifted by `align_offset_ns`.
#[derive(Debug)]
pub struct CalendarBarBuilder {
    spec: BarSpec,
    align_offset_ns: i64,
    label: SourceLabel,
    /// Window key: months-since-epoch / step (Month) or year / step (Year).
    window: Option<i64>,
    partial: Option<Partial>,
    /// The second of two bars one record can finish; see [`TimeBarBuilder`].
    pending: Option<OhlcvBar>,
}

impl CalendarBarBuilder {
    pub fn new(spec: BarSpec, align_offset_ns: i64, label: SourceLabel) -> Self {
        Self { spec, align_offset_ns, label, window: None, partial: None, pending: None }
    }

    /// Whether *ts* is the last instant its window can contain.
    ///
    /// The same rule as [`TimeBarBuilder::completes_window`], over calendar
    /// boundaries: a close-labelled record stamped at midnight on the first
    /// of a month ends the month before it.
    fn completes_window(&self, ts: Timestamp, window: i64) -> bool {
        self.label == SourceLabel::Close && ts == self.window_end(window)
    }

    fn window_of(&self, ts: Timestamp) -> i64 {
        // Same bias, same reason as `TimeBarBuilder::window_of`: a record
        // stamped at midnight closes the month that ends there.
        let shifted = ts + self.align_offset_ns - self.label.boundary_bias_ns();
        let days = shifted.div_euclid(NS_PER_DAY);
        let (year, month) = civil_from_days(days);
        let key = match self.spec.unit {
            AggregationUnit::Month => year * 12 + (month as i64 - 1),
            _ => year,
        };
        key.div_euclid(self.spec.step as i64)
    }

    /// Start of the window *after* `window`, in the source clock.
    fn window_end(&self, window: i64) -> Timestamp {
        let next = (window + 1) * self.spec.step as i64;
        let days = match self.spec.unit {
            AggregationUnit::Month => {
                days_from_civil(next.div_euclid(12), next.rem_euclid(12) as u32 + 1)
            }
            _ => days_from_civil(next, 1),
        };
        days * NS_PER_DAY - self.align_offset_ns
    }
}

impl BarBuilder for CalendarBarBuilder {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        let window = self.window_of(rec.timestamp);
        let mut done = match (self.window, &mut self.partial) {
            (Some(current), Some(partial)) if window == current => {
                partial.absorb(rec);
                None
            }
            (Some(current), partial_slot) if window != current => {
                let done = partial_slot.take().map(|p| p.into_bar(self.window_end(current)));
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
                done
            }
            _ => {
                self.window = Some(window);
                self.partial = Some(Partial::start(rec));
                None
            }
        };
        if self.completes_window(rec.timestamp, window) {
            let finished = self.partial.take().map(|p| p.into_bar(self.window_end(window)));
            self.window = None;
            match done {
                Some(_) => self.pending = finished,
                None => done = finished,
            }
        }
        done
    }

    fn next_pending(&mut self) -> Option<OhlcvBar> {
        self.pending.take()
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        let window = self.window.take()?;
        self.partial.take().map(|p| p.into_bar(self.window_end(window)))
    }
}

/// What a count-threshold builder accumulates toward.
#[derive(Debug, Clone, Copy)]
enum Threshold {
    Ticks(u64),
    Volume(f64),
    Value(f64),
}

/// Emits a bar every N records / units of volume / units of value.
///
/// The record that crosses the threshold is *included* in the emitted bar,
/// which is stamped with that record's timestamp. Oversized records are not
/// split — a single trade larger than the threshold emits one bar.
#[derive(Debug)]
pub struct CountBarBuilder {
    threshold: Threshold,
    accumulated: f64,
    ticks: u64,
    last_ts: Timestamp,
    partial: Option<Partial>,
}

impl CountBarBuilder {
    pub fn ticks(n: u64) -> Self {
        Self::new(Threshold::Ticks(n.max(1)))
    }

    pub fn volume(v: f64) -> Self {
        Self::new(Threshold::Volume(v.max(f64::MIN_POSITIVE)))
    }

    pub fn value(v: f64) -> Self {
        Self::new(Threshold::Value(v.max(f64::MIN_POSITIVE)))
    }

    fn new(threshold: Threshold) -> Self {
        Self { threshold, accumulated: 0.0, ticks: 0, last_ts: 0, partial: None }
    }

    fn crossed(&self) -> bool {
        match self.threshold {
            Threshold::Ticks(n) => self.ticks >= n,
            Threshold::Volume(v) => self.accumulated >= v,
            Threshold::Value(v) => self.accumulated >= v,
        }
    }
}

impl BarBuilder for CountBarBuilder {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        match &mut self.partial {
            Some(partial) => partial.absorb(rec),
            None => self.partial = Some(Partial::start(rec)),
        }
        self.ticks += 1;
        self.accumulated += match self.threshold {
            Threshold::Ticks(_) => 0.0,
            Threshold::Volume(_) => rec.volume,
            Threshold::Value(_) => rec.close * rec.volume,
        };
        self.last_ts = rec.timestamp;

        if self.crossed() {
            self.ticks = 0;
            self.accumulated = 0.0;
            self.partial.take().map(|p| p.into_bar(self.last_ts))
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        self.ticks = 0;
        self.accumulated = 0.0;
        self.partial.take().map(|p| p.into_bar(self.last_ts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(ts: i64, price: f64, vol: f64) -> SourceRecord {
        SourceRecord::trade(ts, price, vol)
    }

    #[test]
    fn time_windows_emit_on_boundary_with_end_stamp() {
        // 10ns windows.
        let mut b = TimeBarBuilder::new(10, 0, SourceLabel::Open);
        assert!(b.push(&rec(1, 100.0, 1.0)).is_none());
        assert!(b.push(&rec(5, 102.0, 2.0)).is_none());
        assert!(b.push(&rec(9, 99.0, 1.0)).is_none());

        // ts=12 starts the next window; the first bar emits, stamped 10.
        let bar = b.push(&rec(12, 101.0, 1.0)).expect("bar");
        assert_eq!(bar.timestamp, 10);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 99.0);
        assert_eq!(bar.close, 99.0);
        assert_eq!(bar.volume, 4.0);

        // Flush emits the in-progress second window.
        let last = b.flush().expect("flush");
        assert_eq!(last.timestamp, 20);
        assert_eq!(last.volume, 1.0);
        assert!(b.flush().is_none());
    }

    #[test]
    fn a_close_labelled_record_on_the_boundary_closes_the_window_ending_there() {
        // The same records under both labels. Only the one landing exactly
        // on the boundary moves, and it moves a whole window.
        let on_boundary = rec(10, 99.0, 1.0);

        let mut open_labelled = TimeBarBuilder::new(10, 0, SourceLabel::Open);
        assert!(open_labelled.push(&rec(4, 100.0, 1.0)).is_none());
        // Under `Open` ts=10 opens the next window, so the first bar closes
        // on the record before it.
        let bar = open_labelled.push(&on_boundary).expect("bar");
        assert_eq!(bar.timestamp, 10);
        assert_eq!(bar.close, 100.0);

        let mut close_labelled = TimeBarBuilder::new(10, 0, SourceLabel::Close);
        assert!(close_labelled.push(&rec(4, 100.0, 1.0)).is_none());
        // Under `Close` it belongs to the window ending at 10 -- and ends
        // it, so the bar comes out on this push rather than the next.
        let bar = close_labelled.push(&on_boundary).expect("bar");
        assert_eq!(bar.timestamp, 10);
        assert_eq!(bar.close, 99.0);
        assert_eq!(bar.volume, 2.0);
        assert!(close_labelled.next_pending().is_none());
    }

    #[test]
    fn a_close_labelled_window_is_emitted_on_its_own_boundary_not_the_next_record() {
        // The timing matters on its own: a consumer that reacts to a
        // composite bar acts one primary bar late if the bar arrives late,
        // whatever price it carries.
        let mut b = TimeBarBuilder::new(10, 0, SourceLabel::Close);
        assert!(b.push(&rec(4, 100.0, 1.0)).is_none());
        assert!(b.push(&rec(10, 99.0, 1.0)).is_some(), "emitted by its own boundary");
        // The next record starts a fresh window and completes nothing.
        assert!(b.push(&rec(11, 98.0, 1.0)).is_none());
    }

    #[test]
    fn one_close_labelled_record_can_finish_two_windows_oldest_first() {
        // A gap leaves a window open; the record that ends a later window
        // closes both. `push` returns one bar, so the second must be
        // drained -- losing it would drop a whole composite bar.
        let mut b = TimeBarBuilder::new(10, 0, SourceLabel::Close);
        assert!(b.push(&rec(4, 100.0, 1.0)).is_none());
        let first = b.push(&rec(20, 99.0, 2.0)).expect("the window the gap left open");
        assert_eq!(first.timestamp, 10);
        assert_eq!(first.close, 100.0);
        let second = b.next_pending().expect("the window ending at 20");
        assert_eq!(second.timestamp, 20);
        assert_eq!(second.close, 99.0);
        assert!(b.next_pending().is_none());
        assert!(b.flush().is_none(), "nothing left in progress");
    }

    #[test]
    fn the_label_moves_nothing_that_is_not_on_a_boundary() {
        // Every record strictly inside a window buckets identically under
        // either label; a difference anywhere else would be a bug in the
        // bias, not a convention.
        let inside = [1_i64, 4, 9, 11, 19, 21];
        for width in [10_i64, 7, 1_000] {
            for ts in inside {
                if ts % width == 0 {
                    continue;
                }
                let open = TimeBarBuilder::new(width, 0, SourceLabel::Open).window_of(ts);
                let close = TimeBarBuilder::new(width, 0, SourceLabel::Close).window_of(ts);
                assert_eq!(open, close, "ts={ts} width={width}");
            }
        }
    }

    #[test]
    fn close_labelled_minutes_aggregate_to_the_nautilus_hourly_close() {
        // The defect this label exists for, at the size it was found:
        // close-labelled 1m bars into 1h. The minute stamped 10:00 is the
        // last minute of the hour ending 10:00, and the hourly bar labeled
        // 10:00 must close on it.
        const MIN: i64 = 60 * 1_000_000_000;
        const HOUR: i64 = 60 * MIN;
        let mut b = TimeBarBuilder::new(HOUR, 0, SourceLabel::Close);
        for minute in 1..=60 {
            let bar = OhlcvBar {
                timestamp: 9 * HOUR + minute * MIN,
                open: 100.0,
                high: 100.0 + minute as f64,
                low: 99.0,
                close: 100.0 + minute as f64,
                volume: 1.0,
            };
            let emitted = b.push(&SourceRecord::from_bar(&bar));
            if minute == 60 {
                let hourly = emitted.expect("the 10:00 minute closes the 10:00 hour");
                assert_eq!(hourly.timestamp, 10 * HOUR);
                assert_eq!(hourly.close, 160.0, "closes on the 10:00 minute, not 09:59");
                assert_eq!(hourly.volume, 60.0, "all sixty minutes, not fifty-nine");
                return;
            }
            assert!(emitted.is_none(), "minute {minute}");
        }
        // The sixtieth minute is stamped 10:00 and closes the hour, so the
        // loop above already emitted it on that push rather than here.
        unreachable!("the sixtieth minute returns");
    }

    #[test]
    fn empty_windows_are_skipped_not_emitted() {
        let mut b = TimeBarBuilder::new(10, 0, SourceLabel::Open);
        assert!(b.push(&rec(1, 100.0, 1.0)).is_none());
        // Jump three windows ahead: exactly one bar comes out.
        let bar = b.push(&rec(35, 101.0, 1.0)).expect("bar");
        assert_eq!(bar.timestamp, 10);
        assert!(b.flush().is_some());
    }

    #[test]
    fn composite_from_bars_preserves_ohlc_identity() {
        // Two 5-unit bars into one 10-unit bar.
        let mut b = TimeBarBuilder::new(10, 0, SourceLabel::Open);
        let bar1 =
            OhlcvBar { timestamp: 3, open: 10.0, high: 12.0, low: 9.0, close: 11.0, volume: 5.0 };
        let bar2 =
            OhlcvBar { timestamp: 8, open: 11.0, high: 15.0, low: 10.5, close: 14.0, volume: 7.0 };
        assert!(b.push(&SourceRecord::from_bar(&bar1)).is_none());
        assert!(b.push(&SourceRecord::from_bar(&bar2)).is_none());
        let out = b.flush().expect("bar");
        assert_eq!(out.timestamp, 10);
        assert_eq!(out.open, 10.0);
        assert_eq!(out.high, 15.0);
        assert_eq!(out.low, 9.0);
        assert_eq!(out.close, 14.0);
        assert_eq!(out.volume, 12.0);
    }

    #[test]
    fn tick_count_bars() {
        let mut b = CountBarBuilder::ticks(3);
        assert!(b.push(&rec(1, 100.0, 1.0)).is_none());
        assert!(b.push(&rec(2, 101.0, 1.0)).is_none());
        let bar = b.push(&rec(3, 99.0, 1.0)).expect("bar");
        assert_eq!(bar.timestamp, 3);
        assert_eq!(bar.high, 101.0);
        assert_eq!(bar.volume, 3.0);
        // Counter reset: next threshold takes three more.
        assert!(b.push(&rec(4, 100.0, 1.0)).is_none());
    }

    #[test]
    fn volume_bars_include_crossing_record() {
        let mut b = CountBarBuilder::volume(10.0);
        assert!(b.push(&rec(1, 100.0, 4.0)).is_none());
        let bar = b.push(&rec(2, 101.0, 7.0)).expect("bar");
        assert_eq!(bar.volume, 11.0);
        assert_eq!(bar.timestamp, 2);
    }

    #[test]
    fn value_bars_accumulate_price_times_volume() {
        let mut b = CountBarBuilder::value(1_000.0);
        assert!(b.push(&rec(1, 100.0, 4.0)).is_none()); // 400
        let bar = b.push(&rec(2, 100.0, 7.0)).expect("bar"); // +700 = 1100
        assert_eq!(bar.volume, 11.0);
    }

    #[test]
    fn calendar_month_bars_use_civil_boundaries() {
        // 2024-01-31 12:00 UTC and 2024-02-01 00:30 UTC straddle a month
        // boundary. Epoch days: 2024-01-01 is 19723.
        let jan31_noon = (19_723 + 30) * NS_PER_DAY + 12 * 3_600_000_000_000;
        let feb01 = (19_723 + 31) * NS_PER_DAY + 1_800_000_000_000;
        let spec = BarSpec::new(1, AggregationUnit::Month).unwrap();
        let mut b = builder_for(spec, 0).unwrap();
        assert!(b.push(&rec(jan31_noon, 100.0, 1.0)).is_none());
        let bar = b.push(&rec(feb01, 101.0, 1.0)).expect("month bar");
        // Stamped at the start of February.
        assert_eq!(bar.timestamp, (19_723 + 31) * NS_PER_DAY);
        assert!(b.flush().is_some());
    }

    #[test]
    fn a_close_labelled_month_ends_on_the_first_of_the_next() {
        // The calendar builder keeps the same rule as the time builder: a
        // record stamped exactly at midnight on the 1st is the last instant
        // January can contain, so it closes January rather than opening
        // February -- and closes it on that record's own price.
        let jan31_noon = (19_723 + 30) * NS_PER_DAY + 12 * 3_600_000_000_000;
        let feb01 = (19_723 + 31) * NS_PER_DAY;
        let spec = BarSpec::new(1, AggregationUnit::Month).unwrap();
        let mut b = CalendarBarBuilder::new(spec, 0, SourceLabel::Close);
        assert!(b.push(&rec(jan31_noon, 100.0, 1.0)).is_none());
        let bar = b.push(&rec(feb01, 101.0, 2.0)).expect("January closed");
        assert_eq!(bar.timestamp, feb01, "stamped at its own close");
        assert_eq!(bar.close, 101.0, "closes on the boundary record");
        assert_eq!(bar.volume, 3.0, "which it also contains");
        assert!(b.next_pending().is_none());
        assert!(b.flush().is_none(), "nothing carried into February");
    }

    #[test]
    fn one_close_labelled_record_can_finish_two_months_oldest_first() {
        // The gap case, on calendar windows: nothing prints in February, and
        // the 1st of March closes both January and February's successor --
        // the month the gap left open, then its own.
        let jan15 = (19_723 + 14) * NS_PER_DAY;
        let mar01 = (19_723 + 60) * NS_PER_DAY; // 2024 is a leap year
        let spec = BarSpec::new(1, AggregationUnit::Month).unwrap();
        let mut b = CalendarBarBuilder::new(spec, 0, SourceLabel::Close);
        assert!(b.push(&rec(jan15, 100.0, 1.0)).is_none());
        let first = b.push(&rec(mar01, 101.0, 1.0)).expect("January");
        assert_eq!(first.timestamp, (19_723 + 31) * NS_PER_DAY);
        assert_eq!(first.close, 100.0);
        let second = b.next_pending().expect("February, which the 1st closed");
        assert_eq!(second.timestamp, mar01);
        assert_eq!(second.close, 101.0);
        assert!(b.next_pending().is_none());
    }

    #[test]
    fn calendar_year_bars() {
        // 2023-12-31 and 2024-01-02 (epoch day 19722 = 2023-12-31).
        let dec31 = 19_722 * NS_PER_DAY + NS_PER_DAY / 2;
        let jan02 = (19_723 + 1) * NS_PER_DAY;
        let spec = BarSpec::new(1, AggregationUnit::Year).unwrap();
        let mut b = builder_for(spec, 0).unwrap();
        assert!(b.push(&rec(dec31, 100.0, 1.0)).is_none());
        let bar = b.push(&rec(jan02, 101.0, 1.0)).expect("year bar");
        assert_eq!(bar.timestamp, 19_723 * NS_PER_DAY); // 2024-01-01 00:00 UTC
    }

    #[test]
    fn ist_alignment_groups_by_trading_date() {
        // Two NSE sessions: 2024-01-02 10:00 IST and 2024-01-03 10:00 IST.
        // In UTC both are 04:30, on consecutive UTC days — but a *UTC*-
        // aligned day window would also split e.g. an overnight MCX session
        // at 05:30 IST. Verify the boundary sits at IST midnight.
        let day0 = 19_724 * NS_PER_DAY; // 2024-01-02 00:00 UTC
        let ist_0930 = day0 + 4 * 3_600_000_000_000; // 09:30 IST
        let ist_2330 = day0 + 18 * 3_600_000_000_000; // 23:30 IST same trading date
        let next_ist_morning = day0 + 20 * 3_600_000_000_000; // 01:30 IST next date

        let spec = BarSpec::new(1, AggregationUnit::Day).unwrap();
        let mut b = builder_for(spec, IST_OFFSET_NS).unwrap();
        assert!(b.push(&rec(ist_0930, 100.0, 1.0)).is_none());
        // 23:30 IST is still the same IST date (would be next UTC day!).
        assert!(b.push(&rec(ist_2330, 101.0, 1.0)).is_none());
        let bar = b.push(&rec(next_ist_morning, 102.0, 1.0)).expect("day bar");
        assert_eq!(bar.volume, 2.0);
        // Stamped at IST midnight expressed in UTC ns.
        assert_eq!((bar.timestamp + IST_OFFSET_NS) % NS_PER_DAY, 0);

        // Contrast: UTC alignment sees all three records inside one UTC day
        // (04:00, 18:00, 20:00 UTC) and emits nothing where IST rolled over.
        let mut utc = builder_for(spec, 0).unwrap();
        assert!(utc.push(&rec(ist_0930, 100.0, 1.0)).is_none());
        assert!(utc.push(&rec(ist_2330, 101.0, 1.0)).is_none());
        assert!(utc.push(&rec(next_ist_morning, 102.0, 1.0)).is_none());
    }

    #[test]
    fn every_declared_unit_now_builds() {
        // The enum was declared ahead of its implementations; nothing in it
        // should still return Unimplemented.
        use AggregationUnit::*;
        for unit in [
            Millisecond,
            Second,
            Minute,
            Hour,
            Day,
            Week,
            Month,
            Year,
            Tick,
            Volume,
            Value,
            Renko,
            TickImbalance,
            TickRuns,
            VolumeImbalance,
            VolumeRuns,
            ValueImbalance,
            ValueRuns,
        ] {
            let spec = BarSpec::new(1, unit).unwrap();
            assert!(builder_for(spec, 0).is_ok(), "{unit:?} should build");
        }
    }

    #[test]
    fn builder_for_builds_renko() {
        let spec = BarSpec::new(1, AggregationUnit::Renko).unwrap();
        assert!(builder_for(spec, 0).is_ok(), "step becomes the brick height");
        let params = BuilderParams { brick_size: 0.05, ..Default::default() };
        assert!(builder_for_with(spec, 0, params).is_ok());
        let bad = BuilderParams { brick_size: -1.0, ..Default::default() };
        // A negative brick falls back to `step`, which is valid.
        assert!(builder_for_with(spec, 0, bad).is_ok());
    }
}
