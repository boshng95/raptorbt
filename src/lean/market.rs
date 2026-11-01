//! One symbol's primary bars and the venue book each one leaves behind.
//!
//! The kernel's book is the last print of the bars it has replayed, sized as
//! [`crate::execution::fill`] slices a bar's volume. Orders never consume it,
//! so the book after every bar is a function of the bars alone: computed
//! here once per symbol and shared by every cell run on it.

use super::rules::{highest, lowest};
use crate::core::lots::LotGrid;
use crate::core::performance::DailyPerformanceCollector;

/// A bar's prints as the kernel replays them: open, high, low, close, each
/// skipped where it does not move the price. `unbounded` for a bar whose
/// volume puts no bound on what it fills.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tape {
    pub unbounded: bool,
    pub prints: [(f64, f64); 4],
    pub n: usize,
}

pub struct Market {
    pub ts: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub lot: f64,
    pub(crate) grid: LotGrid,
    pub slices: f64,
    /// Whether every bar is stamped after the one before.
    pub rising: bool,
    /// The highest and lowest close, NaN where any is.
    pub top_close: f64,
    pub low_close: f64,
    /// The book (price, size) after each bar.
    book_px: Vec<f64>,
    book_sz: Vec<f64>,
}

impl Market {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ts: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
        lot: f64,
        slices: f64,
    ) -> Self {
        let n = ts.len();
        let rising = ts.windows(2).all(|pair| pair[0] < pair[1]);
        let mut market = Self {
            ts,
            open,
            high,
            low,
            close,
            volume,
            lot,
            grid: LotGrid::new(lot),
            slices,
            rising,
            top_close: f64::NAN,
            low_close: f64::NAN,
            book_px: Vec::with_capacity(n),
            book_sz: Vec::with_capacity(n),
        };
        market.top_close = highest(&market.close);
        market.low_close = lowest(&market.close);
        let mut book: Option<(f64, f64)> = None;
        for i in 0..n {
            let tape = market.replay(book, i);
            if tape.unbounded {
                book = Some((market.close[i], f64::INFINITY));
            } else if tape.n > 0 {
                book = Some(tape.prints[tape.n - 1]);
            }
            let (px, sz) = book.expect("a bar always leaves a book");
            market.book_px.push(px);
            market.book_sz.push(sz);
        }
        market
    }

    pub fn len(&self) -> usize {
        self.ts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    /// `share_on_grid`: what one print of a bar of `volume` offers.
    #[inline]
    pub fn share(&self, volume: f64) -> f64 {
        let lot = self.lot;
        if !(volume >= 0.0) || !volume.is_finite() || !(self.slices > 0.0) {
            f64::INFINITY
        } else if lot > 0.0 && lot.is_finite() {
            self.grid.floor(volume / self.slices).max(lot)
        } else if volume > 0.0 {
            volume / self.slices
        } else {
            f64::INFINITY
        }
    }

    /// `last_share_from`: what the closing print offers, the volume the
    /// other slices left.
    #[inline]
    pub fn last_share(&self, volume: f64, share: f64) -> f64 {
        if !share.is_finite() {
            return share;
        }
        let lot = self.lot;
        let claimed = (self.slices - 1.0) * share;
        if claimed >= volume {
            return lot.max(0.0);
        }
        let floor = if lot > 0.0 { lot } else { 0.0 };
        self.grid.snap(volume - claimed).max(floor)
    }

    /// `BarTape::replay_on_grid` of bar `i` against `book`.
    fn replay(&self, book: Option<(f64, f64)>, i: usize) -> Tape {
        let volume = self.volume[i];
        let share = self.share(volume);
        let mut tape = Tape::default();
        if !share.is_finite() {
            tape.unbounded = true;
            return tape;
        }
        let (open, high, low, close) = (self.open[i], self.high[i], self.low[i], self.close[i]);
        let mut here = book.map_or(open, |b| b.0);
        if book.is_none() || here != open {
            tape.prints[tape.n] = (open, share);
            tape.n += 1;
            here = open;
        }
        if high > here {
            tape.prints[tape.n] = (high, share);
            tape.n += 1;
            here = high;
        }
        if low < here {
            tape.prints[tape.n] = (low, share);
            tape.n += 1;
            here = low;
        }
        if close != here {
            tape.prints[tape.n] = (close, self.last_share(volume, share));
            tape.n += 1;
        }
        tape
    }

    /// The bars the daily curve keeps under `transitions`: the last of each
    /// local day, as `DailyPerformanceCollector` keys days -- found by
    /// running it over the bars with each bar's index for its equity.
    pub fn day_ends(&self, transitions: Vec<(i64, i64)>) -> Vec<usize> {
        let mut collector = DailyPerformanceCollector::new(transitions);
        for (i, &ts) in self.ts.iter().enumerate() {
            collector.observe(ts, i as f64);
        }
        collector.finish().equity.into_iter().map(|i| i as usize).collect()
    }

    /// The book before bar `i` steps; none before the first bar.
    #[inline]
    pub fn book_before(&self, i: usize) -> Option<(f64, f64)> {
        (i > 0).then(|| (self.book_px[i - 1], self.book_sz[i - 1]))
    }

    /// The book bar `i` leaves.
    #[inline]
    pub fn book_after(&self, i: usize) -> (f64, f64) {
        (self.book_px[i], self.book_sz[i])
    }

    /// Bar `i`'s prints, as the kernel replays them when an order works.
    #[inline]
    pub fn tape(&self, i: usize) -> Tape {
        self.replay(self.book_before(i), i)
    }
}
