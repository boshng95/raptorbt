//! Summaries of one primary stream that answer a run of its bars without
//! walking it.
//!
//! A kernel stepping an idle run reads the run's extreme closes, highs and
//! lows, whether its timestamps rise, and the newest bar whose volume shows
//! no finite share. Walked directly, that is a pass over every bar a run
//! covers, and a sweep walks the whole stream once per backtest. Built once
//! per stream, these summaries answer each run in time logarithmic in its
//! length, with the same results the walks give.

use crate::execution::fill::FiniteShare;

/// Units of one level of [`BarIndex::levels`] per unit of the next, and
/// bars per unit of the finest.
const FANOUT: usize = 8;
/// `FANOUT` as a power of two, so that a unit's index and alignment are a
/// shift and a mask rather than a division per unit folded.
const FANOUT_BITS: u32 = FANOUT.trailing_zeros();
const _: () = assert!(FANOUT.is_power_of_two());
/// Levels of [`BarIndex::levels`]: units of 8, 64, ..., 8^LEVELS bars. A
/// run of `n` bars folds at most `2 * (FANOUT - 1)` units of each level
/// below its own scale, so the decision runs of a minute stream resampled
/// to hours fold a dozen summaries instead of sixty bars.
const LEVELS: usize = 5;

/// The extremes a run of bars marks, as the kernel's idle-run walk folds
/// them: a NaN never compares, so it marks nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Extremes {
    pub lowest_close: f64,
    pub highest_close: f64,
    pub high: f64,
    pub low: f64,
}

impl Extremes {
    pub(crate) const EMPTY: Self = Self {
        lowest_close: f64::INFINITY,
        highest_close: f64::NEG_INFINITY,
        high: f64::NEG_INFINITY,
        low: f64::INFINITY,
    };

    /// Fold one bar in.
    #[inline]
    pub(crate) fn push(&mut self, close: f64, high: f64, low: f64) {
        if close < self.lowest_close {
            self.lowest_close = close;
        }
        if close > self.highest_close {
            self.highest_close = close;
        }
        if high > self.high {
            self.high = high;
        }
        if low < self.low {
            self.low = low;
        }
    }

    /// Fold in the extremes of the bars that follow.
    ///
    /// A strict comparison keeps the first of equal values, so a block's
    /// fold keeps the first bar of its extreme and merging blocks in order
    /// keeps the first of theirs: the bar a single walk would have kept,
    /// down to the sign of a zero.
    #[inline]
    pub(crate) fn merge(&mut self, later: &Self) {
        self.push_extremes(later.lowest_close, later.highest_close, later.high, later.low);
    }

    #[inline]
    fn push_extremes(&mut self, lowest_close: f64, highest_close: f64, high: f64, low: f64) {
        if lowest_close < self.lowest_close {
            self.lowest_close = lowest_close;
        }
        if highest_close > self.highest_close {
            self.highest_close = highest_close;
        }
        if high > self.high {
            self.high = high;
        }
        if low < self.low {
            self.low = low;
        }
    }
}

/// Extremes of one stream's aligned units, the instants its timestamps fail
/// to rise, and the bars whose volume shows no finite share.
#[derive(Debug, Clone)]
pub struct BarIndex {
    len: usize,
    /// `levels[l][i]`: the extremes of bars `i * 8^(l+1)..(i+1) * 8^(l+1)`,
    /// the last unit of each level cut at the stream's end.
    levels: Vec<Vec<Extremes>>,
    /// Every `i` with `!(timestamp[i] < timestamp[i + 1])`, ascending.
    falls: Vec<usize>,
    /// Bars whose volume fails `0 <= volume < inf`, ascending.
    no_nonnegative_share: Vec<usize>,
    /// Bars whose volume fails `0 < volume < inf`, ascending.
    no_positive_share: Vec<usize>,
}

impl BarIndex {
    /// Summarize a stream's columns, which must share one length.
    pub fn build(
        timestamp: &[i64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Self {
        let len = timestamp.len();
        assert!(
            [high.len(), low.len(), close.len(), volume.len()].iter().all(|&n| n == len),
            "bar columns must share one length"
        );
        let mut levels: Vec<Vec<Extremes>> = Vec::with_capacity(LEVELS);
        levels.push(
            (0..len)
                .step_by(FANOUT)
                .map(|start| {
                    let mut unit = Extremes::EMPTY;
                    for idx in start..(start + FANOUT).min(len) {
                        unit.push(close[idx], high[idx], low[idx]);
                    }
                    unit
                })
                .collect(),
        );
        while levels.len() < LEVELS {
            let coarser = levels[levels.len() - 1]
                .chunks(FANOUT)
                .map(|chunk| {
                    let mut unit = Extremes::EMPTY;
                    for finer in chunk {
                        unit.merge(finer);
                    }
                    unit
                })
                .collect();
            levels.push(coarser);
        }
        let falls = timestamp
            .windows(2)
            .enumerate()
            .filter(|(_, pair)| !(pair[0] < pair[1]))
            .map(|(idx, _)| idx)
            .collect();
        let failing = |passes: fn(f64) -> bool| -> Vec<usize> {
            volume
                .iter()
                .enumerate()
                .filter(|(_, &value)| !passes(value))
                .map(|(idx, _)| idx)
                .collect()
        };
        Self {
            len,
            levels,
            falls,
            no_nonnegative_share: failing(|volume| (0.0..f64::INFINITY).contains(&volume)),
            no_positive_share: failing(|volume| (volume > 0.0) & (volume < f64::INFINITY)),
        }
    }

    /// Bars summarized.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the stream is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The extremes of bars `start..end`, as folding each bar of them in
    /// turn from [`Extremes::EMPTY`] finds them. `close`, `high` and `low`
    /// are the columns this index was built from.
    pub(crate) fn extremes(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
        start: usize,
        end: usize,
    ) -> Extremes {
        debug_assert!(start <= end && end <= self.len);
        let mut acc = Extremes::EMPTY;
        let mut idx = start;
        // The run in aligned units, in order: up the levels while each
        // unit's boundary lies inside the run, then down them. `size` bars,
        // `1 << bits`, make a unit of `level - 1`, or a bar when `level` is 0.
        let take = |acc: &mut Extremes, level: usize, bits: u32, idx: usize| match level {
            0 => acc.push(close[idx], high[idx], low[idx]),
            _ => acc.merge(&self.levels[level - 1][idx >> bits]),
        };
        let (mut level, mut bits) = (0, 0);
        let mut size = 1;
        while level < LEVELS {
            let within = (size << FANOUT_BITS) - 1;
            while idx & within != 0 && idx + size <= end {
                take(&mut acc, level, bits, idx);
                idx += size;
            }
            if idx & within != 0 {
                break;
            }
            level += 1;
            bits += FANOUT_BITS;
            size <<= FANOUT_BITS;
        }
        loop {
            while idx + size <= end {
                take(&mut acc, level, bits, idx);
                idx += size;
            }
            if level == 0 {
                break;
            }
            level -= 1;
            bits -= FANOUT_BITS;
            size >>= FANOUT_BITS;
        }
        acc
    }

    /// Whether the timestamps of bars `start..end` strictly rise, as
    /// `timestamp[start..end].windows(2).all(|pair| pair[0] < pair[1])`.
    pub(crate) fn rising(&self, start: usize, end: usize) -> bool {
        debug_assert!(start <= end && end <= self.len);
        let first = self.falls.partition_point(|&idx| idx < start);
        self.falls.get(first).is_none_or(|&idx| idx + 1 >= end)
    }

    /// The newest of bars `start..end` whose volume shows no finite share
    /// under `share`, as [`crate::execution::fill::BarLiquidity`] finds it
    /// walking their volumes.
    pub(crate) fn last_without_finite_share(
        &self,
        share: FiniteShare,
        start: usize,
        end: usize,
    ) -> Option<usize> {
        debug_assert!(start <= end && end <= self.len);
        let failing = match share {
            FiniteShare::NonNegative => &self.no_nonnegative_share,
            FiniteShare::Positive => &self.no_positive_share,
            FiniteShare::Never => return end.checked_sub(1).filter(|&last| last >= start),
        };
        let below = failing.partition_point(|&idx| idx < end);
        below.checked_sub(1).map(|at| failing[at]).filter(|&idx| idx >= start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bars per unit of the coarsest level.
    const TOP: usize = FANOUT.pow(LEVELS as u32);

    fn walk(close: &[f64], high: &[f64], low: &[f64], start: usize, end: usize) -> Extremes {
        let mut acc = Extremes::EMPTY;
        for idx in start..end {
            acc.push(close[idx], high[idx], low[idx]);
        }
        acc
    }

    fn same(a: &Extremes, b: &Extremes) -> bool {
        let bits = |e: &Extremes| {
            [e.lowest_close.to_bits(), e.highest_close.to_bits(), e.high.to_bits(), e.low.to_bits()]
        };
        bits(a) == bits(b)
    }

    /// Timestamp, high, low, close and volume columns.
    type Columns = (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// A deterministic stream with ties, signed zeros, NaNs, infinities,
    /// zero and negative volumes and a repeated timestamp.
    fn stream(n: usize) -> Columns {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let special = [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, 1.0];
        let value = |r: u64| {
            if r % 97 < special.len() as u64 {
                special[(r % 97) as usize]
            } else {
                ((r >> 8) % 1000) as f64 / 8.0
            }
        };
        let mut ts = Vec::with_capacity(n);
        let (mut high, mut low, mut close, mut volume) = (vec![], vec![], vec![], vec![]);
        let mut t = 0i64;
        for _ in 0..n {
            let r = next();
            t += if r % 1009 == 0 { 0 } else { 60 };
            ts.push(t);
            close.push(value(next()));
            high.push(value(next()));
            low.push(value(next()));
            volume.push(match next() % 211 {
                0 => 0.0,
                1 => -0.0,
                2 => f64::NAN,
                3 => f64::INFINITY,
                4 => -1.0,
                r => r as f64,
            });
        }
        (ts, high, low, close, volume)
    }

    #[test]
    fn answers_every_run_as_the_walks_do() {
        let n = 3 * TOP + 777;
        let (ts, high, low, close, volume) = stream(n);
        let index = BarIndex::build(&ts, &high, &low, &close, &volume);
        let mut state = 7u64;
        let mut ranges = vec![(0, n), (0, 0), (n, n), (1, n - 1), (TOP, 2 * TOP)];
        let mut draw = |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize % bound
        };
        for _ in 0..2000 {
            let (a, b) = (draw(n + 1), draw(n + 1));
            ranges.push((a.min(b), a.max(b)));
        }
        // Runs of a few hundred bars at any offset, as decisions cut them,
        // and runs either side of every scale's unit boundaries.
        for _ in 0..4000 {
            let start = draw(n + 1);
            ranges.push((start, (start + draw(300)).min(n)));
        }
        for size in (0..=LEVELS as u32).map(|level| FANOUT.pow(level)) {
            for at in (size..n).step_by(size * 5 + 3).take(40) {
                let at = at - at % size;
                for (lo, hi) in [(1, 1), (0, 1), (1, 0), (size, size), (0, size + 1)] {
                    ranges.push((at.saturating_sub(lo), (at + hi).min(n)));
                }
            }
        }
        for (start, end) in ranges {
            let walked = walk(&close, &high, &low, start, end);
            let indexed = index.extremes(&close, &high, &low, start, end);
            assert!(same(&walked, &indexed), "{start}..{end}: {walked:?} vs {indexed:?}");
            assert_eq!(
                index.rising(start, end),
                ts[start..end].windows(2).all(|pair| pair[0] < pair[1]),
                "{start}..{end}"
            );
            let volumes = &volume[start..end];
            let nonnegative = volumes
                .iter()
                .rposition(|&v| !(0.0..f64::INFINITY).contains(&v))
                .map(|at| start + at);
            let positive =
                volumes.iter().rposition(|&v| !(v > 0.0 && v < f64::INFINITY)).map(|at| start + at);
            assert_eq!(
                index.last_without_finite_share(FiniteShare::NonNegative, start, end),
                nonnegative
            );
            assert_eq!(
                index.last_without_finite_share(FiniteShare::Positive, start, end),
                positive
            );
            assert_eq!(
                index.last_without_finite_share(FiniteShare::Never, start, end),
                volumes.len().checked_sub(1).map(|at| start + at)
            );
        }
    }

    #[test]
    fn keeps_the_first_of_equal_extremes() {
        // A walk keeps +0.0 ahead of a later -0.0 and the reverse.
        let n = 2 * TOP;
        let mut close = vec![1.0; n];
        close[5] = 0.0;
        close[TOP + 5] = -0.0;
        let ts: Vec<i64> = (0..n as i64).collect();
        let index = BarIndex::build(&ts, &close, &close, &close, &close);
        let found = index.extremes(&close, &close, &close, 0, n);
        assert_eq!(found.lowest_close.to_bits(), 0.0f64.to_bits());
        let found = index.extremes(&close, &close, &close, 6, n);
        assert_eq!(found.lowest_close.to_bits(), (-0.0f64).to_bits());
    }
}
