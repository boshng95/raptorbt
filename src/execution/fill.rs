//! Order fill simulation models.

use crate::core::lots::{floor_to_lot, snap_to_lot_grid};
use crate::core::types::{Direction, FillTiming, OhlcvBar, Price};

/// Fill price model determining at what price orders are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillPrice {
    /// Execute at close price (end of bar).
    #[default]
    Close,
    /// Execute at open price (start of next bar).
    Open,
    /// Execute at OHLC average.
    Average,
    /// Execute at typical price (H+L+C)/3.
    Typical,
    /// Execute at VWAP (if available, otherwise typical).
    Vwap,
    /// Execute at worst price (high for buys, low for sells).
    Worst,
    /// Execute at best price (low for buys, high for sells).
    Best,
}

impl FillPrice {
    /// Price source implied by an execution-timing policy.
    ///
    /// `NextBarOpen` and `SameBarOpenLookahead` both read the open — the
    /// policy decides *which bar* that open belongs to, which is the
    /// kernel's job, not this table's.
    pub fn for_timing(timing: FillTiming) -> Self {
        match timing {
            FillTiming::SameBarClose => FillPrice::Close,
            FillTiming::NextBarOpen | FillTiming::SameBarOpenLookahead => FillPrice::Open,
        }
    }

    /// Get execution price from OHLCV bar.
    ///
    /// # Arguments
    /// * `bar` - OHLCV bar data
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Execution price
    pub fn get_price(&self, bar: &OhlcvBar, direction: Direction, is_entry: bool) -> Price {
        match self {
            FillPrice::Close => bar.close,
            FillPrice::Open => bar.open,
            FillPrice::Average => (bar.open + bar.high + bar.low + bar.close) / 4.0,
            FillPrice::Typical => (bar.high + bar.low + bar.close) / 3.0,
            FillPrice::Vwap => (bar.high + bar.low + bar.close) / 3.0, // Simplified
            FillPrice::Worst => {
                // Worst price for the trade
                match (direction, is_entry) {
                    (Direction::Long, true) => bar.high,   // Buy high
                    (Direction::Long, false) => bar.low,   // Sell low
                    (Direction::Short, true) => bar.low,   // Short at low (bad)
                    (Direction::Short, false) => bar.high, // Cover at high (bad)
                }
            }
            FillPrice::Best => {
                // Best price for the trade
                match (direction, is_entry) {
                    (Direction::Long, true) => bar.low,   // Buy low
                    (Direction::Long, false) => bar.high, // Sell high
                    (Direction::Short, true) => bar.high, // Short at high (good)
                    (Direction::Short, false) => bar.low, // Cover at low (good)
                }
            }
        }
    }

    /// Get execution price from separate arrays.
    ///
    /// # Arguments
    /// * `open` - Open price
    /// * `high` - High price
    /// * `low` - Low price
    /// * `close` - Close price
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Execution price
    pub fn get_price_from_arrays(
        &self,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        direction: Direction,
        is_entry: bool,
    ) -> Price {
        match self {
            FillPrice::Close => close,
            FillPrice::Open => open,
            FillPrice::Average => (open + high + low + close) / 4.0,
            FillPrice::Typical => (high + low + close) / 3.0,
            FillPrice::Vwap => (high + low + close) / 3.0,
            FillPrice::Worst => match (direction, is_entry) {
                (Direction::Long, true) => high,
                (Direction::Long, false) => low,
                (Direction::Short, true) => low,
                (Direction::Short, false) => high,
            },
            FillPrice::Best => match (direction, is_entry) {
                (Direction::Long, true) => low,
                (Direction::Long, false) => high,
                (Direction::Short, true) => high,
                (Direction::Short, false) => low,
            },
        }
    }
}

/// Deterministic SplitMix64 stream for stochastic fills.
///
/// Hand-rolled (like the Monte Carlo module's generator) to avoid an RNG
/// dependency; the same seed always produces the same fill sequence.
#[derive(Debug, Clone)]
pub struct FillRng(u64);

impl FillRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next uniform draw in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Number of prints a bar is replayed as: open, high, low, close.
pub const PRINTS_PER_BAR: usize = 4;

/// One print on the tape: the price that traded, and the size shown at it.
pub type Print = (Price, f64);

/// What an order does once the last print offered to it runs out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tail {
    /// Nothing. The order is queued at a price the market only touched, so
    /// it takes what printed and keeps waiting for the rest.
    #[default]
    Rests,
    /// The market traded *through* a resting order. It was at that price
    /// before the move, so the whole remainder fills at its own price -- it
    /// is the one being traded against, not the one crossing.
    Through,
    /// An aggressive order emptied the book at the price it took. Whatever
    /// is left crosses one increment worse, in one fill, unbounded.
    Sweep,
}

/// What a bar still owes an order after the print it has just taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NextPrint {
    /// Another print at the order's own price, of the size this depth says.
    Same(FillDepth),
    /// The market moved through a resting order: the rest fills at its price.
    Through,
    /// The book is empty at the order's price. Whatever is left crosses one
    /// increment worse, in one fill, unbounded by the bar.
    Sweep,
}

/// What a bar offered one order: the prints it gets, in order, and what
/// becomes of any remainder.
///
/// Read it as a schedule rather than a quantity. Each entry is one fill at
/// the order's price; [`tail`] says what happens to what is left once they
/// run out. Consuming it is a fold: take [`cap`], fill that much, then
/// follow [`next`] until it runs out.
///
/// [`cap`]: FillDepth::cap
/// [`next`]: FillDepth::next
/// [`tail`]: FillDepth::tail
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillDepth {
    /// Sizes of the prints still to take, soonest first. Only the first
    /// `prints` entries are real.
    sizes: [f64; PRINTS_PER_BAR],
    /// How many prints remain.
    prints: u32,
    /// What the remainder does once those prints are spent.
    tail: Tail,
}

impl FillDepth {
    /// Depth that bounds nothing: one fill, for whatever was asked.
    pub const UNLIMITED: Self = Self::single(f64::INFINITY, Tail::Rests);

    /// A bar that offered the order nothing at all.
    pub const NONE: Self = Self { sizes: [0.0; PRINTS_PER_BAR], prints: 0, tail: Tail::Rests };

    /// One print of `size`, with `tail` describing the remainder.
    pub const fn single(size: f64, tail: Tail) -> Self {
        let mut sizes = [0.0; PRINTS_PER_BAR];
        sizes[0] = size;
        Self { sizes, prints: 1, tail }
    }

    /// The same print offered `times` over, with `tail` after the last.
    ///
    /// A book that is read more than once without moving offers the same
    /// size each time; `times` is clamped to what a bar can hold.
    pub const fn repeated(size: f64, times: usize, tail: Tail) -> Self {
        let prints = if times > PRINTS_PER_BAR { PRINTS_PER_BAR } else { times };
        let mut sizes = [0.0; PRINTS_PER_BAR];
        let mut i = 0;
        while i < prints {
            sizes[i] = size;
            i += 1;
        }
        Self { sizes, prints: prints as u32, tail }
    }

    /// Whether the bar offered this order no print at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prints == 0
    }

    /// How much the next fill may take. Zero once the bar is spent.
    #[inline]
    pub fn cap(&self) -> f64 {
        if self.prints == 0 { 0.0 } else { self.sizes[0] }
    }

    /// What follows the print [`cap`] describes, if anything.
    ///
    /// [`cap`]: FillDepth::cap
    pub fn next(&self) -> Option<NextPrint> {
        match self.prints {
            0 => None,
            1 => match self.tail {
                Tail::Rests => None,
                Tail::Through => Some(NextPrint::Through),
                Tail::Sweep => Some(NextPrint::Sweep),
            },
            _ => {
                let mut sizes = [0.0; PRINTS_PER_BAR];
                sizes[..PRINTS_PER_BAR - 1].copy_from_slice(&self.sizes[1..]);
                Some(NextPrint::Same(Self { sizes, prints: self.prints - 1, ..*self }))
            }
        }
    }
}

/// How much of a bar's traded volume one aggressive order may consume.
///
/// A bar is a summary, not a tape: it says a range was traded and how much
/// changed hands, but not in what order or against whose resting size. An
/// engine that fills every order for its full size is therefore assuming
/// unbounded depth at the touch, which is fine for a liquid instrument and
/// a small order and wrong for the case the model exists to catch -- an
/// order large relative to what actually traded.
///
/// [`Unlimited`] is that historical assumption and stays the default, so
/// enabling a bound is always a deliberate choice.
///
/// [`Unlimited`]: BarLiquidity::Unlimited
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BarLiquidity {
    /// Volume is ignored: every order fills for its whole size.
    #[default]
    Unlimited,
    /// The bar is replayed as `slices` prints and an order takes only what
    /// the prints it is marketable against actually show.
    ///
    /// Every print but the bar's last shows `volume / slices` floored onto
    /// the size grid; the last carries the remainder, so the prints sum to
    /// the bar's volume exactly rather than losing up to `slices - 1`
    /// increments to rounding. Neither may round away to nothing: a print
    /// of zero size is a print that did not happen, so both floor at one
    /// increment.
    ///
    /// This is Nautilus Trader's bar-execution model at `slices = 4`: it
    /// synthesizes four ticks per bar, one per OHLC price. Matching that
    /// number is what lets a Raptor run agree with a Nautilus run on fill
    /// *sizes* and not merely on prices.
    VolumeShare {
        /// Prints per bar. Must be positive; `4.0` mirrors Nautilus.
        slices: f64,
    },
}

impl BarLiquidity {
    /// Nautilus Trader's bar-execution depth: four prints per bar.
    pub const NAUTILUS: Self = Self::VolumeShare { slices: 4.0 };

    /// Whether this model bounds a fill at all.
    #[inline]
    pub fn is_bounded(&self) -> bool {
        matches!(self, Self::VolumeShare { .. })
    }

    /// Size every print but the bar's last shows, on the `quantum` size grid.
    ///
    /// Returns [`f64::INFINITY`] when no bound applies, so callers can clamp
    /// unconditionally with `size.min(..)`.
    ///
    /// A bar whose volume is missing or negative carries no information
    /// about depth, and treating "unknown" as "none available" would
    /// silently halt a whole run on a data gap. Such a bar is left
    /// unconstrained instead. A volume of *zero* is not unknown -- it says
    /// nothing traded -- and yields the smallest print the grid allows,
    /// which is what a venue that cannot quote a fraction of a lot shows.
    #[inline]
    pub fn share(&self, volume: f64, quantum: f64) -> f64 {
        match self {
            Self::Unlimited => f64::INFINITY,
            Self::VolumeShare { slices } => {
                if !(volume >= 0.0) || !volume.is_finite() || !(*slices > 0.0) {
                    f64::INFINITY
                } else if quantum > 0.0 && quantum.is_finite() {
                    floor_to_lot(volume / slices, quantum).max(quantum)
                } else if volume > 0.0 {
                    volume / slices
                } else {
                    // No grid to floor onto and nothing traded: there is no
                    // smallest print to fall back to, so bound nothing.
                    f64::INFINITY
                }
            }
        }
    }

    /// Size the bar's closing print shows: everything the others left over.
    ///
    /// Snapped back onto the size grid, because the subtraction that finds
    /// the remainder is exact in decimal and not in binary -- and this is
    /// the print an order most often meets, since an order submitted during
    /// a bar matches against its close.
    ///
    /// The earlier prints are floored *up* to one increment each, so on a
    /// nearly empty bar they can between them claim more than the volume.
    /// The close then shows one increment rather than a negative size.
    #[inline]
    pub fn last_share(&self, volume: f64, quantum: f64) -> f64 {
        let share = self.share(volume, quantum);
        match self {
            Self::VolumeShare { slices } if share.is_finite() => {
                let claimed = (slices - 1.0) * share;
                if claimed >= volume {
                    return quantum.max(0.0);
                }
                let floor = if quantum > 0.0 { quantum } else { 0.0 };
                snap_to_lot_grid(volume - claimed, quantum).max(floor)
            }
            _ => share,
        }
    }
}

/// The prints one step put on the tape, in the order they traded.
///
/// `unbounded` marks a tape produced under a liquidity model that bounds
/// nothing: it still records where the market went, but every order it is
/// offered to fills for whatever it asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tape {
    prints: [Print; PRINTS_PER_BAR],
    len: usize,
    unbounded: bool,
}

impl Tape {
    /// A step under a liquidity model that bounds nothing.
    pub const UNBOUNDED: Self =
        Self { prints: [(0.0, f64::INFINITY); PRINTS_PER_BAR], len: 0, unbounded: true };

    const EMPTY: Self = Self { prints: [(0.0, 0.0); PRINTS_PER_BAR], len: 0, unbounded: false };

    fn push(&mut self, print: Print) {
        if self.len < PRINTS_PER_BAR {
            self.prints[self.len] = print;
            self.len += 1;
        }
    }

    /// The prints this step put up, soonest first.
    pub fn prints(&self) -> &[Print] {
        &self.prints[..self.len]
    }

    /// What this step offered an order already resting at `price`.
    ///
    /// Every print the order is marketable against shows one print's size,
    /// and the order takes that much and no more: it is queued behind the
    /// size already resting at its price. The first print that trades
    /// *through* the order is different -- the order was there before the
    /// market moved, so the whole remainder fills at the order's own price.
    ///
    /// `immediate` marks an order canceled the instant its first fill lands
    /// (IOC, FOK). It never survives to take a second print or a tail.
    pub fn offered(&self, price: Price, buying: bool, immediate: bool) -> FillDepth {
        if self.unbounded {
            return FillDepth::UNLIMITED;
        }
        let mut sizes = [0.0; PRINTS_PER_BAR];
        let mut prints = 0u32;
        for &(print_price, print_size) in self.prints() {
            let marketable = if buying { print_price <= price } else { print_price >= price };
            if !marketable {
                continue;
            }
            sizes[prints as usize] = print_size;
            prints += 1;
            if immediate {
                return FillDepth { sizes, prints, tail: Tail::Rests };
            }
            if print_price != price {
                return FillDepth { sizes, prints, tail: Tail::Through };
            }
        }
        FillDepth { sizes, prints, tail: Tail::Rests }
    }
}

/// Which kind of step is being replayed onto the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// A bar, replayed as up to four prints.
    Bar,
    /// A single trade print, which always trades whatever its price.
    Print,
}

/// The venue's tape: the last price to trade, and the size showing there.
///
/// A bar is replayed as up to four prints -- open, high, low, close -- but
/// only a price that *moves* the tape prints at all: the open is skipped
/// when it repeats where the tape already sits, the high and the low when
/// they do not extend past it, the close when it is already there. A bar
/// whose whole range collapses onto the last traded price therefore prints
/// nothing, matches no resting order, and leaves the book exactly as it
/// was -- so an order arriving on such a bar is filled against a size that
/// may be many bars old.
///
/// That carry-over is why this is state and not a function of one bar, and
/// it is not a detail: on a quiet instrument the last real print's size and
/// the quiet bar's own volume routinely differ by two orders of magnitude,
/// and the fill sizes follow the former.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BarTape {
    book: Option<Print>,
}

impl BarTape {
    /// The book this tape currently shows, if anything has traded yet.
    pub fn book(&self) -> Option<Print> {
        self.book
    }

    /// Size the book currently shows, unbounded until something trades.
    pub fn book_size(&self) -> f64 {
        self.book.map_or(f64::INFINITY, |(_, size)| size)
    }

    /// Replay one step onto the tape and return the prints it put up.
    pub fn replay(
        &mut self,
        bar: &OhlcvBar,
        liquidity: BarLiquidity,
        quantum: f64,
        kind: StepKind,
    ) -> Tape {
        let share = liquidity.share(bar.volume, quantum);
        if !share.is_finite() {
            // Nothing bounds a fill here, but the tape still tracks where
            // the market is so a later bounded step starts from the truth.
            self.book = Some((bar.close, f64::INFINITY));
            return Tape::UNBOUNDED;
        }
        let mut tape = Tape::EMPTY;
        if kind == StepKind::Print {
            // A trade is a trade: it prints at its price whether or not the
            // tape already sits there.
            tape.push((bar.close, share));
            self.book = Some((bar.close, share));
            return tape;
        }
        // Where the tape sits, walked forward one print at a time. Nothing
        // that is already here prints again.
        let mut here = self.book.map_or(bar.open, |(price, _)| price);
        if self.book.is_none() || here != bar.open {
            tape.push((bar.open, share));
            here = bar.open;
        }
        if bar.high > here {
            tape.push((bar.high, share));
            here = bar.high;
        }
        if bar.low < here {
            tape.push((bar.low, share));
            here = bar.low;
        }
        if bar.close != here {
            tape.push((bar.close, liquidity.last_share(bar.volume, quantum)));
        }
        if let Some(&print) = tape.prints().last() {
            self.book = Some(print);
        }
        tape
    }

    /// What the book offers an order arriving marketable against it.
    ///
    /// This is the aggressive side of the same tape: an order submitted
    /// while a bar was being observed meets whatever the last print left
    /// showing and takes it.
    ///
    /// What happens to the remainder depends on where the order was priced.
    /// An order priced *through* the book crosses it, and the rest of the
    /// size pays one increment worse. An order priced *at* the book has
    /// nothing to cross: it joins the queue at that price. It does get one
    /// further bite of the same size, because the venue settles the new
    /// order and then walks its book once more at the same instant, and the
    /// book has not moved in between -- but only one, and only at its own
    /// price.
    ///
    /// An immediate-or-cancel order never sees that second walk: it is
    /// killed the moment its first fill lands.
    pub fn offered(&self, price: Price, buying: bool, immediate: bool) -> FillDepth {
        let Some((book_price, size)) = self.book else {
            return FillDepth::NONE;
        };
        let marketable = if buying { book_price <= price } else { book_price >= price };
        if !marketable {
            return FillDepth::NONE;
        }
        if !size.is_finite() {
            return FillDepth::UNLIMITED;
        }
        match (immediate, book_price == price) {
            (true, _) => FillDepth::single(size, Tail::Rests),
            (false, true) => FillDepth::repeated(size, 2, Tail::Rests),
            (false, false) => FillDepth::single(size, Tail::Sweep),
        }
    }
}

/// Fill model combining price model with execution rules.
#[derive(Debug, Clone)]
pub struct FillModel {
    /// Price model for fills.
    pub fill_price: FillPrice,
    /// How much of a bar's traded volume one aggressive order may consume.
    pub bar_liquidity: BarLiquidity,
    /// The instrument's size grid, which a bar's prints are floored onto.
    /// `0.0` (default) leaves print sizes unrounded.
    pub size_quantum: f64,
    /// Adverse price adjustment applied to limit fills, as a fraction of
    /// the limit price. `0.0` (default) fills exactly at the limit.
    ///
    /// Models being filled only when the market is about to move through
    /// you — adverse selection on a resting order. It does *not* apply
    /// when the queue model granted the fill: observed volume traded ahead
    /// of you is evidence you genuinely held that price.
    pub limit_slippage: f64,
}

impl Default for FillModel {
    fn default() -> Self {
        Self {
            fill_price: FillPrice::Close,
            bar_liquidity: BarLiquidity::Unlimited,
            size_quantum: 0.0,
            limit_slippage: 0.0,
        }
    }
}

impl FillModel {
    /// Create a fill model that executes at close.
    pub fn at_close() -> Self {
        Self { fill_price: FillPrice::Close, ..Self::default() }
    }

    /// Set the bar liquidity model.
    pub fn with_bar_liquidity(mut self, liquidity: BarLiquidity) -> Self {
        self.bar_liquidity = liquidity;
        self
    }

    /// Check if a limit order would be filled.
    ///
    /// # Arguments
    /// * `limit_price` - Limit price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// True if order would be filled
    pub fn would_fill_limit(
        &self,
        limit_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> bool {
        match (direction, is_entry) {
            // Long entry: buy at or below limit
            (Direction::Long, true) => bar.low <= limit_price,
            // Long exit: sell at or above limit
            (Direction::Long, false) => bar.high >= limit_price,
            // Short entry: sell at or above limit
            (Direction::Short, true) => bar.high >= limit_price,
            // Short exit: buy at or below limit
            (Direction::Short, false) => bar.low <= limit_price,
        }
    }

    /// Get fill price for a limit order.
    ///
    /// Returns limit price if filled, None if not filled.
    ///
    /// # Arguments
    /// * `limit_price` - Limit price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Fill price or None
    pub fn get_limit_fill_price(
        &self,
        limit_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> Option<Price> {
        if !self.would_fill_limit(limit_price, bar, direction, is_entry) {
            return None;
        }
        if self.limit_slippage == 0.0 {
            return Some(limit_price);
        }
        // Slip against the trader: a buy pays more, a sell receives less.
        let adjust = limit_price * self.limit_slippage;
        Some(match (direction, is_entry) {
            (Direction::Long, true) | (Direction::Short, false) => limit_price + adjust,
            _ => limit_price - adjust,
        })
    }

    /// Check if a stop order would be triggered.
    ///
    /// # Arguments
    /// * `stop_price` - Stop price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// True if stop would be triggered
    pub fn would_trigger_stop(
        &self,
        stop_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> bool {
        match (direction, is_entry) {
            // Long entry stop: buy when price rises to stop
            (Direction::Long, true) => bar.high >= stop_price,
            // Long exit stop: sell when price falls to stop
            (Direction::Long, false) => bar.low <= stop_price,
            // Short entry stop: sell when price falls to stop
            (Direction::Short, true) => bar.low <= stop_price,
            // Short exit stop: buy when price rises to stop
            (Direction::Short, false) => bar.high >= stop_price,
        }
    }

    /// Get fill price for a stop order.
    ///
    /// Returns fill price if triggered, None if not.
    /// Uses worst-case scenario (stop price or worse).
    ///
    /// # Arguments
    /// * `stop_price` - Stop price
    /// * `bar` - OHLCV bar
    /// * `direction` - Trade direction
    /// * `is_entry` - Whether this is an entry or exit
    ///
    /// # Returns
    /// Fill price or None
    pub fn get_stop_fill_price(
        &self,
        stop_price: Price,
        bar: &OhlcvBar,
        direction: Direction,
        is_entry: bool,
    ) -> Option<Price> {
        if !self.would_trigger_stop(stop_price, bar, direction, is_entry) {
            return None;
        }

        // Check for gap through stop
        match (direction, is_entry) {
            (Direction::Long, true) => {
                // Buy stop: fill at stop or worse (gap up through stop)
                if bar.open >= stop_price {
                    Some(bar.open) // Gap up, fill at open
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Long, false) => {
                // Sell stop: fill at stop or worse (gap down through stop)
                if bar.open <= stop_price {
                    Some(bar.open) // Gap down, fill at open
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Short, true) => {
                // Short stop: fill at stop or worse (gap down through stop)
                if bar.open <= stop_price {
                    Some(bar.open)
                } else {
                    Some(stop_price)
                }
            }
            (Direction::Short, false) => {
                // Cover stop: fill at stop or worse (gap up through stop)
                if bar.open >= stop_price {
                    Some(bar.open)
                } else {
                    Some(stop_price)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_fills_at_the_limit_by_default() {
        let model = FillModel::default();
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 98.0,
            close: 100.0,
            volume: 1.0,
        };
        assert_eq!(model.get_limit_fill_price(99.0, &bar, Direction::Long, true), Some(99.0));
    }

    /// Sizes a depth hands out, in order, and what its remainder does.
    fn schedule(depth: FillDepth) -> (Vec<f64>, Tail) {
        let mut sizes = Vec::new();
        let mut step = depth;
        loop {
            if step.is_empty() {
                return (sizes, Tail::Rests);
            }
            sizes.push(step.cap());
            match step.next() {
                Some(NextPrint::Same(rest)) => step = rest,
                Some(NextPrint::Through) => return (sizes, Tail::Through),
                Some(NextPrint::Sweep) => return (sizes, Tail::Sweep),
                None => return (sizes, Tail::Rests),
            }
        }
    }

    /// One bar's worth of tape, replayed onto a fresh (empty) book.
    fn replayed(prints: [Price; PRINTS_PER_BAR], volume: f64, quantum: f64) -> Tape {
        BarTape::default().replay(&priced(prints, volume), BarLiquidity::NAUTILUS, quantum, StepKind::Bar)
    }

    fn priced(prints: [Price; PRINTS_PER_BAR], volume: f64) -> OhlcvBar {
        OhlcvBar {
            timestamp: 0,
            open: prints[0],
            high: prints[1],
            low: prints[2],
            close: prints[3],
            volume,
        }
    }

    const BAR: [Price; PRINTS_PER_BAR] = [100.0, 110.0, 90.0, 105.0];

    #[test]
    fn unlimited_liquidity_never_bounds_a_fill() {
        assert_eq!(BarLiquidity::Unlimited.share(1_000.0, 0.0), f64::INFINITY);
        let mut tape = BarTape::default();
        let step = tape.replay(&priced(BAR, 1_000.0), BarLiquidity::Unlimited, 0.0, StepKind::Bar);
        assert_eq!(step.offered(90.0, true, false), FillDepth::UNLIMITED);
        // The book still says where the market is -- an order it cannot
        // reach is unfilled, not unbounded -- but it bounds no size.
        assert_eq!(tape.offered(105.0, true, false), FillDepth::UNLIMITED);
        assert_eq!(tape.offered(90.0, true, false), FillDepth::NONE);
    }

    #[test]
    fn a_volume_share_is_one_print_of_the_bar() {
        // The rule Nautilus implements: four ticks, a quarter of the volume
        // each.
        assert_eq!(BarLiquidity::NAUTILUS.share(1_000.0, 0.0), 250.0);
        assert_eq!(BarLiquidity::VolumeShare { slices: 2.0 }.share(1_000.0, 0.0), 500.0);
    }

    #[test]
    fn the_closing_print_carries_what_the_others_rounded_away() {
        // A quarter of 0.3847 is 0.096175, which is off a 0.00001 grid: the
        // first three prints show 0.09617 and the close absorbs the 0.00002
        // they left, so the four sum to the bar's volume exactly.
        let nautilus = BarLiquidity::NAUTILUS;
        assert_eq!(nautilus.share(0.3847, 0.00001), 0.09617);
        assert!((nautilus.last_share(0.3847, 0.00001) - 0.09619).abs() < 1e-12);
        let total = 3.0 * nautilus.share(0.3847, 0.00001) + nautilus.last_share(0.3847, 0.00001);
        assert!((total - 0.3847).abs() < 1e-12);
    }

    #[test]
    fn a_grid_the_volume_already_sits_on_leaves_no_remainder() {
        let nautilus = BarLiquidity::NAUTILUS;
        assert_eq!(nautilus.share(1_000.0, 0.001), 250.0);
        assert_eq!(nautilus.last_share(1_000.0, 0.001), 250.0);
    }

    #[test]
    fn no_print_can_round_away_to_nothing() {
        // A quarter of this bar is under one increment. A print of zero
        // size is a print that did not happen, so every print floors at the
        // smallest size the grid can express -- and the three that do so
        // between them claim the whole bar, leaving the close the same
        // floor rather than a negative size.
        let nautilus = BarLiquidity::NAUTILUS;
        assert_eq!(nautilus.share(3.0, 1.0), 1.0);
        assert_eq!(nautilus.last_share(3.0, 1.0), 1.0);
        // A bar on which nothing traded still shows the smallest print.
        assert_eq!(nautilus.share(0.0, 1.0), 1.0);
        assert_eq!(nautilus.last_share(0.0, 1.0), 1.0);
    }

    #[test]
    fn a_bar_with_no_volume_bounds_nothing() {
        // "Unknown" must not read as "none available": a data gap would
        // otherwise halt every fill for the rest of the run.
        for volume in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                replayed(BAR, volume, 0.0).offered(90.0, true, false),
                FillDepth::UNLIMITED,
                "volume {volume} should not bound a fill"
            );
        }
    }

    #[test]
    fn a_buy_resting_on_the_low_takes_only_what_the_low_printed() {
        // The bar never traded below 90, so nothing was ever offered under
        // the order: it gets the one print that touched it and stays put.
        let depth = replayed(BAR, 1_000.0, 0.0).offered(90.0, true, false);
        assert_eq!(schedule(depth), (vec![250.0], Tail::Rests));
    }

    #[test]
    fn a_bar_that_trades_through_a_resting_order_fills_it_at_its_own_price() {
        // 90 is below the order's 95. The order was at 95 before the market
        // went there, so it is the one being traded against: the whole
        // remainder fills at 95 rather than crossing for a worse price.
        let depth = replayed(BAR, 1_000.0, 0.0).offered(95.0, true, false);
        assert_eq!(schedule(depth), (vec![250.0], Tail::Through));
    }

    #[test]
    fn every_print_at_the_order_price_adds_one_before_the_market_moves_through() {
        // The open sat exactly on the order and the low went through it:
        // two prints of size, and then the rest at the order's price.
        let depth = replayed([95.0, 110.0, 90.0, 105.0], 1_000.0, 0.0).offered(95.0, true, false);
        assert_eq!(schedule(depth), (vec![250.0, 250.0], Tail::Through));
    }

    #[test]
    fn a_price_the_tape_already_sits_on_does_not_print() {
        // A bar that closed on its low prints that price once, not twice --
        // the close is where the low already left the tape. Separated
        // repeats are a different matter: the open and the low below are
        // both 90, and the high between them makes each its own print.
        let closed_on_low = replayed([100.0, 110.0, 90.0, 90.0], 1_000.0, 0.0);
        assert_eq!(schedule(closed_on_low.offered(90.0, true, false)), (vec![250.0], Tail::Rests));

        let returned_to_the_open = replayed([90.0, 110.0, 90.0, 105.0], 1_000.0, 0.0);
        assert_eq!(
            schedule(returned_to_the_open.offered(90.0, true, false)),
            (vec![250.0, 250.0], Tail::Rests)
        );
    }

    #[test]
    fn a_bar_that_never_leaves_the_last_traded_price_prints_nothing() {
        // Every one of the four prices is where the tape already sits, so
        // none of them moves it: no print, no match, and the book keeps the
        // size it was showing -- which is how a quiet bar fills an order
        // against liquidity that traded long before it.
        let mut tape = BarTape::default();
        let opening = tape.replay(&priced([10.0; 4], 4_000.0), BarLiquidity::NAUTILUS, 0.01, StepKind::Bar);
        assert_eq!(schedule(opening.offered(10.0, true, false)), (vec![1_000.0], Tail::Rests));
        assert_eq!(tape.book(), Some((10.0, 1_000.0)));

        let quiet = tape.replay(&priced([10.0; 4], 8.0), BarLiquidity::NAUTILUS, 0.01, StepKind::Bar);
        assert_eq!(quiet.prints(), &[]);
        assert_eq!(quiet.offered(10.0, true, false), FillDepth::NONE);
        // The order arriving on the quiet bar still meets the old book.
        assert_eq!(tape.book(), Some((10.0, 1_000.0)));
        assert_eq!(
            schedule(tape.offered(10.0, true, false)),
            (vec![1_000.0, 1_000.0], Tail::Rests)
        );
    }

    #[test]
    fn an_order_priced_at_the_book_gets_a_second_bite_of_it_but_no_more() {
        // The venue settles the incoming order against the book, then walks
        // the book once more at the same instant before any new data. The
        // book has not moved in between, so an order that could only take
        // what it showed takes that much again -- at its own price, never
        // crossing -- and then waits.
        let bar = priced([10.0; 4], 4_000.0);
        let mut tape = BarTape::default();
        tape.replay(&bar, BarLiquidity::NAUTILUS, 0.01, StepKind::Bar);

        assert_eq!(
            schedule(tape.offered(10.0, true, false)),
            (vec![1_000.0, 1_000.0], Tail::Rests)
        );
        // Immediate-or-cancel is killed the moment its first fill lands, so
        // the second walk finds nothing of it left.
        assert_eq!(schedule(tape.offered(10.0, true, true)), (vec![1_000.0], Tail::Rests));
        // Priced through the book, it crosses instead, and the crossing
        // fills the whole remainder at once.
        assert_eq!(schedule(tape.offered(10.01, true, false)), (vec![1_000.0], Tail::Sweep));
    }

    #[test]
    fn a_trade_print_trades_wherever_the_tape_already_is() {
        // A bar summarizes; a trade is a fact. It prints at its price even
        // when that is exactly where the tape already sits.
        let mut tape = BarTape::default();
        tape.replay(&priced([10.0; 4], 4_000.0), BarLiquidity::NAUTILUS, 0.01, StepKind::Bar);
        let tick = tape.replay(&priced([10.0; 4], 8.0), BarLiquidity::NAUTILUS, 0.01, StepKind::Print);
        assert_eq!(tick.prints(), &[(10.0, 2.0)]);
        assert_eq!(tape.book(), Some((10.0, 2.0)));
    }

    #[test]
    fn a_sell_reads_the_high_the_way_a_buy_reads_the_low() {
        let bar = replayed(BAR, 1_000.0, 0.0);
        assert_eq!(schedule(bar.offered(110.0, false, false)), (vec![250.0], Tail::Rests));
        assert_eq!(schedule(bar.offered(105.0, false, false)), (vec![250.0], Tail::Through));
    }

    #[test]
    fn a_bar_that_never_reaches_the_order_offers_nothing() {
        let depth = replayed(BAR, 1_000.0, 0.0).offered(80.0, true, false);
        assert_eq!(depth, FillDepth::NONE);
        assert_eq!(schedule(depth), (vec![], Tail::Rests));
    }

    #[test]
    fn an_immediate_order_never_walks_past_the_first_print() {
        // IOC and FOK are canceled the instant their first fill lands, so
        // trading through them buys them nothing.
        let depth = replayed(BAR, 1_000.0, 0.0).offered(95.0, true, true);
        assert_eq!(schedule(depth), (vec![250.0], Tail::Rests));
    }

    #[test]
    fn an_order_submitted_mid_bar_meets_only_the_book_the_bar_left() {
        // The open, high and low are behind it; reading them would be
        // look-ahead. What it meets is the closing print, which is also the
        // one carrying the bar's rounding remainder.
        let nautilus = BarLiquidity::NAUTILUS;
        let last = nautilus.last_share(0.3847, 0.00001);
        let mut tape = BarTape::default();
        tape.replay(&priced([87_900.0, 88_100.0, 87_800.0, 88_000.0], 0.3847), nautilus, 0.00001, StepKind::Bar);

        // Priced at the book, the order joins the queue there: it takes
        // what the book shows, twice, and never crosses.
        assert_eq!(
            schedule(tape.offered(88_000.0, true, false)),
            (vec![last, last], Tail::Rests)
        );

        // A limit through that close crosses: it empties the book beneath
        // it and sweeps one increment worse for the rest.
        assert_eq!(schedule(tape.offered(88_100.0, true, false)), (vec![last], Tail::Sweep));

        // And a book the order cannot reach offers it nothing.
        assert_eq!(tape.offered(87_900.0, true, false), FillDepth::NONE);
    }

    #[test]
    fn limit_slippage_moves_against_the_trader() {
        let model = FillModel { limit_slippage: 0.01, ..FillModel::default() };
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 98.0,
            close: 100.0,
            volume: 1.0,
        };
        // A buy pays more than its limit.
        assert_eq!(model.get_limit_fill_price(100.0, &bar, Direction::Long, true), Some(101.0));
        // A sell receives less.
        assert_eq!(model.get_limit_fill_price(100.0, &bar, Direction::Short, true), Some(99.0));
    }

    #[test]
    fn limit_slippage_does_not_create_fills() {
        // A price the market never reached stays unfilled however the
        // slippage is configured.
        let model = FillModel { limit_slippage: 0.05, ..FillModel::default() };
        let bar = OhlcvBar {
            timestamp: 0,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.0,
            volume: 1.0,
        };
        assert_eq!(model.get_limit_fill_price(90.0, &bar, Direction::Long, true), None);
    }

    fn test_bar() -> OhlcvBar {
        OhlcvBar { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 }
    }

    #[test]
    fn test_fill_price_close() {
        let bar = test_bar();
        let fp = FillPrice::Close;
        assert!((fp.get_price(&bar, Direction::Long, true) - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_price_worst() {
        let bar = test_bar();
        let fp = FillPrice::Worst;

        // Long entry: high (105)
        assert!((fp.get_price(&bar, Direction::Long, true) - 105.0).abs() < 1e-10);

        // Long exit: low (95)
        assert!((fp.get_price(&bar, Direction::Long, false) - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_limit_fill() {
        let fill = FillModel::default();
        let bar = test_bar();

        // Limit buy at 96 should fill (low is 95)
        assert!(fill.would_fill_limit(96.0, &bar, Direction::Long, true));

        // Limit buy at 94 should not fill (low is 95)
        assert!(!fill.would_fill_limit(94.0, &bar, Direction::Long, true));
    }

    #[test]
    fn test_stop_fill() {
        let fill = FillModel::default();
        let bar = test_bar();

        // Stop sell at 96 should trigger (low is 95)
        assert!(fill.would_trigger_stop(96.0, &bar, Direction::Long, false));

        // Stop sell at 94 should not trigger (low is 95)
        assert!(!fill.would_trigger_stop(94.0, &bar, Direction::Long, false));
    }

    #[test]
    fn test_gap_through_stop() {
        let fill = FillModel::default();

        // Gap down through stop
        let gap_bar = OhlcvBar {
            timestamp: 0,
            open: 90.0, // Gap down from stop at 95
            high: 92.0,
            low: 88.0,
            close: 91.0,
            volume: 1000.0,
        };

        let fill_price = fill.get_stop_fill_price(95.0, &gap_bar, Direction::Long, false);
        // Should fill at open (90) not stop (95)
        assert_eq!(fill_price, Some(90.0));
    }
}
