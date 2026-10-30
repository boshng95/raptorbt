//! Twins of the engine's talipp indicators (`src/indicators/talipp_indicators.py`
//! in the engine): talipp's pure-Python state machines, which the engine's
//! `TalippIndicator` wrapper feeds one bar at a time and reads back.
//!
//! talipp's `Indicator.add` gives every sub-indicator the input first,
//! appends it, computes a new output only for an input that is not `None`,
//! repeats the last output where the computation gives `None`, and hands the
//! output to its listeners. An output list is therefore a run of `None`
//! followed by values, so `has_valid_values(seq, w)` holds once `seq` has
//! taken `w` values and `exact=True` holds on exactly the `w`-th: the blocks
//! below count values instead of keeping lists.
//!
//! The wrapper purges talipp's lists to their newest `keep` items once they
//! hold `2 * keep`. A purge takes the same number of items off the front of
//! every list. Every list here but two is fed on every bar from a start
//! within the warm-up, which `keep` is four times over, so what a purge
//! leaves covers every look-back and no reading changes. The two exceptions
//! start or skip on the data (`ChaikinOsc` waits for a bar with a range,
//! `EMV` skips a bar it cannot divide by): they count what their averages
//! hold and decline the purge that would cut into what those read.

use super::nautilus::{boxed, require};
use super::window::Window;
use super::{div, first_max, first_min, Bar, Declined, Params, Step, Twin};
use crate::metrics::exact::{py_pow, py_sum};

const VALUE: &[&str] = &["value"];
const TREND: &[&str] = &["trend", "value"];

/// The wrapper's `_KEEP_MIN`.
const KEEP_MIN: usize = 2048;

/// Python's `float ** float`, declining where Python raises.
fn pow(base: f64, exponent: f64) -> Result<f64, Declined> {
    py_pow(base, exponent)
        .map_err(|error| Declined::new(format!("{base} ** {exponent} raises ({error:?})")))
}

extern "C" {
    /// The C library's `exp`, which `math.exp` calls.
    #[link_name = "exp"]
    fn c_exp(x: f64) -> f64;
}

/// `math.exp` of a float that is not positive, which cannot overflow.
fn exp(x: f64) -> f64 {
    // SAFETY: `exp` is a pure function of a double.
    unsafe { c_exp(std::hint::black_box(x)) }
}

/// The output talipp appends: the new one, or the last where there is none.
fn carry<T: Copy>(last: &mut Option<T>, new: Option<T>) -> Option<T> {
    if new.is_some() {
        *last = new;
    }
    *last
}

/// `SMA`.
struct Sma {
    period: usize,
    inputs: Window,
    seen: usize,
    last: Option<f64>,
}

impl Sma {
    fn new(period: usize) -> Self {
        Sma { period, inputs: Window::new(period + 1), seen: 0, last: None }
    }

    fn add(&mut self, input: Option<f64>) -> Option<f64> {
        let new = input.and_then(|x| {
            self.inputs.push(x);
            self.seen += 1;
            let period = self.period;
            if self.seen > period {
                Some(self.last? - (self.inputs.first() - x) / period as f64)
            } else if self.seen == period {
                Some(py_sum(self.inputs.as_slice()) / period as f64)
            } else {
                None
            }
        });
        carry(&mut self.last, new)
    }
}

/// `EMA`.
struct Ema {
    period: usize,
    firsts: Vec<f64>,
    seen: usize,
    last: Option<f64>,
}

impl Ema {
    fn new(period: usize) -> Self {
        Ema { period, firsts: Vec::with_capacity(period), seen: 0, last: None }
    }

    fn add(&mut self, input: Option<f64>) -> Option<f64> {
        let new = input.and_then(|x| {
            self.seen += 1;
            let period = self.period;
            if self.seen < period {
                self.firsts.push(x);
                None
            } else if self.seen == period {
                self.firsts.push(x);
                let first = py_sum(&self.firsts) / period as f64;
                self.firsts = Vec::new();
                Some(first)
            } else {
                let mult = 2.0 / (period as f64 + 1.0);
                Some(mult * x + (1.0 - mult) * self.last?)
            }
        });
        carry(&mut self.last, new)
    }
}

/// `WMA`.
struct Wma {
    period: usize,
    denom_sum: f64,
    inputs: Window,
    seen: usize,
    last: Option<f64>,
}

impl Wma {
    fn new(period: usize) -> Self {
        let denom_sum = (period * (period + 1)) as f64 / 2.0;
        Wma { period, denom_sum, inputs: Window::new(period), seen: 0, last: None }
    }

    fn add(&mut self, input: Option<f64>) -> Option<f64> {
        let new = input.and_then(|x| {
            self.inputs.push(x);
            self.seen += 1;
            if self.seen < self.period {
                return None;
            }
            // Newest first, the newest weighted by the period.
            let inputs = self.inputs.as_slice();
            let mut s = 0.0;
            for i in (1..=self.period).rev() {
                s += inputs[i - 1] * i as f64;
            }
            Some(s / self.denom_sum)
        });
        carry(&mut self.last, new)
    }
}

/// `ATR`, of whole bars.
struct Atr {
    period: usize,
    count: usize,
    close: f64,
    trs: Vec<f64>,
    last: Option<f64>,
}

impl Atr {
    fn new(period: usize) -> Self {
        Atr { period, count: 0, close: 0.0, trs: Vec::with_capacity(period), last: None }
    }

    fn add(&mut self, bar: &Bar) -> Option<f64> {
        self.count += 1;
        let range = bar.high - bar.low;
        let tr = if self.count == 1 {
            range
        } else {
            first_max([range, (bar.high - self.close).abs(), (bar.low - self.close).abs()])
        };
        self.close = bar.close;
        let period = self.period;
        let new = if self.count < period {
            self.trs.push(tr);
            None
        } else if self.count == period {
            self.trs.push(tr);
            let first = py_sum(&self.trs) / period as f64;
            self.trs = Vec::new();
            Some(first)
        } else {
            self.last.map(|last| (last * (period - 1) as f64 + tr) / period as f64)
        };
        carry(&mut self.last, new)
    }
}

/// `StdDev`.
struct StdDev {
    period: usize,
    inputs: Window,
    seen: usize,
    squares: Vec<f64>,
    last: Option<f64>,
}

impl StdDev {
    fn new(period: usize) -> Self {
        StdDev {
            period,
            inputs: Window::new(period),
            seen: 0,
            squares: Vec::with_capacity(period),
            last: None,
        }
    }

    fn add(&mut self, x: f64) -> Result<Option<f64>, Declined> {
        self.inputs.push(x);
        self.seen += 1;
        let new = if self.seen < self.period {
            None
        } else {
            let inputs = self.inputs.as_slice();
            let mean = py_sum(inputs) / self.period as f64;
            self.squares.clear();
            for &item in inputs {
                self.squares.push(pow(item - mean, 2.0)?);
            }
            Some((py_sum(&self.squares) / self.period as f64).sqrt())
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `ROC`.
struct Roc {
    period: usize,
    inputs: Window,
    seen: usize,
    last: Option<f64>,
}

impl Roc {
    fn new(period: usize) -> Self {
        Roc { period, inputs: Window::new(period + 1), seen: 0, last: None }
    }

    fn add(&mut self, x: f64) -> Result<Option<f64>, Declined> {
        self.inputs.push(x);
        self.seen += 1;
        let new = if self.seen <= self.period {
            None
        } else {
            let old = self.inputs.first();
            Some(div(100.0 * (x - old), old)?)
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `RSI`.
struct Rsi {
    period: usize,
    firsts: Vec<f64>,
    seen: usize,
    prev: f64,
    avg_gain: f64,
    avg_loss: f64,
    last: Option<f64>,
}

impl Rsi {
    fn new(period: usize) -> Self {
        Rsi {
            period,
            firsts: Vec::with_capacity(period),
            seen: 0,
            prev: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            last: None,
        }
    }

    fn add(&mut self, x: f64) -> Result<Option<f64>, Declined> {
        self.seen += 1;
        let period = self.period;
        let new = if self.seen <= period {
            self.firsts.push(x);
            None
        } else {
            if self.seen == period + 1 {
                // The first averages, of the changes among the first
                // `period` inputs only.
                let changes: Vec<f64> =
                    (1..period).map(|i| self.firsts[i] - self.firsts[i - 1]).collect();
                let gains: Vec<f64> =
                    changes.iter().copied().filter(|&change| change > 0.0).collect();
                let losses: Vec<f64> =
                    changes.iter().filter(|&&change| change < 0.0).map(|&change| -change).collect();
                self.avg_gain = div(py_sum(&gains), (period - 1) as f64)?;
                self.avg_loss = div(py_sum(&losses), (period - 1) as f64)?;
                self.firsts = Vec::new();
            }
            let change = x - self.prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };
            self.avg_gain = (self.avg_gain * (period - 1) as f64 + gain) / period as f64;
            self.avg_loss = (self.avg_loss * (period - 1) as f64 + loss) / period as f64;
            Some(if self.avg_loss == 0.0 {
                100.0
            } else {
                let rs = self.avg_gain / self.avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            })
        };
        self.prev = x;
        Ok(carry(&mut self.last, new))
    }
}

/// `AccuDist`.
#[derive(Default)]
struct AccuDist {
    last: Option<f64>,
}

impl AccuDist {
    fn add(&mut self, bar: &Bar) -> Option<f64> {
        let new = (bar.high != bar.low).then(|| {
            let mfm = ((bar.close - bar.low) - (bar.high - bar.close)) / (bar.high - bar.low);
            let mfv = mfm * bar.volume;
            match self.last {
                None => mfv,
                Some(last) => last + mfv,
            }
        });
        carry(&mut self.last, new)
    }
}

/// A talipp indicator behind the engine's `TalippIndicator` wrapper.
trait Inner {
    /// talipp's `add` of one bar, then the wrapper's `_read` of the newest
    /// output: whether it took a reading.
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined>;
    /// The wrapper's attributes `read` writes, in its order.
    fn names(&self) -> &'static [&'static str];
    fn read(&self, out: &mut [f64]);
    /// `purge_oldest(size)`.
    fn purge(&mut self, _size: usize) -> Step {
        Ok(())
    }
}

/// A talipp indicator whose output is one float, read as `value`.
trait Scalar {
    /// talipp's `add` of one bar: the newest output.
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined>;
    /// `purge_oldest(size)`.
    fn purge(&mut self, _size: usize) -> Step {
        Ok(())
    }
}

struct ValueOf<T> {
    inner: T,
    value: f64,
}

impl<T: Scalar> Inner for ValueOf<T> {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        Ok(match self.inner.add(bar)? {
            Some(value) => {
                self.value = value;
                true
            }
            None => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }

    fn purge(&mut self, size: usize) -> Step {
        self.inner.purge(size)
    }
}

/// `TalippIndicator.handle_bar`: add, purge, read, and initialize once the
/// readings pass `_skip`.
struct Wrapped<T> {
    inner: T,
    keep: usize,
    held: usize,
    skip: usize,
    readings: usize,
    initialized: bool,
}

impl<T: Inner> Twin for Wrapped<T> {
    fn update(&mut self, bar: &Bar) -> Step {
        // The newest output is read after the purge, which leaves it be.
        let read = self.inner.step(bar)?;
        self.held += 1;
        if self.held >= 2 * self.keep {
            self.inner.purge(self.held - self.keep)?;
            self.held = self.keep;
        }
        if read {
            self.readings += 1;
            if !self.initialized && self.readings > self.skip {
                self.initialized = true;
            }
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        self.inner.names()
    }

    fn read(&self, out: &mut [f64]) {
        self.inner.read(out);
    }
}

fn wrapped(
    inner: impl Inner + 'static,
    warmup: usize,
    skip: usize,
) -> Result<Box<dyn Twin>, Declined> {
    boxed(Wrapped {
        inner,
        keep: KEEP_MIN.max(4 * warmup),
        held: 0,
        skip,
        readings: 0,
        initialized: false,
    })
}

fn scalar(inner: impl Scalar + 'static, warmup: usize) -> Result<Box<dyn Twin>, Declined> {
    wrapped(ValueOf { inner, value: 0.0 }, warmup, 0)
}

/// Purge `size` items off an average fed `fed` values, declining where that
/// cuts into the `need` newest it reads.
fn purge_fed(fed: &mut usize, size: usize, need: usize) -> Step {
    let lost = size.min(*fed);
    *fed -= lost;
    require(lost == 0 || *fed >= need, "a purge cuts into an average that started late")
}

#[derive(Clone, Copy)]
struct Sar {
    value: f64,
    up: bool,
    ep: f64,
    accel_factor: f64,
}

/// `ParabolicSAR`.
struct ParabolicSar {
    init_accel_factor: f64,
    accel_factor_inc: f64,
    max_accel_factor: f64,
    highs: Window,
    lows: Window,
    seen: usize,
    last: Option<Sar>,
    value: f64,
    trend: f64,
}

impl ParabolicSar {
    const SAR_INIT_LEN: usize = 5;

    fn new(init_accel_factor: f64, accel_factor_inc: f64, max_accel_factor: f64) -> Self {
        ParabolicSar {
            init_accel_factor,
            accel_factor_inc,
            max_accel_factor,
            highs: Window::new(Self::SAR_INIT_LEN),
            lows: Window::new(Self::SAR_INIT_LEN),
            seen: 0,
            last: None,
            value: 0.0,
            trend: 0.0,
        }
    }

    fn next(&self, prev: Sar, bar: &Bar) -> Sar {
        let mut sar = prev.value + prev.accel_factor * (prev.ep - prev.value);
        let mut up = prev.up;
        let mut ep = prev.ep;
        let mut accel_factor = prev.accel_factor;
        let (high2, high3) = (self.highs.back(2), self.highs.back(3));
        let (low2, low3) = (self.lows.back(2), self.lows.back(3));
        if prev.up && sar > first_min([low2, low3]) {
            sar = first_min([low2, low3]);
        } else if !prev.up && sar < first_max([high2, high3]) {
            sar = first_max([high2, high3]);
        }
        if prev.up && bar.high > prev.ep {
            ep = bar.high;
        } else if !prev.up && bar.low < prev.ep {
            ep = bar.low;
        }
        if ep != prev.ep {
            accel_factor += self.accel_factor_inc;
            if accel_factor > self.max_accel_factor {
                accel_factor = self.max_accel_factor;
            }
        }
        if prev.up && sar > bar.low {
            sar = first_max([prev.ep, bar.high]);
            ep = bar.low;
            up = false;
            accel_factor = self.init_accel_factor;
        } else if !prev.up && sar < bar.high {
            sar = first_min([prev.ep, bar.low]);
            ep = bar.high;
            up = true;
            accel_factor = self.init_accel_factor;
        }
        Sar { value: sar, up, ep, accel_factor }
    }
}

impl Inner for ParabolicSar {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        self.highs.push(bar.high);
        self.lows.push(bar.low);
        self.seen += 1;
        let new = if self.seen < Self::SAR_INIT_LEN {
            None
        } else if self.seen == Self::SAR_INIT_LEN {
            Some(Sar {
                value: first_min(self.lows.as_slice().iter().copied()),
                up: true,
                ep: first_max(self.highs.as_slice().iter().copied()),
                accel_factor: self.init_accel_factor,
            })
        } else {
            self.last.map(|prev| self.next(prev, bar))
        };
        Ok(match carry(&mut self.last, new) {
            Some(sar) => {
                self.value = sar.value;
                self.trend = if sar.up { 1.0 } else { -1.0 };
                true
            }
            None => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        TREND
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.trend;
        out[1] = self.value;
    }
}

/// The last two items of a list talipp starts with an `int` 0.
#[derive(Clone, Copy, Default)]
struct LastTwo {
    len: usize,
    prev: f64,
    last: f64,
}

impl LastTwo {
    fn push(&mut self, value: f64) {
        self.prev = self.last;
        self.last = value;
        self.len += 1;
    }
}

/// `SuperTrend`.
struct SuperTrend {
    multiplier: f64,
    atr: Atr,
    prev_close: f64,
    fub: LastTwo,
    flb: LastTwo,
    last: Option<(f64, bool)>,
    value: f64,
    trend: f64,
}

impl Inner for SuperTrend {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let atr = self.atr.add(bar);
        let new = match atr {
            None => None,
            Some(atr) => {
                let close = bar.close;
                let hla = (bar.high + bar.low) / 2.0;
                let bub = hla + self.multiplier * atr;
                let blb = hla - self.multiplier * atr;
                let fub = if self.fub.len == 0 {
                    0.0
                } else if bub < self.fub.last || self.prev_close > self.fub.last {
                    bub
                } else {
                    self.fub.last
                };
                self.fub.push(fub);
                let flb = if self.flb.len == 0 {
                    0.0
                } else if blb > self.flb.last || self.prev_close < self.flb.last {
                    blb
                } else {
                    self.flb.last
                };
                self.flb.push(flb);
                let (fub, flb) = (self.fub, self.flb);
                let supertrend = match self.last {
                    None => 0.0,
                    Some((value, _)) if value == fub.prev && close <= fub.last => fub.last,
                    Some((value, _)) if value == fub.prev && close > fub.last => flb.last,
                    Some((value, _)) if value == flb.prev && close >= flb.last => flb.last,
                    Some((value, _)) if value == flb.prev && close < flb.last => fub.last,
                    // Python's `supertrend` is left unbound and raises.
                    Some(_) => return Err(Declined::new("SuperTrend leaves its value unbound")),
                };
                Some((supertrend, close > supertrend))
            }
        };
        self.prev_close = bar.close;
        Ok(match carry(&mut self.last, new) {
            Some((value, up)) => {
                self.value = value;
                self.trend = if up { 1.0 } else { -1.0 };
                true
            }
            None => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        TREND
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.trend;
        out[1] = self.value;
    }
}

/// `ChandeKrollStop`.
struct ChandeKrollStop {
    atr_period: usize,
    atr_mult: f64,
    period: usize,
    atr: Atr,
    highs: Window,
    lows: Window,
    seen: usize,
    high_stops: Window,
    low_stops: Window,
    stops: usize,
    last: Option<(f64, f64)>,
    short_stop: f64,
    long_stop: f64,
}

impl Inner for ChandeKrollStop {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let atr = self.atr.add(bar);
        self.highs.push(bar.high);
        self.lows.push(bar.low);
        self.seen += 1;
        let new = match atr {
            Some(atr) if self.seen >= self.atr_period => {
                let highest = first_max(self.highs.as_slice().iter().copied());
                let lowest = first_min(self.lows.as_slice().iter().copied());
                self.high_stops.push(highest - atr * self.atr_mult);
                self.low_stops.push(lowest + atr * self.atr_mult);
                self.stops += 1;
                (self.stops >= self.period).then(|| {
                    (
                        first_max(self.high_stops.as_slice().iter().copied()),
                        first_min(self.low_stops.as_slice().iter().copied()),
                    )
                })
            }
            _ => None,
        };
        Ok(match carry(&mut self.last, new) {
            Some((short_stop, long_stop)) => {
                self.short_stop = short_stop;
                self.long_stop = long_stop;
                true
            }
            None => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        &["long_stop", "short_stop"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.long_stop;
        out[1] = self.short_stop;
    }
}

/// `TEMA`.
struct Tema {
    ema: Ema,
    ema_ema: Ema,
    ema_ema_ema: Ema,
    last: Option<f64>,
}

impl Scalar for Tema {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let e1 = self.ema.add(Some(bar.close));
        let e2 = self.ema_ema.add(e1);
        let e3 = self.ema_ema_ema.add(e2);
        let new = match (e1, e2, e3) {
            (Some(e1), Some(e2), Some(e3)) => Some(3.0 * e1 - 3.0 * e2 + e3),
            _ => None,
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `T3`.
struct T3 {
    emas: [Ema; 6],
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    last: Option<f64>,
}

impl T3 {
    fn new(period: usize, factor: f64) -> Result<Self, Declined> {
        Ok(T3 {
            emas: std::array::from_fn(|_| Ema::new(period)),
            c1: -pow(factor, 3.0)?,
            c2: 3.0 * pow(factor, 2.0)? + 3.0 * pow(factor, 3.0)?,
            c3: -6.0 * pow(factor, 2.0)? - 3.0 * factor - 3.0 * pow(factor, 3.0)?,
            c4: 1.0 + 3.0 * factor + pow(factor, 3.0)? + 3.0 * pow(factor, 2.0)?,
            last: None,
        })
    }
}

impl Scalar for T3 {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let mut outputs = [None; 6];
        let mut input = Some(bar.close);
        for (ema, output) in self.emas.iter_mut().zip(outputs.iter_mut()) {
            input = ema.add(input);
            *output = input;
        }
        let new = match outputs {
            [_, _, Some(e3), Some(e4), Some(e5), Some(e6)] => {
                Some(self.c1 * e6 + self.c2 * e5 + self.c3 * e4 + self.c4 * e3)
            }
            _ => None,
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `ALMA`.
struct Alma {
    period: usize,
    weights: Vec<f64>,
    weight_sum: f64,
    inputs: Window,
    seen: usize,
    last: Option<f64>,
}

impl Alma {
    fn new(period: usize, offset: f64, sigma: f64) -> Self {
        let s = period as f64 / sigma;
        let m = (period - 1) as f64 * offset;
        let mut weights = Vec::with_capacity(period);
        let mut weight_sum = 0.0;
        for i in 0..period {
            let d = i as f64 - m;
            let weight = exp(-d * d / (2.0 * s * s));
            weights.push(weight);
            weight_sum += weight;
        }
        Alma { period, weights, weight_sum, inputs: Window::new(period), seen: 0, last: None }
    }
}

impl Scalar for Alma {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        self.inputs.push(bar.close);
        self.seen += 1;
        let new = if self.seen < self.period {
            None
        } else {
            let mut alma = 0.0;
            for (&x, &weight) in self.inputs.as_slice().iter().zip(&self.weights) {
                alma += x * weight;
            }
            Some(div(alma, self.weight_sum)?)
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `_ZLEMA`: `ZLEMA` with the lag `(period - 1) // 2`.
struct Zlema {
    lag: usize,
    inputs: Window,
    seen: usize,
    ema: Ema,
    last: Option<f64>,
}

impl Scalar for Zlema {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let x = bar.close;
        self.inputs.push(x);
        self.seen += 1;
        let new = if self.seen <= self.lag {
            None
        } else {
            self.ema.add(Some(x + (x - self.inputs.first())))
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `_VWMA`: `VWMA`, with no reading where the volume sums to zero.
struct Vwma {
    period: usize,
    closes: Window,
    volumes: Window,
    seen: usize,
    last: Option<f64>,
}

impl Scalar for Vwma {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        self.closes.push(bar.close);
        self.volumes.push(bar.volume);
        self.seen += 1;
        let new = if self.seen < self.period {
            None
        } else {
            let mut s = 0.0;
            let mut v = 0.0;
            for (&close, &volume) in self.closes.as_slice().iter().zip(self.volumes.as_slice()) {
                s += close * volume;
                v += volume;
            }
            (v != 0.0).then(|| s / v)
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `McGinleyDynamic`.
struct McGinley {
    period: usize,
    firsts: Vec<f64>,
    seen: usize,
    last: Option<f64>,
}

impl Scalar for McGinley {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let x = bar.close;
        self.seen += 1;
        let period = self.period as f64;
        let new = if self.seen < self.period {
            self.firsts.push(x);
            None
        } else if self.seen == self.period {
            self.firsts.push(x);
            let first = py_sum(&self.firsts) / period;
            self.firsts = Vec::new();
            Some(first)
        } else {
            match self.last {
                None => None,
                Some(out) => {
                    let ratio = div(x, out)?;
                    Some(out + div(x - out, period * pow(ratio, 4.0)?)?)
                }
            }
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `TRIX`.
struct Trix {
    emas: [Ema; 3],
    count: usize,
    prev: f64,
    last: Option<f64>,
}

impl Scalar for Trix {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let e1 = self.emas[0].add(Some(bar.close));
        let e2 = self.emas[1].add(e1);
        let new = match self.emas[2].add(e2) {
            None => None,
            Some(e3) => {
                self.count += 1;
                let prev = std::mem::replace(&mut self.prev, e3);
                if self.count < 2 {
                    None
                } else {
                    Some(div(10000.0 * (e3 - prev), prev)?)
                }
            }
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `TSI`.
struct Tsi {
    slow_ma: Ema,
    fast_ma: Ema,
    abs_slow_ma: Ema,
    abs_fast_ma: Ema,
    prev: Option<f64>,
    last: Option<f64>,
}

impl Scalar for Tsi {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let x = bar.close;
        let new = self.prev.and_then(|prev| {
            let change = x - prev;
            let fast = self.fast_ma.add(self.slow_ma.add(Some(change)));
            let abs_fast = self.abs_fast_ma.add(self.abs_slow_ma.add(Some(change.abs())));
            match (fast, abs_fast) {
                // A zero denominator repeats the last output.
                (Some(fast), Some(abs_fast)) if abs_fast != 0.0 => Some(100.0 * (fast / abs_fast)),
                _ => None,
            }
        });
        self.prev = Some(x);
        Ok(carry(&mut self.last, new))
    }
}

/// `KST`'s output.
#[derive(Clone, Copy)]
struct KstVal {
    kst: f64,
    signal: Option<f64>,
}

/// `KST`.
struct Kst {
    rocs: [Roc; 4],
    roc_mas: [Sma; 4],
    signal_line: Sma,
    last: Option<KstVal>,
    value: f64,
    signal: f64,
}

impl Inner for Kst {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let mut mas = [None; 4];
        for ((roc, roc_ma), ma) in
            self.rocs.iter_mut().zip(self.roc_mas.iter_mut()).zip(mas.iter_mut())
        {
            *ma = roc_ma.add(roc.add(bar.close)?);
        }
        let new = match mas {
            [Some(m1), Some(m2), Some(m3), Some(m4)] => {
                let kst = 1.0 * m1 + 2.0 * m2 + 3.0 * m3 + 4.0 * m4;
                let signal = self.signal_line.add(Some(kst));
                Some(KstVal { kst, signal })
            }
            _ => None,
        };
        Ok(match carry(&mut self.last, new) {
            Some(KstVal { kst, signal: Some(signal) }) => {
                self.value = kst;
                self.signal = signal;
                true
            }
            _ => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        &["signal", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.signal;
        out[1] = self.value;
    }
}

/// `CoppockCurve`.
struct Coppock {
    fast_roc: Roc,
    slow_roc: Roc,
    wma: Wma,
    last: Option<f64>,
}

impl Scalar for Coppock {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let fast = self.fast_roc.add(bar.close)?;
        let slow = self.slow_roc.add(bar.close)?;
        let new = match (fast, slow) {
            (Some(fast), Some(slow)) => self.wma.add(Some(slow + fast)),
            _ => None,
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `StochRSI`'s output.
#[derive(Clone, Copy)]
struct StochRsiVal {
    k: Option<f64>,
    d: Option<f64>,
}

/// `StochRSI`.
struct StochRsi {
    stoch_period: usize,
    rsi: Rsi,
    recent_rsi: Window,
    rsis: usize,
    smoothed_k: Sma,
    values_d: Sma,
    last: Option<StochRsiVal>,
    value_k: f64,
    value_d: f64,
}

impl Inner for StochRsi {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let new = match self.rsi.add(bar.close)? {
            None => None,
            Some(rsi) => {
                self.recent_rsi.push(rsi);
                self.rsis += 1;
                (self.rsis >= self.stoch_period).then(|| {
                    let recent = self.recent_rsi.as_slice();
                    let max_high = first_max(recent.iter().copied());
                    let min_low = first_min(recent.iter().copied());
                    let k = if max_high == min_low {
                        100.0
                    } else {
                        100.0 * (rsi - min_low) / (max_high - min_low)
                    };
                    let k = self.smoothed_k.add(Some(k));
                    let d = self.values_d.add(k);
                    StochRsiVal { k, d }
                })
            }
        };
        Ok(match carry(&mut self.last, new) {
            Some(StochRsiVal { k: Some(k), d: Some(d) }) => {
                self.value_k = k;
                self.value_d = d;
                true
            }
            _ => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        &["value_d", "value_k"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value_d;
        out[1] = self.value_k;
    }
}

/// `_VTX`: `VTX`, with no reading where the true ranges sum to zero.
struct Vtx {
    period: usize,
    atr: Atr,
    atrs: Window,
    atr_count: usize,
    prev: Option<(f64, f64)>,
    plus_vm: Window,
    minus_vm: Window,
    vms: usize,
    last: Option<(f64, f64)>,
    plus_vi: f64,
    minus_vi: f64,
}

impl Inner for Vtx {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        if let Some(atr) = self.atr.add(bar) {
            self.atrs.push(atr);
            self.atr_count += 1;
        }
        let new = match self.prev {
            None => None,
            Some((high2, low2)) => {
                self.plus_vm.push((bar.high - low2).abs());
                self.minus_vm.push((bar.low - high2).abs());
                self.vms += 1;
                if self.atr_count < self.period || self.vms < self.period {
                    None
                } else {
                    let atr_sum = py_sum(self.atrs.as_slice());
                    (atr_sum != 0.0).then(|| {
                        (
                            py_sum(self.plus_vm.as_slice()) / atr_sum,
                            py_sum(self.minus_vm.as_slice()) / atr_sum,
                        )
                    })
                }
            }
        };
        self.prev = Some((bar.high, bar.low));
        Ok(match carry(&mut self.last, new) {
            Some((plus_vi, minus_vi)) => {
                self.plus_vi = plus_vi;
                self.minus_vi = minus_vi;
                true
            }
            None => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        &["minus_vi", "plus_vi"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.minus_vi;
        out[1] = self.plus_vi;
    }
}

/// `IBS`.
#[derive(Default)]
struct Ibs {
    last: Option<f64>,
}

impl Scalar for Ibs {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let new = (bar.high != bar.low).then(|| (bar.close - bar.low) / (bar.high - bar.low));
        Ok(carry(&mut self.last, new))
    }
}

/// `_BOP`: an SMA of where each bar closed against where it opened.
struct Bop {
    ma: Sma,
}

impl Scalar for Bop {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let span = bar.high - bar.low;
        let x = if span != 0.0 { (bar.close - bar.open) / span } else { 0.0 };
        Ok(self.ma.add(Some(x)))
    }
}

/// `AO`.
struct Ao {
    ma_fast: Sma,
    ma_slow: Sma,
    last: Option<f64>,
}

impl Scalar for Ao {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let median = (bar.high + bar.low) / 2.0;
        let fast = self.ma_fast.add(Some(median));
        let slow = self.ma_slow.add(Some(median));
        let new = match (fast, slow) {
            (Some(fast), Some(slow)) => Some(fast - slow),
            _ => None,
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `DPO`.
struct Dpo {
    half: usize,
    ma: Sma,
    inputs: Window,
    seen: usize,
    last: Option<f64>,
}

impl Scalar for Dpo {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let ma = self.ma.add(Some(bar.close));
        self.inputs.push(bar.close);
        self.seen += 1;
        let new = match ma {
            Some(ma) if self.seen >= self.half + 2 => Some(self.inputs.first() - ma),
            _ => None,
        };
        Ok(carry(&mut self.last, new))
    }
}

/// `TTM`'s output.
#[derive(Clone, Copy)]
struct TtmVal {
    squeeze: bool,
    histogram: Option<f64>,
}

/// `TTM`.
struct Ttm {
    period: usize,
    bb_mult: f64,
    kc_mult: f64,
    // `BB`'s central band and `TTM`'s own `ma` are both an SMA of the closes.
    sma: Sma,
    std_dev: StdDev,
    kc_ema: Ema,
    kc_atr: Atr,
    highs: Window,
    lows: Window,
    seen: usize,
    deltas: Window,
    delta_count: usize,
    mean_x: f64,
    denom: f64,
    last: Option<TtmVal>,
    value: f64,
    squeeze: f64,
}

impl Ttm {
    fn new(period: usize, bb_mult: f64, kc_mult: f64) -> Result<Self, Declined> {
        let mean_x = ((period * (period - 1)) / 2) as f64 / period as f64;
        let mut denom = 0.0;
        for x in 0..period {
            denom += pow(x as f64 - mean_x, 2.0)?;
        }
        Ok(Ttm {
            period,
            bb_mult,
            kc_mult,
            sma: Sma::new(period),
            std_dev: StdDev::new(period),
            kc_ema: Ema::new(period),
            kc_atr: Atr::new(period),
            highs: Window::new(period),
            lows: Window::new(period),
            seen: 0,
            deltas: Window::new(period),
            delta_count: 0,
            mean_x,
            denom,
            last: None,
            value: 0.0,
            squeeze: 0.0,
        })
    }

    fn histogram(&self) -> Result<f64, Declined> {
        let deltas = self.deltas.as_slice();
        let mean_y = py_sum(deltas) / self.period as f64;
        let mut numer = 0.0;
        for (x, &y) in deltas.iter().enumerate() {
            numer += (x as f64 - self.mean_x) * (y - mean_y);
        }
        let a = div(numer, self.denom)?;
        let b = mean_y - (a * self.mean_x);
        Ok(a * (self.period - 1) as f64 + b)
    }
}

impl Inner for Ttm {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let close = bar.close;
        let ma = self.sma.add(Some(close));
        let std_dev = self.std_dev.add(close)?;
        self.highs.push(bar.high);
        self.lows.push(bar.low);
        self.seen += 1;
        let kc_cb = self.kc_ema.add(Some(close));
        let kc_atr = self.kc_atr.add(bar);
        let warm = self.seen >= self.period;
        let new = match (ma, std_dev, kc_cb, kc_atr) {
            (Some(cb), Some(sd), Some(kc_cb), Some(atr)) if warm => {
                let (bb_lb, bb_ub) = (cb - self.bb_mult * sd, cb + self.bb_mult * sd);
                let (kc_lb, kc_ub) = (kc_cb - self.kc_mult * atr, kc_cb + self.kc_mult * atr);
                let squeeze = bb_ub < kc_ub && bb_lb > kc_lb;
                let dc_cb = (first_max(self.highs.as_slice().iter().copied())
                    + first_min(self.lows.as_slice().iter().copied()))
                    / 2.0;
                self.deltas.push(close - (dc_cb + cb) / 2.0);
                self.delta_count += 1;
                let histogram =
                    if self.delta_count >= self.period { Some(self.histogram()?) } else { None };
                Some(TtmVal { squeeze, histogram })
            }
            _ => None,
        };
        Ok(match carry(&mut self.last, new) {
            Some(TtmVal { squeeze, histogram: Some(histogram) }) => {
                self.value = histogram;
                self.squeeze = if squeeze { 1.0 } else { 0.0 };
                true
            }
            _ => false,
        })
    }

    fn names(&self) -> &'static [&'static str] {
        &["squeeze", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.squeeze;
        out[1] = self.value;
    }
}

/// `ChaikinOsc`.
struct ChaikinOsc {
    slow_period: usize,
    accu_dist: AccuDist,
    ma_fast: Ema,
    ma_slow: Ema,
    fed: usize,
    last: Option<f64>,
}

impl Scalar for ChaikinOsc {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let new = self.accu_dist.add(bar).and_then(|accu_dist| {
            let fast = self.ma_fast.add(Some(accu_dist));
            let slow = self.ma_slow.add(Some(accu_dist));
            self.fed += 1;
            Some(fast? - slow?)
        });
        Ok(carry(&mut self.last, new))
    }

    fn purge(&mut self, size: usize) -> Step {
        purge_fed(&mut self.fed, size, self.slow_period + 1)
    }
}

/// `ForceIndex`.
struct ForceIndex {
    ma: Ema,
    prev_close: Option<f64>,
    last: Option<f64>,
}

impl Scalar for ForceIndex {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let new =
            self.prev_close.and_then(|close2| self.ma.add(Some((bar.close - close2) * bar.volume)));
        self.prev_close = Some(bar.close);
        Ok(carry(&mut self.last, new))
    }
}

/// `_EMV`: `EMV`, with no reading (and nothing for its average) on a bar
/// with a range but no volume.
struct Emv {
    period: usize,
    volume_div: f64,
    sma: Sma,
    prev: Option<(f64, f64)>,
    fed: usize,
    last: Option<f64>,
}

impl Scalar for Emv {
    fn add(&mut self, bar: &Bar) -> Result<Option<f64>, Declined> {
        let new = self.prev.and_then(|(high2, low2)| {
            let emv = if bar.high != bar.low {
                let distance = (bar.high + bar.low) / 2.0 - (high2 + low2) / 2.0;
                let box_ratio = (bar.volume / self.volume_div) / (bar.high - bar.low);
                if box_ratio == 0.0 {
                    return None;
                }
                distance / box_ratio
            } else {
                0.0
            };
            self.fed += 1;
            self.sma.add(Some(emv))
        });
        self.prev = Some((bar.high, bar.low));
        Ok(carry(&mut self.last, new))
    }

    fn purge(&mut self, size: usize) -> Step {
        purge_fed(&mut self.fed, size, self.period + 1)
    }
}

/// `AccumulationDistribution`: `AccuDist`, less its reading `period` bars
/// back when `period` is set.
struct AccumulationDistribution {
    period: usize,
    accu_dist: AccuDist,
    history: Window,
    values: usize,
    value: f64,
}

impl Inner for AccumulationDistribution {
    fn step(&mut self, bar: &Bar) -> Result<bool, Declined> {
        let Some(output) = self.accu_dist.add(bar) else {
            return Ok(false);
        };
        self.history.push(output);
        self.values += 1;
        if self.period == 0 {
            self.value = output;
        } else if self.values > self.period {
            self.value = output - self.history.first();
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

/// `_below`: the fast period must be the shorter.
fn below(fast: usize, slow: usize) -> Step {
    require(fast < slow, "the fast period is not below the slow")
}

/// A float argument that must be positive.
fn positive_float(params: &Params, name: &str, default: f64) -> Result<f64, Declined> {
    let value = params.float_or(name, default)?;
    require(value > 0.0, "a multiplier is not positive")?;
    Ok(value)
}

/// The twin of the engine's talipp indicator class *kind*, if it has one.
pub(super) fn build(kind: &str, params: &Params) -> Option<Result<Box<dyn Twin>, Declined>> {
    let built = match kind {
        "ParabolicSAR" => (|| {
            let af_start = params.float_or("af_start", 0.02)?;
            let af_step = params.float_or("af_step", 0.02)?;
            let af_max = params.float_or("af_max", 0.2)?;
            require(
                0.0 < af_start && af_start <= af_max && af_step > 0.0,
                "bad acceleration factors",
            )?;
            wrapped(ParabolicSar::new(af_start, af_step, af_max), ParabolicSar::SAR_INIT_LEN, 0)
        })(),
        "SuperTrend" => (|| {
            let period = params.period_or("period", 10)?;
            let multiplier = positive_float(params, "multiplier", 3.0)?;
            let inner = SuperTrend {
                multiplier,
                atr: Atr::new(period),
                prev_close: 0.0,
                fub: LastTwo::default(),
                flb: LastTwo::default(),
                last: None,
                value: 0.0,
                trend: 0.0,
            };
            wrapped(inner, period + 2, 1)
        })(),
        "ChandeKrollStop" => (|| {
            let atr_period = params.period_or("atr_period", 10)?;
            let atr_mult = positive_float(params, "atr_mult", 1.0)?;
            let period = params.period_or("period", 9)?;
            let inner = ChandeKrollStop {
                atr_period,
                atr_mult,
                period,
                atr: Atr::new(atr_period),
                highs: Window::new(atr_period),
                lows: Window::new(atr_period),
                seen: 0,
                high_stops: Window::new(period),
                low_stops: Window::new(period),
                stops: 0,
                last: None,
                short_stop: 0.0,
                long_stop: 0.0,
            };
            wrapped(inner, atr_period + period, 0)
        })(),
        "TripleExponentialMovingAverage" => (|| {
            let period = params.period_or("period", 30)?;
            let inner = Tema {
                ema: Ema::new(period),
                ema_ema: Ema::new(period),
                ema_ema_ema: Ema::new(period),
                last: None,
            };
            scalar(inner, 3 * period)
        })(),
        "TillsonT3" => (|| {
            let period = params.period_or("period", 5)?;
            let volume_factor = params.float_or("volume_factor", 0.7)?;
            scalar(T3::new(period, volume_factor)?, 6 * period)
        })(),
        "ArnaudLegouxMovingAverage" => (|| {
            let period = params.period_or("period", 9)?;
            let offset = params.float_or("offset", 0.85)?;
            let sigma = positive_float(params, "sigma", 6.0)?;
            scalar(Alma::new(period, offset, sigma), period)
        })(),
        "ZeroLagExponentialMovingAverage" => (|| {
            let period = params.period_or("period", 20)?;
            let lag = (period - 1) / 2;
            let inner = Zlema {
                lag,
                inputs: Window::new(lag + 1),
                seen: 0,
                ema: Ema::new(period),
                last: None,
            };
            scalar(inner, period + lag)
        })(),
        "VolumeWeightedMovingAverage" => (|| {
            let period = params.period_or("period", 20)?;
            let inner = Vwma {
                period,
                closes: Window::new(period),
                volumes: Window::new(period),
                seen: 0,
                last: None,
            };
            scalar(inner, period)
        })(),
        "McGinleyDynamic" => (|| {
            let period = params.period_or("period", 14)?;
            scalar(
                McGinley { period, firsts: Vec::with_capacity(period), seen: 0, last: None },
                period,
            )
        })(),
        "Trix" => (|| {
            let period = params.period_or("period", 15)?;
            let inner = Trix {
                emas: std::array::from_fn(|_| Ema::new(period)),
                count: 0,
                prev: 0.0,
                last: None,
            };
            scalar(inner, 3 * period + 1)
        })(),
        "TrueStrengthIndex" => (|| {
            let fast = params.period_or("fast_period", 13)?;
            let slow = params.period_or("slow_period", 25)?;
            below(fast, slow)?;
            let inner = Tsi {
                slow_ma: Ema::new(slow),
                fast_ma: Ema::new(fast),
                abs_slow_ma: Ema::new(slow),
                abs_fast_ma: Ema::new(fast),
                prev: None,
                last: None,
            };
            scalar(inner, fast + slow + 1)
        })(),
        "KnowSureThing" => (|| {
            let rocs = [
                params.period_or("roc1_period", 10)?,
                params.period_or("roc2_period", 15)?,
                params.period_or("roc3_period", 20)?,
                params.period_or("roc4_period", 30)?,
            ];
            let mas = [
                params.period_or("roc1_ma_period", 10)?,
                params.period_or("roc2_ma_period", 10)?,
                params.period_or("roc3_ma_period", 10)?,
                params.period_or("roc4_ma_period", 15)?,
            ];
            let signal_period = params.period_or("signal_period", 9)?;
            let legs = (0..4).map(|i| rocs[i] + mas[i]).max().unwrap_or(0);
            let inner = Kst {
                rocs: rocs.map(Roc::new),
                roc_mas: mas.map(Sma::new),
                signal_line: Sma::new(signal_period),
                last: None,
                value: 0.0,
                signal: 0.0,
            };
            wrapped(inner, legs + signal_period, 0)
        })(),
        "CoppockCurve" => (|| {
            let fast = params.period_or("fast_roc_period", 11)?;
            let slow = params.period_or("slow_roc_period", 14)?;
            let wma = params.period_or("wma_period", 10)?;
            let inner = Coppock {
                fast_roc: Roc::new(fast),
                slow_roc: Roc::new(slow),
                wma: Wma::new(wma),
                last: None,
            };
            scalar(inner, fast.max(slow) + wma)
        })(),
        "StochasticRSI" => (|| {
            let rsi_period = params.period_or("rsi_period", 14)?;
            let stoch_period = params.period_or("stoch_period", 14)?;
            let k_period = params.period_or("k_period", 3)?;
            let d_period = params.period_or("d_period", 3)?;
            require(rsi_period >= 2, "rsi_period < 2")?;
            let inner = StochRsi {
                stoch_period,
                rsi: Rsi::new(rsi_period),
                recent_rsi: Window::new(stoch_period),
                rsis: 0,
                smoothed_k: Sma::new(k_period),
                values_d: Sma::new(d_period),
                last: None,
                value_k: 0.0,
                value_d: 0.0,
            };
            wrapped(inner, rsi_period + stoch_period + k_period + d_period, 0)
        })(),
        "VortexIndicator" => (|| {
            let period = params.period_or("period", 14)?;
            let inner = Vtx {
                period,
                atr: Atr::new(1),
                atrs: Window::new(period),
                atr_count: 0,
                prev: None,
                plus_vm: Window::new(period),
                minus_vm: Window::new(period),
                vms: 0,
                last: None,
                plus_vi: 0.0,
                minus_vi: 0.0,
            };
            wrapped(inner, period + 1, 0)
        })(),
        "InternalBarStrength" => scalar(Ibs::default(), 1),
        "BalanceOfPower" => (|| {
            let period = params.period_or("period", 14)?;
            scalar(Bop { ma: Sma::new(period) }, period)
        })(),
        "AwesomeOscillator" => (|| {
            let fast = params.period_or("fast_period", 5)?;
            let slow = params.period_or("slow_period", 34)?;
            below(fast, slow)?;
            scalar(Ao { ma_fast: Sma::new(fast), ma_slow: Sma::new(slow), last: None }, slow)
        })(),
        "DetrendedPriceOscillator" => (|| {
            let period = params.period_or("period", 20)?;
            let half = period / 2;
            let inner = Dpo {
                half,
                ma: Sma::new(period),
                inputs: Window::new(half + 2),
                seen: 0,
                last: None,
            };
            scalar(inner, period.max(half + 2))
        })(),
        "TTMSqueeze" => (|| {
            let period = params.period_or("period", 20)?;
            require(period >= 2, "period < 2")?;
            let bb_mult = positive_float(params, "bb_mult", 2.0)?;
            let kc_mult = positive_float(params, "kc_mult", 1.5)?;
            wrapped(Ttm::new(period, bb_mult, kc_mult)?, 2 * period - 1, 0)
        })(),
        "ChaikinOscillator" => (|| {
            let fast = params.period_or("fast_period", 3)?;
            let slow = params.period_or("slow_period", 10)?;
            below(fast, slow)?;
            let inner = ChaikinOsc {
                slow_period: slow,
                accu_dist: AccuDist::default(),
                ma_fast: Ema::new(fast),
                ma_slow: Ema::new(slow),
                fed: 0,
                last: None,
            };
            scalar(inner, slow + 1)
        })(),
        "ForceIndex" => (|| {
            let period = params.period_or("period", 13)?;
            scalar(ForceIndex { ma: Ema::new(period), prev_close: None, last: None }, period + 1)
        })(),
        "EaseOfMovement" => (|| {
            let period = params.period_or("period", 14)?;
            let volume_divisor = params.int_or("volume_divisor", 10_000)?;
            require(volume_divisor >= 1, "volume_divisor < 1")?;
            let inner = Emv {
                period,
                volume_div: volume_divisor as f64,
                sma: Sma::new(period),
                prev: None,
                fed: 0,
                last: None,
            };
            scalar(inner, period + 1)
        })(),
        "AccumulationDistribution" => (|| {
            let period = params.int_or("period", 0)?;
            let period = usize::try_from(period)
                .ok()
                .filter(|&period| period <= 1 << 24)
                .ok_or_else(|| Declined::new("period is not a usable look-back"))?;
            let inner = AccumulationDistribution {
                period,
                accu_dist: AccuDist::default(),
                history: Window::new(period + 1),
                values: 0,
                value: 0.0,
            };
            wrapped(inner, period + 1, 0)
        })(),
        _ => return None,
    };
    Some(built)
}

#[cfg(test)]
mod tests {
    use super::super::{compute, Columns, Param, Params};
    use super::*;

    fn params(items: &[(&str, Param)]) -> Params {
        Params::new(items.iter().map(|(name, value)| (name.to_string(), value.clone())).collect())
    }

    fn bar(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Bar {
        Bar { open, high, low, close, volume }
    }

    #[test]
    fn a_flat_bar_repeats_the_last_reading() {
        let mut ibs = Ibs::default();
        assert_eq!(ibs.add(&bar(1.0, 1.0, 1.0, 1.0, 1.0)).unwrap(), None);
        assert_eq!(ibs.add(&bar(1.0, 2.0, 1.0, 1.5, 1.0)).unwrap(), Some(0.5));
        assert_eq!(ibs.add(&bar(1.0, 1.0, 1.0, 1.0, 1.0)).unwrap(), Some(0.5));
    }

    #[test]
    fn the_sma_rolls_its_first_mean_forward() {
        let mut sma = Sma::new(2);
        assert_eq!(sma.add(None), None);
        assert_eq!(sma.add(Some(1.0)), None);
        assert_eq!(sma.add(Some(2.0)), Some(1.5));
        assert_eq!(sma.add(Some(4.0)), Some(1.5 - (1.0 - 4.0) / 2.0));
    }

    #[test]
    fn supertrend_skips_its_first_reading() {
        let n = 8;
        let close: Vec<f64> = (0..n).map(|i| 10.0 + f64::from(i)).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let volume = vec![1.0; n as usize];
        let columns =
            Columns { open: &close, high: &high, low: &low, close: &close, volume: &volume };
        let series =
            compute("SuperTrend", &params(&[("period", Param::Int(2))]), &columns).unwrap();
        // ATR(2) first reads on bar 2, the dummy reading; bar 3 initializes.
        assert_eq!(series.initialized, vec![false, false, true, true, true, true, true, true]);
    }

    #[test]
    fn a_purge_into_a_late_average_declines() {
        let mut fed = 10;
        assert!(purge_fed(&mut fed, 4, 6).is_ok());
        assert_eq!(fed, 6);
        assert!(purge_fed(&mut fed, 1, 6).is_err());
        let mut empty = 0;
        assert!(purge_fed(&mut empty, 4, 6).is_ok());
    }

    #[test]
    fn chaikin_declines_when_the_first_range_comes_too_late() {
        // 2 * 2048 flat bars before the first with a range: the first purge
        // finds the averages a handful of values in.
        let n = 2 * KEEP_MIN + 1;
        let mut close = vec![1.0; n];
        let mut high = vec![1.0; n];
        close[n - 3] = 1.5;
        high[n - 3] = 2.0;
        let low = vec![1.0; n];
        let volume = vec![1.0; n];
        let columns =
            Columns { open: &close, high: &high, low: &low, close: &close, volume: &volume };
        assert!(compute("ChaikinOscillator", &params(&[]), &columns).is_err());
    }
}
