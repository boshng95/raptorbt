//! Twins of nautilus_trader's Cython indicators (`nautilus_trader/indicators`).
//!
//! Each `update` is its original's `update_raw`, statement for statement:
//! the C arithmetic of the `.pyx`, the builtin `max`/`min`/`sum` it calls on
//! its deques, and numpy's where it hands a deque to numpy. The Cython is
//! compiled without `cdivision`, so each of its divisions raises on a zero
//! divisor; `div` declines there.

use std::collections::VecDeque;

use super::window::Window;
use super::{div, first_max, first_min, Bar, Declined, Params, Step, Twin};
use crate::metrics::exact::{np_sum, py_sum};

/// `fast_mean` (`nautilus_trader.core.stats`): a C loop from 0.0.
fn fast_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for &value in values {
        total += value;
    }
    total / values.len() as f64
}

/// `fast_std_with_mean`.
fn fast_std_with_mean(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for &value in values {
        let v = value - mean;
        total += v * v;
    }
    (total / values.len() as f64).sqrt()
}

/// `fast_mad_with_mean`.
fn fast_mad_with_mean(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for &value in values {
        total += (value - mean).abs();
    }
    total / values.len() as f64
}

/// Cython's `max(a, b)` of two C doubles: `b if b > a else a`.
#[inline]
fn c_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Cython's `min(a, b)` of two C doubles: `b if b < a else a`.
#[inline]
fn c_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline]
fn flag(condition: bool) -> f64 {
    if condition {
        1.0
    } else {
        0.0
    }
}

/// `np.average(values, weights=weights)` of two equal-length float64 arrays.
fn np_average(values: &[f64], weights: &[f64]) -> Result<f64, Declined> {
    let scale = np_sum(weights);
    if scale == 0.0 {
        return Err(Declined::new("weights sum to zero"));
    }
    let products: Vec<f64> = values.iter().zip(weights).map(|(v, w)| v * w).collect();
    Ok(np_sum(&products) / scale)
}

/// `MovingAverage`'s count, and the flags `_increment_count` sets.
#[derive(Clone, Debug)]
struct Count {
    period: usize,
    count: usize,
    initialized: bool,
    has_inputs: bool,
}

impl Count {
    fn new(period: usize) -> Self {
        Count { period, count: 0, initialized: false, has_inputs: false }
    }

    /// `_increment_count`.
    fn increment(&mut self) {
        self.count += 1;
        if !self.initialized {
            self.has_inputs = true;
            if self.count >= self.period {
                self.initialized = true;
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Sma {
    count: Count,
    inputs: Window,
    value: f64,
}

impl Sma {
    fn new(period: usize) -> Self {
        Sma { count: Count::new(period), inputs: Window::new(period), value: 0.0 }
    }

    fn update(&mut self, value: f64) {
        self.inputs.push(value);
        self.value = fast_mean(self.inputs.as_slice());
        self.count.increment();
    }
}

/// `ExponentialMovingAverage` (alpha `2 / (period + 1)`) and
/// `WilderMovingAverage` (alpha `1 / period`), which differ only there.
#[derive(Clone, Debug)]
pub(super) struct Ema {
    count: Count,
    alpha: f64,
    value: f64,
}

impl Ema {
    pub(super) fn exponential(period: usize) -> Self {
        Ema { count: Count::new(period), alpha: 2.0 / (period as f64 + 1.0), value: 0.0 }
    }

    fn wilder(period: usize) -> Self {
        Ema { count: Count::new(period), alpha: 1.0 / period as f64, value: 0.0 }
    }

    pub(super) fn update(&mut self, value: f64) {
        if !self.count.has_inputs {
            self.value = value;
        }
        self.value = self.alpha * value + ((1.0 - self.alpha) * self.value);
        self.count.increment();
    }

    pub(super) fn value(&self) -> f64 {
        self.value
    }

    pub(super) fn initialized(&self) -> bool {
        self.count.initialized
    }
}

/// The averages `MovingAverageFactory.create` builds inside the indicators
/// here, by their `MovingAverageType`.
#[derive(Clone, Debug)]
enum Average {
    Simple(Sma),
    Exponential(Ema),
}

impl Average {
    fn simple(period: usize) -> Self {
        Average::Simple(Sma::new(period))
    }

    fn exponential(period: usize) -> Self {
        Average::Exponential(Ema::exponential(period))
    }

    fn wilder(period: usize) -> Self {
        Average::Exponential(Ema::wilder(period))
    }

    fn update(&mut self, value: f64) {
        match self {
            Average::Simple(average) => average.update(value),
            Average::Exponential(average) => average.update(value),
        }
    }

    fn value(&self) -> f64 {
        match self {
            Average::Simple(average) => average.value,
            Average::Exponential(average) => average.value,
        }
    }

    fn initialized(&self) -> bool {
        match self {
            Average::Simple(average) => average.count.initialized,
            Average::Exponential(average) => average.count.initialized,
        }
    }
}

const VALUE: &[&str] = &["value"];
const BANDS: &[&str] = &["lower", "middle", "upper"];

/// An indicator whose one output is a moving average's `value`.
struct AverageTwin(Average);

impl Twin for AverageTwin {
    fn update(&mut self, bar: &Bar) -> Step {
        self.0.update(bar.close);
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.0.initialized()
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.0.value();
    }
}

struct Dema {
    ma1: Ema,
    ma2: Ema,
    value: f64,
    initialized: bool,
}

impl Twin for Dema {
    fn update(&mut self, bar: &Bar) -> Step {
        self.ma1.update(bar.close);
        self.ma2.update(self.ma1.value);
        self.value = 2.0 * self.ma1.value - self.ma2.value;
        if !self.initialized && self.ma2.count.initialized {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

#[derive(Clone, Debug)]
struct Wma {
    count: Count,
    inputs: Window,
    weights: Vec<f64>,
    value: f64,
}

impl Wma {
    /// `WeightedMovingAverage(period, weights)`, with its constructor's checks.
    fn new(period: usize, weights: Vec<f64>) -> Result<Self, Declined> {
        if weights.len() != period || !(f64::EPSILON < np_sum(&weights)) {
            return Err(Declined::new("weights the constructor refuses"));
        }
        Ok(Wma { count: Count::new(period), inputs: Window::new(period), weights, value: 0.0 })
    }

    fn update(&mut self, value: f64) -> Step {
        self.inputs.push(value);
        let inputs = self.inputs.as_slice();
        let weights = &self.weights[self.weights.len() - inputs.len()..];
        self.value = np_average(inputs, weights)?;
        self.count.increment();
        Ok(())
    }
}

impl Twin for Wma {
    fn update(&mut self, bar: &Bar) -> Step {
        Wma::update(self, bar.close)
    }

    fn initialized(&self) -> bool {
        self.count.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Hma {
    count: Count,
    ma1: Wma,
    ma2: Wma,
    ma3: Wma,
    value: f64,
}

impl Hma {
    fn new(period: usize) -> Result<Self, Declined> {
        let halved = (period as f64 / 2.0) as usize;
        let root = (period as f64).sqrt() as usize;
        Ok(Hma {
            count: Count::new(period),
            ma1: Wma::new(halved, Self::weights(halved))?,
            ma2: Wma::new(period, Self::weights(period))?,
            ma3: Wma::new(root, Self::weights(root))?,
            value: 0.0,
        })
    }

    /// `_get_weights`: `arange(1, size + 1)` over its own numpy sum.
    fn weights(size: usize) -> Vec<f64> {
        let raw: Vec<f64> = (1..=size).map(|i| i as f64).collect();
        let total = np_sum(&raw);
        raw.iter().map(|w| w / total).collect()
    }
}

impl Twin for Hma {
    fn update(&mut self, bar: &Bar) -> Step {
        self.ma1.update(bar.close)?;
        self.ma2.update(bar.close)?;
        self.ma3.update(self.ma1.value * 2.0 - self.ma2.value)?;
        self.value = self.ma3.value;
        self.count.increment();
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.count.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

/// `EfficiencyRatio`, inside the adaptive average.
struct EfficiencyRatio {
    period: usize,
    inputs: Window,
    deltas: Window,
    initialized: bool,
    value: f64,
}

impl EfficiencyRatio {
    fn new(period: usize) -> Self {
        EfficiencyRatio {
            period,
            inputs: Window::new(period),
            deltas: Window::new(period),
            initialized: false,
            value: 0.0,
        }
    }

    fn update(&mut self, price: f64) {
        self.inputs.push(price);
        if !self.initialized {
            if self.inputs.len() < 2 {
                return;
            } else if self.inputs.len() >= self.period {
                self.initialized = true;
            }
        }
        self.deltas.push((self.inputs.last() - self.inputs.back(2)).abs());
        let net_diff = (self.inputs.first() - self.inputs.last()).abs();
        let sum_deltas = py_sum(self.deltas.as_slice());
        self.value = if sum_deltas > 0.0 { net_diff / sum_deltas } else { 0.0 };
    }
}

struct Ama {
    count: Count,
    alpha_slow: f64,
    alpha_diff: f64,
    efficiency_ratio: EfficiencyRatio,
    value: f64,
}

impl Twin for Ama {
    fn update(&mut self, bar: &Bar) -> Step {
        let value = bar.close;
        if !self.count.has_inputs {
            self.value = value;
        }
        self.efficiency_ratio.update(value);
        let prior = self.value;
        let root = self.efficiency_ratio.value * self.alpha_diff + self.alpha_slow;
        // `pow(root, 2)`, which the C compiler folds to a product.
        let sc = root * root;
        self.value = prior + sc * (value - prior);
        self.count.increment();
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.count.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

/// `ChandeMomentumOscillator`, on its own or inside VIDYA.
struct Cmo {
    average_gain: Average,
    average_loss: Average,
    has_inputs: bool,
    initialized: bool,
    previous_close: f64,
    value: f64,
}

impl Cmo {
    fn new(average_gain: Average, average_loss: Average) -> Self {
        Cmo {
            average_gain,
            average_loss,
            has_inputs: false,
            initialized: false,
            previous_close: 0.0,
            value: 0.0,
        }
    }

    fn update(&mut self, close: f64) -> Step {
        if !self.has_inputs {
            self.has_inputs = true;
            self.previous_close = close;
        }
        let gain = close - self.previous_close;
        if gain > 0.0 {
            self.average_gain.update(gain);
            self.average_loss.update(0.0);
        } else if gain < 0.0 {
            self.average_gain.update(0.0);
            self.average_loss.update(-gain);
        } else {
            self.average_gain.update(0.0);
            self.average_loss.update(0.0);
        }
        if !self.initialized && self.average_gain.initialized() && self.average_loss.initialized() {
            self.initialized = true;
        }
        if self.initialized {
            let divisor = self.average_gain.value() + self.average_loss.value();
            self.value = if divisor == 0.0 {
                0.0
            } else {
                div(100.0 * (self.average_gain.value() - self.average_loss.value()), divisor)?
            };
        }
        self.previous_close = close;
        Ok(())
    }
}

impl Twin for Cmo {
    fn update(&mut self, bar: &Bar) -> Step {
        Cmo::update(self, bar.close)
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Vidya {
    count: Count,
    cmo: Cmo,
    cmo_pct: f64,
    alpha: f64,
    value: f64,
}

impl Twin for Vidya {
    fn update(&mut self, bar: &Bar) -> Step {
        let value = bar.close;
        self.cmo.update(value)?;
        self.cmo_pct = (self.cmo.value / 100.0).abs();
        if self.count.initialized {
            self.value =
                self.alpha * self.cmo_pct * value + (1.0 - self.alpha * self.cmo_pct) * self.value;
        }
        if !self.count.initialized && self.cmo.initialized {
            self.count.initialized = true;
        }
        self.count.increment();
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.count.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Rsi {
    average_gain: Ema,
    average_loss: Ema,
    has_inputs: bool,
    initialized: bool,
    last_value: f64,
    value: f64,
}

impl Twin for Rsi {
    fn update(&mut self, bar: &Bar) -> Step {
        const RSI_MAX: f64 = 1.0;
        let value = bar.close;
        if !self.has_inputs {
            self.last_value = value;
            self.has_inputs = true;
        }
        let gain = value - self.last_value;
        if gain > 0.0 {
            self.average_gain.update(gain);
            self.average_loss.update(0.0);
        } else if gain < 0.0 {
            self.average_loss.update(-gain);
            self.average_gain.update(0.0);
        } else {
            self.average_gain.update(0.0);
            self.average_loss.update(0.0);
        }
        if !self.initialized
            && self.average_gain.count.initialized
            && self.average_loss.count.initialized
        {
            self.initialized = true;
        }
        if self.average_loss.value == 0.0 {
            self.value = RSI_MAX;
            self.last_value = value;
            return Ok(());
        }
        let rs = div(self.average_gain.value, self.average_loss.value)?;
        self.value = RSI_MAX - div(RSI_MAX, 1.0 + rs)?;
        self.last_value = value;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Roc {
    period: usize,
    prices: Window,
    initialized: bool,
    value: f64,
}

impl Twin for Roc {
    fn update(&mut self, bar: &Bar) -> Step {
        let price = bar.close;
        self.prices.push(price);
        if !self.initialized && self.prices.len() >= self.period {
            self.initialized = true;
        }
        let first = self.prices.first();
        self.value = div(price - first, first)?;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Stochastics {
    period_k: usize,
    ratio: bool,
    highs: Window,
    lows: Window,
    c_sub_l: Window,
    h_sub_l: Window,
    slowing_ma: Option<Ema>,
    d_ma: Option<Ema>,
    initialized: bool,
    value_k: f64,
    value_d: f64,
}

impl Twin for Stochastics {
    fn update(&mut self, bar: &Bar) -> Step {
        let (high, low, close) = (bar.high, bar.low, bar.close);
        self.highs.push(high);
        self.lows.push(low);
        if !self.initialized
            && self.highs.len() == self.period_k
            && self.lows.len() == self.period_k
            && self.slowing_ma.is_none()
            && self.ratio
        {
            self.initialized = true;
        }
        let k_max_high = first_max(self.highs.as_slice().iter().copied());
        let k_min_low = first_min(self.lows.as_slice().iter().copied());
        if self.ratio {
            self.c_sub_l.push(close - k_min_low);
            self.h_sub_l.push(k_max_high - k_min_low);
        }
        if k_max_high == k_min_low {
            return Ok(());
        }
        let raw_k = 100.0 * div(close - k_min_low, k_max_high - k_min_low)?;
        let slowed_k = match &mut self.slowing_ma {
            Some(average) => {
                average.update(raw_k);
                average.value
            }
            None => raw_k,
        };
        self.value_k = slowed_k;
        if self.ratio {
            let sum_h_sub_l = py_sum(self.h_sub_l.as_slice());
            self.value_d = if sum_h_sub_l == 0.0 {
                0.0
            } else {
                100.0 * div(py_sum(self.c_sub_l.as_slice()), sum_h_sub_l)?
            };
        } else if let Some(average) = &mut self.d_ma {
            average.update(slowed_k);
            self.value_d = average.value;
        }
        if !self.initialized {
            let base_ready = self.highs.len() == self.period_k;
            let slowing_ready = self.slowing_ma.as_ref().is_none_or(|a| a.count.initialized);
            let d_ready = self.ratio || self.d_ma.as_ref().is_none_or(|a| a.count.initialized);
            if base_ready && slowing_ready && d_ready {
                self.initialized = true;
            }
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["value_d", "value_k"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value_d;
        out[1] = self.value_k;
    }
}

struct Cci {
    scalar: f64,
    prices: Window,
    ma: Sma,
    initialized: bool,
    value: f64,
}

impl Twin for Cci {
    fn update(&mut self, bar: &Bar) -> Step {
        let typical_price = (bar.high + bar.low + bar.close) / 3.0;
        self.prices.push(typical_price);
        self.ma.update(typical_price);
        let mad = fast_mad_with_mean(self.prices.as_slice(), self.ma.value);
        if self.ma.count.initialized {
            self.value = if mad > 0.0 {
                div(typical_price - self.ma.value, self.scalar * mad)?
            } else {
                0.0
            };
        }
        if !self.initialized && self.ma.count.initialized {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Rvi {
    scalar: f64,
    sqrt_period: f64,
    sqrt_period_less_one: f64,
    prices: Window,
    ma: Sma,
    pos_ma: Ema,
    neg_ma: Ema,
    previous_close: f64,
    initialized: bool,
    value: f64,
}

impl Twin for Rvi {
    fn update(&mut self, bar: &Bar) -> Step {
        let close = bar.close;
        self.prices.push(close);
        self.ma.update(close);
        let std = fast_std_with_mean(self.prices.as_slice(), self.ma.value);
        // numpy scalars: a zero divisor gives inf, not an exception.
        let std = std * self.sqrt_period / self.sqrt_period_less_one;
        if self.ma.count.initialized {
            if close > self.previous_close {
                self.pos_ma.update(std);
                self.neg_ma.update(0.0);
            } else if close < self.previous_close {
                self.pos_ma.update(0.0);
                self.neg_ma.update(std);
            } else {
                self.pos_ma.update(0.0);
                self.neg_ma.update(0.0);
            }
            let total = self.pos_ma.value + self.neg_ma.value;
            self.value =
                if total == 0.0 { 0.0 } else { div(self.scalar * self.pos_ma.value, total)? };
        }
        self.previous_close = close;
        if !self.initialized && self.pos_ma.count.initialized {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct PsychologicalLine {
    ma: Sma,
    has_inputs: bool,
    initialized: bool,
    previous_close: f64,
    value: f64,
}

impl Twin for PsychologicalLine {
    fn update(&mut self, bar: &Bar) -> Step {
        let close = bar.close;
        if !self.has_inputs {
            self.previous_close = close;
        }
        let diff = close - self.previous_close;
        self.ma.update(if diff <= 0.0 { 0.0 } else { 1.0 });
        self.value = 100.0 * self.ma.value;
        if !self.initialized {
            self.has_inputs = true;
            if self.ma.count.initialized {
                self.initialized = true;
            }
        }
        self.previous_close = close;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Archer {
    signal_period: usize,
    fast_ma: Ema,
    slow_ma: Ema,
    fast_ma_price: Window,
    slow_ma_price: Window,
    initialized: bool,
    long_run: bool,
    short_run: bool,
}

impl Twin for Archer {
    fn update(&mut self, bar: &Bar) -> Step {
        self.fast_ma.update(bar.close);
        self.slow_ma.update(bar.close);
        if self.slow_ma.count.initialized {
            self.fast_ma_price.push(self.fast_ma.value);
            self.slow_ma_price.push(self.slow_ma.value);
            let fast = self.fast_ma_price.last() - self.fast_ma_price.first();
            let slow = self.slow_ma_price.last() - self.slow_ma_price.first();
            self.long_run = fast > 0.0 && slow < 0.0;
            self.long_run = (fast > 0.0 && slow > 0.0) || self.long_run;
            self.short_run = fast < 0.0 && slow > 0.0;
            self.short_run = (fast < 0.0 && slow < 0.0) || self.short_run;
        }
        if !self.initialized
            && self.slow_ma_price.len() > self.signal_period
            && self.slow_ma.count.initialized
        {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["long_run", "short_run"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = flag(self.long_run);
        out[1] = flag(self.short_run);
    }
}

struct Aroon {
    period: usize,
    highs: Window,
    lows: Window,
    initialized: bool,
    aroon_up: f64,
    aroon_down: f64,
    value: f64,
}

impl Aroon {
    /// `np.argmax` (or `argmin`, with *better* `<`) of the deque the
    /// indicator fills with `appendleft`, newest first: the index of the
    /// newest extreme, counted back from the newest bar.
    fn bars_since(oldest_first: &[f64], better: impl Fn(f64, f64) -> bool) -> f64 {
        let mut newest_first = oldest_first.iter().rev();
        let mut best = *newest_first.next().unwrap_or(&f64::NAN);
        let mut index = 0usize;
        for (i, &x) in newest_first.enumerate() {
            if better(x, best) {
                best = x;
                index = i + 1;
            }
        }
        index as f64
    }
}

impl Twin for Aroon {
    fn update(&mut self, bar: &Bar) -> Step {
        self.highs.push(bar.high);
        self.lows.push(bar.low);
        let periods_from_hh = Self::bars_since(self.highs.as_slice(), |x, best| x > best);
        let periods_from_ll = Self::bars_since(self.lows.as_slice(), |x, best| x < best);
        let period = self.period as f64;
        self.aroon_up = 100.0 * (1.0 - div(periods_from_hh, period)?);
        self.aroon_down = 100.0 * (1.0 - div(periods_from_ll, period)?);
        self.value = self.aroon_up - self.aroon_down;
        if !self.initialized && self.highs.len() > self.period {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["aroon_down", "aroon_up", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.aroon_down;
        out[1] = self.aroon_up;
        out[2] = self.value;
    }
}

struct DirectionalMovement {
    pos_ma: Ema,
    neg_ma: Ema,
    has_inputs: bool,
    initialized: bool,
    previous_high: f64,
    previous_low: f64,
    pos: f64,
    neg: f64,
}

impl Twin for DirectionalMovement {
    fn update(&mut self, bar: &Bar) -> Step {
        let (high, low) = (bar.high, bar.low);
        if !self.has_inputs {
            self.previous_high = high;
            self.previous_low = low;
        }
        let up = high - self.previous_high;
        let dn = self.previous_low - low;
        // A C int times a double: a false flag gives -0.0 for a negative move.
        self.pos_ma.update(flag(up > dn && up > 0.0) * up);
        self.neg_ma.update(flag(dn > up && dn > 0.0) * dn);
        self.pos = self.pos_ma.value;
        self.neg = self.neg_ma.value;
        self.previous_high = high;
        self.previous_low = low;
        if !self.initialized {
            self.has_inputs = true;
            if self.neg_ma.count.initialized {
                self.initialized = true;
            }
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["neg", "pos", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.neg;
        out[1] = self.pos;
        // Declared, never assigned.
        out[2] = 0.0;
    }
}

/// `MovingAverageConvergenceDivergence`, on its own or inside the project's
/// MACD histogram and Schaff cycle.
pub(super) struct Macd {
    fast_ma: Ema,
    slow_ma: Ema,
    initialized: bool,
    value: f64,
}

impl Macd {
    /// With the constructor's checks.
    pub(super) fn new(fast_period: usize, slow_period: usize) -> Result<Self, Declined> {
        require(slow_period > fast_period, "slow_period <= fast_period")?;
        Ok(Macd {
            fast_ma: Ema::exponential(fast_period),
            slow_ma: Ema::exponential(slow_period),
            initialized: false,
            value: 0.0,
        })
    }

    pub(super) fn update(&mut self, close: f64) {
        self.fast_ma.update(close);
        self.slow_ma.update(close);
        self.value = self.fast_ma.value - self.slow_ma.value;
        if !self.initialized && self.fast_ma.count.initialized && self.slow_ma.count.initialized {
            self.initialized = true;
        }
    }

    pub(super) fn value(&self) -> f64 {
        self.value
    }
}

impl Twin for Macd {
    fn update(&mut self, bar: &Bar) -> Step {
        Macd::update(self, bar.close);
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

/// The project's `IchimokuCloud` (`src/indicators/ichimoku.py`): Nautilus's,
/// with `initialized` held back until `senkou_period + displacement` bars.
/// Nautilus's own update re-raises the flag before it reads it, so holding
/// it back changes nothing inside.
struct Ichimoku {
    tenkan_period: usize,
    kijun_period: usize,
    senkou_period: usize,
    displacement: usize,
    highs_tenkan: Window,
    lows_tenkan: Window,
    highs_kijun: Window,
    lows_kijun: Window,
    highs_senkou: Window,
    lows_senkou: Window,
    senkou_a: Window,
    senkou_b: Window,
    chikou: Window,
    bars: usize,
    tenkan_sen: f64,
    kijun_sen: f64,
    senkou_span_a: f64,
    senkou_span_b: f64,
    chikou_span: f64,
}

impl Ichimoku {
    fn new(tenkan: usize, kijun: usize, senkou: usize, displacement: usize) -> Self {
        Ichimoku {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_period: senkou,
            displacement,
            highs_tenkan: Window::new(tenkan),
            lows_tenkan: Window::new(tenkan),
            highs_kijun: Window::new(kijun),
            lows_kijun: Window::new(kijun),
            highs_senkou: Window::new(senkou),
            lows_senkou: Window::new(senkou),
            senkou_a: Window::new(displacement),
            senkou_b: Window::new(displacement),
            chikou: Window::new(displacement),
            bars: 0,
            tenkan_sen: 0.0,
            kijun_sen: 0.0,
            senkou_span_a: 0.0,
            senkou_span_b: 0.0,
            chikou_span: 0.0,
        }
    }

    fn middle(highs: &Window, lows: &Window) -> f64 {
        (first_max(highs.as_slice().iter().copied()) + first_min(lows.as_slice().iter().copied()))
            / 2.0
    }
}

impl Twin for Ichimoku {
    fn update(&mut self, bar: &Bar) -> Step {
        let (high, low, close) = (bar.high, bar.low, bar.close);
        self.highs_tenkan.push(high);
        self.lows_tenkan.push(low);
        self.highs_kijun.push(high);
        self.lows_kijun.push(low);
        self.highs_senkou.push(high);
        self.lows_senkou.push(low);
        let nautilus_initialized = self.highs_tenkan.len() >= self.tenkan_period
            && self.highs_kijun.len() >= self.kijun_period
            && self.highs_senkou.len() >= self.senkou_period;
        if self.highs_tenkan.len() >= self.tenkan_period {
            self.tenkan_sen = Self::middle(&self.highs_tenkan, &self.lows_tenkan);
        }
        if self.highs_kijun.len() >= self.kijun_period {
            self.kijun_sen = Self::middle(&self.highs_kijun, &self.lows_kijun);
        }
        let mut mid52 = 0.0;
        if self.highs_senkou.len() >= self.senkou_period {
            mid52 = Self::middle(&self.highs_senkou, &self.lows_senkou);
        }
        if nautilus_initialized {
            if self.senkou_a.len() == self.displacement {
                self.senkou_span_a = self.senkou_a.first();
            }
            self.senkou_a.push((self.tenkan_sen + self.kijun_sen) / 2.0);
            if self.senkou_b.len() == self.displacement {
                self.senkou_span_b = self.senkou_b.first();
            }
            self.senkou_b.push(mid52);
            if self.chikou.len() == self.displacement {
                self.chikou_span = self.chikou.first();
            }
            self.chikou.push(close);
        }
        self.bars += 1;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.highs_senkou.len() >= self.senkou_period
            && self.highs_kijun.len() >= self.kijun_period
            && self.highs_tenkan.len() >= self.tenkan_period
            && self.bars >= self.senkou_period + self.displacement
    }

    fn names(&self) -> &'static [&'static str] {
        &["chikou_span", "kijun_sen", "senkou_span_a", "senkou_span_b", "tenkan_sen"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.chikou_span;
        out[1] = self.kijun_sen;
        out[2] = self.senkou_span_a;
        out[3] = self.senkou_span_b;
        out[4] = self.tenkan_sen;
    }
}

struct Swings {
    high_inputs: Window,
    low_inputs: Window,
    initialized: bool,
    direction: i64,
    high_price: f64,
    low_price: f64,
    length: f64,
    duration: i64,
    since_high: i64,
    since_low: i64,
}

impl Twin for Swings {
    fn update(&mut self, bar: &Bar) -> Step {
        let (high, low) = (bar.high, bar.low);
        self.high_inputs.push(high);
        self.low_inputs.push(low);
        let max_high = first_max(self.high_inputs.as_slice().iter().copied());
        let min_low = first_min(self.low_inputs.as_slice().iter().copied());
        let is_swing_high = high >= max_high && low >= min_low;
        let is_swing_low = low <= min_low && high <= max_high;
        if is_swing_high && !is_swing_low {
            self.high_price = high;
            self.direction = 1;
            self.since_high = 0;
            self.since_low += 1;
        } else if is_swing_low && !is_swing_high {
            self.low_price = low;
            self.direction = -1;
            self.since_low = 0;
            self.since_high += 1;
        } else {
            self.since_high += 1;
            self.since_low += 1;
        }
        if !self.initialized {
            if self.high_price != 0.0 && self.low_price != 0.0 {
                self.initialized = true;
            }
        } else {
            self.length = self.high_price - self.low_price;
            self.duration = if self.direction == 1 { self.since_low } else { self.since_high };
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["direction", "duration", "high_price", "length", "low_price", "since_high", "since_low"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.direction as f64;
        out[1] = self.duration as f64;
        out[2] = self.high_price;
        out[3] = self.length;
        out[4] = self.low_price;
        out[5] = self.since_high as f64;
        out[6] = self.since_low as f64;
    }
}

/// `AverageTrueRange` with `use_previous=True`, inside Keltner and Pressure.
struct Atr {
    ma: Average,
    value_floor: f64,
    has_inputs: bool,
    initialized: bool,
    previous_close: f64,
    value: f64,
}

impl Atr {
    fn new(ma: Average, value_floor: f64) -> Self {
        Atr {
            ma,
            value_floor,
            has_inputs: false,
            initialized: false,
            previous_close: 0.0,
            value: 0.0,
        }
    }

    fn update(&mut self, high: f64, low: f64, close: f64) {
        if !self.has_inputs {
            self.previous_close = close;
        }
        self.ma.update(c_max(self.previous_close, high) - c_min(low, self.previous_close));
        self.previous_close = close;
        let ma_value = self.ma.value();
        self.value = if self.value_floor == 0.0 || self.value_floor < ma_value {
            ma_value
        } else {
            self.value_floor
        };
        if !self.initialized {
            self.has_inputs = true;
            if self.ma.initialized() {
                self.initialized = true;
            }
        }
    }
}

struct Bollinger {
    period: usize,
    k: f64,
    ma: Sma,
    prices: Window,
    initialized: bool,
    upper: f64,
    middle: f64,
    lower: f64,
}

impl Twin for Bollinger {
    fn update(&mut self, bar: &Bar) -> Step {
        let typical = (bar.high + bar.low + bar.close) / 3.0;
        self.prices.push(typical);
        self.ma.update(typical);
        if !self.initialized && self.prices.len() >= self.period {
            self.initialized = true;
        }
        let std = fast_std_with_mean(self.prices.as_slice(), self.ma.value);
        self.upper = self.ma.value + (self.k * std);
        self.middle = self.ma.value;
        self.lower = self.ma.value - (self.k * std);
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        BANDS
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.lower;
        out[1] = self.middle;
        out[2] = self.upper;
    }
}

struct Donchian {
    period: usize,
    upper_prices: Window,
    lower_prices: Window,
    initialized: bool,
    upper: f64,
    middle: f64,
    lower: f64,
}

impl Twin for Donchian {
    fn update(&mut self, bar: &Bar) -> Step {
        self.upper_prices.push(bar.high);
        self.lower_prices.push(bar.low);
        if !self.initialized
            && self.upper_prices.len() >= self.period
            && self.lower_prices.len() >= self.period
        {
            self.initialized = true;
        }
        self.upper = first_max(self.upper_prices.as_slice().iter().copied());
        self.lower = first_min(self.lower_prices.as_slice().iter().copied());
        self.middle = (self.upper + self.lower) / 2.0;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        BANDS
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.lower;
        out[1] = self.middle;
        out[2] = self.upper;
    }
}

struct Keltner {
    k_multiplier: f64,
    ma: Ema,
    atr: Atr,
    initialized: bool,
    upper: f64,
    middle: f64,
    lower: f64,
}

impl Twin for Keltner {
    fn update(&mut self, bar: &Bar) -> Step {
        let typical_price = (bar.high + bar.low + bar.close) / 3.0;
        self.ma.update(typical_price);
        self.atr.update(bar.high, bar.low, bar.close);
        self.upper = self.ma.value + (self.atr.value * self.k_multiplier);
        self.middle = self.ma.value;
        self.lower = self.ma.value - (self.atr.value * self.k_multiplier);
        if !self.initialized && self.ma.count.initialized {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        BANDS
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.lower;
        out[1] = self.middle;
        out[2] = self.upper;
    }
}

/// The builtin `sum` over a deque holding floats and the int `0`
/// (`None`): ints add exactly until the first float, which starts the
/// compensated float sum; later ints add to it uncompensated.
#[derive(Clone, Copy, Debug, Default)]
struct MixedSum {
    float: bool,
    total: f64,
    compensation: f64,
}

impl MixedSum {
    fn add(&mut self, item: Option<f64>) {
        match item {
            None => {
                if self.float {
                    self.total += 0.0;
                }
            }
            Some(x) if !self.float => {
                self.float = true;
                self.total = 0.0 + x;
            }
            Some(x) => {
                let t = self.total + x;
                if self.total.abs() >= x.abs() {
                    self.compensation += (self.total - t) + x;
                } else {
                    self.compensation += (x - t) + self.total;
                }
                self.total = t;
            }
        }
    }

    /// What the call returns, as the C double it is assigned to.
    fn result(&self) -> f64 {
        if !self.float {
            return 0.0;
        }
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.total + self.compensation
        } else {
            self.total
        }
    }
}

struct Obv {
    period: usize,
    obv: VecDeque<Option<f64>>,
    running: MixedSum,
    initialized: bool,
    value: f64,
}

impl Twin for Obv {
    fn update(&mut self, bar: &Bar) -> Step {
        let item = if bar.close > bar.open {
            Some(bar.volume)
        } else if bar.close < bar.open {
            Some(-bar.volume)
        } else {
            None
        };
        if self.period == 0 {
            // Nothing leaves an unbounded deque, so its sum folds forward.
            self.running.add(item);
            self.value = self.running.result();
            self.initialized = true;
        } else {
            if self.obv.len() == self.period {
                self.obv.pop_front();
            }
            self.obv.push_back(item);
            let mut sum = MixedSum::default();
            for &kept in &self.obv {
                sum.add(kept);
            }
            self.value = sum.result();
            if !self.initialized && self.obv.len() >= self.period {
                self.initialized = true;
            }
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Kvo {
    fast_ma: Ema,
    slow_ma: Ema,
    signal_ma: Ema,
    previous_hlc3: f64,
    initialized: bool,
    value: f64,
}

impl Twin for Kvo {
    fn update(&mut self, bar: &Bar) -> Step {
        let hlc3 = (bar.high + bar.low + bar.close) / 3.0;
        let volume = bar.volume;
        if hlc3 > self.previous_hlc3 {
            self.fast_ma.update(volume);
            self.slow_ma.update(volume);
        } else if hlc3 < self.previous_hlc3 {
            self.fast_ma.update(-volume);
            self.slow_ma.update(-volume);
        } else {
            self.fast_ma.update(0.0);
            self.slow_ma.update(0.0);
        }
        if self.slow_ma.count.initialized {
            self.signal_ma.update(self.fast_ma.value - self.slow_ma.value);
            self.value = self.signal_ma.value;
        }
        if !self.initialized && self.signal_ma.count.initialized {
            self.initialized = true;
        }
        self.previous_hlc3 = hlc3;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        VALUE
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

struct Pressure {
    atr: Atr,
    average_volume: Ema,
    initialized: bool,
    value: f64,
    value_cumulative: f64,
}

impl Twin for Pressure {
    fn update(&mut self, bar: &Bar) -> Step {
        let (high, low, close, volume) = (bar.high, bar.low, bar.close, bar.volume);
        self.atr.update(high, low, close);
        self.average_volume.update(volume);
        if !self.initialized && self.atr.initialized {
            self.initialized = true;
        }
        if self.average_volume.value == 0.0 || self.atr.value == 0.0 {
            self.value = 0.0;
            return Ok(());
        }
        let relative_volume = div(volume, self.average_volume.value)?;
        let buy_pressure = div(close - low, self.atr.value)? * relative_volume;
        let sell_pressure = div(high - close, self.atr.value)? * relative_volume;
        self.value = buy_pressure - sell_pressure;
        self.value_cumulative += self.value;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["value", "value_cumulative"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
        out[1] = self.value_cumulative;
    }
}

/// `LinearRegression`'s slope, intercept, value and cfo; its `R2` and
/// `degree` go through `statistics.mean` and `np.arctan`, not transcribed.
struct LinearRegression {
    period: usize,
    inputs: Window,
    initialized: bool,
    slope: f64,
    intercept: f64,
    cfo: f64,
    value: f64,
}

impl Twin for LinearRegression {
    fn update(&mut self, bar: &Bar) -> Step {
        self.inputs.push(bar.close);
        if !self.initialized {
            if self.inputs.len() >= self.period {
                self.initialized = true;
            } else {
                return Ok(());
            }
        }
        let period = self.period as f64;
        let x_sum = 0.5 * period * (self.period + 1) as f64;
        let x2_sum = x_sum * (2 * self.period + 1) as f64 / 3.0;
        let divisor = period * x2_sum - x_sum * x_sum;
        // The builtin `sum` of numpy scalars adds them in turn, from 0.
        let y = self.inputs.as_slice();
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        for (i, &value) in y.iter().enumerate() {
            y_sum += value;
            xy_sum += (i + 1) as f64 * value;
        }
        self.slope = div(period * xy_sum - x_sum * y_sum, divisor)?;
        self.intercept = div(y_sum * x2_sum - x_sum * xy_sum, divisor)?;
        let last = y[y.len() - 1];
        let residual = self.slope * period + self.intercept - last;
        self.value = residual + last;
        // numpy scalars: a zero close gives inf or NaN, not an exception.
        self.cfo = 100.0 * residual / last;
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["cfo", "intercept", "slope", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.cfo;
        out[1] = self.intercept;
        out[2] = self.slope;
        out[3] = self.value;
    }
}

pub(super) fn require(condition: bool, reason: &str) -> Step {
    if condition {
        Ok(())
    } else {
        Err(Declined::new(reason))
    }
}

pub(super) fn boxed(twin: impl Twin + 'static) -> Result<Box<dyn Twin>, Declined> {
    Ok(Box::new(twin))
}

/// The twin of Nautilus's indicator class *kind*, if it has one here.
pub(super) fn build(kind: &str, params: &Params) -> Option<Result<Box<dyn Twin>, Declined>> {
    let built = match kind {
        "SimpleMovingAverage" => {
            params.period("period").and_then(|p| boxed(AverageTwin(Average::simple(p))))
        }
        "ExponentialMovingAverage" => {
            params.period("period").and_then(|p| boxed(AverageTwin(Average::exponential(p))))
        }
        "WilderMovingAverage" => {
            params.period("period").and_then(|p| boxed(AverageTwin(Average::wilder(p))))
        }
        "DoubleExponentialMovingAverage" => params.period("period").and_then(|p| {
            boxed(Dema {
                ma1: Ema::exponential(p),
                ma2: Ema::exponential(p),
                value: 0.0,
                initialized: false,
            })
        }),
        "WeightedMovingAverage" => (|| {
            let period = params.period("period")?;
            boxed(Wma::new(period, params.floats("weights")?)?)
        })(),
        "HullMovingAverage" => params.period("period").and_then(|p| boxed(Hma::new(p)?)),
        "AdaptiveMovingAverage" => (|| {
            let period_er = params.period("period_er")?;
            let fast = params.period("period_alpha_fast")?;
            let slow = params.period("period_alpha_slow")?;
            require(slow > fast, "period_alpha_slow <= period_alpha_fast")?;
            let alpha_fast = 2.0 / (fast as f64 + 1.0);
            let alpha_slow = 2.0 / (slow as f64 + 1.0);
            boxed(Ama {
                count: Count::new(period_er),
                alpha_slow,
                alpha_diff: alpha_fast - alpha_slow,
                efficiency_ratio: EfficiencyRatio::new(period_er),
                value: 0.0,
            })
        })(),
        "VariableIndexDynamicAverage" => params.period("period").and_then(|p| {
            boxed(Vidya {
                count: Count::new(p),
                cmo: Cmo::new(Average::simple(p), Average::simple(p)),
                cmo_pct: 0.0,
                alpha: 2.0 / (p as f64 + 1.0),
                value: 0.0,
            })
        }),
        "ChandeMomentumOscillator" => params
            .period("period")
            .and_then(|p| boxed(Cmo::new(Average::wilder(p), Average::wilder(p)))),
        "RelativeStrengthIndex" => params.period("period").and_then(|p| {
            boxed(Rsi {
                average_gain: Ema::exponential(p),
                average_loss: Ema::exponential(p),
                has_inputs: false,
                initialized: false,
                last_value: 0.0,
                value: 0.0,
            })
        }),
        "RateOfChange" => (|| {
            let period = params.period("period")?;
            require(period > 1, "period <= 1")?;
            boxed(Roc { period, prices: Window::new(period), initialized: false, value: 0.0 })
        })(),
        "Stochastics" => (|| {
            let period_k = params.period("period_k")?;
            let period_d = params.period("period_d")?;
            let slowing = usize::try_from(params.int_or("slowing", 1)?)
                .ok()
                .filter(|&s| (1..=1 << 24).contains(&s))
                .ok_or_else(|| Declined::new("slowing is not a usable period"))?;
            let ratio = match params.text_or("d_method", "ratio")?.as_str() {
                "ratio" => true,
                "moving_average" => false,
                _ => return Err(Declined::new("unknown d_method")),
            };
            boxed(Stochastics {
                period_k,
                ratio,
                highs: Window::new(period_k),
                lows: Window::new(period_k),
                c_sub_l: Window::new(period_d),
                h_sub_l: Window::new(period_d),
                slowing_ma: (slowing > 1).then(|| Ema::exponential(slowing)),
                d_ma: (!ratio).then(|| Ema::exponential(period_d)),
                initialized: false,
                value_k: 0.0,
                value_d: 0.0,
            })
        })(),
        "CommodityChannelIndex" => (|| {
            let period = params.period("period")?;
            boxed(Cci {
                scalar: params.float_or("scalar", 0.015)?,
                prices: Window::new(period),
                ma: Sma::new(period),
                initialized: false,
                value: 0.0,
            })
        })(),
        "RelativeVolatilityIndex" => (|| {
            let period = params.period("period")?;
            require(period > 1, "period <= 1")?;
            boxed(Rvi {
                scalar: params.float_or("scalar", 100.0)?,
                sqrt_period: (period as f64).sqrt(),
                sqrt_period_less_one: ((period - 1) as f64).sqrt(),
                prices: Window::new(period),
                ma: Sma::new(period),
                pos_ma: Ema::exponential(period),
                neg_ma: Ema::exponential(period),
                previous_close: 0.0,
                initialized: false,
                value: 0.0,
            })
        })(),
        "PsychologicalLine" => params.period("period").and_then(|p| {
            boxed(PsychologicalLine {
                ma: Sma::new(p),
                has_inputs: false,
                initialized: false,
                previous_close: 0.0,
                value: 0.0,
            })
        }),
        "ArcherMovingAveragesTrends" => (|| {
            let fast = params.period("fast_period")?;
            let slow = params.period("slow_period")?;
            let signal = params.period("signal_period")?;
            require(slow > fast, "fast_period >= slow_period")?;
            boxed(Archer {
                signal_period: signal,
                fast_ma: Ema::exponential(fast),
                slow_ma: Ema::exponential(slow),
                fast_ma_price: Window::new(signal + 1),
                slow_ma_price: Window::new(signal + 1),
                initialized: false,
                long_run: false,
                short_run: false,
            })
        })(),
        "AroonOscillator" => params.period("period").and_then(|p| {
            boxed(Aroon {
                period: p,
                highs: Window::new(p + 1),
                lows: Window::new(p + 1),
                initialized: false,
                aroon_up: 0.0,
                aroon_down: 0.0,
                value: 0.0,
            })
        }),
        "DirectionalMovement" => params.period("period").and_then(|p| {
            boxed(DirectionalMovement {
                pos_ma: Ema::exponential(p),
                neg_ma: Ema::exponential(p),
                has_inputs: false,
                initialized: false,
                previous_high: 0.0,
                previous_low: 0.0,
                pos: 0.0,
                neg: 0.0,
            })
        }),
        "MovingAverageConvergenceDivergence" => {
            (|| boxed(Macd::new(params.period("fast_period")?, params.period("slow_period")?)?))()
        }
        "IchimokuCloud" => (|| {
            let tenkan = usize::try_from(params.int_or("tenkan_period", 9)?).unwrap_or(0);
            let kijun = usize::try_from(params.int_or("kijun_period", 26)?).unwrap_or(0);
            let senkou = usize::try_from(params.int_or("senkou_period", 52)?).unwrap_or(0);
            let displacement = usize::try_from(params.int_or("displacement", 26)?).unwrap_or(0);
            let limit = 1 << 24;
            require(
                [tenkan, kijun, senkou, displacement].iter().all(|&p| (1..=limit).contains(&p)),
                "periods the constructor refuses",
            )?;
            require(kijun >= tenkan && senkou >= kijun, "periods out of order")?;
            boxed(Ichimoku::new(tenkan, kijun, senkou, displacement))
        })(),
        "Swings" => params.period("period").and_then(|p| {
            boxed(Swings {
                high_inputs: Window::new(p),
                low_inputs: Window::new(p),
                initialized: false,
                direction: 0,
                high_price: 0.0,
                low_price: 0.0,
                length: 0.0,
                duration: 0,
                since_high: 0,
                since_low: 0,
            })
        }),
        "BollingerBands" => (|| {
            let period = params.period("period")?;
            boxed(Bollinger {
                period,
                k: params.float("k")?,
                ma: Sma::new(period),
                prices: Window::new(period),
                initialized: false,
                upper: 0.0,
                middle: 0.0,
                lower: 0.0,
            })
        })(),
        "DonchianChannel" => params.period("period").and_then(|p| {
            boxed(Donchian {
                period: p,
                upper_prices: Window::new(p),
                lower_prices: Window::new(p),
                initialized: false,
                upper: 0.0,
                middle: 0.0,
                lower: 0.0,
            })
        }),
        "KeltnerChannel" => (|| {
            let period = params.period("period")?;
            boxed(Keltner {
                k_multiplier: params.float("k_multiplier")?,
                ma: Ema::exponential(period),
                atr: Atr::new(Average::simple(period), params.float_or("atr_floor", 0.0)?),
                initialized: false,
                upper: 0.0,
                middle: 0.0,
                lower: 0.0,
            })
        })(),
        "OnBalanceVolume" => (|| {
            let period = usize::try_from(params.int_or("period", 0)?)
                .ok()
                .filter(|&p| p <= 1 << 24)
                .ok_or_else(|| Declined::new("period is not a usable length"))?;
            boxed(Obv {
                period,
                obv: VecDeque::new(),
                running: MixedSum::default(),
                initialized: false,
                value: 0.0,
            })
        })(),
        "KlingerVolumeOscillator" => (|| {
            boxed(Kvo {
                fast_ma: Ema::exponential(params.period("fast_period")?),
                slow_ma: Ema::exponential(params.period("slow_period")?),
                signal_ma: Ema::exponential(params.period("signal_period")?),
                previous_hlc3: 0.0,
                initialized: false,
                value: 0.0,
            })
        })(),
        "Pressure" => (|| {
            let period = params.period("period")?;
            boxed(Pressure {
                atr: Atr::new(Average::exponential(period), params.float_or("atr_floor", 0.0)?),
                average_volume: Ema::exponential(period),
                initialized: false,
                value: 0.0,
                value_cumulative: 0.0,
            })
        })(),
        "LinearRegression" => (|| {
            let period = params.period("period")?;
            require(period > 1, "period <= 1 divides by zero")?;
            boxed(LinearRegression {
                period,
                inputs: Window::new(period),
                initialized: false,
                slope: 0.0,
                intercept: 0.0,
                cfo: 0.0,
                value: 0.0,
            })
        })(),
        _ => return None,
    };
    Some(built)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mixed_sum_starts_compensating_at_the_first_float() {
        let mut sum = MixedSum::default();
        sum.add(None);
        assert_eq!(sum.result(), 0.0);
        for item in [Some(1e16), Some(1.0), None, Some(-1e16)] {
            sum.add(item);
        }
        // The builtin sum of [0, 1e16, 1.0, 0, -1e16] is 1.0, not 0.0.
        assert_eq!(sum.result(), 1.0);
    }

    #[test]
    fn aroon_counts_back_to_the_newest_extreme() {
        // Newest first: [3, 5, 5, 1] -> argmax 1, the newer of the two fives.
        assert_eq!(Aroon::bars_since(&[1.0, 5.0, 5.0, 3.0], |x, best| x > best), 1.0);
        assert_eq!(Aroon::bars_since(&[1.0, 5.0, 5.0, 3.0], |x, best| x < best), 3.0);
    }

    #[test]
    fn a_false_flag_times_a_falling_move_is_negative_zero() {
        let product = flag(false) * -2.5;
        assert!(product == 0.0 && product.is_sign_negative());
    }
}
