//! Twins of the engine's own indicators (`src/indicators` in the engine),
//! which are Python: their arithmetic is CPython's float arithmetic.

use super::nautilus::{boxed, require, Ema, Macd};
use super::window::Window;
use super::{div, first_max, first_min, Bar, Declined, Params, Step, Twin};

const BANDS: &[&str] = &["lower", "middle", "upper"];

struct CloseChannel {
    period: usize,
    closes: Window,
    initialized: bool,
    upper: f64,
    middle: f64,
    lower: f64,
}

impl Twin for CloseChannel {
    fn update(&mut self, bar: &Bar) -> Step {
        self.closes.push(bar.close);
        if !self.initialized && self.closes.len() >= self.period {
            self.initialized = true;
        }
        self.upper = first_max(self.closes.as_slice().iter().copied());
        self.lower = first_min(self.closes.as_slice().iter().copied());
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

struct MacdHistogram {
    macd_line: Macd,
    signal_ma: Ema,
    initialized: bool,
    macd: f64,
    signal: f64,
    value: f64,
}

impl Twin for MacdHistogram {
    fn update(&mut self, bar: &Bar) -> Step {
        self.macd_line.update(bar.close);
        self.macd = self.macd_line.value();
        if !Twin::initialized(&self.macd_line) {
            return Ok(());
        }
        self.signal_ma.update(self.macd);
        self.signal = self.signal_ma.value();
        self.value = self.macd - self.signal;
        if !self.initialized && self.signal_ma.initialized() {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["macd", "signal", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.macd;
        out[1] = self.signal;
        out[2] = self.value;
    }
}

/// `_SmoothedStochastic` (`schaff_trend_cycle.py`).
struct SmoothedStochastic {
    period: usize,
    window: Window,
    factor: f64,
    k: f64,
    value: f64,
}

impl SmoothedStochastic {
    const FLAT: f64 = 1e-9;

    fn new(period: usize, factor: f64) -> Self {
        SmoothedStochastic { period, window: Window::new(period), factor, k: 0.0, value: 0.0 }
    }

    fn update(&mut self, x: f64) -> Result<bool, Declined> {
        self.window.push(x);
        if self.window.len() < self.period {
            return Ok(false);
        }
        let lo = first_min(self.window.as_slice().iter().copied());
        let hi = first_max(self.window.as_slice().iter().copied());
        // The builtin `max(a, b)`: `a` unless `b > a`.
        let (a, b) = (hi.abs(), lo.abs());
        let larger = if b > a { b } else { a };
        if hi - lo > Self::FLAT * larger {
            self.k = div(100.0 * (x - lo), hi - lo)?;
        }
        self.value += self.factor * (self.k - self.value);
        Ok(true)
    }
}

struct SchaffTrendCycle {
    warmup_period: usize,
    macd: Macd,
    macd_stoch: SmoothedStochastic,
    cycle: SmoothedStochastic,
    closes: usize,
    initialized: bool,
    value: f64,
}

impl Twin for SchaffTrendCycle {
    fn update(&mut self, bar: &Bar) -> Step {
        self.closes += 1;
        self.macd.update(bar.close);
        if self.macd_stoch.update(self.macd.value())? && self.cycle.update(self.macd_stoch.value)? {
            self.value = self.cycle.value;
        }
        if !self.initialized && self.closes >= self.warmup_period {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.value;
    }
}

/// `SignalLine`: an EMA of one output of another indicator's twin.
struct SignalLine {
    source: Box<dyn Twin>,
    field: usize,
    readings: Vec<f64>,
    signal_ma: Ema,
    initialized: bool,
    line: f64,
    signal: f64,
    value: f64,
}

impl Twin for SignalLine {
    fn update(&mut self, bar: &Bar) -> Step {
        self.source.update(bar)?;
        if !self.source.initialized() {
            return Ok(());
        }
        self.source.read(&mut self.readings);
        let line = self.readings[self.field];
        if line.is_nan() {
            return Ok(());
        }
        self.line = line;
        self.signal_ma.update(self.line);
        self.signal = self.signal_ma.value();
        self.value = self.line - self.signal;
        if !self.initialized && self.signal_ma.initialized() {
            self.initialized = true;
        }
        Ok(())
    }

    fn initialized(&self) -> bool {
        self.initialized
    }

    fn names(&self) -> &'static [&'static str] {
        &["line", "signal", "value"]
    }

    fn read(&self, out: &mut [f64]) {
        out[0] = self.line;
        out[1] = self.signal;
        out[2] = self.value;
    }
}

/// The twin of the engine's indicator class *kind*, if it has one here.
pub(super) fn build(kind: &str, params: &Params) -> Option<Result<Box<dyn Twin>, Declined>> {
    let built = match kind {
        "CloseChannel" => (|| {
            let period = params.period_or("period", 20)?;
            boxed(CloseChannel {
                period,
                closes: Window::new(period),
                initialized: false,
                upper: 0.0,
                middle: 0.0,
                lower: 0.0,
            })
        })(),
        "MACDHistogram" => (|| {
            let fast = params.period_or("fast_period", 12)?;
            let slow = params.period_or("slow_period", 26)?;
            let signal = params.period_or("signal_period", 9)?;
            boxed(MacdHistogram {
                macd_line: Macd::new(fast, slow)?,
                signal_ma: Ema::exponential(signal),
                initialized: false,
                macd: 0.0,
                signal: 0.0,
                value: 0.0,
            })
        })(),
        "SchaffTrendCycle" => (|| {
            let fast = params.period_or("fast_period", 23)?;
            let slow = params.period_or("slow_period", 50)?;
            let stoch = params.period_or("stoch_period", 10)?;
            let factor = params.float_or("factor", 0.5)?;
            require(stoch >= 2, "stoch_period < 2")?;
            require(0.0 < factor && factor <= 1.0, "factor outside (0, 1]")?;
            boxed(SchaffTrendCycle {
                warmup_period: slow + 2 * stoch - 2,
                macd: Macd::new(fast, slow)?,
                macd_stoch: SmoothedStochastic::new(stoch, factor),
                cycle: SmoothedStochastic::new(stoch, factor),
                closes: 0,
                initialized: false,
                value: 0.0,
            })
        })(),
        "SignalLine" => (|| {
            let source = params.text_or("source", "Trix")?;
            let ema_period = params.period_or("ema_period", 9)?;
            let source_field = params.text_or("source_field", "value")?;
            require(source != "SignalLine", "a SignalLine of a SignalLine")?;
            let twin = super::build(&source, &params.rest())?;
            let field =
                twin.names().iter().position(|&name| name == source_field).ok_or_else(|| {
                    Declined::new(format!("{source} has no twin output {source_field}"))
                })?;
            let readings = vec![0.0; twin.names().len()];
            boxed(SignalLine {
                source: twin,
                field,
                readings,
                signal_ma: Ema::exponential(ema_period),
                initialized: false,
                line: 0.0,
                signal: 0.0,
                value: 0.0,
            })
        })(),
        _ => return None,
    };
    Some(built)
}
