//! Exact twins of the indicators the engine's precomputed lanes walk.
//!
//! The engine precomputes a composable strategy's indicators by walking each
//! one over its bars as a strategy would (`parity/indicators.py` in the
//! engine): one Python call per bar into Nautilus's Cython indicators,
//! talipp's pure-Python ones, or the project's own. A twin here is the same
//! state machine. It reads the same bar fields and performs the same
//! floating-point operations in the same order -- C's for the Cython,
//! CPython's for the Python (`crate::metrics::exact`) -- so its readings are
//! the original's to the bit, without the per-bar Python call.
//!
//! A twin declines (`Declined`) wherever its original would raise, such as a
//! division by zero Python checks, or would take a path not transcribed here;
//! the engine then walks the original. The engine also checks each twin
//! against its original on probe bars before trusting it, because how a C
//! compiler built the Cython (a multiply-add fused on one architecture and
//! not another) is the one thing a transcription cannot see.

mod nautilus;
mod project;
mod talipp;
mod window;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// One bar's fields, as the indicators read them (`bar.close.as_double()`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Why a twin cannot stand in for its original.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declined(pub String);

impl Declined {
    pub fn new(reason: impl Into<String>) -> Self {
        Declined(reason.into())
    }
}

pub type Step = Result<(), Declined>;

/// An indicator's state machine, fed one bar at a time.
pub trait Twin {
    /// Take one bar, as the original's `handle_bar` does.
    fn update(&mut self, bar: &Bar) -> Step;
    /// The original's `initialized` after the last bar.
    fn initialized(&self) -> bool;
    /// The outputs `read` writes, in its order: the original's attribute names.
    fn names(&self) -> &'static [&'static str];
    /// The outputs' current values, as the original's attributes hold them.
    fn read(&self, out: &mut [f64]);
}

/// A constructor argument, as Python passed it.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Int(i64),
    Float(f64),
    Text(String),
    Floats(Vec<f64>),
}

/// An indicator's constructor arguments, each to be taken exactly once.
///
/// A twin takes the arguments it understands; one left over is an argument
/// the twin does not transcribe, so the build declines (`finish`).
pub struct Params {
    values: BTreeMap<String, Param>,
    taken: RefCell<BTreeSet<String>>,
}

impl Params {
    pub fn new(values: BTreeMap<String, Param>) -> Self {
        Params { values, taken: RefCell::new(BTreeSet::new()) }
    }

    fn get(&self, name: &str) -> Option<&Param> {
        let found = self.values.get(name);
        if found.is_some() {
            self.taken.borrow_mut().insert(name.to_string());
        }
        found
    }

    /// A Python `int` argument.
    pub fn int(&self, name: &str) -> Result<i64, Declined> {
        match self.get(name) {
            Some(Param::Int(value)) => Ok(*value),
            Some(_) => Err(Declined::new(format!("{name} is not an int"))),
            None => Err(Declined::new(format!("{name} is missing"))),
        }
    }

    /// A Python `int` argument, or *default* where none was passed.
    pub fn int_or(&self, name: &str, default: i64) -> Result<i64, Declined> {
        if self.values.contains_key(name) {
            self.int(name)
        } else {
            Ok(default)
        }
    }

    /// A positive `int` argument, as a length.
    pub fn period(&self, name: &str) -> Result<usize, Declined> {
        let value = self.int(name)?;
        usize::try_from(value)
            .ok()
            .filter(|&period| (1..=1 << 24).contains(&period))
            .ok_or_else(|| Declined::new(format!("{name} is not a usable period")))
    }

    /// A positive `int` argument, as a length, or *default* where none was
    /// passed.
    pub fn period_or(&self, name: &str, default: usize) -> Result<usize, Declined> {
        if self.values.contains_key(name) {
            self.period(name)
        } else {
            Ok(default)
        }
    }

    /// A Python `float` argument.
    pub fn float(&self, name: &str) -> Result<f64, Declined> {
        match self.get(name) {
            Some(Param::Float(value)) => Ok(*value),
            Some(_) => Err(Declined::new(format!("{name} is not a float"))),
            None => Err(Declined::new(format!("{name} is missing"))),
        }
    }

    /// A Python `float` argument, or *default* where none was passed.
    pub fn float_or(&self, name: &str, default: f64) -> Result<f64, Declined> {
        if self.values.contains_key(name) {
            self.float(name)
        } else {
            Ok(default)
        }
    }

    /// A Python `str` argument, or *default* where none was passed.
    pub fn text_or(&self, name: &str, default: &str) -> Result<String, Declined> {
        match self.get(name) {
            Some(Param::Text(value)) => Ok(value.clone()),
            Some(_) => Err(Declined::new(format!("{name} is not a str"))),
            None => Ok(default.to_string()),
        }
    }

    /// A list of floats.
    pub fn floats(&self, name: &str) -> Result<Vec<f64>, Declined> {
        match self.get(name) {
            Some(Param::Floats(values)) => Ok(values.clone()),
            Some(_) => Err(Declined::new(format!("{name} is not a list of floats"))),
            None => Err(Declined::new(format!("{name} is missing"))),
        }
    }

    /// The arguments not taken, as their own `Params` (a source's, for one
    /// indicator built over another).
    pub fn rest(&self) -> Params {
        let taken = self.taken.borrow();
        let values = self
            .values
            .iter()
            .filter(|(name, _)| !taken.contains(*name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        drop(taken);
        for name in values.keys() {
            self.taken.borrow_mut().insert(name.clone());
        }
        Params::new(values)
    }

    /// Decline if an argument was passed that no twin took.
    pub fn finish(&self) -> Step {
        let taken = self.taken.borrow();
        match self.values.keys().find(|name| !taken.contains(*name)) {
            Some(name) => Err(Declined::new(format!("{name} is not transcribed"))),
            None => Ok(()),
        }
    }
}

/// The admitted bars' columns.
pub struct Columns<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
}

impl Columns<'_> {
    fn len(&self) -> usize {
        self.close.len()
    }

    fn bar(&self, index: usize) -> Bar {
        Bar {
            open: self.open[index],
            high: self.high[index],
            low: self.low[index],
            close: self.close[index],
            volume: self.volume[index],
        }
    }
}

/// A twin's readings after each admitted bar.
///
/// `initialized` is the original's flag after each bar; each output holds
/// the attribute's value after each bar where that flag is set and NaN
/// elsewhere, as the walk records them.
pub struct Series {
    pub initialized: Vec<bool>,
    pub outputs: Vec<(&'static str, Vec<f64>)>,
}

/// Build the twin of the indicator class *kind* from its arguments.
pub fn build(kind: &str, params: &Params) -> Result<Box<dyn Twin>, Declined> {
    let twin = nautilus::build(kind, params)
        .or_else(|| project::build(kind, params))
        .or_else(|| talipp::build(kind, params))
        .unwrap_or_else(|| Err(Declined::new(format!("{kind} has no twin"))))?;
    params.finish()?;
    Ok(twin)
}

/// *kind*'s readings over *columns*, or why its twin declined.
pub fn compute(kind: &str, params: &Params, columns: &Columns<'_>) -> Result<Series, Declined> {
    let lengths = [columns.open.len(), columns.high.len(), columns.low.len(), columns.volume.len()];
    if lengths.iter().any(|&length| length != columns.len()) {
        return Err(Declined::new("columns differ in length"));
    }
    let mut twin = build(kind, params)?;
    let names = twin.names();
    let count = columns.len();
    let mut initialized = vec![false; count];
    let mut outputs: Vec<Vec<f64>> = names.iter().map(|_| vec![f64::NAN; count]).collect();
    let mut current = vec![0.0; names.len()];
    for index in 0..count {
        twin.update(&columns.bar(index))?;
        if twin.initialized() {
            initialized[index] = true;
            twin.read(&mut current);
            for (output, &value) in outputs.iter_mut().zip(&current) {
                output[index] = value;
            }
        }
    }
    Ok(Series { initialized, outputs: names.iter().copied().zip(outputs).collect() })
}

/// `a / b` where Python (or Cython without `cdivision`) raises
/// `ZeroDivisionError` on a zero divisor.
#[inline]
pub(crate) fn div(a: f64, b: f64) -> Result<f64, Declined> {
    if b == 0.0 {
        Err(Declined::new("division by zero"))
    } else {
        Ok(a / b)
    }
}

/// The builtin `max` of a non-empty run: the first of the largest.
#[inline]
pub(crate) fn first_max(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut iter = values.into_iter();
    let first = iter.next().unwrap_or(f64::NAN);
    iter.fold(first, |best, x| if x > best { x } else { best })
}

/// The builtin `min` of a non-empty run: the first of the smallest.
#[inline]
pub(crate) fn first_min(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut iter = values.into_iter();
    let first = iter.next().unwrap_or(f64::NAN);
    iter.fold(first, |best, x| if x < best { x } else { best })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(items: &[(&str, Param)]) -> Params {
        Params::new(items.iter().map(|(name, value)| (name.to_string(), value.clone())).collect())
    }

    #[test]
    fn an_argument_no_twin_takes_declines() {
        let taken = params(&[("period", Param::Int(3))]);
        assert!(build("SimpleMovingAverage", &taken).is_ok());
        let extra = params(&[("period", Param::Int(3)), ("price_type", Param::Int(1))]);
        assert!(build("SimpleMovingAverage", &extra).is_err());
        assert!(build("NoSuchIndicator", &taken).is_err());
    }

    #[test]
    fn readings_are_nan_until_warm() {
        let close = [1.0, 2.0, 3.0, 4.0];
        let zeros = [0.0; 4];
        let columns =
            Columns { open: &close, high: &close, low: &close, close: &close, volume: &zeros };
        let series =
            compute("SimpleMovingAverage", &params(&[("period", Param::Int(3))]), &columns)
                .unwrap();
        assert_eq!(series.initialized, vec![false, false, true, true]);
        let (name, values) = &series.outputs[0];
        assert_eq!(*name, "value");
        assert!(values[1].is_nan());
        assert_eq!(values[2], 2.0);
        assert_eq!(values[3], 3.0);
    }
}
