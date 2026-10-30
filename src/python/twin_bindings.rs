//! The engine's indicator walks, run by their exact twins (`crate::twins`).
//!
//! `indicator_twin` takes an indicator class's twin kind, its constructor
//! arguments and the admitted bars' columns, and answers the twin's readings
//! after each bar, or why it declined. The engine checks each twin against
//! the original on probe bars before it uses one.

use std::collections::BTreeMap;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};

use crate::twins::{compute, Columns, Param, Params};

/// A Python argument as a twin takes it: exactly an `int`, a `float`, a
/// `str`, or a list or tuple of `float`s. Anything else -- a `bool`, a
/// numpy scalar, `None` -- has no twin, since the original's arithmetic on
/// it may differ.
fn param(value: &PyAny) -> Option<Param> {
    if value.is_instance_of::<PyBool>() {
        return None;
    }
    if value.is_exact_instance_of::<PyLong>() {
        return value.extract::<i64>().ok().map(Param::Int);
    }
    if value.is_exact_instance_of::<PyFloat>() {
        return value.extract::<f64>().ok().map(Param::Float);
    }
    if value.is_exact_instance_of::<PyString>() {
        return value.extract::<String>().ok().map(Param::Text);
    }
    if value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>() {
        let mut floats = Vec::new();
        for item in value.iter().ok()? {
            let item = item.ok()?;
            if !item.is_exact_instance_of::<PyFloat>() {
                return None;
            }
            floats.push(item.extract::<f64>().ok()?);
        }
        return Some(Param::Floats(floats));
    }
    None
}

/// The twin of indicator *kind* built from *params*, over the bars' columns.
///
/// Returns `(initialized, {output: values})` -- the original's
/// `initialized` after each bar, and each output's value after each bar
/// where it is set (NaN elsewhere) -- or, where the twin declines, why.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn indicator_twin(
    py: Python<'_>,
    kind: &str,
    params: &PyDict,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let mut values = BTreeMap::new();
    for (name, value) in params.iter() {
        let name: String = name.extract()?;
        match param(value) {
            Some(value) => {
                values.insert(name, value);
            }
            None => return Ok(format!("{name} is not an argument a twin takes").into_py(py)),
        }
    }
    let (Ok(open), Ok(high), Ok(low), Ok(close), Ok(volume)) =
        (open.as_slice(), high.as_slice(), low.as_slice(), close.as_slice(), volume.as_slice())
    else {
        return Ok("the columns are not contiguous".into_py(py));
    };
    let params = Params::new(values);
    let columns = Columns { open, high, low, close, volume };
    let series = py.allow_threads(move || compute(kind, &params, &columns));
    match series {
        Err(declined) => Ok(declined.0.into_py(py)),
        Ok(series) => {
            let outputs = PyDict::new(py);
            for (name, values) in series.outputs {
                outputs.set_item(name, values.into_pyarray(py))?;
            }
            Ok((series.initialized.into_pyarray(py), outputs).into_py(py))
        }
    }
}
