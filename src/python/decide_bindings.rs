//! A traded series' decisions (`crate::lean::decide`), for the lean
//! portfolio lane: a pair's or basket's rules walked once over its series --
//! or a rank's over each member's bars -- with the side each intends.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use super::lean_bindings::{declined, tree, TreeTuple};
use crate::lean::decide::{decide_rank, decide_series, Keys, Member, Rules};

/// Where a decision walk changes side: the bars (or instants) and, per
/// change, the sides it changes to.
type Changes<Sides> = (Py<PyArray1<i64>>, Py<Sides>);

/// One decider's keys: `(values, ready, gated)` -- each key's values and
/// readiness over its bars, the first `gated` its indicators', any after
/// them its bar's prices.
type KeysTuple<'py> = (Vec<PyReadonlyArray1<'py, f64>>, Vec<PyReadonlyArray1<'py, bool>>, usize);

fn keys<'a>(spec: &'a KeysTuple<'_>) -> PyResult<Keys<'a>> {
    let (values, ready, gated) = spec;
    Ok(Keys {
        values: values.iter().map(|v| v.as_slice()).collect::<Result<_, _>>()?,
        ready: ready.iter().map(|r| r.as_slice()).collect::<Result<_, _>>()?,
        gated: *gated,
    })
}

/// The bars at which a series' intended side changes, and the sides it
/// changes to (+1, -1, 0), from flat (`decide_series`). *tree* is the lean
/// lane's (its dust terms unread), *entry_sides* each entry rule's side.
/// Raises `LeanDeclined` where the rules do not fit the keys.
#[pyfunction]
#[pyo3(signature = (tree, entry_sides, keys, *, each_bar, legacy))]
pub fn lean_decide_series(
    py: Python<'_>,
    tree: TreeTuple,
    entry_sides: Vec<i8>,
    keys: KeysTuple<'_>,
    each_bar: bool,
    legacy: bool,
) -> PyResult<Changes<PyArray1<i8>>> {
    let tree = self::tree(&tree)?;
    let keys = self::keys(&keys)?;
    let rules = Rules { tree: &tree, entry_sides: &entry_sides, each_bar, legacy };
    let changes = py.allow_threads(|| decide_series(&rules, &keys)).map_err(declined)?;
    let (bars, sides): (Vec<i64>, Vec<i8>) =
        changes.into_iter().map(|(bar, side)| (i64::from(bar), side)).unzip();
    Ok((bars.into_pyarray(py).into(), sides.into_pyarray(py).into()))
}

/// The instants at which a rank's book changes, and each member's side held
/// there, one row an instant (`decide_rank`). *members* are each member's
/// `(keys, score, scored)` over the bars it is decided on, *delivered* each
/// member's bar at every instant (-1: carried), *by_name* the members in
/// instrument-name order, and *permits* `(long, short)`.
#[pyfunction]
#[pyo3(signature = (tree, entry_sides, members, delivered, by_name, top, permits, *, each_bar, legacy))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn lean_decide_rank(
    py: Python<'_>,
    tree: TreeTuple,
    entry_sides: Vec<i8>,
    members: Vec<(KeysTuple<'_>, PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, bool>)>,
    delivered: Vec<PyReadonlyArray1<'_, i64>>,
    by_name: Vec<usize>,
    top: usize,
    permits: (bool, bool),
    each_bar: bool,
    legacy: bool,
) -> PyResult<Changes<PyArray2<i8>>> {
    let tree = self::tree(&tree)?;
    let rules = Rules { tree: &tree, entry_sides: &entry_sides, each_bar, legacy };
    let members = members
        .iter()
        .map(|(spec, score, scored)| {
            Ok(Member { keys: keys(spec)?, score: score.as_slice()?, scored: scored.as_slice()? })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let delivered = delivered.iter().map(|d| d.as_slice()).collect::<Result<Vec<_>, _>>()?;
    let n = members.len();
    let changes = py
        .allow_threads(|| decide_rank(&rules, &members, &delivered, &by_name, top, permits))
        .map_err(declined)?;
    let instants: Vec<i64> = changes.iter().map(|(at, _)| i64::from(*at)).collect();
    let held: Vec<i8> = changes.iter().flat_map(|(_, held)| held.iter().copied()).collect();
    let rows = instants.len();
    let held = Array2::from_shape_vec((rows, n), held)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((instants.into_pyarray(py).into(), held.into_pyarray(py).into()))
}
