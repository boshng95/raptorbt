//! The [`BarIndex`] of each primary stream a process steps, built once.
//!
//! A sweep hands the kernel the same arrays backtest after backtest. Each
//! stepping call looks them up here by identity and borrows the summaries
//! built on the first; a stream the cache does not hold is walked as before.
//!
//! Only read-only arrays are summarized, so the summaries cannot go stale
//! under a writer holding the same objects. The cache keeps weak references
//! only: it never holds a stream alive, and an entry whose arrays died is
//! dropped before a new one is built.

use std::cell::RefCell;
use std::sync::Arc;

use numpy::npyffi::flags::NPY_ARRAY_WRITEABLE;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3::sync::GILProtected;

use crate::portfolio::bar_index::BarIndex;

/// Streams shorter than this are walked: summaries save them nothing.
const MIN_BARS: usize = 4096;
/// Streams held per process. A sweep steps one primary at a time; a few
/// more cover a driver alternating between timeframes' gap-clocked copies.
const CAPACITY: usize = 8;

struct Entry {
    /// Weak references to the timestamp, high, low, close and volume
    /// arrays the index summarizes.
    arrays: [PyObject; 5],
    index: Arc<BarIndex>,
}

static INDEXES: GILProtected<RefCell<Vec<Entry>>> = GILProtected::new(RefCell::new(Vec::new()));

/// The stream's columns, as Python objects and as the slices they hold.
pub(crate) struct Stream<'a> {
    pub objects: [&'a PyAny; 5],
    pub timestamp: &'a [i64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
}

/// The summaries of `stream`, built on its first call; `None` for a short
/// or writable stream.
pub(crate) fn shared(py: Python<'_>, stream: &Stream<'_>) -> PyResult<Option<Arc<BarIndex>>> {
    if stream.timestamp.len() < MIN_BARS || !stream.objects.iter().all(|&array| read_only(array)) {
        return Ok(None);
    }
    let cell = INDEXES.get(py);
    let mut hit = None;
    {
        let entries = cell.borrow();
        for (at, entry) in entries.iter().enumerate() {
            if holds(py, entry, &stream.objects)? {
                hit = Some(at);
                break;
            }
        }
    }
    if let Some(at) = hit {
        let mut entries = cell.borrow_mut();
        let entry = entries.remove(at);
        let index = Arc::clone(&entry.index);
        entries.insert(0, entry);
        return Ok(Some(index));
    }
    let weak = py.import("weakref")?.getattr("ref")?;
    let mut arrays = Vec::with_capacity(5);
    for &array in &stream.objects {
        match weak.call1((array,)) {
            Ok(reference) => arrays.push(reference.into_py(py)),
            // An array that takes no weak reference is walked.
            Err(_) => return Ok(None),
        }
    }
    let arrays: [PyObject; 5] = match arrays.try_into() {
        Ok(arrays) => arrays,
        Err(_) => return Ok(None),
    };
    let index = Arc::new(BarIndex::build(
        stream.timestamp,
        stream.high,
        stream.low,
        stream.close,
        stream.volume,
    ));
    let mut alive = Vec::with_capacity(CAPACITY);
    {
        let entries = std::mem::take(&mut *cell.borrow_mut());
        for entry in entries {
            if alive.len() + 1 < CAPACITY && live(py, &entry)? {
                alive.push(entry);
            }
        }
    }
    alive.insert(0, Entry { arrays, index: Arc::clone(&index) });
    *cell.borrow_mut() = alive;
    Ok(Some(index))
}

fn read_only(array: &PyAny) -> bool {
    match array.downcast::<PyUntypedArray>() {
        // SAFETY: a live numpy array object, read under the GIL.
        Ok(array) => unsafe { (*array.as_array_ptr()).flags & NPY_ARRAY_WRITEABLE == 0 },
        Err(_) => false,
    }
}

/// Whether `entry` summarizes these very arrays.
fn holds(py: Python<'_>, entry: &Entry, objects: &[&PyAny; 5]) -> PyResult<bool> {
    for (reference, &object) in entry.arrays.iter().zip(objects) {
        if !reference.as_ref(py).call0()?.is(object) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether every array `entry` summarizes is still alive.
fn live(py: Python<'_>, entry: &Entry) -> PyResult<bool> {
    for reference in &entry.arrays {
        if reference.as_ref(py).call0()?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}
