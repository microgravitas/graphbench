#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::iter::Map;

pub mod graph;
pub mod iterators;
pub mod parse;
pub mod pygraph;

#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pymodule]
fn graphbench(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pygraph::PyGraph>()?;
    // m.add_wrapped(wrap_pyfunction!(from_pid))?;

    Ok(())
}
