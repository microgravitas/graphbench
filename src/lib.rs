#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::iter::Map;

pub mod graph;
pub mod editgraph;
pub mod dtfgraph;
pub mod ordgraph;

pub mod iterators;
pub mod parse;

pub mod pygraph;
pub mod pyordgraph;

#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pymodule]
fn graphbench(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pygraph::PyGraph>()?;
    m.add_class::<pygraph::PyVertexMapDegree>()?;
    m.add_class::<pygraph::PyVertexMapBool>()?;
    m.add_class::<pyordgraph::PyOrdGraph>()?;
    // m.add_wrapped(wrap_pyfunction!(from_pid))?;

    Ok(())
}
