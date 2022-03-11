use fnv::{FnvHashSet, FnvHashMap};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyIterProtocol;
use pyo3::PyObjectProtocol;
use pyo3::class::iter::IterNextOutput;
use pyo3::exceptions;

use std::collections::HashSet;

use crate::ordgraph::*;
use crate::graph::*;

use std::cell::{Cell, RefCell};

use crate::iterators::*;

/*
    Python-specific methods
*/
#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pyproto]
impl PyObjectProtocol for PyOrdGraph {
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("OrdGraph (n={},m={})]", self.G.num_vertices(), self.G.num_edges() ))
    }
}

/*
    Delegation methods
*/
#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pymethods]
impl PyOrdGraph {
    pub fn num_vertices(&self) -> PyResult<usize> {
        Ok(self.G.num_vertices())
    }

    pub fn num_edges(&self) -> PyResult<usize> {
        Ok(self.G.num_edges())
    }

    pub fn adjacent(&self, u:Vertex, v:Vertex) -> PyResult<bool> {
        Ok(self.G.adjacent(&u, &v))
    }

    pub fn degree(&self, u:Vertex) -> PyResult<u32> {
        Ok(self.G.degree(&u))
    }

    pub fn contains(&mut self, u:Vertex) -> PyResult<bool> {
        Ok(self.G.contains(&u))
    }

    pub fn vertices(&self) -> PyResult<VertexSet> {
        Ok(self.G.vertices().cloned().collect())
    }

    pub fn edges(&self) -> PyResult<Vec<Edge>> {
        Ok(self.G.edges().collect())
    }


    /*
        Neighbourhood methods
    */
    pub fn neighbours(&self, u:Vertex) -> PyResult<VertexSet> {
        Ok(self.G.neighbours(&u).cloned().collect())
    }

    pub fn neighbourhood(&self, c:HashSet<u32>) -> PyResult<VertexSet> {
        Ok(self.G.neighbourhood(c.iter()))
    }

    pub fn closed_neighbourhood(&self, c:HashSet<u32>) -> PyResult<VertexSet> {
        Ok(self.G.closed_neighbourhood(c.iter()))
    }

    pub fn r_neighbours(&self, u:Vertex, r:usize) -> PyResult<VertexSet> {
        Ok(self.G.r_neighbours(&u, r))
    }
}

#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pyclass(name="OrdGraph")]
pub struct PyOrdGraph {
    pub(crate) G: OrdGraph
}
