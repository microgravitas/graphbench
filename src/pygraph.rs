use fnv::{FnvHashSet, FnvHashMap};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyIterProtocol;
use pyo3::PyObjectProtocol;
use pyo3::PyClass;
use pyo3::class::iter::IterNextOutput;
use pyo3::exceptions;

use std::collections::HashSet;

use crate::graph::*;
use crate::ordgraph::OrdGraph;
use crate::editgraph::*;

use std::cell::{Cell, RefCell};
use std::iter::IntoIterator;

use crate::iterators::*;

use crate::pyordgraph::*;


fn to_vertex_list(obj:&PyAny) -> PyResult<Vec<u32>>  {
    let vec:Vec<_> = obj.iter()?.map(|i| i.and_then(PyAny::extract::<u32>).unwrap()).collect();
    Ok(vec)
}


/*
    TODO:
    - Lazy iteration. Right now everything has to be cloned :/
*/

/*
    Python-specific methods
*/
#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pyproto]
impl PyObjectProtocol for PyGraph {
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("EditGraph (n={},m={})]", self.G.num_vertices(), self.G.num_edges() ))
    }
}

/*
    Delegation methods
*/
#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pymethods]
impl PyGraph {
    #[new]
    pub fn new() -> PyGraph {
        PyGraph{G: EditGraph::new()}
    }

    pub fn to_ordered(&self) -> PyResult<PyOrdGraph> {
        Ok(PyOrdGraph{G: OrdGraph::with_degeneracy_order(&self.G)})
    }

    pub fn normalize(&mut self) -> FnvHashMap<Vertex, Vertex>{
        let (GG, mapping) = self.G.normalize();
        self.G = GG;
        mapping
    }

    #[staticmethod]
    pub fn from_file(filename:&str) -> PyResult<PyGraph> {
        if &filename[filename.len()-3..] == ".gz" {
            match EditGraph::from_gzipped(filename) {
                Ok(G) => Ok(PyGraph{G}),
                Err(_) => Err(PyErr::new::<exceptions::IOError, _>("IO-Error"))
            }
        } else {
            match EditGraph::from_txt(filename) {
                Ok(G) => Ok(PyGraph{G}),
                Err(_) => Err(PyErr::new::<exceptions::IOError, _>("IO-Error"))
            }
        }
    }

    pub fn degeneracy_ordering(&self) -> PyResult<Vec<Vertex>> {
        Ok(self.G.degeneracy_ordering())
    }

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

    pub fn neighbourhood(&self, vertices:&PyAny) -> PyResult<VertexSet> {
        let vertices = to_vertex_list(vertices)?;
        Ok(self.G.neighbourhood(vertices.iter()))
    }

    pub fn closed_neighbourhood(&self, vertices:&PyAny) -> PyResult<VertexSet> {
        let vertices = to_vertex_list(vertices)?;
        Ok(self.G.closed_neighbourhood(vertices.iter()))
    }

    pub fn r_neighbours(&self, u:Vertex, r:usize) -> PyResult<VertexSet> {
        Ok(self.G.r_neighbours(&u, r))
    }

    pub fn add_vertex(&mut self, u:Vertex) -> PyResult<()> {
        self.G.add_vertex(&u);
        Ok(())
    }

    pub fn add_edge(&mut self, u:Vertex, v:Vertex) -> PyResult<bool> {
        Ok( self.G.add_edge(&u, &v) )
    }

    pub fn remove_edge(&mut self, u:Vertex, v:Vertex) -> PyResult<bool> {
        Ok( self.G.remove_edge(&u, &v) )
    }

    pub fn remove_vertex(&mut self, u:Vertex) -> PyResult<bool> {
        Ok( self.G.remove_vertex(&u) )
    }

    pub fn remove_loops(&mut self) -> PyResult<usize> {
        Ok( self.G.remove_loops() )
    }

    pub fn remove_isolates(&mut self) -> PyResult<usize> {
        Ok( self.G.remove_isolates() )
    }

    /*
        Advanced operations
    */

    pub fn contract(&mut self, vertices:&PyAny) -> PyResult<Vertex> {
        let vertices = to_vertex_list(vertices)?;
        Ok( self.G.contract(vertices.iter()) )
    }

    pub fn contract_into(&mut self, center:Vertex, vertices:&PyAny) -> PyResult<()> {
        let vertices = to_vertex_list(vertices)?;
        self.G.contract_into(&center, vertices.iter());
        Ok(())
    }

    /*
        Subgraphs and components
    */
    pub fn copy(&self) -> PyResult<PyGraph> {
        Ok(PyGraph{G: self.G.clone()})
    }

    pub fn subgraph(&self, vertices:&PyAny) -> PyResult<PyGraph> {
        let vertices = to_vertex_list(vertices)?;
        Ok(PyGraph{G: self.G.subgraph(vertices.iter())})
    }

    pub fn components(&self) -> PyResult<Vec<VertexSet>> {
        Ok(self.G.components())
    }
}

#[cfg(not(test))] // pyclass and pymethods break `cargo test`
#[pyclass(name=EditGraph)]
pub struct PyGraph {
    G: EditGraph
}
