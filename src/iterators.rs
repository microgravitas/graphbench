use fxhash::{FxHashMap, FxHashSet};

use crate::graph::*;
use std::hash::Hash;

use crate::editgraph::EditGraph;

use crate::dtfgraph::DTFGraph;
use crate::dtfgraph::DTFVertexIterator;
use crate::dtfgraph::InArcIterator;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FxHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

/// Neighbourhood iterators for graphs and digraphs. At each step, the iterator
/// returns a pair `(v,N(v))` (or `(v,N^-(v))` or `(v,N^+(V))` for digraphs).
pub struct NeighIterator<'a, G> where G: Graph  {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>
}

impl<'a, G> NeighIterator<'a, G> where G: Graph  {
    pub fn new(graph: &'a G) -> NeighIterator<'a, G> {
        NeighIterator { graph, v_it: graph.vertices() }
    }
}

impl<'a, G> Iterator for NeighIterator<'a, G> where G: Graph {
    type Item = (Vertex, Box<dyn Iterator<Item=&'a Vertex> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v_it.next()?.clone();
        let N = self.graph.neighbours(&v);

        Some((v, N))
    }
}

// Note: It would be nice if we could just re-use NeighItertor here, but stable Rust
// currently does not support overlapping `impl` blocks.
pub struct DiNeighIterator<'a, G> where G: Graph {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>,
    mode: DiNeighIteratorMode
}

enum DiNeighIteratorMode {IN, OUT}

impl<'a, D: Digraph> DiNeighIterator<'a, D> {
    pub fn new_in(graph: &'a D) -> DiNeighIterator<'a, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::IN, v_it: graph.vertices() }
    }

    pub fn new_out(graph: &'a D) -> DiNeighIterator<'a, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::OUT, v_it: graph.vertices() }
    }
}

impl<'a, D> Iterator for DiNeighIterator<'a, D> where D: Digraph {
    type Item = (Vertex, Box<dyn Iterator<Item=&'a Vertex> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v_it.next()?.clone();
        match &self.mode {
            DiNeighIteratorMode::IN =>  {let N = self.graph.in_neighbours(&v);
                    Some((v, N))},
            DiNeighIteratorMode::OUT => {let N = self.graph.out_neighbours(&v);
                    Some((v, N))}
        }
    }
}

pub trait NeighIterable<G> where G: Graph {
    fn neighbourhoods(&self) -> NeighIterator<G>;
}

impl<G> NeighIterable<G> for G where G: Graph {
    fn neighbourhoods(&self) -> NeighIterator<G> {
        NeighIterator::<G>::new(self)
    }
}

pub trait DiNeighIterable<D> where D: Digraph {
    fn in_neighbourhoods(&self) -> DiNeighIterator<D>;
    fn out_neighbourhoods(&self) -> DiNeighIterator<D>;
}

impl<D> DiNeighIterable<D> for D where D: Digraph {
    fn in_neighbourhoods(&self) -> DiNeighIterator<D> {
        DiNeighIterator::<D>::new_in(self)
    }

    fn out_neighbourhoods(&self) -> DiNeighIterator<D> {
        DiNeighIterator::<D>::new_out(self)
    }
}

/// Edge iterator for graphs. This iterator uses an `NeighIterator`
/// internally, so in order to break ties between edge `{u,v}` and `{v,u}`
/// we need the vertex type `V` to implement `std::cmp::Ord`.
/// The associated trait EdgeIterable is implemented for generic graphs
///  to provide the method `edges(...)` to create an `EdgeIterator`.
pub struct EdgeIterator<'a, G> where G: Graph {
    N_it: NeighIterator<'a, G>,
    curr_v: Option<Vertex>,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a, G> EdgeIterator<'a, G> where G: Graph {
    pub fn new(graph: &'a G) -> EdgeIterator<'a, G> {
        let mut res = EdgeIterator {
            N_it: graph.neighbourhoods(),
            curr_v: None,
            curr_it: None,
        };
        res.advance();
        res
    }

    fn advance(&mut self) {
        if let Some((v, it)) = self.N_it.next() {
            self.curr_v = Some(v);
            self.curr_it = Some(it);
        } else {
            self.curr_it = None;
        }
    }
}

impl<'a, G> Iterator for EdgeIterator<'a, G> where G: Graph {
    type Item = (Vertex, Vertex);

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_it.is_some() {
            let uu = {
                let uu = self.curr_it.as_mut().unwrap().next();
                if uu.is_none() {
                    self.advance();
                    continue;
                }
                uu
            };

            // Tie-breaking so we only return every edge once
            let u = uu.unwrap().clone();
            if self.curr_v.as_ref().unwrap() > &u {
                continue;
            }
            return Some((self.curr_v.as_ref().unwrap().clone(), u));
        }

        None
    }
}

pub trait EdgeIterable<G> where G: Graph {
    fn edges(&self) -> EdgeIterator<G>;
}

impl<G> EdgeIterable<G> for G where G: Graph {
    fn edges(&self) -> EdgeIterator<G> {
        EdgeIterator::<G>::new(self)
    }
}


/// Similar to `EdgeIterator`, but for directed graphs.
pub struct ArcIterator<'a, D> where D: Digraph {
    N_it: DiNeighIterator<'a, D>,
    curr_v: Option<Vertex>,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a, D> ArcIterator<'a, D> where D: Digraph{
    pub fn new(graph: &'a D) -> ArcIterator<'a, D> {
        let mut res = ArcIterator {
            N_it: graph.in_neighbourhoods(),
            curr_v: None,
            curr_it: None,
        };
        res.advance();
        res
    }

    fn advance(&mut self) {
        if let Some((v, it)) = self.N_it.next() {
            self.curr_v = Some(v);
            self.curr_it = Some(it);
        } else {
            self.curr_it = None;
        }
    }
}

impl<'a, D> Iterator for ArcIterator<'a, D> where D: Digraph  {
    type Item = (Vertex, Vertex);

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_it.is_some() {
            let uu = {
                let uu = self.curr_it.as_mut().unwrap().next();
                if uu.is_none() {
                    self.advance();
                    continue;
                }
                uu
            };

            // Tie-breaking so we only return every edge once
            let u = uu.unwrap().clone();
            return Some((self.curr_v.as_ref().unwrap().clone(), u));
        }

        None
    }
}

pub trait ArcIterable<D> where D: Digraph {
    fn arcs(&self) -> ArcIterator<D>;
}

impl<D> ArcIterable<D> for D where D: Digraph {
    fn arcs(&self) -> ArcIterator<D> {
        ArcIterator::<D>::new(self)
    }
}

/*
    Neighbourhood iterator for dtf graphs. At each step,
    the iterator returns a pair (v,N(v)).
*/
pub struct DTFNIterator<'a> {
    G: &'a DTFGraph,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>,
    depth: Option<usize>,
}

impl<'a> DTFNIterator<'a> {
    pub fn all_depths(G: &'a DTFGraph) -> DTFNIterator<'a> {
        DTFNIterator {
            G,
            v_it: G.vertices(),
            depth: None,
        }
    }

    pub fn fixed_depth(G: &'a DTFGraph, depth: usize) -> DTFNIterator<'a> {
        DTFNIterator {
            G,
            v_it: G.vertices(),
            depth: Some(depth),
        }
    }
}

impl<'a> Iterator for DTFNIterator<'a> {
    type Item = (Vertex, Box<dyn Iterator<Item=&'a Vertex> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(depth) = self.depth {
            let v = *self.v_it.next()?;
            let N = self.G.in_neighbours_at(&v, depth);

            Some((v, N))
        } else {
            let v = *self.v_it.next()?;
            let N = self.G.in_neighbours(&v);

            Some((v, N))
        }
    }
}

/*
    Arc iterator for DTF graphs.
*/
pub struct DTFArcIterator<'a> {
    N_it: DTFNIterator<'a>,
    curr_v: Vertex,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a> DTFArcIterator<'a> {
    pub fn all_depths(G: &'a DTFGraph) -> DTFArcIterator {
        let mut res = DTFArcIterator {
            N_it: G.in_neighbourhoods_iter(),
            curr_v: std::u32::MAX,
            curr_it: None,
        };
        res.advance();
        res
    }

    pub fn fixed_depth(G: &'a DTFGraph, depth:usize) -> DTFArcIterator {
        let mut res = DTFArcIterator {
            N_it: G.in_neighbourhoods_iter_at(depth),
            curr_v: std::u32::MAX,
            curr_it: None,
        };
        res.advance();
        res
    }

    fn advance(&mut self) {
        if let Some((v, it)) = self.N_it.next() {
            self.curr_v = v;
            self.curr_it = Some(it);
        } else {
            self.curr_it = None;
        }
    }
}

impl<'a> Iterator for DTFArcIterator<'a> {
    type Item = Arc;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_it.is_some() {
            let uu = self.curr_it.as_mut().unwrap().next();
            if uu.is_none() {
                self.advance();
                continue;
            }

            // Tie-breaking so we only return every edge once
            let u = *uu.unwrap();
            if self.curr_v > u {
                continue;
            }
            return Some((self.curr_v, u));
        }

        None
    }
}
