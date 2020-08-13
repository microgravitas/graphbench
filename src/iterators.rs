use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::*;

use crate::editgraph::{Vertex, Edge, Arc};
use crate::editgraph::EditGraph;
use crate::editgraph::VertexSet;

use crate::dtfgraph::DTFGraph;
use crate::dtfgraph::DTFVertexIterator;
use crate::dtfgraph::InArcIterator;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FnvHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

/// Neighbourhood iterators for graphs and digraphs. At each step, the iterator
/// returns a pair `(v,N(v))` (or `(v,N^-(v))` or `(v,N^+(V))` for digraphs).
pub struct NeighIterator<'a, V, G: Graph<V>> where V: Clone {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a V> + 'a>
}

impl<'a, V, G> NeighIterator<'a, V, G> where V: Clone, G: Graph<V> {
    pub fn new(graph: &'a G) -> NeighIterator<'a, V, G> {
        NeighIterator { graph, v_it: graph.vertices() }
    }
}

impl<'a, V, G> Iterator for NeighIterator<'a, V, G> where V: Clone, G: Graph<V> {
    type Item = (V, Box<dyn Iterator<Item=&'a V> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v_it.next()?.clone();
        let N = self.graph.neighbours(&v);

        Some((v, N))
    }
}

// Note: It would be nice if we could just re-use NeighItertor here, but stable Rust
// currently does not support overlapping `impl` blocks.
pub struct DiNeighIterator<'a, V, G: Graph<V>> where V: Clone {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a V> + 'a>,
    mode: DiNeighIteratorMode
}

enum DiNeighIteratorMode {IN, OUT}

impl<'a, V, D> DiNeighIterator<'a, V, D> where V: Clone, D: Digraph<V> {
    pub fn new_in(graph: &'a D) -> DiNeighIterator<'a, V, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::IN, v_it: graph.vertices() }
    }

    pub fn new_out(graph: &'a D) -> DiNeighIterator<'a, V, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::OUT, v_it: graph.vertices() }
    }
}

impl<'a, V, D> Iterator for DiNeighIterator<'a, V, D> where V: Clone, D: Digraph<V> {
    type Item = (V, Box<dyn Iterator<Item=&'a V> + 'a>);

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

pub trait NeighIterable<V, G> where V: Clone, G: Graph<V> {
    fn neighbourhoods(&self) -> NeighIterator<V,G>;
}

impl<V, G> NeighIterable<V,G> for G where V: Clone, G: Graph<V> {
    fn neighbourhoods(&self) -> NeighIterator<V, G> {
        NeighIterator::<V, G>::new(self)
    }
}

pub trait DiNeighIterable<V, D> where V: Clone, D: Digraph<V> {
    fn in_neighbourhoods(&self) -> DiNeighIterator<V, D>;
    fn out_neighbourhoods(&self) -> DiNeighIterator<V, D>;
}

impl<V, D> DiNeighIterable<V, D> for D where V: Clone, D: Digraph<V> {
    fn in_neighbourhoods(&self) -> DiNeighIterator<V, D> {
        DiNeighIterator::<V, D>::new_in(self)
    }

    fn out_neighbourhoods(&self) -> DiNeighIterator<V, D> {
        DiNeighIterator::<V, D>::new_out(self)
    }
}

/// Edge iterator for graphs. This iterator uses an `NeighIterator`
/// internally, so in order to break ties between edge `{u,v}` and `{v,u}`
/// we need the vertex type `V` to implement `std::cmp::Ord`.
/// The associated trait EdgeIterable is implemented for generic graphs
///  to provide the method `edges(...)` to create an `EdgeIterator`.
pub struct EdgeIterator<'a, V, G> where V: Ord + Clone, G: Graph<V> {
    N_it: NeighIterator<'a, V, G>,
    curr_v: Option<V>,
    curr_it: Option<Box<dyn Iterator<Item=&'a V> + 'a>>,
}

impl<'a, V, G> EdgeIterator<'a, V, G> where V: Ord + Clone, G: Graph<V> {
    pub fn new(graph: &'a G) -> EdgeIterator<'a, V, G> {
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

impl<'a, V, G> Iterator for EdgeIterator<'a, V, G> where V: Ord + Clone, G: Graph<V> {
    type Item = (V, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_it.is_some() {
            let uu = {
                let uu = self.curr_it.as_mut().unwrap().next();
                if uu.is_none() {
                    self.advance();
                    continue;
                }
                uu.clone()
            };

            // Tie-breaking so we only return every edge once
            let u = uu.unwrap().clone();
            if self.curr_v.as_ref().unwrap() > &u {
                continue;
            }
            return Some((self.curr_v.as_ref().unwrap().clone(), u.clone()));
        }

        return None;
    }
}

pub trait EdgeIterable<V: Ord, G: Graph<V>> where V: Clone {
    fn edges(&self) -> EdgeIterator<V,G>;
}

impl<V, G> EdgeIterable<V,G> for G where V: Ord + Clone, G: Graph<V> {
    fn edges(&self) -> EdgeIterator<V,G> {
        EdgeIterator::<V,G>::new(self)
    }
}


/// Similar to `EdgeIterator`, but for directed graphs.
pub struct ArcIterator<'a, V, D> where V: Ord + Clone, D: Digraph<V> {
    N_it: DiNeighIterator<'a, V, D>,
    curr_v: Option<V>,
    curr_it: Option<Box<dyn Iterator<Item=&'a V> + 'a>>,
}

impl<'a, V, D> ArcIterator<'a, V, D> where V: Ord + Clone, D: Digraph<V> {
    pub fn new(graph: &'a D) -> ArcIterator<'a, V, D> {
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

impl<'a, V, D> Iterator for ArcIterator<'a, V, D> where V: Ord + Clone, D: Digraph<V> {
    type Item = (V, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_it.is_some() {
            let uu = {
                let uu = self.curr_it.as_mut().unwrap().next();
                if uu.is_none() {
                    self.advance();
                    continue;
                }
                uu.clone()
            };

            // Tie-breaking so we only return every edge once
            let u = uu.unwrap().clone();
            return Some((self.curr_v.as_ref().unwrap().clone(), u));
        }

        return None;
    }
}

pub trait ArcIterable<V, D> where V: Clone + Ord, D: Digraph<V> {
    fn arcs(&self) -> ArcIterator<V, D>;
}

impl<V, D> ArcIterable<V, D> for D where V: Ord + Clone, D: Digraph<V> {
    fn arcs(&self) -> ArcIterator<V, D> {
        ArcIterator::<V, D>::new(self)
    }
}

/*
    Neighbourhood iterator for dtf graphs. At each step,
    the iterator returns a pair (v,N(v)).
*/
pub struct DTFNIterator<'a> {
    G: &'a DTFGraph,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>,
    depth: usize,
}

impl<'a> DTFNIterator<'a> {
    pub fn new(G: &'a DTFGraph, depth: usize) -> DTFNIterator<'a> {
        DTFNIterator {
            G,
            v_it: G.vertices(),
            depth,
        }
    }
}

impl<'a> Iterator for DTFNIterator<'a> {
    type Item = (Vertex, Box<dyn Iterator<Item=&'a Vertex> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        let N = self.G.in_neighbours_at(&v, self.depth);

        Some((v, N))
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
    pub fn new(G: &'a DTFGraph, depth:usize) -> DTFArcIterator {
        let mut res = DTFArcIterator {
            N_it: G.in_neighbourhoods_iter(depth),
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

        return None;
    }
}
