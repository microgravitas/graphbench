use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::Graph;

use crate::editgraph::{Vertex, Edge, Arc};
use crate::editgraph::EditGraph;
use crate::editgraph::VertexSet;

use crate::dtfgraph::DTFGraph;
use crate::dtfgraph::DTFVertexIterator;
use crate::dtfgraph::InArcIterator;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FnvHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

/*
    Neighbourhood iterator for normal graphs. At each step,
    the iterator returns a pair (v,N(v)).
*/
pub struct NIterator<'a, V, G: Graph<V>> where V: Clone {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a V> + 'a>
}

impl<'a, V, G:Graph<V>> NIterator<'a, V, G> where V: Clone {
    pub fn new(graph: &'a G) -> NIterator<'a, V, G> {
        NIterator {
            graph,
            v_it: graph.vertices(),
        }
    }
}

impl<'a, V: Clone, G:Graph<V>> Iterator for NIterator<'a, V, G> where V: Clone {
    type Item = (V, Box<dyn Iterator<Item=&'a V> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v_it.next()?.clone();
        let N = self.graph.neighbours(&v);

        Some((v, N))
    }
}

trait NIterable<V, G: Graph<V>> where V: Clone {
    fn neighbourhoods(&self) -> NIterator<V,G>;
}

impl<V,G:Graph<V>> NIterable<V,G> for G where V: Clone {
    fn neighbourhoods(&self) -> NIterator<V,G> {
        NIterator::<V,G>::new(self)
    }
}

/*
    Edge iterator for normal graphs.
*/
pub struct EdgeIterator<'a, V, G: Graph<V>> where V: Ord + Clone {
    N_it: NIterator<'a, V, G>,
    curr_v: Option<V>,
    curr_it: Option<Box<dyn Iterator<Item=&'a V> + 'a>>,
}

impl<'a, V, G: Graph<V>> EdgeIterator<'a, V, G> where V: Ord, V: Clone {
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

trait EdgeIterable<V: Ord, G: Graph<V>> where V: Clone {
    fn edges(&self) -> EdgeIterator<V,G>;
}

impl<V, G> EdgeIterable<V,G> for G where V: Ord + Clone, G: Graph<V> {
    fn edges(&self) -> EdgeIterator<V,G> {
        EdgeIterator::<V,G>::new(self)
    }
}

/*
    Neighbourhood iterator for normal graphs. At each step,
    the iterator returns a pair (v,N(v)).
*/
pub struct DTFNIterator<'a> {
    G: &'a DTFGraph,
    v_it: DTFVertexIterator<'a>,
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
    type Item = (Vertex, InArcIterator<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        let N = self.G.in_neighbours_at(v, self.depth);

        Some((v, N))
    }
}

/*
    Arc iterator for DTF graphs.
*/
pub struct DTFArcIterator<'a> {
    N_it: DTFNIterator<'a>,
    curr_v: Vertex,
    curr_it: Option<InArcIterator<'a>>,
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
