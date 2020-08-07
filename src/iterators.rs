use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::{Vertex, Edge, Arc};
use crate::graph::Graph;
use crate::graph::VertexSet;

use crate::dtfgraph::DTFGraph;
use crate::dtfgraph::DTFVertexIterator;
use crate::dtfgraph::InArcIterator;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FnvHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

/*
    Neighbourhood iterator for normal graphs. At each step,
    the iterator returns a pair (v,N(v)).
*/
pub struct NIterator<'a> {
    G: &'a Graph,
    v_it: VertexIterator<'a>,
}

impl<'a> NIterator<'a> {
    pub fn new(G: &'a Graph) -> NIterator<'a> {
        NIterator {
            G,
            v_it: G.vertices(),
        }
    }
}

impl<'a> Iterator for NIterator<'a> {
    type Item = (Vertex, NVertexIterator<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        let N = self.G.neighbours(v);

        Some((v, N))
    }
}

/*
    Edge iterator for normal graphs.
*/
pub struct EdgeIterator<'a> {
    N_it: NIterator<'a>,
    curr_v: Vertex,
    curr_it: Option<NVertexIterator<'a>>,
}

impl<'a> EdgeIterator<'a> {
    pub fn new(G: &'a Graph) -> EdgeIterator {
        let mut res = EdgeIterator {
            N_it: G.neighbours_iter(),
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

impl<'a> Iterator for EdgeIterator<'a> {
    type Item = Edge;

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
