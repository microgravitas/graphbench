use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::Graph;
use crate::graph::Vertex;
use crate::graph::Edge;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FnvHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

pub struct NIterator<'a> {
    G: &'a Graph,
    v_it: VertexIterator<'a>,
}

impl<'a> NIterator<'a> {
    pub fn new(G: &'a Graph) -> NIterator<'a> {
        NIterator{G, v_it: G.vertices()}
    }
}

impl<'a> Iterator for NIterator<'a> {
    type Item = (Vertex, NVertexIterator<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        let N = self.G.neighbours(v);

        Some((v,N))
    }
}

pub struct EdgeIterator<'a> {
    N_it: NIterator<'a>,
    curr_v: Vertex,
    curr_it: Option<NVertexIterator<'a>>
}

impl<'a> EdgeIterator<'a> {
    pub fn new(G: &'a Graph) -> EdgeIterator {
        let mut res = EdgeIterator{N_it: G.neighbours_iter(), curr_v: 99999, curr_it: None};
        res.advance();
        res
    }

    fn advance(&mut self) {
        if let Some((v,it)) = self.N_it.next() {
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
                continue
            }
            return Some((self.curr_v, u))
        }

        return None
    }
}
