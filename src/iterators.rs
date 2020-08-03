use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::Graph;
use crate::graph::Vertex;

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
        EdgeIterator{N_it: G.neighbours_iter(), curr_v: 0, curr_it: None}
    }

    fn advance(&mut self) {
        let res = self.N_it.next();
        if let Some((v,it)) = res {
            self.curr_v = v;
            self.curr_it = Some(it);
        } else {
            self.curr_it = None;
        }
    }
}

impl<'a> Iterator for EdgeIterator<'a> {
    type  Item = (Vertex, Vertex);

    fn next(&mut self) -> Option<Self::Item> {
        self.advance();

        while self.curr_it.is_some() {
            let u = self.curr_it.as_mut().unwrap().next();
            if u.is_none() {
                self.advance();
                continue;
            }
            return Some((self.curr_v, *u.unwrap()))
        }

        return None
    }
}
