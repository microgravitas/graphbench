use std::collections::HashSet;
use union_find_rs::prelude::*;

use crate::graph::*;
use crate::iterators::*;

pub trait GraphAlgorithms {
    fn components(&self) -> Vec<VertexSet>;
}

impl<G> GraphAlgorithms for G where G: Graph {
    fn components(&self) -> Vec<VertexSet> {
        let mut dsets:DisjointSets<u32> = DisjointSets::new();

        for v in self.vertices() {
            dsets.make_set(*v);
        }

        for (u,v) in self.edges() {
            dsets.union(&u, &v);
        }

        // Convertex HashSet<Vertex> into VertexSet
        let mut res = Vec::new();
        for comp in dsets {
            res.push(comp.iter().cloned().collect())
        }
        res
    }
}


