use std::borrow::Borrow;
use std::iter::Sum;
use itertools::max;
use fxhash::{FxHashMap, FxHashSet};

use crate::iterators::*;
use crate::graph::*;

/// An implementation of the [MutableDigraph] trait with additional convenient editing and generating functions.
#[derive(Debug)]
pub struct EditDigraph {
    in_adj: FxHashMap<Vertex, VertexSet>,
    out_adj: FxHashMap<Vertex, VertexSet>,
    out_degs: FxHashMap<Vertex, u32>,
    in_degs: FxHashMap<Vertex, u32>,
    m: usize
}

impl PartialEq for EditDigraph {
    fn eq(&self, other: &Self) -> bool {
        if self.num_vertices() != other.num_vertices() {
            return false
        }
        if self.num_edges() != other.num_edges() {
            return false
        }
        if self.out_degs != other.out_degs {
            return false
        }
        self.out_adj == other.out_adj
    }
}
impl Eq for EditDigraph {}

impl Clone for EditDigraph {
    fn clone(&self) -> EditDigraph {
        let mut G = EditDigraph::new();
        for v in self.vertices() {
            G.add_vertex(v);
            for u in self.out_neighbours(v) {
                G.add_arc(v, u);
            }
        }

        G
    }
}

impl Graph for EditDigraph {
    /*
        Basic properties and queries
    */
    fn num_vertices(&self) -> usize {
        self.out_adj.len()
    }

    fn num_edges(&self) -> usize {
        self.m
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        self.has_arc(u, v) || self.has_arc(u, v)
    }

    fn degree(&self, u:&Vertex) -> u32 {
        *self.in_degs.get(u).unwrap_or(&0) + *self.out_degs.get(u).unwrap_or(&0)
    }

    /*
        Iteration and access
    */
    fn contains(&self, u:&Vertex) -> bool {
        self.out_adj.contains_key(u)
    }

    // fn vertices(&self) -> Box<dyn Iterator<Item=&Vertex>>;
    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        Box::new(self.out_adj.keys())
    }

    // fn neighbours(&self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex>>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        if !self.contains(u) {
            panic!("Vertex not contained in EditDigraph")
        }

        let N_in = &self.in_adj[u];
        let N_out = &self.out_adj[u];

        Box::new(N_in.iter().chain(N_out.iter()))
    }
}

impl Digraph for EditDigraph {
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool {
        match self.out_adj.get(u) {
            Some(N) => N.contains(v),
            _ => false
        }
    }

    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        match self.out_adj.get(u) {
            Some(N) => Box::new(N.iter()),
            None => panic!("Vertex not contained in EditDigraph")
        }
    }

    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        match self.in_adj.get(u) {
            Some(N) => Box::new(N.iter()),
            None => panic!("Vertex not contained in EditDigraph")
        }
    }

    /// Returns the number of arcs which point to `u` in the digraph.
    fn in_degree(&self, u:&Vertex) -> u32 {
        *self.in_degs.get(u).expect("Vertex not contained in EditDigraph")
    }

    /// Returns the number of arcs which point away from `u` in the digraph.
    fn out_degree(&self, u:&Vertex) -> u32 {
        *self.out_degs.get(u).expect("Vertex not contained in EditDigraph")
    }
}

impl FromIterator<Edge> for EditDigraph {
    fn from_iter<T: IntoIterator<Item = Edge>>(iter: T) -> Self {
        let mut res = EditDigraph::new();
        for (u,v) in iter {
            res.add_arc(&u, &v);
        }
        res
    }
}

impl MutableDigraph for EditDigraph {
    fn new() -> EditDigraph {
        EditDigraph{
              in_adj: FxHashMap::default(),
              out_adj: FxHashMap::default(),
              in_degs: FxHashMap::default(),
              out_degs: FxHashMap::default(),
              m: 0}
    }

    fn with_capacity(n_guess:usize) -> Self {
        EditDigraph {
            in_adj: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            out_adj: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            in_degs: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            out_degs: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            m :0
        }
    }

    fn add_vertex(&mut self, u:&Vertex) -> bool {
        if !self.out_adj.contains_key(u) {
            self.in_adj.insert(*u, FxHashSet::default());
            self.out_adj.insert(*u, FxHashSet::default());
            self.in_degs.insert(*u, 0);
            self.out_degs.insert(*u, 0);
            true
        } else {
            false
        }
    }

    fn add_arc(&mut self, u:&Vertex, v:&Vertex) -> bool {
        self.add_vertex(u);
        self.add_vertex(v);

        if !self.has_arc(u, v) {
            self.out_adj.get_mut(u).unwrap().insert(*v);
            self.in_adj.get_mut(v).unwrap().insert(*u);
            self.out_degs.insert(*u, self.out_degs[u] + 1);
            self.in_degs.insert(*v, self.in_degs[v] + 1);
            self.m += 1;
            true
        } else {
            false
        }
    }

    fn remove_arc(&mut self, u:&Vertex, v:&Vertex) -> bool {
        if self.has_arc(u, v) {
            self.out_adj.get_mut(u).unwrap().remove(v);
            self.in_adj.get_mut(v).unwrap().remove(u);
            self.out_degs.insert(*u, self.out_degs[u] - 1);
            self.in_degs.insert(*v, self.in_degs[v] - 1);
            self.m -= 1;
            true
        } else {
            false
        }
    }

    fn remove_vertex(&mut self, u:&Vertex) -> bool {
        if !self.contains(u) {
            false
        } else {
            let N = self.out_adj.get(u).unwrap().clone();
            for v in &N {
                self.remove_arc(u, v);
            }
            let N = self.in_adj.get(u).unwrap().clone();
            for v in &N {
                self.remove_arc(v, u);
            }

            self.out_adj.remove(u);
            self.in_adj.remove(u);
            self.out_degs.remove(u);
            self.in_degs.remove(u);

            true
        }
    }

}

impl EditDigraph {
    /// Generates a directed path on `n` vertices.
    pub fn path(n:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(n as usize);
        for u in 0..(n-1) {
            let v = u+1;
            res.add_arc(&u,&v);
        }

        res
    }

    /// Generates a directed cycle on `n` vertices.
    pub fn cycle(n:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(n as usize);
        for u in 0..n {
            let v = (u+1) % n;
            res.add_arc(&u,&v);
        }

        res
    }

    /// Generates a directed matching on `2n` vertices.
    pub fn matching(n:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(n as usize);
        for u in 0..n {
            let v = u+n;
            res.add_arc(&u,&v);
        }

        res
    }

    /// Generates a star with `n` leaves, so `n+1` vertices total.
    pub fn star(n:u32) -> EditDigraph {
        EditDigraph::biclique(1, n)
    }

    /// Generates a directed complete graph (clique). The edges are directed according
    /// to a linear ordering.
    pub fn clique(n:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(n as usize);
        for u in 0..n {
            for v in (u+1)..n {
                res.add_arc(&u,&v);
            }
        }

        res
    }

    /// Generates an empty directed graph (independent set) on `n` vertices.
    pub fn independent(n:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(n as usize);
        for u in 0..n {
            res.add_vertex(&u);
        }

        res
    }

    /// Generates a complete bipartite graph (biclique) on `s`+`t` vertices.
    /// Arcs go from the s-side to the t-side.
    pub fn biclique(s:u32, t:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity((s+t) as usize);
        for u in 0..s {
            for v in s..(s+t) {
                res.add_arc(&u,&v);
            }
        }

        res
    }

    /// Generates a directed grid with s rows and t columns. Arcs go from
    /// left to right / top to bottom.
    pub fn grid(s:u32, t:u32) -> EditDigraph {
        let mut res = EditDigraph::with_capacity((s*t) as usize);
        if s == 0 || t == 0 {
            return res;
        }

        // s = 3, t = 4
        // 0 -- 1 -- 2 -- 3
        // |    |    |    |
        // 4 -- 5 -- 6 -- 7
        // |    |    |    |
        // 8 -- 9 -- 10 -- 11
        for y in 0..s-1 { // 0..2
            for x in 0..t-1 { // 0..3
                let u = y * t + x;
                let right = u + 1;
                let down = u + t;
                res.add_arc(&u, &right);
                res.add_arc(&u, &down);
            }
            // Last vertex of this row
            let u = y*t + t-1;
            let down = u + t;
            res.add_arc(&u, &down);
        }

        // Last row
        for x in 0..t-1 { // 0..3
            let u = (s-1) * t + x; // (s-1) * t = 2*4 = 8 + 0..3 = 8..11
            let right = u + 1;
            res.add_arc(&u, &right);
        }

        res
    }

    /// Creates a new graph that is the disjoint union of `self` and `graph`.
    /// The vertices of the second graph are relabelled to avoid index clashes.
    pub fn disj_union(&self, graph: &impl Graph) -> EditDigraph {
        let mut res = EditDigraph::with_capacity(self.len() + graph.len());

        let offset:Vertex = self.vertices().max().map(|x| x+1).unwrap_or(0);

        res.add_vertices(self.vertices().cloned());
        res.add_arcs(self.edges());

        res.add_vertices(graph.vertices().map(|v| v+offset));
        res.add_arcs(graph.edges().map(|(u,v)| (u+offset,v+offset) ));

        res
    }

    /// Computes the disjoint unions of all graphs in the iterator `it`.
    pub fn disj_unions<'a,I,G>(it: I) -> EditDigraph where I: Iterator<Item = &'a G>, G: Graph + 'a {
        let mut res = EditDigraph::new();
        for graph in it {
            res = res.disj_union(graph);
        }
        res
    }

    /// Creates a copy of the graph in which vertices are labelled from $0$ to $n-1$,
    /// where $n$ is the number of vertices in the graph. The relative order of the indices
    /// is preserved, e.g. the smallest vertex from the original graph will be labelled $0$ and
    /// the largest one $n-1$.
    ///
    /// Returns a tuple (`graph`, `map`) where `graph` is the relabelled graph and
    /// `map` stores the mapping from new vertices to old vertices.
    pub fn normalize(&self) -> (EditDigraph, FxHashMap<Vertex, Vertex>) {
        let mut res = EditDigraph::with_capacity(self.num_vertices());
        let mut order:Vec<_> = self.vertices().collect();
        order.sort_unstable();

        let id2vex:FxHashMap<Vertex, Vertex> = order.iter()
                                .enumerate().map(|(i,v)| (i as Vertex, **v) )
                                .collect();
        let vex2id:FxHashMap<Vertex, Vertex> = id2vex.iter()
                                .map(|(i,v)| (*v, *i))
                                .collect();
        for u in self.vertices() {
            res.add_vertex(&vex2id[u]);
        }
        for (u, v) in self.edges() {
            res.add_arc(&vex2id[&u], &vex2id[&v]);
        }
        (res, id2vex)
    }
}



//  #######
//     #    ######  ####  #####  ####
//     #    #      #        #   #
//     #    #####   ####    #    ####
//     #    #           #   #        #
//     #    #      #    #   #   #    #
//     #    ######  ####    #    ####


#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;
    use crate::algorithms::GraphAlgorithms;

    #[test]
    fn add_remove_arcs() {
        let mut G = EditDigraph::new();
        G.add_arc(&0, &1);
        G.add_arc(&0, &2);
        G.add_arc(&0, &3);

        assert_eq!(G.out_degree(&0), 3);
        assert_eq!(G.in_degree(&0), 0);
        assert_eq!(G.out_neighbours(&0).cloned().collect::<Vec<_>>(), vec![1,2,3]);

        G.remove_arc(&0, &3);
        assert_eq!(G.out_degree(&0), 2);
        assert_eq!(G.in_degree(&0), 0);
        assert_eq!(G.out_neighbours(&0).cloned().collect::<Vec<_>>(), vec![1,2]);

        G.remove_arc(&0, &2);
        assert_eq!(G.out_degree(&0), 1);
        assert_eq!(G.in_degree(&0), 0);
        assert_eq!(G.out_neighbours(&0).cloned().collect::<Vec<_>>(), vec![1]);

        G.remove_arc(&0, &1);
        assert_eq!(G.out_degree(&0), 0);
        assert_eq!(G.in_degree(&0), 0);
    }

    #[test]
    fn remove_vertex() {
        let mut G = EditDigraph::new();
        G.add_arc(&0, &1);
        G.add_arc(&0, &2);
        G.add_arc(&0, &3);

        G.add_arc(&1, &0);
        G.add_arc(&2, &0);
        G.add_arc(&3, &0);

        G.remove_vertex(&0);

        assert_eq!(G.num_edges(), 0);
    }


    #[test]
    fn neighbourhoods() {
        let mut G = EditDigraph::new();
        // 0 --> 1 --> 2 --> 3 --> 4
        G.add_arc(&0, &1);
        G.add_arc(&1, &2);
        G.add_arc(&2, &3);
        G.add_arc(&3, &4);

        // Check that undirected graph neighbourhoods work as intended
        assert_eq!( G.neighbourhood([2].iter()), [1,3].iter().cloned().collect());
        assert_eq!( G.neighbourhood([1,2].iter()), [0,3].iter().cloned().collect());
        assert_eq!( G.neighbourhood([1,3].iter()), [0,2,4].iter().cloned().collect());

        // Check that directed graph neighbourhoods work as intended
        assert_eq!( G.in_neighbourhood([2].iter()), [1].iter().cloned().collect());
        assert_eq!( G.out_neighbourhood([2].iter()), [3].iter().cloned().collect());
        assert_eq!( G.in_neighbourhood([1,2].iter()), [0].iter().cloned().collect());
        assert_eq!( G.out_neighbourhood([1,2].iter()), [3].iter().cloned().collect());
        assert_eq!( G.in_neighbourhood([1,3].iter()), [0,2].iter().cloned().collect());
        assert_eq!( G.out_neighbourhood([1,3].iter()), [2,4].iter().cloned().collect());
    }

    #[test]
    fn arcs() {
        let mut G = EditDigraph::new();
        // 0 --> 1 --> 2 --> 3 --> 4
        G.add_arc(&0, &1);
        G.add_arc(&1, &2);
        G.add_arc(&2, &3);
        G.add_arc(&3, &4);

        assert_eq!( G.num_edges(), 4);
        let arcs = EdgeSet::from_iter(vec![(0,1),(1,2),(2,3),(3,4)]);
        let arcs_graph = G.arcs().collect();
        assert_eq!(arcs, arcs_graph);
    }
}
