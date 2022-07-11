//! 
//! This is a versatile graph data structure that allows various modifications. It uses hash maps
//! to store adjacency lists internally, therefore it is not very memory- or cash-efficient. 
//! 
//! Graphs can either be loaded from file (see [`graphbench::io`](crate::io)) or constructed
//! by manually adding vertices and edges. The struct offers a few constructors for named graphs:
//! 
//! ```rust
//! use graphbench::graph::*;
//! use graphbench::iterators::*;
//! use graphbench::editgraph::EditGraph;
//! 
//! fn main() {
//!     let graph = EditGraph::path(5);
//!     let edges:EdgeSet = vec![(0,1),(1,2),(2,3),(3,4)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! 
//!     let graph = EditGraph::cycle(5);
//!     let edges:EdgeSet = vec![(0,1),(1,2),(2,3),(3,4),(0,4)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! 
//!     let graph = EditGraph::matching(4);
//!     let edges:EdgeSet = vec![(0,4),(1,5),(2,6),(3,7)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! 
//!     let graph = EditGraph::clique(4);
//!     let edges:EdgeSet = vec![(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! 
//!     let graph = EditGraph::biclique(2,3);
//!     let edges:EdgeSet = vec![(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! 
//!     let graph = EditGraph::complete_kpartite(vec![1,2,2].iter());
//!     let edges:EdgeSet = vec![(0,1),(0,2),(0,3),(0,4),(1,3),(1,4),(2,3),(2,4)].into_iter().collect();
//!     assert_eq!(graph.edges().collect::<EdgeSet>(), edges);
//! }
//! ```
//! 
//! ## Editing operations
//! 
//! Vertices and edges can be added an removed from the graph in $O(1)$ time (see basic example on the [`graphbench`](crate) page).
//! These operations can also be applied in bulk:
//! 
//! ```rust
//! use graphbench::graph::*;
//! use graphbench::editgraph::EditGraph;
//! 
//! fn main() {
//!     let mut graph = EditGraph::new();
//!     graph.add_vertices(vec![0,1,2,3].into_iter());   
//!     graph.add_edges(vec![(0,1),(1,2),(2,3)].into_iter());
//! 
//!     println!("Graph has {} vertices and {} edges", graph.num_vertices(), graph.num_edges());
//! }
//! 
//! ```
//! The data structure further supports the *contraction* or *identification* of vertices. This operation
//! takes a set of vertices $X$ and turns it into a single vertex whose neighbourhood are the neighbours
//! of $X$. In graph-theoretic terms, the difference between a contraction and identification is that for
//! a contraction we demand that $G\[X\]$ is connected. The methods offered here will *not* check connectivity.
//! 
//! ```rust
//! use graphbench::graph::*;
//! use graphbench::iterators::*;
//! use graphbench::editgraph::EditGraph;
//! 
//! fn main() {
//!     let mut graph = EditGraph::path(4);
//!     graph.contract_pair(&1, &2);
//!     assert!(graph.contains(&1));
//!     assert!(!graph.contains(&2));
//!     assert_eq!(graph.neighbours(&1).collect::<VertexSetRef>(),
//!                 [0,3].iter().collect());
//! 
//!     // Identify vertices on left side of a matching
//!     let mut graph = EditGraph::matching(3);
//!     graph.contract_into(&0, vec![1,2].iter());
//!     assert!(graph.contains(&0));
//!     assert!(!graph.contains(&1));
//!     assert!(!graph.contains(&2));
//!     assert_eq!(graph.neighbours(&0).collect::<VertexSetRef>(),
//!                 [3,4,5].iter().collect());
//! 
//!     // The following is equivalent
//!     let mut graph_other = EditGraph::matching(3);
//!     graph_other.contract(vec![0,1,2].iter());
//!     assert_eq!(graph, graph_other);
//! }
//! ```

use std::iter::Sum;
use itertools::max;
use fxhash::{FxHashMap, FxHashSet};

use crate::iterators::*;
use crate::graph::*;

/// An implementation of the [MutableGraph] trait with additional convenient editing and generating functions.
#[derive(Debug)]
pub struct EditGraph {
    adj: FxHashMap<Vertex, VertexSet>,
    degs: FxHashMap<Vertex, u32>,
    m: usize
}

impl PartialEq for EditGraph {
    fn eq(&self, other: &Self) -> bool {
        if self.num_vertices() != other.num_vertices() {
            return false
        }
        if self.num_edges() != other.num_edges() {
            return false
        }
        if self.degs != other.degs {
            return false
        }
        self.adj == other.adj
    }
}
impl Eq for EditGraph {}

impl Clone for EditGraph {
    fn clone(&self) -> EditGraph {
        let mut G = EditGraph::new();
        for v in self.vertices() {
            G.add_vertex(v);
            for u in self.neighbours(v) {
                G.add_edge(u, v);
            }
        }

        G
    }
}

impl Graph for EditGraph {
    /*
        Basic properties and queries
    */
    fn num_vertices(&self) -> usize {
        self.adj.len()
    }

    fn num_edges(&self) -> usize {
        self.m as usize
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        match self.adj.get(u) {
            Some(N) => N.contains(v),
            _ => false
        }
    }

    fn degree(&self, u:&Vertex) -> u32 {
        *self.degs.get(u).unwrap_or(&0) 
    }

    /*
        Iteration and access
    */
    fn contains(&self, u:&Vertex) -> bool {
        self.adj.contains_key(u)
    }

    // fn vertices(&self) -> Box<dyn Iterator<Item=&Vertex>>;
    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.adj.keys())
    }

    // fn neighbours(&self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex>>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        match self.adj.get(u) {
            Some(N) => Box::new(N.iter()),
            None => panic!("Vertex not contained in EditGraph")
        }
    }
}

impl MutableGraph for EditGraph {
    fn new() -> EditGraph {
        EditGraph{adj: FxHashMap::default(),
              degs: FxHashMap::default(),
              m: 0}
    }

    fn with_capacity(n_guess:usize) -> Self {
        EditGraph {
            adj: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            degs: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            m :0
        }
    }

    fn add_vertex(&mut self, u:&Vertex) -> bool {
        if !self.adj.contains_key(&u) {
            self.adj.insert(*u, FxHashSet::default());
            self.degs.insert(*u, 0);
            true
        } else {
            false
        }
    }

    fn add_edge(&mut self, u:&Vertex, v:&Vertex) -> bool {
        self.add_vertex(u);
        self.add_vertex(v);
        if !self.adjacent(u, v) {
            self.adj.get_mut(u).unwrap().insert(*v);
            self.adj.get_mut(v).unwrap().insert(*u);
            self.degs.insert(*u, self.degs[u] + 1);
            self.degs.insert(*v, self.degs[v] + 1);
            self.m += 1;
            true
        } else {
            false
        }
    }

    fn remove_edge(&mut self, u:&Vertex, v:&Vertex) -> bool {
        if self.adjacent(u, v) {
            self.adj.get_mut(&u).unwrap().remove(&v);
            self.adj.get_mut(&v).unwrap().remove(&u);
            self.degs.insert(*u, self.degs[&u] - 1);
            self.degs.insert(*v, self.degs[&v] - 1);
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
            let N = self.adj.get(u).unwrap().clone();
            for v in &N {
                self.adj.get_mut(v).unwrap().remove(u);
                self.degs.insert(*v, self.degs[&v] - 1);
                self.m -= 1;
            }

            self.adj.remove(&u);
            self.degs.remove(&u);

            true
        }
    }

}

impl EditGraph {
    /// Generates a path on `n` vertices.
    pub fn path(n:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity(n as usize);
        for u in 0..(n-1) {
            let v = u+1;
            res.add_edge(&u,&v);
        }

        res
    }

    /// Generates a cycle on `n` vertices.
    pub fn cycle(n:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity(n as usize);
        for u in 0..n {
            let v = (u+1) % n;
            res.add_edge(&u,&v);
        }

        res
    }

    /// Generates a matching on `2n` vertices.
    pub fn matching(n:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity(n as usize);
        for u in 0..n {
            let v = u+n;
            res.add_edge(&u,&v);
        }

        res
    }

    /// Generates a star with `n` leaves, so `n+1` vertices total.
    pub fn star(n:u32) -> EditGraph {
        EditGraph::biclique(1, n)
    }

    /// Generates a complete graph (clique) on `n` vertices.
    pub fn clique(n:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity(n as usize);
        for u in 0..n {
            for v in (u+1)..n {
                res.add_edge(&u,&v);
            }
        }

        res        
    }

    /// Generates a complete bipartite graph (biclique) on `s`+`t` vertices.
    pub fn biclique(s:u32, t:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity((s+t) as usize);
        for u in 0..s {
            for v in s..(s+t) {
                res.add_edge(&u,&v);
            }
        }

        res        
    }

    /// Generates a complete k-partite graph. 
    /// 
    /// # Arguments
    /// - `sizes` - The sizes of each partite set as a sequence of integers.
    pub fn complete_kpartite<'a, I>(sizes:I) -> EditGraph where I: IntoIterator<Item=&'a u32> {
        let sizes:Vec<u32> = sizes.into_iter().cloned().collect();

        if sizes.len() == 0 { 
            return EditGraph::new();
        } else if sizes.len() == 1 {
            return EditGraph::clique(sizes[0]);
        }
        
        let mut ranges:Vec<(u32,u32)> = Vec::new();
        let mut left = 0;
        for size in sizes {
            ranges.push((left, left+size));
            left += size;
        }

        let n = ranges.last().unwrap().1;

        let mut res = EditGraph::with_capacity(n as usize);

        for i in 0..ranges.len() {
            let (leftA,rightA) = ranges[i];
            for j in (i+1)..ranges.len() {
                let (leftB,rightB) = ranges[j];
                for u in leftA..rightA { 
                    for v in leftB..rightB {
                        res.add_edge(&u, &v);
                    }
                }
            }
        }

        res
    }

    /// Creates a new graph that is the disjoint union of `self` and `graph`.
    /// The vertices of the second graph are relabelled to avoid index clashes.
    pub fn disj_union(&self, graph: &impl Graph) -> EditGraph {
        let mut res = EditGraph::with_capacity(self.len() + graph.len());

        let offset:Vertex = self.vertices().max().unwrap_or(&0) + 1;

        res.add_vertices(self.vertices().cloned());
        res.add_edges(self.edges());

        res.add_vertices(graph.vertices().map(|v| v+offset));
        res.add_edges(graph.edges().map(|(u,v)| (u+offset,v+offset) ));

        res
    }

    /// Creates a copy of the graph in which vertices are labelled from $0$ to $n-1$,
    /// where $n$ is the number of vertices in the graph. The relative order of the indices
    /// is preserved, e.g. the smallest vertex from the original graph will be labelled $0$ and
    /// the largest one $n-1$.
    /// 
    /// Returns a tuple (`graph`, `map`) where `graph` is the relabelled graph and
    /// `map` stores the mapping from new vertices to old vertices.
    pub fn normalize(&self) -> (EditGraph, FxHashMap<Vertex, Vertex>) {
        let mut res = EditGraph::with_capacity(self.num_vertices());
        let mut order:Vec<_> = self.vertices().collect();
        order.sort_unstable();

        let id2vex:FxHashMap<Vertex, Vertex> = order.iter()
                                .enumerate().map(|(i,v)| (i as Vertex, **v) )
                                .collect();
        let vex2id:FxHashMap<Vertex, Vertex> = id2vex.iter()
                                .map(|(i,v)| (*v, *i))
                                .collect();
        for u in self.vertices() {
            res.add_vertex(&vex2id[&u]);
        }
        for (u, v) in self.edges() {
            res.add_edge(&vex2id[&u], &vex2id[&v]);
        }
        (res, id2vex)
    }

    /// Contracts all `vertices` into the first vertex of the sequence. The contracted vertex has
    /// as its neighbours all vertices that were adjacent to at least one vertex in `vertices`.
    /// 
    /// This function panics if the sequence is empty.
    /// 
    /// Returns the contracted vertex.
    pub fn contract<'a, I>(&mut self, mut vertices:I) -> Vertex where I: Iterator<Item=&'a Vertex> {
        // TODO: handle case when I is empty
        let u = vertices.next().unwrap();
        self.contract_into(u, vertices);
        *u
    }

    /// Contracts the pair $\{u,v\}$ be identifying $v$ with $u$. The operation removes $v$
    /// from the graph and adds $v$'s neighbours to $u$.
    /// 
    /// Panics if either of the two vertices does not exist.
    pub fn contract_pair(&mut self, u:&Vertex, v:&Vertex)  {
        if !self.contains(u) || !self.contains(v) {
            panic!("Pair {u},{v} not contained in graph.");
        }
        let mut N:VertexSet = self.neighbours(v).cloned().collect();
        N.remove(u); // Avoid adding a self-loop

        for x in N {
            self.add_edge(u, &x);
        }
        self.remove_vertex(v);
    }    

    /// Contracts all `vertices` into the `center` vertex. The contracted vertex has
    /// as its neighbours all vertices that were adjacent to at least one vertex in `vertices`.
    pub fn contract_into<'a, I>(&mut self, center:&Vertex, vertices:I) where I: Iterator<Item=&'a Vertex> {
        let mut contract:VertexSet = vertices.cloned().collect();
        contract.remove(&center);

        let mut N = self.neighbourhood(contract.iter());
        N.remove(&center);

        for u in N {
            self.add_edge(center, &u);
        }

        for v in contract {
            self.remove_vertex(&v);
        }
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
    use super::*;
    use crate::algorithms::GraphAlgorithms;

    // #[test]
    fn components() {
        let mut G = EditGraph::new();
        let n:u32 = 10;
        for i in 0..(n/2) {
            G.add_edge(&i, &(5+i));
        }

        assert_eq!(G.components().len(), G.edges().count());

        for comp in G.components() {
            println!("{:?}", comp);
        }
    }

    #[test]
    fn neighbourhoods() {
        let mut G = EditGraph::new();
        // 0 -- 1 -- 2 -- 3 -- 4
        G.add_edge(&0, &1);
        G.add_edge(&1, &2);
        G.add_edge(&2, &3);
        G.add_edge(&3, &4);

        assert_eq!( G.neighbourhood([2].iter()), [1,3].iter().cloned().collect());
        assert_eq!( G.neighbourhood([1,2].iter()), [0,3].iter().cloned().collect());
        assert_eq!( G.neighbourhood([1,3].iter()), [0,2,4].iter().cloned().collect());
    }

    #[test]
    fn equality() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&1, &2);
        G.add_edge(&2, &3);
        G.add_edge(&3, &0);

        let mut H = G.subgraph([0,1,2,3].iter());
        assert_eq!(G, H);
        H.add_edge(&0, &2);
        assert_ne!(G, H);
        H.remove_edge(&0, &2);
        assert_eq!(G, H);
    }

    #[test]
    fn isolates_and_loops() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&0, &0);
        G.add_edge(&2, &2);

        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 3);

        G.remove_loops();

        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 1);

        G.remove_isolates();
        assert_eq!(G.num_vertices(), 2);
        assert_eq!(G.num_edges(), 1);

        G.remove_edge(&0, &1);
        G.remove_isolates();

        assert_eq!(G, EditGraph::new());
    }

    #[test]
    fn N_iteration() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&0, &2);
        G.add_edge(&0, &3);
        G.add_edge(&0, &4);
        G.add_edge(&0, &5);

        for (v,N) in G.neighbourhoods() {
            if v == 0 {
                assert_eq!(N.cloned().collect::<VertexSet>(), [1,2,3,4,5].iter().cloned().collect());
            } else {
                assert_eq!(N.cloned().collect::<VertexSet>(), [0].iter().cloned().collect());
            }
        }
    }

    #[test]
    fn edge_iteration() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&0, &2);
        G.add_edge(&0, &3);
        G.add_edge(&0, &4);
        G.add_edge(&0, &5);

        for e in G.edges() {
            println!("{:?}", e);
        }

        assert_eq!(G.edges().count(), 5);
    }

    #[test]
    fn basic_operations() {
        let mut G = EditGraph::new();
        G.add_vertex(&0);
        G.add_vertex(&1);
        G.add_vertex(&2);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(&0, &1);
        assert_eq!(G.degree(&0), 1);
        assert_eq!(G.degree(&1), 1);
        assert_eq!(G.degree(&2), 0);
        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 1);

        G.remove_edge(&0, &1);
        assert_eq!(G.degree(&0), 0);
        assert_eq!(G.degree(&1), 0);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(&0, &1);
        G.add_edge(&0, &2);
        G.add_edge(&1, &2);
        assert_eq!(G.degree(&0), 2);
        assert_eq!(G.num_edges(), 3);

        G.remove_vertex(&2);
        assert_eq!(G.degree(&0), 1);
        assert_eq!(G.num_vertices(), 2);
        assert_eq!(G.num_edges(), 1);

        G.remove_vertex(&1);
        assert_eq!(G.degree(&0), 0);
        assert_eq!(G.num_vertices(), 1);
        assert_eq!(G.num_edges(), 0);

        G.remove_vertex(&0);
        assert_eq!(G.num_vertices(), 0);
        assert_eq!(G.num_edges(), 0);
    }

    #[test]
    fn contract_pair() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&1, &2);
        G.add_edge(&2, &3);

        G.contract_pair(&1, &2);
        assert!(G.contains(&1));
        assert!(!G.contains(&2));
        assert_eq!(G.neighbours(&1).collect::<VertexSetRef>(),
                    [0,3].iter().collect());
    }

    #[test]
    fn contract() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&1, &2);
        G.add_edge(&2, &0);
        G.add_edge(&0, &3);
        G.add_edge(&1, &4);
        G.add_edge(&2, &5);

        {
            let mut H = G.clone();
            H.contract_into(&0, [0, 1, 2].iter());
            assert_eq!(H.num_vertices(), 4);
            assert_eq!(H.vertices().collect::<VertexSetRef>(),
                        [0,3,4,5].iter().collect());

            let mut HH = G.clone();
            HH.contract([0, 1, 2].iter()); // Contracts into first vertex of collection
            assert_eq!(H, HH);
        }

        {
            let mut H = G.clone();
            H.contract_into(&0, [0, 1].iter());
            assert_eq!(H.num_vertices(), 5);
            assert_eq!(H.neighbours(&0).collect::<VertexSetRef>(),
                        [2,3,4].iter().collect());
            assert_eq!(H.vertices().collect::<VertexSetRef>(),
                        [0,2,3,4,5].iter().collect());
        }
    }
}
