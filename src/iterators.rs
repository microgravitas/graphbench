use fxhash::{FxHashMap, FxHashSet};

use crate::graph::*;
use crate::reachgraph::{ReachGraph, Reachables};
use std::hash::Hash;

use crate::editgraph::EditGraph;

pub type VertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, FxHashSet<Vertex>>;
pub type NVertexIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;

use crate::dtfgraph::{DTFNode, DTFGraph};

pub type InArcIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;
pub type DTFVertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, DTFNode>;

/// Neighbourhood iterators for graphs. At each step, the iterator
/// returns a pair $(v,N(v))$.
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
        let v = *self.v_it.next()?;
        let N = self.graph.neighbours(&v);

        Some((v, N))
    }
}

/// Neighbourhood iterators for digraphs which eithe returns all in- or all
/// out-neighbourhoods. At each step, the iterator returns a pair $(v,N^-(v))$ when in
/// in-neighbourhood mode and $(v,N^+(V))$ when in out-neighbourhood mode.
pub struct DiNeighIterator<'a, G> where G: Graph {
    graph: &'a G,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>,
    mode: DiNeighIteratorMode
}

enum DiNeighIteratorMode {In, Out}

impl<'a, D: Digraph> DiNeighIterator<'a, D> {
    pub fn new_in(graph: &'a D) -> DiNeighIterator<'a, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::In, v_it: graph.vertices() }
    }

    pub fn new_out(graph: &'a D) -> DiNeighIterator<'a, D> {
        DiNeighIterator { graph, mode: DiNeighIteratorMode::Out, v_it: graph.vertices() }
    }
}

impl<'a, D> Iterator for DiNeighIterator<'a, D> where D: Digraph {
    type Item = (Vertex, Box<dyn Iterator<Item=&'a Vertex> + 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        match &self.mode {
            DiNeighIteratorMode::In =>  {let N = self.graph.in_neighbours(&v);
                    Some((v, N))},
            DiNeighIteratorMode::Out => {let N = self.graph.out_neighbours(&v);
                    Some((v, N))}
        }
    }
}

/// Allows construction of a [NeighIterator].
///
/// Has a default implementation for [Graph].
pub trait NeighIterable<G> where G: Graph {
    /// Returns a [NeighIterator] for the graph.
    fn neighbourhoods(&self) -> NeighIterator<G>;
}

impl<G> NeighIterable<G> for G where G: Graph {
    fn neighbourhoods(&self) -> NeighIterator<G> {
        NeighIterator::<G>::new(self)
    }
}


/// Allows construction of a [DiNeighIterator].
///
/// Has a default implementation for [Digraph].
pub trait DiNeighIterable<D> where D: Digraph {
    /// Returns a [DiNeighIterator] over all in-neighbourhoods of this graph.
    fn in_neighbourhoods(&self) -> DiNeighIterator<D>;

    /// Returns a [DiNeighIterator] over all out-neighbourhoods of this graph.
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

/// Edge iterator for graphs. Each edge is returned with the smaller
/// vertex first, so the edge $\\{15,3\\}$ would be returned as $(3,15)$.
///
/// The associated trait EdgeIterable is implemented for generic graphs
/// to provide the method `edges(...)` to create an `EdgeIterator`.
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

impl<G> Iterator for EdgeIterator<'_, G> where G: Graph {
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
            let u = *uu.unwrap();
            let v = self.curr_v.as_ref().unwrap();
            if v > &u {
                continue;
            }
            return Some((*v, u));
        }

        None
    }
}

/// Allows construction of [EdgeIterator].
///
/// This trait has a default implementation for [Graph].
pub trait EdgeIterable<G> where G: Graph {
    /// Returns an [EdgeIterator] over the edges of this graph.
    fn edges(&self) -> EdgeIterator<G>;
}

impl<G> EdgeIterable<G> for G where G: Graph {
    fn edges(&self) -> EdgeIterator<G> {
        EdgeIterator::<G>::new(self)
    }
}

/// An iterator that returns all vertices and edges of the graph.
///
/// This trait has a default implementation for [Graph].
pub struct MixedIterator<'a, G> where G: Graph {
    N_it: NeighIterator<'a, G>,
    returned_v: bool,
    curr_v: Option<Vertex>,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a, G> MixedIterator<'a, G> where G: Graph {
    pub fn new(graph: &'a G) -> MixedIterator<'a, G> {
        let mut res = MixedIterator {
            N_it: graph.neighbourhoods(),
            returned_v: false,
            curr_v: None,
            curr_it: None,
        };
        res.advance();
        res
    }

    fn advance(&mut self) {
        self.returned_v = false;
        if let Some((v, it)) = self.N_it.next() {
            self.curr_v = Some(v);
            self.curr_it = Some(it);
        } else {
            self.curr_v = None;
            self.curr_it = None;
        }
    }
}

impl<G> Iterator for MixedIterator<'_, G> where G: Graph {
    type Item = VertexOrEdge;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(v), Some(it)) = (self.curr_v, &mut self.curr_it) {
            // Return vertex
            if !self.returned_v {
                self.returned_v = true;
                return Some(VertexOrEdge::V(v));
            }

            // Otherwise return edge
            for u in it.by_ref() {
                // Tie-break so we do not return edges multiple times
                if v < *u {
                    return Some(VertexOrEdge::E( (v,*u) ));
                }
            }

            self.advance();
        }

        None
    }
}

pub trait MixedIterable<G> where G: Graph {
    /// Returns an [EdgeIterator] over the edges of this graph.
    fn vertices_and_edges(&self) -> MixedIterator<G>;
}

impl<G> MixedIterable<G> for G where G: Graph {
    fn vertices_and_edges(&self) -> MixedIterator<G> {
        MixedIterator::<G>::new(self)
    }
}


/// Similar to [EdgeIterator] but for arcs of digraphs.
pub struct ArcIterator<'a, D> where D: Digraph {
    N_it: DiNeighIterator<'a, D>,
    curr_v: Option<Vertex>,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a, D> ArcIterator<'a, D> where D: Digraph{
    pub fn new(graph: &'a D) -> ArcIterator<'a, D> {
        let mut res = ArcIterator {
            N_it: graph.out_neighbourhoods(),
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

impl<D> Iterator for ArcIterator<'_, D> where D: Digraph  {
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
            let u = *uu.unwrap();
            return Some((*self.curr_v.as_ref().unwrap(), u));
        }

        None
    }
}

/// Allows construction of [ArcIterator].
///
/// This trait has a default implementation for [Digraph].
pub trait ArcIterable<D> where D: Digraph {
    /// Returns an [ArcIterator] over the arcs of this digraph.
    fn arcs(&self) -> ArcIterator<D>;
}

impl<D> ArcIterable<D> for D where D: Digraph {
    fn arcs(&self) -> ArcIterator<D> {
        ArcIterator::<D>::new(self)
    }
}

/// Neighbourhood iterator for [DTFGraph]. At each step, the iterator returns
/// a pair $(v,X)$ where $X$ is a certain subset of the in-neighbours of $v$.
/// If the iterator is in 'all depths' mode, $X$ is simply $v$'s in-neighbourhood.
/// If the iterator operates on one specific depth $d$, then $X$ contains all
/// vertices that can reach $v$ via an arc of weight $d$.
pub struct DTFNIterator<'a> {
    G: &'a DTFGraph,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>,
    depth: Option<usize>,
}

impl<'a> DTFNIterator<'a> {
    /// Constructs a [DTFNIterator] in all-depths mode.
    pub fn all_depths(G: &'a DTFGraph) -> DTFNIterator<'a> {
        DTFNIterator {
            G,
            v_it: G.vertices(),
            depth: None,
        }
    }

    /// Constructs a [DTFNIterator] which only returns in-neighbours at `depth`.
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

/// Arc iterator for [DTFGraph]. If the iterator is in 'all depths' mode it
/// iterates over all arcs of the augmentation. If the iterator operates on one
/// specific depth $d$ then it only return arcs with weight (depth) $d$.
pub struct DTFArcIterator<'a> {
    N_it: DTFNIterator<'a>,
    curr_v: Vertex,
    curr_it: Option<Box<dyn Iterator<Item=&'a Vertex> + 'a>>,
}

impl<'a> DTFArcIterator<'a> {
    pub fn all_depths(G: &'a DTFGraph) -> DTFArcIterator<'a> {
        let mut res = DTFArcIterator {
            N_it: G.in_neighbourhoods_iter(),
            curr_v: u32::MAX,
            curr_it: None,
        };
        res.advance();
        res
    }

    pub fn fixed_depth(G: &'a DTFGraph, depth:usize) -> DTFArcIterator<'a> {
        let mut res = DTFArcIterator {
            N_it: G.in_neighbourhoods_iter_at(depth),
            curr_v: u32::MAX,
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

impl Iterator for DTFArcIterator<'_> {
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


/// Left-neighbourhood iterators for linear graphs. At each step, the iterator
/// returns a pair $(v,L(v))$.
pub struct LeftNeighIterator<'a, L> where L: LinearGraph  {
    graph: &'a L,
    v_it: Box<dyn Iterator<Item=&'a Vertex> + 'a>
}

impl<'a, L> LeftNeighIterator<'a, L> where L: LinearGraph  {
    pub fn new(graph: &'a L) -> LeftNeighIterator<'a, L> {
        LeftNeighIterator { graph, v_it: graph.vertices() }
    }
}

impl<L> Iterator for LeftNeighIterator<'_, L> where L: LinearGraph  {
    type Item = (Vertex, Vec<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.v_it.next()?;
        let N = self.graph.left_neighbours(&v);

        Some((v, N))
    }
}

/// Allows construction of a [LeftNeighIterator].
///
/// Has a default implementation for [LinearGraph].
pub trait LeftNeighIterable<L> where L: LinearGraph {
    /// Returns a [NeighIterator] for the graph.
    fn left_neighbourhoods(&self) -> LeftNeighIterator<L>;
}

impl<L> LeftNeighIterable<L> for L where L: LinearGraph {
    fn left_neighbourhoods(&self) -> LeftNeighIterator<L> {
        LeftNeighIterator::<L>::new(self)
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
    use crate::graph::*;
    use crate::editgraph::*;
    use crate::ordgraph::*;
    use crate::io::*;

    #[test]
    fn edge_iterator() {
        let G = EditGraph::clique(4);
        let edges:EdgeSet = G.edges().collect();
        let test:EdgeSet = vec![(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)].into_iter().collect();
        assert_eq!(edges,test);
    }

    #[test]
    fn N_iterator() {
        let G = EditGraph::biclique(2,2);
        for (v,N) in G.neighbourhoods() {
            let N:VertexSet = N.cloned().collect();
            assert_eq!(G.degree(&v) as usize, N.len());
            for u in N {
                assert!(G.adjacent(&u, &v));
            }
        }
    }

    #[test]
    fn mixed_iterator() {
        let G = EditGraph::biclique(4,5);
        let members:MixedSet = G.vertices_and_edges().collect();
        let vertices:VertexSet = members.iter()
                        .cloned()
                        .filter_map(VertexOrEdge::as_vertex)
                        .collect();
        let edges:EdgeSet = members.iter()
                        .filter_map(|m| if let VertexOrEdge::E((u,v)) = m { Some((*u,*v)) } else { None })
                        .collect();
        assert_eq!(vertices, VertexSet::from_iter(G.vertices().cloned()));
        assert_eq!(edges, EdgeSet::from_iter(G.edges().map(|e| e )));
    }

    #[test]
    fn wreach_iterator() {
        const R:usize = 5;
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let O = OrdGraph::by_degeneracy(&G);
        let W = O.to_wreach_graph::<R>();

        for (v,reachables) in W.iter() {
            assert_eq!(v, reachables.from);
            assert_eq!(reachables, W.reachables(&v));
        }

    }
}
