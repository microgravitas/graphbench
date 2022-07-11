//! Defines several traits for graph data structures. 
//! 
//! We distinguish between undirected vs directed graphs (digraphs), as well as static vs mutable graphs.
//! Static graphs do not allow modifications and can therefore be implemented with a 
//! smaller memory footprint and faster access times.
//! 
//! This library uses 32 bits to represent vertices, therefore graphs cannot contain more than
//! $2^{32} \approx 4.29~\text{Billion}$ vertices.
use fxhash::{FxHashMap, FxHashSet};

use std::{hash::Hash, collections::HashMap, ops::Add};

/// A vertex in a graph.
pub type Vertex = u32;

/// An edge in a graph.
pub type Edge = (Vertex, Vertex);

pub enum VertexOrEdge {
    V(Vertex),
    E(Edge)
}

/// An arc in a digraph.
pub type Arc = (Vertex, Vertex);

/// A set of vertices (implemented as a hashset).
pub type VertexSet = FxHashSet<Vertex>;

/// A set of vertices and edges (implemented as a hashset).
pub type MixedSet = FxHashSet<VertexOrEdge>;

/// A hashmap with vertices as keys.
pub type VertexMap<T> = FxHashMap<Vertex, T>;

/// Alias for a reference to a vertex set.
pub type VertexSetRef<'a> = FxHashSet<&'a Vertex>;

///  Alias for a reference to a mixed set.
pub type MixedSetRef<'a> = FxHashSet<&'a VertexOrEdge>;

/// A set of edges (implemented as a hashset).
pub type EdgeSet = FxHashSet<Edge>;

/// Trait for static graphs.
pub trait Graph {
    /// Returns the number of vertices in the graph.
    fn num_vertices(&self) -> usize;

    /// Returns the number of edges in the graph.
    fn num_edges(&self) -> usize;

    /// Returns whether the given vertex is contained in the graph.
    fn contains(&self, u:&Vertex) -> bool;

    /// Returns whether vertices `u` and `v` are connected by an edge.
    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool;

    /// Returns the number of edges incident to `u` in the graph.
    fn degree(&self, u:&Vertex) -> u32;

    /// Returns the degrees of all vertices in the graph as a map.
    fn degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.degree(v));
        }
        res
    }

    /// Alias for `Graph::num_vertices()`
    fn len(&self) -> usize {
        self.num_vertices()
    }

    /// Returns an iterator to this graph's vertices.
    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    /// Returns an iterator over the neighbours of `u`.
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    /// Given an iterator `vertices` over vertices, returns all vertices of the graph
    /// which are neighbours of those vertices but not part of `vertices` themselves.
    fn neighbourhood<'a, I>(&self, vertices:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        let centers:FxHashSet<Vertex> = vertices.cloned().collect();

        for v in &centers {
            res.extend(self.neighbours(v).cloned());
        }

        res.retain(|u| !centers.contains(&u));
        res
    }

    /// Given an iterator `vertices` over vertices, returns all vertices of the graph
    /// which are neighbours of those vertices as well as all vertices contained in `vertices`.
    fn closed_neighbourhood<'a, I>(&self, vertices:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        for v in vertices {
            res.extend(self.neighbours(v).cloned());
        }

        res
    }

    /// Returns all vertices which lie within distance at most `r` to `u`.
    fn r_neighbours(&self, u:&Vertex, r:usize) -> FxHashSet<Vertex>  {
        self.r_neighbourhood([u.clone()].iter(), r)
    }

    /// Given an iterator `vertices` over vertices and a distance `r`, returns all vertices of the graph
    /// which are within distance at most `r` to vertices contained in `vertices`.
    fn r_neighbourhood<'a,I>(&self, vertices:I, r:usize) -> FxHashSet<Vertex>  
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        res.extend(vertices.cloned());
        for _ in 0..r {
            let ext = self.closed_neighbourhood(res.iter());
            res.extend(ext);
        }
        res
    }    

    /// Returns the subgraph induced by the vertices contained in `vertices`.
    fn subgraph<'a, M, I>(&self, vertices:I) -> M 
                where M: MutableGraph, I: Iterator<Item=&'a Vertex> {
        let selected:VertexSet = vertices.cloned().collect();
        let mut G = M::with_capacity(selected.len());
        for v in &selected {
            G.add_vertex(v);
            let Nv:VertexSet = self.neighbours(v).cloned().collect();
            for u in Nv.intersection(&selected) {
                G.add_edge(u, v);
            }
        }   

        G
    }    
}

/// Trait for mutable graphs.
pub trait MutableGraph: Graph{
    /// Creates an emtpy mutable graph.
    fn new() -> Self;

    /// Creates a mutable graph with a hint on how many vertices it will probably contain.
    fn with_capacity(n: usize) -> Self;

    /// Adds the vertex `u` to the graph.
    /// 
    /// Returns `true` if the vertex was added and `false` if it was already contained in the graph.
    fn add_vertex(&mut self, u: &Vertex) -> bool;

    /// Removes the vertex `u` from the graph. 
    /// 
    /// Returns `true` if the vertex was removed and `false` if it was not contained in the graph.
    fn remove_vertex(&mut self, u: &Vertex) -> bool;

    /// Adds the edge `uv` to the graph. 
    /// 
    /// Returns `true` if the edge was added and `false` if it was already contained in the graph.
    fn add_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;

    /// Removes the edge `uv` from the graph. 
    /// 
    /// Returns `true` if the edge was removed and `false` if it was not contained in the graph.    
    fn remove_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;

    /// Adds a collection of `vertices` to the graph.
    ///
    /// Returns the number of vertices added this way.
    fn add_vertices<'a>(&mut self, vertices: impl Iterator<Item=Vertex>) -> u32 {
        let mut count = 0;
        for v in vertices {
            if self.add_vertex(&v) {
                count += 1;
            }
        }
        count
    }

    /// Adds a collection of `edges` to the graph.
    ///
    /// Returns the number of edges added this way.
    fn add_edges<'a>(&mut self, edges: impl Iterator<Item=Edge>) -> u32 {
        let mut count = 0;
        for (u,v) in edges {
            if self.add_edge(&u, &v) {
                count += 1;
            }
        }
        count
    }

    /// Removes all loops from the graph.
    /// 
    /// Returns the number of loops removed.
    fn remove_loops(&mut self) -> usize {
        let mut cands = Vec::new();
        for u in self.vertices() {
            if self.adjacent(u, u) {
                cands.push(u.clone())
            }
        }

        let res = cands.len();
        for u in cands.into_iter() {
            self.remove_edge(&u, &u);
        }

        res
    }    

    /// Removes all isolate vertices, that is, vertices without any neighbours.
    /// 
    /// Returns the number of isolates removed.
    fn remove_isolates(&mut self) -> usize {
        let cands:Vec<_> = self.vertices().filter(|&u| self.degree(u) == 0).cloned().collect();
        let res = cands.len();
        for u in cands.into_iter() {
            self.remove_vertex(&u);
        }

        res
    }
}

/// Trait for static digraphs. The trait inherits the [Graph] trait, all methods from that trait
/// are treating the digraph as an undirected graph. For example, the (undirected) neighbourhood of 
/// a vertex is the union of its in-neighbourhood and its out-neighbourhood in the digraph.
pub trait Digraph: Graph {
    /// Returns whether the arc `uv` exists in the digraph.
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool;

    /// Returns the number of arcs which point to `u` in the digraph.
    fn in_degree(&self, u:&Vertex) -> u32 {
        self.in_neighbours(&u).count() as u32
    }

    /// Returns the number of arcs which point away from `u` in the digraph.
    fn out_degree(&self, u:&Vertex) -> u32 {
        self.out_neighbours(&u).count() as u32
    }

    /// Returns the in-degrees of all vertices in the digraph as a map.
    fn in_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.in_degree(v));
        }
        res
    }

    /// Returns the out-degrees of all vertices in the digraph as a map.
    fn out_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.out_degree(v));
        }
        res
    }

    /// Returns the set of all in- and out-neighbours of `u` as an iterator.
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    /// Returns an iterator over the out-neighbours of `u`.
    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    /// Returns an iterator over the in-neighbours of `u`.
    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
}

/// Trait for mutable digraphs (currently incomplete).
pub trait MutableDigraph: Digraph  {
    /// Creats an empty mutable digraph
    fn new() -> Self;

    /// Adds the vertex `u` to the digraph.
    /// 
    /// Returns `true` if the vertex was added and `false` if it was already contained in the graph.    
    fn add_vertex(&mut self, u: &Vertex) -> bool;

    /// Removes the vertex `u` from the digraph. 
    /// 
    /// Returns `true` if the vertex was removed and `false` if it was not contained in the graph.    
    fn remove_vertex(&mut self, u: &Vertex) -> bool;

    /// Adds the arc `uv` to the digraph. 
    /// 
    /// Returns `true` if the arc was added and `false` if it was already contained in the graph.
    fn add_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;

    /// Removes the arc `uv` from the graph. 
    /// 
    /// Returns `true` if the arc was removed and `false` if it was not contained in the graph.        
    fn remove_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;
}


