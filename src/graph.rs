//! Defines several traits for graph data structures. 
//! 
//! We distinguish between undirected vs directed graphs (digraphs), as well as static vs mutable graphs.
//! Static graphs do not allow modifications and can therefore be implemented with a 
//! smaller memory footprint and faster access times.
//! 
//! This library uses 32 bits to represent vertices, therefore graphs cannot contain more than
//! $2^{32} \approx 4.29~\text{Billion}$ vertices.
use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;

use std::{borrow::Borrow, collections::HashMap, hash::Hash, ops::{Add, Index}};

/// A vertex in a graph.
pub type Vertex = u32;

/// An edge in a graph.
pub type Edge = (Vertex, Vertex);

/// An arc in a digraph.
pub type Arc = (Vertex, Vertex);

/// A sequence of vertices
pub type VertexSequence = Vec<Vertex>;

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

/// An enum which holds either a vertex or an edge. Used to allow
/// [mixed-type sets](MixedSet) and [iteration](crate::iterators::MixedIterator). 
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexOrEdge {
    V(Vertex),
    E(Edge)
}

impl VertexOrEdge {
    pub fn as_vertex(self) -> Option<Vertex> {
        match self {
            VertexOrEdge::V(v) => Some(v),
            VertexOrEdge::E(_) => None,
        }
    }    

    pub fn is_vertex(self) -> bool {
        match self {
            VertexOrEdge::V(_) => true,
            VertexOrEdge::E(_) => false,
        }
    }

    pub fn is_edge(self) -> bool {
        match self {
            VertexOrEdge::V(_) => false,
            VertexOrEdge::E(_) => true,
        }
    }

    pub fn as_edge(self) -> Option<Edge> {
        match self {
            VertexOrEdge::V(_) => None,
            VertexOrEdge::E(e) => Some(e),
        }
    }

    /// Returns true if the two vertex/edge objects intersect, that is,
    /// if they share at least one vertex.
    pub fn intersects(self, other:VertexOrEdge) -> bool {
        self.intersection(other).is_some()
    }

    // Returns the intersection of these two objects, which is 
    // either a vertex (Some(V(x))), an edge (Some(E(e))), or nothing (None).
    pub fn intersection(self, other:VertexOrEdge) -> Option<VertexOrEdge> {
        use VertexOrEdge::*;
        match (self, other) {
            (V(x), V(y)) => {
                if x == y {
                    Some(V(x)) 
                } else { 
                    None
                }
            },
            (V(x), E((a,b))) | (E((a,b)), V(x)) => {
                if x == a || x == b{
                    Some(V(x))
                } else {
                    None
                }
            }
            (E((a,b)), E((c,d))) => {
                if (a,b) == (c,d) || (b,a) == (c,d) {
                    Some(E((a,b)))
                } else if a == c || a == d {
                    Some(V(a))
                } else if b == c || b == d {
                    Some(V(b))
                } else {
                    None
                }
            }
        }
    }    
}

pub trait VertexMapOperations<T> where T: Copy + Hash + Eq {
    // Groups the vertices in this map by their values 
    // and returns the largest set among those. Ties are broken
    // arbitrarily.
    // Returns None if the vertex map is empty.
    fn majority_set(&self) -> Option<(VertexSet, T)>;

    // 
    fn invert(&self) -> FxHashMap<T, VertexSet>;
}


impl<T> VertexMapOperations<T> for VertexMap<T> where T: Copy + Hash + Eq {
    fn majority_set(&self) -> Option<(VertexSet, T)> {
        if self.is_empty() {
            return None;
        }

        let inverted = self.invert();
        let (_, value, set) = inverted.into_iter().map( |(value, set)| (set.len(), value, set) )
            .sorted_by_key(|(len, _, _)| usize::MAX - *len)
            .next().unwrap();
        Some((set, value))
    }
    
    fn invert(&self) -> FxHashMap<T, VertexSet> {
        let mut res:FxHashMap<T, VertexSet> = FxHashMap::default();
        for (&u, &value) in self.iter() {
            res.entry(value).or_default().insert(u);
        }
        res
    }
}

#[derive(Debug,PartialEq,Eq)]
pub struct VertexColouring<T> where T: Copy + Hash + Eq {
    colouring:VertexMap<T>
}

impl<T> VertexColouring<T> where T: Copy + Hash + Eq {
    pub fn from_iter<I>(iter:I) -> Self where I: IntoIterator<Item=(Vertex,T)> {
        let colouring:VertexMap<T> = VertexMap::from_iter(iter.into_iter());
        VertexColouring{ colouring }
    }

    pub fn as_map(self) -> VertexMap<T> {
        self.colouring
    }

    pub fn insert(&mut self, v:Vertex, c:T) -> Option<T> {
        self.colouring.insert(v, c)
    }

    pub fn iter(&self) -> impl Iterator<Item=(&Vertex, &T)> {
        self.colouring.iter()
    }

    pub fn len(&self) -> usize {
        self.colouring.len()
    }

    pub fn vertices(&self) -> impl Iterator<Item=&Vertex>  {
        self.colouring.keys()
    }

    pub fn colours(&self) -> impl Iterator<Item=&T>  {
        self.colouring.values().unique()
    }    

    pub fn disjoint_extend(&mut self, other:&VertexColouring<T>) where T: num::Integer {
        let colours:FxHashSet<T> = self.colouring.values().cloned().collect();
        let offset:T = if colours.is_empty() {
            T::zero()
        } else {
            let mut res = *colours.iter().max().unwrap();
            res.inc();
            res 
        };

        for (u, c) in other.iter() {
            self.colouring.entry(*u).or_insert(*c+offset);
        }
    }
    
    pub fn get(&self, u:&Vertex) -> Option<&T> {
        self.colouring.get(u)
    }

    pub fn contains(&self, u:&Vertex) -> bool {
        self.colouring.contains_key(u)
    }

    pub fn subset<V,I>(&self, vertices:I) -> VertexColouring<T>  
                where V: Borrow<Vertex>, I: IntoIterator<Item=V> {
        let mut res = VertexColouring::default();
        for v in vertices { 
            let v = *v.borrow();
            if let Some(col) = self.get(&v) {
                res.insert(v, *col);
            }
        }
        res
    }
}

impl<T> Index<&Vertex> for VertexColouring<T> where T: Copy + Hash + Eq {
    type Output = T;
    
    fn index(&self, index: &Vertex) -> &Self::Output {
        &self.colouring[index]
    }
}


impl<T> From<VertexMap<T>> for VertexColouring<T> where T: Copy + Hash + Eq {
    fn from(value: VertexMap<T>) -> Self {
        VertexColouring{ colouring: value }
    }
}

impl<T> Default for VertexColouring<T> where T: Copy + Hash + Eq {
    fn default() -> Self {
        Self { colouring: Default::default() }
    }
}

impl<T> VertexMapOperations<T> for VertexColouring<T> where T: Copy + Hash + Eq {
    fn majority_set(&self) -> Option<(VertexSet, T)> {
        self.colouring.majority_set()
    }

    fn invert(&self) -> FxHashMap<T, VertexSet> {
        self.colouring.invert()
    }
}

pub trait IntegerColouring<T> {
    fn from_sets<I>(iter:I) -> Self where I: IntoIterator<Item=VertexSet>;
}

impl IntegerColouring<u32> for VertexColouring<u32> {
    fn from_sets<I>(iter:I) -> Self where I: IntoIterator<Item=VertexSet> {
        let mut colouring:VertexMap<u32> = VertexMap::default();
        for (i,set) in iter.into_iter().enumerate() {
            let i = i as u32;
            for v in &set {
                colouring.insert(*v, i);
            }
        }
        VertexColouring{ colouring }
    }
}


/// Trait for static graphs.
pub trait Graph {
    /// Returns the number of vertices in the graph.
    fn num_vertices(&self) -> usize;

    /// Returns the number of edges in the graph.
    fn num_edges(&self) -> usize;

    /// Returns whether the vertex `u` is contained in the graph.
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

    /// Returns true if the graph contains no vertices
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator to this graph's vertices.
    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&'a Vertex> + 'a>;

    /// Returns an iterator over the neighbours of `u`.
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a>;

    /// Given an iterator `vertices` over vertices, returns all vertices of the graph
    /// which are neighbours of those vertices but not part of `vertices` themselves.
    fn neighbourhood<'a, I>(&self, vertices:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        let centers:FxHashSet<Vertex> = vertices.cloned().collect();

        for v in &centers {
            res.extend(self.neighbours(v).cloned());
        }

        res.retain(|u| !centers.contains(u));
        res
    }

    /// Given an iterator `vertices` over vertices, returns all vertices of the graph
    /// which are neighbours of those vertices as well as all vertices contained in `vertices`.
    fn closed_neighbourhood<'a, I>(&self, vertices:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        for v in vertices {
            res.extend(self.neighbours(v).cloned());
            res.insert(*v);
        }

        res
    }

    /// Returns all vertices which lie within distance at most `r` to `u`.
    fn r_neighbours(&self, u:&Vertex, r:usize) -> FxHashSet<Vertex>  {
        self.r_neighbourhood([*u], r)
    }

    /// Given an iterator `vertices` over vertices and a distance `r`, returns all vertices of the graph
    /// which are within distance at most `r` to vertices contained in `vertices`.
    fn r_neighbourhood<V,I>(&self, vertices:I, r:usize) -> FxHashSet<Vertex>  
                where V: Borrow<Vertex>, I: IntoIterator<Item=V> {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        res.extend(vertices.into_iter().map(|u| *u.borrow()));
        for _ in 0..r {
            let ext = self.closed_neighbourhood(res.iter());
            res.extend(ext);
        }
        res
    }    

    /// Returns the subgraph induced by the vertices contained in `vertices`.
    fn subgraph<M, V, I>(&self, vertices:I) -> M 
                where V: Borrow<Vertex>, I: IntoIterator<Item=V>, M: MutableGraph, {
        let selected:VertexSet = vertices.into_iter().map(|u| *u.borrow()).collect();
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
    fn add_vertices<V>(&mut self, vertices: impl Iterator<Item=V>) -> u32
        where V:Borrow<Vertex> {
        let mut count = 0;
        for v in vertices {
            if self.add_vertex(v.borrow()) {
                count += 1;
            }
        }
        count
    }

    /// Adds a collection of `edges` to the graph.
    ///
    /// Returns the number of edges added this way.
    fn add_edges(&mut self, edges: impl Iterator<Item=Edge>) -> u32 {
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
                cands.push(*u)
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
        self.in_neighbours(u).count() as u32
    }

    /// Returns the number of arcs which point away from `u` in the digraph.
    fn out_degree(&self, u:&Vertex) -> u32 {
        self.out_neighbours(u).count() as u32
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
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    /// Returns an iterator over the out-neighbours of `u`.
    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a>;

    /// Returns an iterator over the in-neighbours of `u`.
    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a>;
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

/// Represents graphs imbued with a linear ordering. The assumption is that this is used for
/// *degenerate* graphs, meaning that each vertex has only few neighbours that appear before it
/// in the ordering (the 'left' neighbourhood). 
/// 
/// As a consequence this trait does not provide a method to query the right neighbourhood of a vertex
/// as algorithms designed for degenerate graphs work exclusively on left neighbourhoods.
pub trait LinearGraph : Graph {
    /// Returns the index of `u` in the ordering.
    fn index_of(&self, u:&Vertex) -> usize;

    /// Returns the left neighbourhood of `u`. 
    fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex>;

    /// Returns the size of `u`'s right neighbourhood.
    fn right_degree(&self, u:&Vertex) -> u32;    

    /// Returns the number of left neigbhours of `u`. Returns 0 if `u`
    /// is not contained in the graph.
    fn left_degree(&self, u:&Vertex) -> u32 {
        if self.contains(u) {
            self.left_neighbours(u).len() as u32
        } else {
            0
        }
    }
    
    /// Returns the sizes of all left neighbourhoods as a map.
    fn left_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            res.insert(*u, self.left_degree(u));
        }
        res
    }

    /// Returns the maximum left degree of the graph. Returns zero if the graph is empty.
    fn max_left_degree(&self) -> u32 {
        self.vertices().map(|u| self.left_degree(u)).max().unwrap_or(0)
    }    
    
    /// Returns the sizes of all right neighbourhood as a map.
    fn right_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            res.insert(*u, self.right_degree(u));
        }
        res
    }

    /// Returns the maximum right degree of the graph. Returns zero if the graph is empty.
    fn max_right_degree(&self) -> u32 {
        self.vertices().map(|u| self.right_degree(u)).max().unwrap_or(0)
    }    

    /// Returns the leftmost (smallest) vertex in the provided collection 
    /// according to the ordering of this graph.
    fn leftmost<V, I>(&self, vertices:I) -> Option<Vertex> 
        where V: Borrow<Vertex>, I: IntoIterator<Item=V> {
        let res = vertices.into_iter().min_by_key(|u| self.index_of(u.borrow()));
        match res {
            Some(x) => Some(*x.borrow()),
            None => None
        }
    }

    /// Returns the leftmost (smallest) vertex in the provided collection 
    /// according to the ordering of this graph.
    fn rightmost<V, I>(&self, vertices:I) -> Option<Vertex> 
        where V: Borrow<Vertex>, I: IntoIterator<Item=V> {
        let res = vertices.into_iter().max_by_key(|u| self.index_of(u.borrow()));
        res.map(|x| *x.borrow())
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
    use crate::{editgraph::*, io::LoadFromFile, ordgraph::OrdGraph};

    #[test]
    fn ordering() {
        let G = EditGraph::path(5);
        let OG = OrdGraph::with_ordering(&G, vec![4,3,2,1,0].iter());

        assert_eq!(OG.leftmost(vec![3,2,1]), Some(3));
        assert_eq!(OG.rightmost(vec![3,2,1]), Some(1));
        assert_eq!(OG.leftmost(vec![0]), Some(0));
    }

    #[test]
    fn vertex_map() {
        let mut map:VertexMap<u32> = VertexMap::default();
        map.insert(0, 100);
        map.insert(1, 100);
        map.insert(2, 100);
        map.insert(3, 200);
        map.insert(4, 200);
        map.insert(5, 300);

        let inverted = map.invert();
        assert_eq!(inverted.get(&100).unwrap(), &vec![0,1,2].into_iter().collect::<VertexSet>() );
        assert_eq!(inverted.get(&200).unwrap(), &vec![3,4].into_iter().collect::<VertexSet>() );
        assert_eq!(inverted.get(&300).unwrap(), &vec![5].into_iter().collect::<VertexSet>() );
        assert_eq!(inverted.keys().cloned().collect::<FxHashSet<u32>>(), vec![100,200,300].into_iter().collect::<FxHashSet<u32>>());

        let (mset, value) = map.majority_set().unwrap();
        assert_eq!(mset, vec![0,1,2].into_iter().collect::<VertexSet>() );
        assert_eq!(value, 100);
    }

    #[test]
    fn vertex_colouring() {    
        let mut colA:VertexColouring<u32> = VertexColouring::from_iter(vec![(1,100),(2,100),(3,200),(4,200),(5,300)]);
        let colB:VertexColouring<u32> = VertexColouring::from_iter(vec![(5,100),(6,100),(7,200),(8,200),(9,300)]);

        assert_eq!(colA.colours().cloned().collect::<VertexSet>(), VertexSet::from_iter(vec![100,200,300]));
        assert_eq!(colA.vertices().cloned().collect::<VertexSet>(), VertexSet::from_iter(vec![1,2,3,4,5]));
        assert_eq!(colA[&1], 100);
        assert_eq!(colA[&2], 100);
        assert_eq!(colA[&3], 200);
        assert_eq!(colA[&4], 200);
        assert_eq!(colA[&5], 300);

        // colB contains a different colour for vertex 5, but disjoint_extend is not supposed to overwrite 
        // existing colours.
        colA.disjoint_extend(&colB);
        assert_eq!(colA[&1], 100);
        assert_eq!(colA[&2], 100);
        assert_eq!(colA[&3], 200);
        assert_eq!(colA[&4], 200);
        assert_eq!(colA[&5], 300);
        assert_eq!(colA.vertices().cloned().collect::<VertexSet>(), VertexSet::from_iter(vec![1,2,3,4,5,6,7,8,9]));
        assert_eq!(colA.colours().count(), 6); // We do not know how colours are remapped, but we know 
                                               // the total number of colours we should get


        let col_sets = vec![VertexSet::from_iter(vec![1,2,3]), VertexSet::from_iter(vec![4,5,6])];
        let colA = VertexColouring::from_sets(col_sets);
        let colB:VertexColouring<u32> = VertexColouring::from_iter(vec![(1,0),(2,0),(3,0),(4,1),(5,1),(6,1)]);

        assert_eq!(colA, colB);
    }
}