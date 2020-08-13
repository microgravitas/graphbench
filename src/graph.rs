use fnv::{FnvHashMap, FnvHashSet};

pub type Vertex = u32;
pub type Edge = (Vertex, Vertex);
pub type Arc = (Vertex, Vertex);
pub type VertexSet = FnvHashSet<Vertex>;
pub type VertexSetRef<'a> = FnvHashSet<&'a Vertex>;
pub type EdgeSet = FnvHashSet<Edge>;

pub trait Graph<V> {
    fn num_vertices(&self) -> usize;
    fn num_edges(&self) -> usize;

    fn contains(&self, u:&V) -> bool;

    fn adjacent(&self, u:&V, v:&V) -> bool;
    fn degree(&self, u:&V) -> usize;

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&V> + 'a>;
    fn neighbours<'a>(&'a self, u:&V) -> Box<dyn Iterator<Item=&V> + 'a>;
}

pub trait MutableGraph<V>: Graph<V> {
    fn new() -> Self;
    fn add_vertex(&mut self, u: &V) -> bool;
    fn remove_vertex(&mut self, u: &V) -> bool;
    fn add_edge(&mut self, u: &V, v: &V) -> bool;
    fn remove_edge(&mut self, u: &V, v: &V) -> bool;
}

pub trait Digraph<V>: Graph<V> {
    fn has_arc(&self, u:&V, v:&V) -> bool;

    fn in_degree(&self, u:&V) -> usize {
        self.in_neighbours(&u).count()
    }

    fn out_degree(&self, u:&V) -> usize {
        self.out_neighbours(&u).count()
    }

    fn neighbours<'a>(&'a self, u:&V) -> Box<dyn Iterator<Item=&V> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    fn out_neighbours<'a>(&'a self, u:&V) -> Box<dyn Iterator<Item=&V> + 'a>;
    fn in_neighbours<'a>(&'a self, u:&V) -> Box<dyn Iterator<Item=&V> + 'a>;
}

pub trait MutableDigraph<V>: Digraph<V> {
    fn new() -> Self;
    fn add_vertex(&mut self, u: &V) -> bool;
    fn remove_vertex(&mut self, u: &V) -> bool;
    fn add_arc(&mut self, u: &V, v: &V) -> bool;
    fn remove_arc(&mut self, u: &V, v: &V) -> bool;
}
