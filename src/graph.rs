use fnv::{FnvHashMap, FnvHashSet};

pub trait Graph<Vertex> {
    fn num_vertices(&self) -> usize;
    fn num_edges(&self) -> usize;

    fn contains(&self, u:&Vertex) -> bool;

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool;
    fn degree(&self, u:&Vertex) -> usize;

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
}

pub trait MutableGraph<Vertex>: Graph<Vertex> {
    fn new() -> Self;
    fn add_vertex(&mut self, u: &Vertex) -> bool;
    fn remove_vertex(&mut self, u: &Vertex) -> bool;
    fn add_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;
    fn remove_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;
}

pub trait Digraph<Vertex>: Graph<Vertex> {
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool;

    fn in_degree(&self, u:&Vertex) -> usize {
        self.in_neighbours(&u).count()
    }

    fn out_degree(&self, u:&Vertex) -> usize {
        self.out_neighbours(&u).count()
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
}

pub trait MutableDigraph<Vertex>: Digraph<Vertex> {
    fn new() -> Self;
    fn add_vertex(&mut self, u: &Vertex) -> bool;
    fn remove_vertex(&mut self, u: &Vertex) -> bool;
    fn add_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;
    fn remove_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;
}
