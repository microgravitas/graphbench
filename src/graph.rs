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

pub trait Diraph<Vertex>: Graph<Vertex> {

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        self.has_arc(u, v) || self.has_arc(v, u)
    }

    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool;

    fn degree(&self, u:&Vertex) -> usize {
        self.out_degree(u) + self.in_degree(u)
    }

    fn out_degree(&self, u:&Vertex) -> usize;
    fn in_degree(&self, u:&Vertex) -> usize;

    // Return type 'impl ...' not allowed yet
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    // fn neighbourhoods(&self) -> Box<dyn Iterator<Item=&Vertex>>;

    // fn edges(&self) -> Box<dyn Iterator<Item=(Vertex,Vertex)>>;
}
