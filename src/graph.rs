use fnv::{FnvHashMap, FnvHashSet};

pub trait Graph<Vertex> {
    fn num_vertices(&self) -> usize;
    fn num_edges(&self) -> usize;

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool;
    fn degree(&self, u:&Vertex) -> usize;

    fn contains(&self, u:&Vertex) -> bool;

    // Return type 'impl ...' not allowed yet
    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    // fn neighbourhoods(&self) -> Box<dyn Iterator<Item=&Vertex>>;

    // fn edges(&self) -> Box<dyn Iterator<Item=(Vertex,Vertex)>>;
}
