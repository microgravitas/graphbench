use fnv::{FnvHashMap, FnvHashSet};

use crate::iterators::*;
// use iterators::EdgeIterator

pub type Vertex = u32;


#[derive(Debug)]
pub struct Graph {
    adj: FnvHashMap<Vertex, FnvHashSet<Vertex>>,
    degs: FnvHashMap<Vertex, u32>,
    _m: u64
}

impl Graph {
    fn new() -> Graph {
        Graph{adj: FnvHashMap::default(),
              degs: FnvHashMap::default(),
              _m: 0}
    }

    /*
        Basic properties
    */
    fn num_vertices(&self) -> usize {
        self.adj.len()
    }

    fn num_edges(&self) -> u64 {
        self._m
    }

    /*
        Iteration and access
    */
    fn contains(&mut self, u:Vertex) -> bool {
        self.adj.contains_key(&u)
    }

    pub fn vertices(&self) -> VertexIterator {
        self.adj.keys()
    }

    pub fn edges(&self) -> EdgeIterator {
        EdgeIterator::new(self)
    }

    pub fn neighbours_iter(&self) -> NIterator {
        NIterator::new(self)
    }

    pub fn neighbours(&self, u:Vertex) -> NVertexIterator {
        match self.adj.get(&u) {
            Some(N) => N.iter(),
            None => panic!("Vertex not contained in graph")
        }
    }

    /*
        Modification
    */
    fn add_vertex(&mut self, u:Vertex) {
        if !self.adj.contains_key(&u) {
            self.adj.insert(u, FnvHashSet::default());
            self.degs.insert(u, 0);
        }
    }

    fn add_edge(&mut self, u:Vertex, v:Vertex) -> bool {
        self.add_vertex(u);
        self.add_vertex(v);
        if !self.adjacent(u, v) {
            self.adj.get_mut(&u).unwrap().insert(v);
            self.adj.get_mut(&v).unwrap().insert(u);
            self.degs.insert(u, self.degs[&u] + 1);
            self.degs.insert(v, self.degs[&v] + 1);
            self._m += 1;
            true
        } else {
            false
        }
    }

    fn remove_edge(&mut self, u:Vertex, v:Vertex) -> bool {
        if self.adjacent(u, v) {
            self.adj.get_mut(&u).unwrap().remove(&v);
            self.adj.get_mut(&v).unwrap().remove(&u);
            self.degs.insert(u, self.degs[&u] - 1);
            self.degs.insert(v, self.degs[&v] - 1);
            self._m -= 1;
            true
        } else {
            false
        }
    }
    // fn remove_loops(&mut self) -> u64 {
    //     let c = 0
    //
    //
    //
    //     c
    // }

    fn remove_node(&mut self, u:Vertex) -> bool {
        if !self.contains(u) {
            false
        } else {
            let N = self.adj.get(&u).unwrap().clone();
            for &v in &N {
                self.adj.get_mut(&v).unwrap().remove(&u);
                self.degs.insert(v, self.degs[&v] - 1);
                self._m -= 1;
            }

            self.adj.remove(&u);

            true
        }
    }

    fn adjacent(&self, u:Vertex, v:Vertex) -> bool {
        match self.adj.get(&u) {
            Some(N) => N.contains(&v),
            _ => false
        }
    }

    fn degree(&self, u:Vertex) -> u32 {
        *self.degs.get(&u).unwrap_or(&0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_iteration() {
        let mut G = Graph::new();
        let n:u32 = 10;
        for i in 0..(n/2) {
            G.add_edge(i, 5+i);
        }

        assert_eq!(G.edges().count(), (n/2) as usize);
        assert_eq!(G.edges().count(), G.num_edges() as usize);        
    }

    #[test]
    fn basic_operations() {
        let mut G = Graph::new();
        G.add_vertex(0);
        G.add_vertex(1);
        G.add_vertex(2);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(0, 1);
        assert_eq!(G.degree(0), 1);
        assert_eq!(G.degree(1), 1);
        assert_eq!(G.degree(2), 0);
        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 1);

        G.remove_edge(0, 1);
        assert_eq!(G.degree(0), 0);
        assert_eq!(G.degree(1), 0);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(0, 1);
        G.add_edge(0, 2);
        G.add_edge(1, 2);
        assert_eq!(G.degree(0), 2);
        assert_eq!(G.num_edges(), 3);

        G.remove_node(2);
        assert_eq!(G.degree(0), 1);
        assert_eq!(G.num_vertices(), 2);
        assert_eq!(G.num_edges(), 1);

        G.remove_node(1);
        assert_eq!(G.degree(0), 0);
        assert_eq!(G.num_vertices(), 1);
        assert_eq!(G.num_edges(), 0);

        G.remove_node(0);
        assert_eq!(G.num_vertices(), 0);
        assert_eq!(G.num_edges(), 0);
    }
}
