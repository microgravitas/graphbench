use fxhash::{FxHashMap, FxHashSet};

use crate::iterators::*;
use crate::graph::*;

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

    /*
        Advanced operations
    */
    pub fn contract<'a, I>(&mut self, mut vertices:I) -> Vertex where I: Iterator<Item=&'a Vertex> {
        // TODO: handle case when I is empty
        let u = vertices.next().unwrap();
        self.contract_into(u, vertices);
        *u
    }

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

    pub fn subgraph<'a, I>(&self, vertices:I) -> EditGraph where I: Iterator<Item=&'a Vertex> {
        let selected:VertexSet = vertices.cloned().collect();
        let mut G = EditGraph::with_capacity(selected.len());
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
