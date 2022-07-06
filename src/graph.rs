use fxhash::{FxHashMap, FxHashSet};

use std::{hash::Hash, collections::HashMap, ops::Add};

pub type Vertex = u32;
pub type Edge = (Vertex, Vertex);
pub type Arc = (Vertex, Vertex);
pub type VertexSet = FxHashSet<Vertex>;
pub type VertexMap<T> = FxHashMap<Vertex, T>;
pub type VertexSetRef<'a> = FxHashSet<&'a Vertex>;
pub type EdgeSet = FxHashSet<Edge>;

pub trait Graph {
    fn num_vertices(&self) -> usize;
    fn num_edges(&self) -> usize;

    fn contains(&self, u:&Vertex) -> bool;

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool;
    fn degree(&self, u:&Vertex) -> u32;

    fn degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.degree(v));
        }
        res
    }

    /// Alias for Graph::num_vertices()
    fn len(&self) -> usize {
        self.num_vertices()
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    fn neighbourhood<'a, I>(&self, it:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        let centers:FxHashSet<Vertex> = it.cloned().collect();

        for v in &centers {
            res.extend(self.neighbours(v).cloned());
        }

        res.retain(|u| !centers.contains(&u));
        res
    }

    fn closed_neighbourhood<'a, I>(&self, it:I) -> FxHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        for v in it {
            res.extend(self.neighbours(v).cloned());
        }

        res
    }

    fn r_neighbours(&self, u:&Vertex, r:usize) -> FxHashSet<Vertex>  {
        self.r_neighbourhood([u.clone()].iter(), r)
    }

    fn r_neighbourhood<'a,I>(&self, it:I, r:usize) -> FxHashSet<Vertex>  
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FxHashSet<Vertex> = FxHashSet::default();
        res.extend(it.cloned());
        for _ in 0..r {
            let ext = self.closed_neighbourhood(res.iter());
            res.extend(ext);
        }
        res
    }    

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

pub trait MutableGraph: Graph{
    fn new() -> Self;
    fn with_capacity(n: usize) -> Self;

    fn add_vertex(&mut self, u: &Vertex) -> bool;
    fn remove_vertex(&mut self, u: &Vertex) -> bool;
    fn add_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;

    fn remove_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;


    fn add_vertices<'a>(&mut self, it: impl Iterator<Item=Vertex>) -> u32 {
        let mut count = 0;
        for v in it {
            if self.add_vertex(&v) {
                count += 1;
            }
        }
        count
    }

    fn add_edges<'a>(&mut self, it: impl Iterator<Item=Edge>) -> u32 {
        let mut count = 0;
        for (u,v) in it {
            if self.add_edge(&u, &v) {
                count += 1;
            }
        }
        count
    }

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

    fn remove_isolates(&mut self) -> usize {
        let cands:Vec<_> = self.vertices().filter(|&u| self.degree(u) == 0).cloned().collect();
        let res = cands.len();
        for u in cands.into_iter() {
            self.remove_vertex(&u);
        }

        res
    }
}

pub trait Digraph: Graph {
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool;

    fn in_degree(&self, u:&Vertex) -> u32 {
        self.in_neighbours(&u).count() as u32
    }

    fn out_degree(&self, u:&Vertex) -> u32 {
        self.out_neighbours(&u).count() as u32
    }

    fn in_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.in_degree(v));
        }
        res
    }

    fn out_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for v in self.vertices() {
            res.insert(*v, self.out_degree(v));
        }
        res
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.in_neighbours(u).chain(self.out_neighbours(u)))
    }

    fn out_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
}

pub trait MutableDigraph: Digraph  {
    fn new() -> Self;
    fn add_vertex(&mut self, u: &Vertex) -> bool;
    fn remove_vertex(&mut self, u: &Vertex) -> bool;
    fn add_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;
    fn remove_arc(&mut self, u: &Vertex, v: &Vertex) -> bool;
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
    use std::matches;
    use crate::{editgraph::EditGraph, iterators::EdgeIterable};

    use rand::{seq::SliceRandom, SeedableRng}; // 0.6.5
    use rand_chacha::ChaChaRng; // 0.1.1

    use itertools::Itertools;

    #[test]
    fn bipartite() {
        let mut G = EditGraph::new();

        G.add_edge(&0,&1);
        let witness = G.is_bipartite();
        assert!(matches!(witness, BipartiteWitness::Bipartition(_, _)));

        G.add_edge(&1,&2);
        let witness = G.is_bipartite();
        assert!(matches!(witness, BipartiteWitness::Bipartition(_, _)));


        G.add_edge(&0,&2);
        let witness = G.is_bipartite();
        assert!(matches!(witness, BipartiteWitness::OddCycle(_)));

        let G = EditGraph::cycle(50);
        let witness = G.is_bipartite();
        assert!(matches!(witness, BipartiteWitness::Bipartition(left, right) if left.len() == right.len()));

        let G = EditGraph::cycle(51);
        let witness = G.is_bipartite();
        assert!(matches!(witness, BipartiteWitness::OddCycle(cycle) if cycle.len() == 51));        



        let karate = EditGraph::from_txt("resources/karate.txt").unwrap();

        let mut edges:Vec<Edge> = karate.edges().collect();
        
        let seed = [4; 32];
        let mut rng = ChaChaRng::from_seed(seed);
        edges.shuffle(&mut rng);

        let mut G = EditGraph::new();
        for (u,v) in edges {
            G.add_edge(&u, &v);
        }

        let witness = G.is_bipartite();
        println!("n = {}, m = {}", G.num_vertices(), G.num_edges());
        println!("Bipartite: {:?}", witness);        
        assert!(matches!(witness, BipartiteWitness::OddCycle(_)));

        if let BipartiteWitness::OddCycle(cycle) = witness {
            for (u,v) in cycle.iter().tuple_windows() {
                assert!(G.adjacent(u, v));
            }
            assert!(G.adjacent(cycle.first().unwrap(), cycle.last().unwrap()));
        }
    }    
}
                                    