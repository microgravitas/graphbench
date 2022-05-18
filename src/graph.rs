use fxhash::{FxHashMap, FxHashSet};

use std::{hash::Hash, collections::HashMap, ops::Add};

pub type Vertex = u32;
pub type Edge = (Vertex, Vertex);
pub type Arc = (Vertex, Vertex);
pub type VertexSet = FxHashSet<Vertex>;
pub type VertexMap<T> = FxHashMap<Vertex, T>;
pub type VertexSetRef<'a> = FxHashSet<&'a Vertex>;
pub type EdgeSet = FxHashSet<Edge>;


pub enum BipartiteWitness {
    BIPARTITION(VertexSet, VertexSet),
    ODD_CYCLE(Vec<Vertex>)
}

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

    fn is_bipartite(&self) -> BipartiteWitness {
        let unprocessed:VertexSet = self.vertices().cloned().collect();

        let conflict:Option<(Vertex, Vertex, Vertex)> = None;

        // Stores colouring information and the _parent_ vertex which caused
        // the colouring. In this way, we can backtrack and find an odd cycle
        // if the colouring does not work.
        let mut colours:VertexMap<(bool,Vertex)> = VertexMap::default();
        while !unprocessed.is_empty() && conflict.is_none() {
            let u = *unprocessed.iter().next().unwrap();
            unprocessed.remove(&u);

            let mut col_queue = vec![(true, u, u)];
            
            while !col_queue.is_empty() {
                let (col, v, parent) = col_queue.pop().unwrap();
                if colours.contains_key(&v) {
                    // TODO
                    let (curr_col, other_parent) = colours.get(&v).unwrap();
                    if *curr_col != col {
                        conflict = Some((parent, v, *other_parent));
                        break;
                    }
                } else {
                    colours.insert(v, (col, parent));
                    // Queue neighbours
                    for u in self.neighbours(&v) {
                        col_queue.push((!col, *u, v));
                    }
                }
                unprocessed.remove(&v);
            }
        }

        if let conflict = Some((parent1, v, parent2)) {
            let odd_cycle:Vec<Vertex> = Vec::default();
            
            // TODO
            let path1:Vec<Vertex> = vec![v, parent1];
            while colours.get(path1.last()).unwrap().1 != path1.last() {
                path1.push()
            }

            let path2:Vec<Vertex> = vec![v, parent2];


            return BipartiteWitness::ODD_CYCLE(odd_cycle);
        };

        ()
    }
}

pub trait MutableGraph: Graph{
    fn new() -> Self;
    fn with_capacity(n: usize) -> Self;

    fn add_vertex(&mut self, u: &Vertex) -> bool;
    fn remove_vertex(&mut self, u: &Vertex) -> bool;
    fn add_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;
    fn remove_edge(&mut self, u: &Vertex, v: &Vertex) -> bool;

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
