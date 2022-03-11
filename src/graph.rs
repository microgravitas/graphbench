use fnv::{FnvHashMap, FnvHashSet};

use std::cmp::{max, min, Eq, };
use std::hash::Hash;

pub type Vertex = u32;
pub type Edge = (Vertex, Vertex);
pub type Arc = (Vertex, Vertex);
pub type VertexSet = FnvHashSet<Vertex>;
pub type VertexMap<T> = FnvHashMap<Vertex, T>;
pub type VertexSetRef<'a> = FnvHashSet<&'a Vertex>;
pub type EdgeSet = FnvHashSet<Edge>;


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

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a>;
    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>;

    fn neighbourhood<'a, I>(&self, it:I) -> FnvHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FnvHashSet<Vertex> = FnvHashSet::default();
        let centers:FnvHashSet<Vertex> = it.cloned().collect();

        for v in &centers {
            res.extend(self.neighbours(v).cloned());
        }

        res.retain(|u| !centers.contains(&u));
        res
    }

    fn closed_neighbourhood<'a, I>(&self, it:I) -> FnvHashSet<Vertex> 
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FnvHashSet<Vertex> = FnvHashSet::default();
        for v in it {
            res.extend(self.neighbours(v).cloned());
        }

        res
    }

    fn r_neighbours(&self, u:&Vertex, r:usize) -> FnvHashSet<Vertex>  {
        self.r_neighbourhood([u.clone()].iter(), r)
    }

    fn r_neighbourhood<'a,I>(&self, it:I, r:usize) -> FnvHashSet<Vertex>  
                where I: Iterator<Item=&'a Vertex>, Vertex: 'a {
        let mut res:FnvHashSet<Vertex> = FnvHashSet::default();
        res.extend(it.cloned());
        for _ in 0..r {
            let ext = self.closed_neighbourhood(res.iter());
            res.extend(ext);
        }
        res
    }    

    fn degeneracy_ordering(&self) -> Vec<Vertex> {
        let mut res:Vec<_> = Vec::new();

        // This index function defines buckets of exponentially increasing
        // size, but all values below `small` (here 32) are put in their own
        // buckets.
        fn calc_index(i: u32) -> usize {
            let small = 2_i32.pow(5);
            min(i, small as u32) as usize
                + (max(0, (i as i32) - small + 1) as u32)
                    .next_power_of_two()
                    .trailing_zeros() as usize
        }

        let mut deg_dict = FnvHashMap::<Vertex, u32>::default();
        let mut buckets = FnvHashMap::<i32, FnvHashSet<Vertex>>::default();

        for v in self.vertices() {
            let d = self.degree(v);
            deg_dict.insert(v.clone(), d);
            buckets
                .entry(calc_index(d) as i32)
                .or_insert_with(FnvHashSet::default)
                .insert(v.clone());
        }

        let mut seen = FnvHashSet::<Vertex>::default();

        for _ in 0..self.num_vertices() {
            // Find non-empty bucket. If this loop executes, we
            // know that |G| > 0 so a non-empty bucket must exist.
            let mut d = 0;
            while !buckets.contains_key(&d) || buckets[&d].is_empty() {
                d += 1
            }

            if !buckets.contains_key(&d) {
                break;
            }

            let v = buckets[&d].iter().next().unwrap().clone();
            buckets.get_mut(&d).unwrap().remove(&v);

            for u in self.neighbours(&v) {
                if seen.contains(u) {
                    continue;
                }

                // Update bucket
                let du = deg_dict[u];
                let old_index = calc_index(du) as i32;
                let new_index = calc_index(du - 1) as i32;

                if old_index != new_index {
                    buckets.entry(old_index).and_modify(|S| {
                        (*S).remove(u);
                    });
                    buckets
                        .entry(new_index)
                        .or_insert_with(FnvHashSet::default)
                        .insert(u.clone());
                }

                // Updated degree
                deg_dict.entry(u.clone()).and_modify(|e| *e -= 1);
            }
            seen.insert(v.clone());
            res.push(v);
        }

        res
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
