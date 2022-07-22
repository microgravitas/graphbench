//! Various basic graph algorithms which operate on the [Graph]() trait.
//! 
//! ## Testing bipartiteness
//! ```
//! use graphbench::editgraph::EditGraph;
//! use graphbench::graph::*;
//! use graphbench::algorithms::*;
//! use std::matches;
//! fn main() {
//!     let mut graph = EditGraph::biclique(2,4); // Bipartite graph
//!     let witness = graph.is_bipartite();
//!     assert!(matches!(witness, BipartiteWitness::Bipartition(_, _)));
//! 
//!     graph.add_edge(&0, &1); // Graph is not bipartite anymore
//!     let witness = graph.is_bipartite(); 
//!     assert!(matches!(witness, BipartiteWitness::OddCycle(_)));
//! }
//! ```
use fxhash::{FxHashMap, FxHashSet};

use std::collections::HashSet;
use union_find_rs::prelude::*;

use std::cmp::{max, min, Eq};

use crate::graph::*;
use crate::iterators::*;

/// Implements various basic graph algorithms for the [Graph](crate::graph::Graph) trait.
pub trait GraphAlgorithms {
    /// Returns the connected components of this graph.
    fn components(&self) -> Vec<VertexSet>;

    /// Computes the degeneracy (approximate), core-numbers and a degeneracy-ordering of this graph.
    /// The return value contains four values:
    /// 1) A lower bound on the degeneracy,
    /// 2) an upper bound on the degeneracy, 
    /// 3) the computed degeneracy ordering, and
    /// 4) the core-numbers of each vertex in this ordering.
    fn degeneracy(&self) -> (u32, u32, Vec<Vertex>,VertexMap<u32>);

    /// Tests whether the graph is bipartite and returns a witness. 
    fn is_bipartite(&self) -> BipartiteWitness;
}

/// A witness which either contains a bipartition of the graph or and odd cycle. 
#[derive(Debug)]
pub enum BipartiteWitness {
    /// A bipartition of the graph.
    Bipartition(VertexSet, VertexSet),

    // An vertes sequence which forms an odd cycle in the graph.
    OddCycle(Vec<Vertex>)
}

impl<G> GraphAlgorithms for G where G: Graph {

    #[allow(unused_must_use)]
    fn components(&self) -> Vec<VertexSet> {
        let mut dsets:DisjointSets<u32> = DisjointSets::with_capacity(self.len());

        for v in self.vertices() {
            // This returns a Result<()> but the potential 'error' (adding
            // an element that already exists) will not happen.
            dsets.make_set(*v);
        }

        for (u,v) in self.edges() {
            // There is a bug in the 0.2.1 version of the crate `union-find-rs`.
            // I opened an issue here: 
            //   https://gitlab.com/rustychoi/union_find/-/issues/1
            // The following check fixes the bug but costs us some performance.            
            if dsets.find_set(&u).unwrap() == dsets.find_set(&v).unwrap() {
                continue
            }

            // This returns a Result<()> but the potential 'error' (joining
            // two already joined elements) does not matter to us.
            dsets.union(&u, &v);
        }

        // Convertex HashSet<Vertex> into VertexSet
        let mut res = Vec::new();
        for comp in dsets {
            res.push(comp.iter().cloned().collect())
        }
        res
    }

    fn degeneracy(&self) -> (u32, u32, Vec<Vertex>, VertexMap<u32>) {
        let mut order:Vec<_> = Vec::new();

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

        let mut deg_dict = VertexMap::default();
        let mut core_numbers = VertexMap::default();        
        let mut buckets = FxHashMap::<i32, FxHashSet<Vertex>>::default();

        for v in self.vertices() {
            let d = self.degree(v);
            deg_dict.insert(*v, d);
            buckets
                .entry(calc_index(d) as i32)
                .or_insert_with(FxHashSet::default)
                .insert(*v);
        }

        let mut core_num = 0;
        for _ in 0..self.num_vertices() {
            // Find non-empty bucket. If this loop executes, we
            // know that |G| > 0 so a non-empty bucket must exist.
            let mut d = 0;
            while !buckets.contains_key(&d) || buckets[&d].is_empty() {
                d += 1
            }

            core_num = max(core_num, d as u32);

            if !buckets.contains_key(&d) {
                break;
            }

            let v = *buckets[&d].iter().next().unwrap();
            buckets.get_mut(&d).unwrap().remove(&v);

            for u in self.neighbours(&v) {
                if core_numbers.contains_key(u) {
                    // Vertex u has already been removed
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
                        .or_insert_with(FxHashSet::default)
                        .insert(*u);
                }

                // Updated degree
                deg_dict.entry(*u).and_modify(|e| *e -= 1);
            }
            core_numbers.insert(v, core_num);
            order.push(v);
        }

        // Compute lower bound for core number. 
        let ix = calc_index(core_num) as u32; 
        let lower = if ix <= 33 {       
            ix
        } else {
            32 + (1 << ((ix - 32 - 1) as u32 ))
        };
        let upper = core_num;                  

        order.reverse(); // The reverse order is more natural to us (small left-degree)
        (lower, upper, order, core_numbers)
    }    

    fn is_bipartite(&self) -> BipartiteWitness {
        use std::collections::hash_map::Entry::*;

        let mut unprocessed:VertexSet = self.vertices().cloned().collect();

        let mut conflict:Option<(Vertex, Vertex, Vertex)> = None;

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
                match colours.entry(v) {
                    Occupied(e) => {
                        let (curr_col, other_parent) =  e.get();
                        if *curr_col != col {
                            conflict = Some((parent, v, *other_parent));
                            break;
                        }
                    }
                    Vacant(e) => {
                        e.insert((col, parent));
                        // Queue neighbours
                        for u in self.neighbours(&v) {
                            col_queue.push((!col, *u, v));
                        }
                    }
                }
                unprocessed.remove(&v);
            }
        }

        // If the colouring failed we construct an odd path witness
        if let Some((parent1, v, parent2)) = conflict {       
            // The first path starts at v and follows it until
            // the 'root' of the colouring via parent1
            let mut path1:Vec<Vertex> = vec![v, parent1];
            loop {
                let last = *path1.last().unwrap();                
                let par = colours.get(&last).unwrap().1;
                if par != last {
                    path1.push(par)
                } else {
                    break
                }
            }

            // The first path starts at parent2 and follows it until
            // the 'root' of the colouring
            let mut path2:Vec<Vertex> = vec![parent2];
            loop {
                let last = *path2.last().unwrap();                
                let par = colours.get(&last).unwrap().1;
                if par != last {
                    path2.push(par)
                } else {
                    break
                }
            }

            // path1: v parent1 ... root
            // path1: parent2 ... root
            assert_eq!(path1.last(), path2.last());

            // If v == parent2 then path1 is already an odd cycle
            if path1.len() > 1 && path1.first() == path1.last() {
                path1.pop();
                return BipartiteWitness::OddCycle(path1);   
            }

            // If v == parent1 then path2 is already an odd cycle
            if path2.len() > 1 && path2.first() == path2.last() {
                path2.pop();
                return BipartiteWitness::OddCycle(path2);   
            }            

            // path2 = parent2 ... x root  (for some vertex x, potentially x = parent2)
            path2.pop();     

            // path2 = parent2 ... x
            path2.reverse(); 
            // path2 = x ... parent2
        

            // Create odd cycle by concatenating paths
            path1.append(&mut path2); // root ... parent1 v parent2 ... x

            return BipartiteWitness::OddCycle(path1);
        };

        let mut left = VertexSet::default();
        let mut right = VertexSet::default();

        for (x, (col, _)) in colours.iter() {
            if *col {
                left.insert(*x);
            } else {
                right.insert(*x);
            }
        }

        BipartiteWitness::Bipartition(left, right)
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
    use std::matches;
    use crate::{editgraph::EditGraph, iterators::EdgeIterable};

    use rand::{seq::SliceRandom, SeedableRng}; // 0.6.5
    use rand_chacha::ChaChaRng; // 0.1.1

    use itertools::Itertools;

    #[test]
    fn components() {
        // let mut G = EditGraph::disj_unions(&self, graph)
    }

    // #[test]
    #[allow(unused_must_use)]
    fn union_find() {
        // There is a bug in the 0.2.1 version of the crate `union-find-rs`.
        // I opened an issue here: 
        //   https://gitlab.com/rustychoi/union_find/-/issues/1
        let mut dsets:DisjointSets<u32> = DisjointSets::new();
        dsets.make_set(0);
        dsets.make_set(1);
        dsets.make_set(2);

        dsets.union(&0, &1);
        dsets.union(&0, &2);
        dsets.union(&1, &2);
    }

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
                                    
