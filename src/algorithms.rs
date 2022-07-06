use fxhash::{FxHashMap, FxHashSet};

use std::collections::HashSet;
use union_find_rs::prelude::*;

use std::cmp::{max, min, Eq};

use crate::graph::*;
use crate::iterators::*;

pub trait GraphAlgorithms {
    fn components(&self) -> Vec<VertexSet>;
    fn degeneracy(&self) -> (u32, u32, Vec<Vertex>,VertexMap<u32>);
    fn is_bipartite(&self) -> BipartiteWitness;
}

#[derive(Debug)]
pub enum BipartiteWitness {
    Bipartition(VertexSet, VertexSet),
    OddCycle(Vec<Vertex>)
}

impl<G> GraphAlgorithms for G where G: Graph {

    #[allow(unused_must_use)]
    fn components(&self) -> Vec<VertexSet> {
        let mut dsets:DisjointSets<u32> = DisjointSets::new();

        for v in self.vertices() {
            // This returns a Result<()> but the potential 'error' (adding
            // an element that already exists) will not happen.
            dsets.make_set(*v);
        }

        for (u,v) in self.edges() {
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
            deg_dict.insert(v.clone(), d);
            buckets
                .entry(calc_index(d) as i32)
                .or_insert_with(FxHashSet::default)
                .insert(v.clone());
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

            let v = buckets[&d].iter().next().unwrap().clone();
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
                        .insert(u.clone());
                }

                // Updated degree
                deg_dict.entry(u.clone()).and_modify(|e| *e -= 1);
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
                if colours.contains_key(&v) {
                    let (curr_col, other_parent) = colours.get(&v).unwrap();
                    if *curr_col != col {
                        conflict = Some((parent, v, *other_parent));
                        break;
                    }
                    // Otherwise the already assigned colour matches and we are done with v
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

        return BipartiteWitness::Bipartition(left, right)
    }    
}


