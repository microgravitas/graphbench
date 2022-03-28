use fxhash::{FxHashMap, FxHashSet};

use std::collections::HashSet;
use union_find_rs::prelude::*;

use std::cmp::{max, min, Eq};

use crate::graph::*;
use crate::iterators::*;

pub trait GraphAlgorithms {
    fn components(&self) -> Vec<VertexSet>;
    fn degeneracy(&self) -> (Vec<Vertex>,VertexMap<u32>);
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

    fn degeneracy(&self) -> (Vec<Vertex>,VertexMap<u32>) {
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

        order.reverse(); // The reverse order is more natural to us (small left-degree)
        (order, core_numbers)
    }    
}


