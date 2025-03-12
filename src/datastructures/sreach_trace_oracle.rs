
use std::borrow::Borrow;

use crate::{algorithms::LinearGraphAlgorithms, degengraph::DegenGraph, graph::*};

use bitvec::prelude::*;
use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;

type VertexBitVec = BitVec<u64,Lsb0>;

#[derive(Debug)]
struct SReachContext {
    context:Vec<Vertex>,
    indices:VertexMap<usize>,
    right_neighbours:FxHashMap<VertexBitVec, usize>
}

impl SReachContext {
    fn new(context:Vec<Vertex>) -> Self {
        let mut indices = VertexMap::default();
        for (i, x) in context.iter().enumerate() {
            indices.insert(*x, i);
        }

        let right_neighbours = Default::default();
        SReachContext{context, indices, right_neighbours}
    }

    fn get_sets(&self) ->  FxHashMap<Vec<Vertex>, usize> {
        self.count_neighbours(&self.context)
    }

    fn insert(&mut self, neighs:&[Vertex]) {
        let key = self.to_bitset(neighs);
        let temp = key.clone();
        *self.right_neighbours.entry(key).or_default() += 1;
    }

    fn count_neighbours(&self, target:&[Vertex]) -> FxHashMap<Vec<Vertex>, usize> {
        let mask = self.to_bitset(target);    

        // We first count using the bit vectors and later convert the found bit vectors
        // to vecs with vertices.     
        let mut counts:FxHashMap<VertexBitVec, usize> = FxHashMap::default();

        for (bits, count) in self.right_neighbours.iter() {
            let bits = mask.clone() & bits;
            *counts.entry(bits).or_default() += count;
        }
        counts.into_iter().map(|(bits,count)| (self.to_vecset(&bits), count)).collect()
    }

    fn to_bitset(&self, set:&[Vertex]) -> VertexBitVec {
        let mut bitvec:VertexBitVec = BitVec::with_capacity(self.context.len());
        bitvec.resize(self.context.len(), false);
        for u in set {
            if let Some(&ix) = self.indices.get(u) {
                bitvec.set(ix, true);
            }
        }
        bitvec
    }

    fn to_vecset(&self, bits:&VertexBitVec) -> Vec<Vertex> {
        let mut res = vec![];
        for ix in bits.iter_ones() {
            res.push(*&self.context[ix]);
        }
        res
    }
}

pub struct SReachTraceOracle {
    contexts:FxHashMap<Vertex, SReachContext>
}

impl SReachTraceOracle {
    pub fn for_graph(graph:&DegenGraph) -> Self {
        let mut contexts:FxHashMap<_,_> = Default::default();

        for x in graph.vertices() {
            let mut S:Vec<Vertex> = graph.sreach_set(x, 2).into_iter().map(|(x,_)| x).collect();
            S.push(*x);
            S.sort_by_key(|x| graph.index_of(x));
            S.dedup();

            contexts.insert(*x, SReachContext::new(S));
        }

        for x in graph.vertices() {
            let mut N = graph.left_neighbours_slice(x).iter().cloned().collect_vec();
            N.sort_by_key(|x| graph.index_of(x));
            
            for i in 1..=N.len() { // All non-empty prefixes of N, including N itself
                let Nprefix = &N[..i];
                let anchor = *&N[i-1];
                let ctx = contexts.get_mut(&anchor).unwrap();
                ctx.insert(Nprefix);
            }
        }

        Self{contexts}
    }

    pub fn compute_traces(&self, inner: &Vec<Vertex>, graph:&DegenGraph) -> FxHashMap<Vec<Vertex>, usize>  {
        // DEBUG: print all sreach info
        // for v in graph.vertices() {
        //     let ctx = &self.contexts[v];
        //     let sreach = &ctx.context;
        //     let subsets = ctx.get_sets().into_iter().collect_vec();
        //     println!("{v} {sreach:?}: {subsets:?}");
        // }

        // Count neighbourhoods
        let mut traces:FxHashMap<Vec<Vertex>, usize> = FxHashMap::default();

        debug_assert!(inner.is_sorted_by_key(|x| graph.index_of(x)));

        traces.insert(vec![], graph.num_vertices());

        // Count neighbourhoods induced by vertices to the right using the 
        // sreach context sets
        for i in 1..=inner.len() { // All non-empty prefixes of `inner`, including `inner` itself
            let prefix = &inner[..i];
            let anchor = *&inner[i-1];
            let ctx = &self.contexts[&anchor];
            
            let counts = ctx.count_neighbours(&prefix);
            for (set, count) in counts.into_iter() {
                if set.is_empty() {
                     continue // TODO: Is there a nicer way to filter this?
                }
                debug_assert!(set.is_sorted_by_key(|x| graph.index_of(x)));

                // println!("  {set:?} ({count})");
                *traces.entry(set.clone()).or_default() += count;

                // Correct count for prefix
                let set_prefix = &set[..set.len()-1];
                *traces.get_mut(set_prefix).unwrap() -= count; // This entry must exist
            }
        }

        // println!("Right traces {traces:?}");        

        // Add neighbourhoods induced by vertices to the left 
        let mut left_neighbours: VertexSet = VertexSet::default();
        for u in inner.iter() {
            left_neighbours.extend(graph.left_neighbours_slice(u).into_iter());
            left_neighbours.insert(*u);
        }

        // Collect traces of left neighbours in the set `inner`
        // Note: It is important that vertices _without_ any trace in `inner`, .e.g
        //       vertices that are themselves in `inner` but have no neighbours there,
        //       are included in `left_traces` with an empty trace. This will correct the 
        //       overall count for the empty trace in the final result.
        let mut left_traces:VertexMap<Vec<u32>> = Default::default();
        for y in left_neighbours {
            let mut N = left_traces.entry(y).or_default();            
            for x in inner.iter() {
                if graph.adjacent(x, &y) {
                    N.push(*x); // We take care of duplicates below
                }
            }
        }

        // println!("Left traces {left_traces:?}");

        for (x, mut N) in left_traces.into_iter() {
            // Prepare trace
            N.sort_unstable();
            N.dedup();
            N.sort_by_key(|y| graph.index_of(y));

            // Only count trace if the source vertices lies outside of `inner`
            if !inner.contains(&x) {
                *traces.entry(N.clone()).or_default() += 1;
            }

            // Correct count of `traces` by removing prefixes. To that end, we first
            // need to determine where x lies in relation to N
            let ix = graph.index_of(&x);
            let part = N.partition_point(|y| graph.index_of(y) < ix);

            let Nprefix = &N[..part]; 
            *traces.get_mut(Nprefix).unwrap() -= 1;
        }
        // println!("Final traces {traces:?}");

        traces
    }

    pub fn is_shattered<V,I>(&self, inner: I, k:u32, graph:&DegenGraph) -> bool where V: Borrow<u32>, I:IntoIterator<Item=V> {
        let mut inner = inner.into_iter().map(|u| *u.borrow()).collect_vec();
        inner.sort_by_key(|x| graph.index_of(x));

        let traces = self.compute_traces(&inner, graph);

        assert!(k >= inner.len() as u32);
        let multiplicity = 2usize.pow(k - inner.len() as u32);
        let num_neighbourhoods = 2usize.pow(inner.len() as u32);

        if traces.len() < num_neighbourhoods {
            return false
        }

        for (_N, count) in traces.into_iter() {
            if count < multiplicity {
                return false;
            }
        }

        true
    }
}


#[cfg(test)]
mod  tests {
    use crate::datastructures::SReachTraceOracle;

    use super::*;    
    use crate::{editgraph::EditGraph, graph::MutableGraph, io::*};
    use rand::prelude::*;
    use std::collections::BTreeSet;


    #[test]
    fn shattered_test_DNC () {
        let mut graph = EditGraph::from_gzipped("resources/DNC-emails.txt.gz").expect("File not found.");
        graph.remove_loops();

        let degen = DegenGraph::from_graph(&graph);
        let sreach_oracle = SReachTraceOracle::for_graph(&degen);

        // [772, 1734, 1042]
        println!("[772, 1734, 1042] ? {}", sreach_oracle.is_shattered([772, 1734, 1042], 3, &degen));
        println!();
        assert!(sreach_oracle.is_shattered([772, 1734, 1042], 3, &degen));

        // [472, 1042, 1313, 1811]
        assert!(sreach_oracle.is_shattered([472, 1042, 1313, 1811], 4, &degen));
        
        // [379, 772, 1248, 1614, 1635]
        assert!(sreach_oracle.is_shattered([379, 772, 1248, 1614, 1635], 5, &degen));
    }

    #[test]
    fn sreach_traces () {
        let mut graph = EditGraph::new();

        graph.add_vertices(vec![1,2,3,4,5,6,7,8,9,10,11].iter());
        graph.add_edges(vec![(1,4),(2,4),(3,4),(1,5),(2,5),(3,5),(1,6),(2,6),(2,7),(3,7),(1,8),(3,8),(1,9),(2,10),(3,11)].into_iter());

        // graph.add_edge(&1,&2);
        // graph.add_edge(&2,&3);

        let mut order = vec![1,2,3,4,5,6,7,8,9,10,11];

        use rand::thread_rng;
        order.shuffle(&mut thread_rng());

        let degen = DegenGraph::with_ordering(&graph, order);
        let sreach_oracle = SReachTraceOracle::for_graph(&degen);

        let mut target = vec![1,2,3];
        target.sort_by_key(|x| degen.index_of(x));
        let traces = helper_compute_traces(&target, &degen);
        let traces_sreach = sreach_oracle.compute_traces(&target, &degen);

        // The sreach_traces might contain 0-count sets which make the test fail
        let traces_sreach:FxHashMap<_,_> = traces_sreach.into_iter().filter(|(k,v)| *v > 0 ).collect();

        println!("traces: {traces:?}");
        println!("traces: {traces_sreach:?}");

        assert_eq!(traces_sreach, traces);
    }


    #[test]
    fn sreach_traces_2 () {
        let mut graph = EditGraph::new();

        // 1 -- 2 -- 3
        graph.add_vertices(vec![1,2,3].iter());
        graph.add_edge(&1, &2);
        graph.add_edge(&2, &3);

        let degen = DegenGraph::with_ordering(&graph, vec![1,2,3]);
        let sreach_oracle = SReachTraceOracle::for_graph(&degen);

        let mut target = vec![1,2,3];
        let traces = helper_compute_traces(&target, &degen);

        let traces_sreach = sreach_oracle.compute_traces(&target, &degen);

        // The sreach_traces might contain 0-count sets which make the test fail
        let traces_sreach:FxHashMap<_,_> = traces_sreach.into_iter().filter(|(k,v)| *v > 0 ).collect();

        println!("traces: {traces:?}");
        println!("traces: {traces_sreach:?}");

        assert_eq!(traces_sreach, traces);        
    }

    #[test]
    fn traces_test_Yeast () {
        let mut graph = EditGraph::from_gzipped("resources/Yeast.txt.gz").expect("File not found.");
        graph.remove_loops();

        let degen = DegenGraph::from_graph(&graph);
        let sreach_oracle = SReachTraceOracle::for_graph(&degen);

        let mut target = vec![379, 772, 1248, 1614, 1635];
        target.sort_by_key(|x| degen.index_of(x));

        let traces = helper_compute_traces(&target, &degen);
        let traces_sreach = sreach_oracle.compute_traces(&target, &degen);
        // println!("traces: {traces:?}");
        // println!("traces: {traces_sreach:?}");

        for (X, &count) in traces.iter() {
            assert!(traces_sreach.contains_key(X));
            assert!(X.is_sorted_by_key(|x| degen.index_of(x)));
            let sreach_count = traces_sreach[X];
            if count != sreach_count {
                println!("Mismatch {X:?} {sreach_count} (should be {count})")
            }
            // assert_eq!(count, traces_sreach[X]);
        }

        assert_eq!(traces_sreach, traces);        
    }

    fn helper_compute_traces(target:&Vec<Vertex>, degen: &DegenGraph) -> FxHashMap<Vec<Vertex>, usize> {
        let mut traces:FxHashMap<Vec<Vertex>, usize> = FxHashMap::default();
        let target:VertexSet = target.iter().cloned().collect();
        for x in degen.vertices() {
            if target.contains(x) {
                continue
            }
            let N:VertexSet = degen.neighbours(x).cloned().collect();
            let mut S:Vec<Vertex> = N.intersection(&target).cloned().collect();
            S.sort_by_key(|x| degen.index_of(x));
            *traces.entry(S).or_default() += 1;
        }
        traces
    }
}