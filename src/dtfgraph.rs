use fnv::{FnvHashMap, FnvHashSet};

use itertools::Itertools;

use crate::graph::Graph;
use crate::editgraph::{EditGraph, Vertex, VertexSet, Arc, EdgeSet};
use crate::iterators::VertexIterator;
use crate::iterators::DTFArcIterator;
use crate::iterators::DTFNIterator;

use std::cmp::{max, min};

pub type InArcIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;
pub type DTFVertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, DTFNode>;

pub struct DTFGraph {
    nodes: FnvHashMap<Vertex, DTFNode>,
    depth: usize,
    m: u64
}

pub struct DTFNode {
    in_arcs: Vec<VertexSet>,
    in_degs: Vec<u32>,
    out_degs: Vec<u32>
}

impl DTFNode {
    pub fn new(depth: usize) -> DTFNode {
        let mut in_arcs = Vec::new();
        let in_degs = vec![0; depth];
        let out_degs =  vec![0; depth];
        for _ in 0..depth {
            in_arcs.push(VertexSet::default());
        }
        DTFNode { in_arcs, in_degs, out_degs }
    }

    pub fn reserve_depth(&mut self, depth: usize) {
        while self.in_arcs.len() < depth {
            self.in_arcs.push(VertexSet::default());
            self.in_degs.push(0);
            self.out_degs.push(0);
        }
    }

    pub fn has_in_neighbour(&self, v:&Vertex) -> bool {
        for N in &self.in_arcs {
            if N.contains(&v) {
                return true
            }
        }
        false
    }

    pub fn get_arc_weight_from(&self, v:&Vertex) -> Option<u32> {
        for (i,N) in self.in_arcs.iter().enumerate() {
            if N.contains(&v) {
                return Some((i+1) as u32)
            }
        }
        return None
    }

    pub fn in_neighbours(&self) -> VertexSet {
        let mut res = VertexSet::default();
        for N in &self.in_arcs {
            res.extend(N.iter());
        }
        res
    }

    pub fn in_neighbours_at(&self, depth:usize) -> InArcIterator {
        self.in_arcs.get(depth-1).unwrap().iter()
    }

    pub fn in_degree(&self) -> u32 {
        self.in_degs.iter().sum1().unwrap()
    }

    pub fn out_degree(&self) -> u32 {
        self.out_degs.iter().sum1().unwrap()
    }

    pub fn degree(&self) -> u32 {
        self.out_degree()+self.in_degree()
    }
}

impl DTFGraph {
    pub fn new() -> DTFGraph {
        DTFGraph { nodes: FnvHashMap::default(), depth: 1, m: 0 }
    }

    pub fn num_vertices(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_arcs(&self) -> usize {
        self.m as usize
    }

    pub fn vertices(&self) -> DTFVertexIterator {
        self.nodes.keys()
    }

    pub fn orient(G: &EditGraph) -> DTFGraph {
        let mut H = DTFGraph::new();

        /*
            This index function defines buckets of exponentially increasing
            size, but all values below `small` (here 32) are put in their own
            buckets.
        */
        fn calc_index(i: usize) -> usize {
            let small = 2_i32.pow(5);
            min(i, small as usize)
                + (max(0, (i as i32) - small + 1) as u32)
                    .next_power_of_two()
                    .trailing_zeros() as usize
        }

        let mut deg_dict = FnvHashMap::<Vertex, usize>::default();
        let mut buckets = FnvHashMap::<i32, VertexSet>::default();

        for v in G.vertices() {
            H.add_vertex(*v);
            let d = G.degree(v);
            deg_dict.insert(*v, d);
            buckets
                .entry(calc_index(d) as i32)
                .or_insert_with(|| VertexSet::default())
                .insert(*v);
        }

        let mut seen = FnvHashSet::<Vertex>::default();

        for _ in 0..G.num_vertices() {
            // Find non-empty bucket. If this loop executes, we
            // know that |G| > 0 so a non-empty bucket must exist.
            let mut d = 0;
            while !buckets.contains_key(&d) || buckets[&d].len() == 0 {
                d += 1
            }

            if !buckets.contains_key(&d) {
                break;
            }

            let v = buckets[&d].iter().next().unwrap().clone();
            buckets.get_mut(&d).unwrap().remove(&v);

            for u in G.neighbours(&v) {
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
                        .or_insert_with(|| VertexSet::default())
                        .insert(*u);
                }

                // Updated degree
                deg_dict.entry(*u).and_modify(|e| *e -= 1);

                // Orient edge towards v
                H.add_arc(*u, v, 1);
            }
            seen.insert(v);
        }

        H
    }

    pub fn add_vertex(&mut self, u: Vertex) {
        let d = self.depth;
        self.nodes.entry(u).or_insert_with(||  DTFNode::new(d));
    }

    fn reserve_depth(&mut self, depth:usize) {
        if self.depth < depth {
            for (_, node) in self.nodes.iter_mut() {
                node.reserve_depth(depth);
            }
        }
        self.depth = depth;
    }

    pub fn add_arc(&mut self, u:Vertex, v:Vertex, w:usize) {
        self.reserve_depth(w);

        let d = self.depth;
        let inserted = {
            let nodeV = self.nodes.entry(v).or_insert_with(||  DTFNode::new(d));
            if nodeV.in_arcs.get_mut(w-1).unwrap().insert(u) {
                nodeV.in_degs[w-1] += 1;
                true
            } else {
                false
            }
        };
        if inserted {
            let nodeU = self.nodes.entry(u).or_insert_with(||  DTFNode::new(d));
            nodeU.out_degs[w-1] += 1;
            self.m += 1
        }
    }

    pub fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        // Returns whether there is an arc uv or vu
        self.has_arc(u,v) || self.has_arc(v,u)
    }

    pub fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool {
        if !self.nodes.contains_key(&v) {
            return false
        }
        self.nodes.get(&v).unwrap().has_in_neighbour(u)
    }

    pub fn get_arc_weight(&self, u:&Vertex, v:&Vertex) -> Option<u32> {
        self.nodes.get(&v).unwrap().get_arc_weight_from(u)
    }

    pub fn arcs_at(&self, depth:usize) -> DTFArcIterator {
        DTFArcIterator::new(self, depth)
    }

    pub fn in_neighbourhoods_iter(&self, depth:usize) -> DTFNIterator {
        DTFNIterator::new(self, depth)
    }

    pub fn in_degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().in_degree()
    }

    pub fn out_degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().out_degree()
    }

    pub fn degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().degree()
    }

    pub fn in_neighbours(&self, u:&Vertex) -> VertexSet {
        self.nodes.get(&u).unwrap().in_neighbours()
    }

    pub fn in_neighbours_at(&self, u:&Vertex, depth:usize) -> InArcIterator {
        self.nodes.get(&u).unwrap().in_neighbours_at(depth)
    }

    pub fn in_neighbours_distance(&self, u:&Vertex) -> FnvHashMap<Vertex, u32> {
        let mut res:FnvHashMap<Vertex, u32> = FnvHashMap::default();
        for d in 1..(self.depth+1) {
            for v in self.in_neighbours_at(&u, d) {
                res.insert(*v, d as u32);
            }
        }
        res
    }

    pub fn small_distance(&self, u:&Vertex, v:&Vertex) -> Option<u32> {
        let mut dist = std::u32::MAX;

        match self.get_arc_weight(u, v) {
            Some(i) => {dist = i},
            None => {}
        }

        match self.get_arc_weight(v, u) {
            Some(i) => {dist = i},
            None => {}
        }

        let Nv = self.in_neighbours_distance(v);
        for d in 1..(self.depth+1) {
            for x in self.in_neighbours_at(u, d) {
                if Nv.contains_key(x) {
                    dist = min(dist, Nv[&x]+d as u32);
                }
            }
        }

        if dist == std::u32::MAX {
            return None
        }
        Some(dist)
    }

    pub fn augment(&mut self, depth:usize) {
        while self.depth < depth {
            let mut fGraph = EditGraph::new();
            let mut tArcs = FnvHashSet::<Arc>::default();

            // This updates self.depth!
            self.reserve_depth(self.depth+1);

            // Collect fraternal edges and transitive arcs
            for u in self.vertices() {
                for x in self._trans_tails(u, self.depth) {
                    tArcs.insert((x, *u));
                }
                if self.depth == 2 {
                    for (x,y) in self._frat_pairs2(u) {
                        fGraph.add_edge(&x,&y);
                    }
                } else {
                    for (x,y) in self._frat_pairs(u, self.depth) {
                        fGraph.add_edge(&x,&y);
                    }
                }
            }

            for (s, t) in &tArcs {
                self.add_arc(*s, *t, self.depth);
                fGraph.remove_edge(s, t);
            }

            // TODO: implement depth-2 ldo
            // fratDigraph = fratGraph.deep_ldo(ldo_depth)
            let fratDigraph = DTFGraph::orient(&fGraph);

            for (s,t) in fratDigraph.arcs_at(1) {
                self.add_arc(s, t, self.depth);
            }
        }
    }

    fn _trans_tails(&self, u:&Vertex, depth:usize) -> VertexSet {
        // Returns vertices y \in N^{--}(u) \setminus N^-(u) such that
        // the weights of arcs (y,x), (x,u) add up to 'weight'

        let mut cands = VertexSet::default();

        for wy in 1..depth {
            let wx = depth - wy;

            for y in self.in_neighbours_at(u, wy) {
                cands.extend(self.in_neighbours_at(y, wx));
            }
        }

        // We finally have to remove all candidates x which
        // already have an arc to u,  x --?--> u.
        let mut Nu:VertexSet = self.in_neighbours(u);
        Nu.insert(*u);

        cands.difference(&Nu).cloned().collect()
    }

    fn _frat_pairs(&self, u:&Vertex, depth:usize) -> EdgeSet {
        assert_ne!(depth, 2);
        // Since depth != 2, we have that 1 != depth-1 and therefore
        // we can easily nest the iteration below without checking for equality.

        let mut res = EdgeSet::default();

        for wx in [1_usize,depth-1].iter() {
            let wy = depth-wx;

            for x in self.in_neighbours_at(u, *wx) {
                for y in self.in_neighbours_at(u, wy) {
                    if !self.adjacent(x, y) {
                        res.insert((*x,*y));
                    }
                }
            }
        }
        res
    }

    fn _frat_pairs2(&self, u:&Vertex) -> EdgeSet {
        let mut res = EdgeSet::default();

        let N = self.in_neighbours_at(u, 1);

        // This is the same as _frat_pairs, but specifically for depth 2
        for (x,y) in N.tuple_combinations() {
            if !self.adjacent(x, y) {
                res.insert((*x,*y));
            }
        }
        res
    }

    fn domset(&mut self, radius:u32) -> VertexSet {
        self.augment(radius as usize);

        let mut domset = VertexSet::default();
        let mut dom_distance = FnvHashMap::<Vertex, u32>::default();
        let mut dom_counter = FnvHashMap::<Vertex, u32>::default();

        let cutoff = (2*radius).pow(2);
        let n = self.num_vertices() as i64;

        // Sort by _decreasing_ in-degree, tie-break by
        // total degree.
        let order:Vec<Vertex> = self.vertices()
                .cloned()
                .sorted_by_key(|u| -(self.in_degree(u) as i64)*n - (self.degree(u) as i64))
                .collect();
        let undominated = radius+1;

        for v in order.iter() {
            dom_distance.insert(*v, undominated);
            dom_counter.insert(*v, 0);
        }

        for v in order {
            // Update domination distance of v via its in-neighbours
            for r in 1..(radius+1) {
                for u in self.in_neighbours_at(&v, r as usize) {
                    *dom_distance.get_mut(&v).unwrap() = min(dom_distance[&v],  r+dom_distance[u]);
                }
            }

            // If v is a already dominated we have nothing else to do
            if dom_distance[&v] <= radius {
                continue
            }

            // Otherwise, we add v to the dominating set
            domset.insert(v);
            dom_distance.insert(v, 0);

            // Update dominationg distance for v's in-neighbours
            for r in 1..(radius+1) {
                for u in self.in_neighbours_at(&v, r as usize) {
                    *dom_counter.get_mut(u).unwrap() += 1;
                    *dom_distance.get_mut(u).unwrap() = min(dom_distance[u], r);

                    // If a vertex has been an in-neigbhour of a domination node for
                    // too many time, we include it in the domset.
                    if dom_counter[&u] > cutoff && !domset.contains(&u) {
                        domset.insert(*u);
                        dom_distance.insert(*u, 0);

                        for rx in 1..(radius+1) {
                            for x in self.in_neighbours_at(u,  rx as usize) {
                                *dom_distance.get_mut(&x).unwrap() += min(dom_distance[&x],  rx);
                            }
                        }
                    }
                }
            }
        }

        domset
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parse;

    // #[test]
    fn iteration() {
        let mut H = DTFGraph::new();
        H.add_arc(1, 0, 1);
        H.add_arc(2, 0, 1);
        H.add_arc(3, 0, 1);

        assert_eq!(H.in_neighbours_at(&0, 1).cloned().collect::<FnvHashSet<Vertex>>(),
                   [1,2,3].iter().cloned().collect::<FnvHashSet<Vertex>>());
    }

    #[test]
    fn domset() {
        let G = EditGraph::from_txt("resources/karate.txt").unwrap();

        let mut H = DTFGraph::orient(&G);

        for r in 1..5 {
            let D = H.domset(r);
            println!("Found Karate {}-domset of size {}", r, D.len());
            assert_eq!(G.r_neighbourhood(D.iter(), r).len(), G.num_vertices());
        }
    }

    // #[test]
    fn distance() {
        let mut G = EditGraph::new();
        G.add_edge(&0, &1);
        G.add_edge(&1, &2);
        G.add_edge(&2, &3);
        G.add_edge(&3, &4);

        let mut H = DTFGraph::orient(&G);

        H.augment(4);
        assert_eq!(H.small_distance(&0,&1), Some(1));
        assert_eq!(H.small_distance(&1,&2), Some(1));
        assert_eq!(H.small_distance(&2,&3), Some(1));
        assert_eq!(H.small_distance(&3,&4), Some(1));

        assert_eq!(H.small_distance(&0,&2), Some(2));
        assert_eq!(H.small_distance(&1,&3), Some(2));
        assert_eq!(H.small_distance(&2,&4), Some(2));

        assert_eq!(H.small_distance(&0,&3), Some(3));
        assert_eq!(H.small_distance(&1,&4), Some(3));

        assert_eq!(H.small_distance(&0,&4), Some(4));
    }
}
