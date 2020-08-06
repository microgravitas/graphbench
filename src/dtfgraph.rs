use fnv::{FnvHashMap, FnvHashSet};

use itertools::Itertools;


use crate::graph::{Graph, Vertex, VertexSet, Arc, EdgeSet};
use crate::iterators::VertexIterator;

use std::cmp::{max, min};

struct DTFGraph {
    layers: Vec<DTFLayer>,
}

struct DTFLayer {
    in_arcs: FnvHashMap<Vertex, VertexSet>,
    in_degs: FnvHashMap<Vertex, u32>,
    out_degs: FnvHashMap<Vertex, u32>,
    m: u64,
}

impl DTFLayer {
    pub fn new() -> DTFLayer {
        DTFLayer {
            in_arcs: FnvHashMap::default(),
            in_degs: FnvHashMap::default(),
            out_degs: FnvHashMap::default(),
            m: 0,
        }
    }
}

impl DTFGraph {
    pub fn new() -> DTFGraph {
        DTFGraph { layers: vec![DTFLayer::new()] }
    }

    pub fn orient(G: &Graph) -> DTFGraph {
        let mut H = DTFGraph::new();

        /*
            This index function defines buckets of exponentially increasing
            size, but all values below `small` (here 32) are put in their own
            buckets.
        */
        fn calc_index(i: u32) -> u32 {
            let small = 2_i32.pow(5);
            min(i, small as u32) as u32
                + (max(0, (i as i32) - small + 1) as u32)
                    .next_power_of_two()
                    .trailing_zeros()
        }

        let mut deg_dict = FnvHashMap::<Vertex, u32>::default();
        let mut buckets = FnvHashMap::<Vertex, VertexSet>::default();

        for v in G.vertices() {
            H.add_vertex(*v);
            let d = G.degree(*v);
            let bucket = buckets
                .entry(calc_index(d))
                .or_insert_with(|| VertexSet::default());
            bucket.insert(*v);
        }

        let mut seen = FnvHashSet::<Vertex>::default();

        for _ in 0..G.num_vertices() {
            // Find non-empty bucket. If this loop executes, we
            // know that |G| > 0 so a non-empty bucket must exist.
            let mut d = 0;
            while buckets.contains_key(&d) && buckets[&d].len() == 0 {
                d += 1
            }

            if !buckets.contains_key(&d) {
                break;
            }

            let v = buckets[&d].iter().cloned().next().unwrap();
            buckets.remove(&v);

            for u in G.neighbours(v) {
                if seen.contains(u) {
                    continue;
                }

                // Update bucket
                let du = deg_dict[u];
                let old_index = calc_index(du);
                let new_index = calc_index(du - 1);

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

    pub fn vertices(&self) -> VertexIterator {
        self.layers.get(0).unwrap().in_arcs.keys()
    }

    pub fn add_vertex(&mut self, u: Vertex) {
        for layer in self.layers.iter_mut() {
            if !layer.in_arcs.contains_key(&u) {
                layer.in_arcs.insert(u, FnvHashSet::default());
                layer.in_degs.insert(u, 0);
                layer.out_degs.insert(u, 0);
            }
        }
    }

    fn reserve_layers(&mut self, w:usize) {
        while self.layers.len() < w {
            self.layers.push(DTFLayer::new())
        }
    }

    pub fn add_arc(&mut self, u:Vertex, v:Vertex, w:usize) {
        self.reserve_layers(w);

        let mut layer = self.layers.get_mut(w-1).unwrap();
        let inserted = layer.in_arcs.entry(v)
            .or_insert_with(|| VertexSet::default())
            .insert(u);
        if inserted {
            *layer.out_degs.get_mut(&u).unwrap() += 1;
            *layer.in_degs.get_mut(&v).unwrap() += 1;
            layer.m += 1;
        }
    }

    pub fn adjacent(&self, u:Vertex, v:Vertex) -> bool {
        // Returns whether there is an arc uv or vu
        self.has_arc(u,v) || self.has_arc(v,u)
    }

    pub fn has_arc(&self, u:Vertex, v:Vertex) -> bool {
        for layer in self.layers.iter() {
            if layer.in_arcs.get(&u).unwrap().contains(&v) {
                return true
            }
        }
        false
    }

    pub fn in_neighbours(&self, u:Vertex) -> VertexSet {
        // Returns all in-neighbours of a vertex
        let mut res = VertexSet::default();
        for layer in self.layers.iter() {
            res.extend(layer.in_arcs.get(&u).unwrap().iter());
        }
        res
    }

    pub fn dtf_step(&mut self, depth:usize) {
        let mut fGraph = Graph::new();
        let mut tArcs = FnvHashSet::<Arc>::default();

        self.reserve_layers(depth+1);
        for u in self.vertices() {
            for x in self._trans_tails(*u, depth) {
                tArcs.insert((*u, x));
            }
            if depth == 2 {
                for (x,y) in self._frat_pairs2(*u) {
                    fGraph.add_edge(x,y);
                }
            } else {
                for (x,y) in self._frat_pairs(*u, depth) {
                    fGraph.add_edge(x,y);
                }
            }
            // ..
        }
    }


        // def dtf_step(self, dist, ldo_depth=2):
        //     fratGraph = Graph()
        //     newTrans = {}
        //
        //     self._reserve_weight(dist+1)
        //     for v in self.nodes:
        //         for x in self._trans_tails(v, dist):
        //             newTrans[(x,v)] = dist
        //         if dist == 2:
        //             for x,y in self._frat_pairs2(v):
        //                 fratGraph.add_edge(x,y)
        //         else:
        //             for x,y in self._frat_pairs(v, dist):
        //                 fratGraph.add_edge(x,y)
        //
        //     for (s, t) in newTrans:
        //         self._add_arc(s, t, dist)
        //         fratGraph.remove_edge(s,t)
        //
        //     fratDigraph = fratGraph.deep_ldo(ldo_depth)
        //
        //     for s,t in fratDigraph.arcs():
                // self._add_arc(s,t,dist)

    fn _trans_tails(&self, u:Vertex, depth:usize) -> VertexSet {
        // Returns vertices y \in N^{--}(u) \setminus N^-(u) such that
        // the weights of arcs (y,x), (x,u) add up to 'weight'

        let mut cands = VertexSet::default();

        for wy in 1..depth {
            let wx = depth - wy;
            let layerY = &self.layers[wy-1];
            let layerX = &self.layers[wx-1];

            // Find x --wx--> y --wy--> u
            for y in layerY.in_arcs.get(&u).unwrap().iter() {
                cands.extend(layerX.in_arcs.get(&y).unwrap().iter())
            }
        }

        // We finally have to remove all candidates x which
        // already have an arc to u,  x --?--> u.
        let mut Nu:VertexSet = self.in_neighbours(u);
        Nu.insert(u);

        cands.difference(&Nu).cloned().collect()
    }

    fn _frat_pairs(&self, u:Vertex, depth:usize) -> EdgeSet {
        assert_ne!(depth, 2);
        // Since depth != 2, we have that 1 != depth-1 and therefore
        // we can easily nest the iteration below without checking for equality.

        let mut res = EdgeSet::default();

        for wx in [1_usize,depth-1].iter() {
            let wy = depth-wx;
            let layerX = &self.layers[*wx-1];
            let layerY = &self.layers[wy-1];

            for x in layerX.in_arcs.get(&u).unwrap().iter() {
                for y in layerY.in_arcs.get(&u).unwrap().iter() {
                    if !self.adjacent(*x,*y) {
                        res.insert((*x,*y));
                    }
                }
            }
        }
        res
    }

    fn _frat_pairs2(&self, u:Vertex) -> EdgeSet {
        let mut res = EdgeSet::default();
        // ...
        for (x,y) in self.layers[0].in_arcs[&u].iter().tuple_combinations() {
            if !self.adjacent(*x,*y) {
                res.insert((*x,*y));
            }
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn convert() {
        let mut G = Graph::new();
        G.add_edge(0, 1);
        G.add_edge(0, 2);
        G.add_edge(0, 3);
        G.add_edge(0, 4);
        G.add_edge(0, 5);

        let mut H = DTFGraph::orient(&G);
        
    }
}
