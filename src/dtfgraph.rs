use crate::algorithms::GraphAlgorithms;
use fxhash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

use itertools::Itertools;

use crate::graph::*;
use crate::iterators::*;
use crate::editgraph::EditGraph;

use std::cmp::{max, min};

pub type InArcIterator<'a> = std::collections::hash_set::Iter<'a, Vertex>;
pub type DTFVertexIterator<'a> = std::collections::hash_map::Keys<'a, Vertex, DTFNode>;

pub struct DTFGraph {
    nodes: FxHashMap<Vertex, DTFNode>,
    depth: usize,
    ms: Vec<usize>
}

pub struct DTFLayer<'a> {
    graph: &'a DTFGraph,
    depth: usize
}

impl<'a> DTFLayer<'a> {
    pub fn new(graph: &'a DTFGraph, depth: usize) -> Self {
        DTFLayer{ graph, depth }
    }
}

impl<'a> Graph for DTFLayer<'a> {
    fn num_vertices(&self) -> usize {
        self.graph.num_vertices()
    }

    fn num_edges(&self) -> usize {
        self.graph.num_arcs_at_depth(self.depth)
    }

    fn contains(&self, u:&Vertex) -> bool {
        self.graph.contains(u)
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        self.graph.adjacent(u, v)
    }

    fn degree(&self, u:&Vertex) -> u32 {
        self.graph.degree(u)
    }

    fn vertices<'b>(&'b self) -> Box<dyn Iterator<Item=&Vertex> + 'b> {
        self.graph.vertices()
    }

    fn neighbours<'b>(&'b self, _:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'b> {
        // DTFGraph does not implement this method for efficiency reasons.
        unimplemented!("DTFGraph does not implement Graph::neighbours");
    }
}

impl<'b> Digraph for DTFLayer<'b> {
  
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool {
        self.graph.has_arc_at(u, v, self.depth)
    }

    fn in_degree(&self, u:&Vertex) -> u32 {
        self.graph.in_degree(u)
    }

    fn out_degree(&self, u:&Vertex) -> u32 {
        self.graph.out_degree(u)
    }

    fn neighbours<'a>(&'a self, _:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        // DTFGraph does not implement this method for efficiency reasons.
        unimplemented!("DTFGraph does not implement Graph::neighbours");
    }

    fn out_neighbours<'a>(&'a self, _:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        // DTFGraph does not implement this method for efficiency reasons.
        unimplemented!("DTFGraph does not implement Graph::out_neighbours");
    }

    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        self.graph.in_neighbours_at(u, self.depth)
    }
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

    pub fn has_in_neighbour_at(&self, v:&Vertex, depth:usize) -> bool {
        match self.in_arcs.get(depth-1) {
            Some(X) => X.contains(v),
            None => false
        }
    }

    pub fn get_arc_depth_from(&self, v:&Vertex) -> Option<u32> {
        for (i,N) in self.in_arcs.iter().enumerate() {
            if N.contains(&v) {
                return Some((i+1) as u32)
            }
        }
        None
    }

    pub fn in_neighbours(&self) -> Box<dyn Iterator<Item=&Vertex> + '_> {
        Box::new(self.in_arcs.iter().flat_map(|N| N.iter()))
    }

    pub fn in_neighbours_at(&self, depth:usize) ->  Box<dyn Iterator<Item=&Vertex> + '_> {
        Box::new(self.in_arcs.get(depth-1).unwrap().iter())
    }

    pub fn in_degree(&self) -> u32 {
        self.in_degs.iter().sum1::<u32>().unwrap() 
    }

    pub fn out_degree(&self) -> u32 {
        self.out_degs.iter().sum1::<u32>().unwrap() 
    }

    pub fn degree(&self) -> u32 {
        self.out_degree()+self.in_degree()
    }
}


impl Graph for DTFGraph {
    fn num_vertices(&self) -> usize {
        self.nodes.len()
    }

    fn num_edges(&self) -> usize {
        self.ms[0]
    }

    fn contains(&self, u:&Vertex) -> bool {
        self.nodes.contains_key(u)
    }

    fn vertices<'a>(&'a self) ->  Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.nodes.keys())
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        // Returns whether there is an arc uv or vu
        self.has_arc(u,v) || self.has_arc(v,u)
    }

    fn neighbours<'a>(&'a self, _:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        unimplemented!("DTFGraph does not implement DiGraph::neighbours");
    }

    fn degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().degree()
    }
}

impl Digraph for DTFGraph {
    fn has_arc(&self, u:&Vertex, v:&Vertex) -> bool {
        if !self.nodes.contains_key(&v) {
            return false
        }
        self.nodes.get(&v).unwrap().has_in_neighbour(u)
    }

    fn in_degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().in_degree()
    }

    fn out_degree(&self, u:&Vertex) -> u32 {
        self.nodes.get(&u).unwrap().out_degree()
    }

    fn out_neighbours<'a>(&'a self, _:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>  {
        unimplemented!("DTFGraph does not implement DiGraph::out_neighbours");
    }

    fn in_neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a>  {
        self.nodes.get(&u).unwrap().in_neighbours()
    }
}

impl DTFGraph {
    fn new() -> DTFGraph {
        DTFGraph { nodes: FxHashMap::default(), depth: 1, ms: vec![0] }
    }

    pub fn get_depth(&self) -> usize { 
        self.depth
    }

    fn with_capacity(n_guess:usize) -> DTFGraph {
        DTFGraph {
            nodes: FxHashMap::with_capacity_and_hasher(n_guess, Default::default()),
            depth: 1,
            ms: vec![0]
        }
    }

    pub fn to_undirected<G>(&self) -> G where G: MutableGraph {
        let mut res = G::new();
        for u in self.vertices() {
            res.add_vertex(u);
        }

        for (u,v) in self.arcs() {
            res.add_edge(&u, &v);
        }

        res
    }

    pub fn layer(&mut self, depth:usize) -> DTFLayer {
        self.reserve_depth(depth);
        DTFLayer{ graph: self, depth }
    }

    pub fn num_arcs_at_depth(&self, depth: usize) -> usize {
        if depth > self.depth {
            0
        } else {
            self.ms[depth-1]
        }
    }

    pub fn add_vertex(&mut self, u: Vertex) {
        let d = self.depth;
        self.nodes.entry(u).or_insert_with(||  DTFNode::new(d));
    }

    pub fn orient_deep<G>(graph: &G, depth:usize) -> DTFGraph where G: Graph {
        let mut augg = DTFGraph::orient(graph);
        if depth <= 1 {
            return augg;
        };

        augg.augment(depth, 0); // FratDepth must be <=1, otherwise we recurse endlessly

        let reoriented = DTFGraph::orient(&augg.to_undirected::<EditGraph>());
        reoriented.edge_subgraph(graph.edges())
    }

    pub fn edge_subgraph<I>(&'_ self, it: I ) -> DTFGraph where I: Iterator<Item=(Vertex,Vertex)> {
        let mut res = DTFGraph::new();
        for v in self.vertices() {
            res.add_vertex(*v);
        }

        for (u,v) in it {
            if self.has_arc(&u, &v) {
                res.add_arc(&u, &v, 1);
            }
            if self.has_arc(&v, &u) {
                res.add_arc(&v, &u, 1);
            }
        }

        res
    }

    pub fn orient<G>(graph: &G) -> DTFGraph where G: Graph {
        let mut H = DTFGraph::with_capacity(graph.num_vertices());

        let (ord, _) = graph.degeneracy();
        let indices:FxHashMap<Vertex, usize> = ord.iter().enumerate()
                                                .map(|(i,u)| (*u,i)).collect();

        for (v,N) in graph.neighbourhoods() {
            let iv = indices[&v];
            H.add_vertex(v);
            for u in N {
                let iu = indices[&u];
                if iu < iv {
                    H.add_arc(&u, &v, 1);
                } else {
                    H.add_arc(&v, &u, 1);
                }
            }
        }

        H
    }

    fn reserve_depth(&mut self, depth:usize) {
        if self.depth < depth {
            for (_, node) in self.nodes.iter_mut() {
                node.reserve_depth(depth);
            }
            self.ms.push(0);
        }
        self.depth = depth;
    }

    pub fn add_arc(&mut self, u:&Vertex, v:&Vertex, depth:usize) {
        self.reserve_depth(depth);

        let d = self.depth;
        let inserted = {
            let nodeV = self.nodes.entry(*v).or_insert_with(||  DTFNode::new(d));
            if nodeV.in_arcs.get_mut(depth-1).unwrap().insert(*u) {
                nodeV.in_degs[depth-1] += 1;
                true
            } else {
                false
            }
        };
        if inserted {
            let nodeU = self.nodes.entry(*u).or_insert_with(||  DTFNode::new(d));
            nodeU.out_degs[depth-1] += 1;
            self.ms[depth-1] += 1
        }
    }

    pub fn has_arc_at(&self, u:&Vertex, v:&Vertex, depth:usize) -> bool {
        if !self.nodes.contains_key(&v) {
            return false
        }
        self.nodes.get(&v).unwrap().has_in_neighbour_at(u, depth)
    }

    pub fn get_arc_depth(&self, u:&Vertex, v:&Vertex) -> Option<u32> {
        self.nodes.get(&v).unwrap().get_arc_depth_from(u)
    }

    pub fn arcs(&self) -> DTFArcIterator {
        DTFArcIterator::all_depths(self)
    }

    pub fn arcs_at(&self, depth:usize) -> DTFArcIterator {
        DTFArcIterator::fixed_depth(self, depth)
    }

    pub fn in_neighbourhoods_iter(&self) -> DTFNIterator {
        DTFNIterator::all_depths(self)
    }

    pub fn in_neighbourhoods_iter_at(&self, depth:usize) -> DTFNIterator {
        DTFNIterator::fixed_depth(self, depth)
    }

    pub fn in_neighbours_at<'a>(&'a self, u:&Vertex, depth:usize) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        self.nodes.get(&u).unwrap().in_neighbours_at(depth)
    }

    pub fn in_neighbours_with_weights(&self, u:&Vertex) -> FxHashMap<Vertex, u32> {
        let mut res:FxHashMap<Vertex, u32> = FxHashMap::default();
        for d in 1..(self.depth+1) {
            for v in self.in_neighbours_at(&u, d) {
                res.insert(*v, d as u32);
            }
        }
        res
    }

    pub fn small_distance(&self, u:&Vertex, v:&Vertex) -> Option<u32> {
        let mut dist = std::u32::MAX;

        if let Some(i) = self.get_arc_depth(u, v) {
            dist = i;
        }

        if let Some(i) = self.get_arc_depth(v, u) {
            dist = i;
        }

        let Nv = self.in_neighbours_with_weights(v);
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

    pub fn augment(&mut self, depth:usize, frat_depth:usize) {
        while self.depth < depth {
            let mut fGraph = EditGraph::new();
            let mut tArcs = FxHashSet::<Arc>::default();

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
                self.add_arc(s, t, self.depth);
                fGraph.remove_edge(s, t);
            }

            let fratDigraph = DTFGraph::orient_deep(&fGraph, frat_depth);

            for (s,t) in fratDigraph.arcs_at(1) {
                self.add_arc(&s, &t, self.depth);
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
        // TODO: Implement this using iterators only.
        let mut Nu:VertexSet = self.in_neighbours(u).cloned().collect();
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

        let N:Vec<_> = self.in_neighbours_at(u, 1).collect();

        // This is the same as _frat_pairs, but specifically for depth 2
        for (x,y) in N.into_iter().tuple_combinations() {
            if !self.adjacent(x, y) {
                res.insert((*x,*y));
            }
        }
        res
    }

    pub fn domset(&mut self, radius:u32) -> VertexSet {
        // A fraternal lookahead of 2 seems good accross the board.
        self.augment(radius as usize, 2);

        let mut domset = VertexSet::default();
        let mut dom_distance = FxHashMap::<Vertex, u32>::default();
        let mut dom_counter = FxHashMap::<Vertex, u32>::default();

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

    #[test]
    fn iteration() {
        let mut H = DTFGraph::new();
        H.add_arc(&1, &0, 1);
        H.add_arc(&2, &0, 1);
        H.add_arc(&3, &0, 1);

        assert_eq!(H.in_neighbours_at(&0, 1).cloned().collect::<FxHashSet<Vertex>>(),
                   [1,2,3].iter().cloned().collect::<FxHashSet<Vertex>>());
    }

    #[test]
    fn edge_subgraph() {
        let mut H = DTFGraph::new();
        let mut G = EditGraph::new();

        G.add_edge(&0, &1);
        G.add_edge(&0, &2);
        H.add_arc(&0, &1, 1);
        H.add_arc(&0, &3, 1);

        let HH = H.edge_subgraph(G.edges());
        assert_eq!(HH.num_edges(), 1);
        assert!(HH.has_arc_at(&0, &1, 1));
        assert!(!HH.has_arc_at(&1, &0, 1));
    }

    #[test]
    fn domset() {
        let G = EditGraph::from_txt("resources/karate.txt").unwrap();

        let mut H = DTFGraph::orient(&G);

        for r in 1..5 {
            let D = H.domset(r);
            println!("Found Karate {}-domset of size {}", r, D.len());
            assert_eq!(G.r_neighbourhood(D.iter(), r as usize).len(), G.num_vertices());
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

        H.augment(4, 2);
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
