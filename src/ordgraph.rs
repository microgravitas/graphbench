use std::collections::BTreeSet;
use std::iter;

use crate::algorithms::GraphAlgorithms;
use crate::algorithms::LinearGraphAlgorithms;
use crate::degengraph::DegenGraph;
use crate::degengraph::DegenGraphBuilder;
use crate::reachgraph::ReachGraph;
use crate::reachgraph::ReachGraphBuilder;
use fxhash::{FxHashMap, FxHashSet};

use crate::graph::*;
use crate::iterators::*;

/// Static graph which has a mutable ordering of its vertices. 
/// 
/// The neighbourhood of each vertex $u$ is divided into a *left* neighbourhood,
/// meaning all members of $N(u)$ which come before $u$ in the ordering, and a
/// *right* neighbourhood. For $d$-degenerate graphs we can compute such an ordering 
/// where every left neighbourhood has size at most $d$. 
/// 
/// Further allows the computation of r-weakly and r-strongly reachable sets under the
/// given ordering. This data structure is intended to explore different strategies
/// for computing orderings with small r-weakly/-strongly reachable sets by
/// modifying the odering.
/// 
pub struct OrdGraph {
    indices: VertexMap<usize>,
    nodes: Vec<OrdNode>,
    m: usize
}

/// A vertex alongside its left and right neighbourhood.
pub struct OrdNode {
    v: Vertex,
    left: VertexSet,
    right: VertexSet
}

impl OrdNode {
    /// Returns the union of left and right neighbours.
    fn neighbours<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new( self.left.iter().chain(self.right.iter()) )
    }
}

impl OrdNode {
    fn new(v:&Vertex) -> Self {
        OrdNode{v: *v, left: VertexSet::default(), right: VertexSet::default() }
    }
}


impl OrdGraph {
    /// Creates an ordered graph from `graph` by computing a degeneracy ordering.
    pub fn by_degeneracy<G>(graph: &G) -> OrdGraph where G: Graph {
        let (_, _, ord, _) = graph.degeneracy();
        OrdGraph::with_ordering(graph, ord.iter())
    }
    
    /// Creates an ordered graphs from `graph` using `order`.
    pub fn with_ordering<'a, G, I>(graph: &G, order:I) -> OrdGraph
        where G: Graph, I: Iterator<Item=&'a Vertex>
    {
        let order:Vec<_> = order.collect();
        let indices:VertexMap<_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (*u,i)).collect();
        let mut nodes:Vec<_> = Vec::with_capacity(order.len());

        for v in &order {
            nodes.push(OrdNode::new(v));
            assert!(indices[v] == nodes.len()-1);
        }

        for (u,v) in graph.edges() {
            assert!(indices.contains_key(&u), "Vertex {} not contained in provided ordering", u);
            assert!(indices.contains_key(&v), "Vertex {} not contained in provided ordering", v);
            let iu = indices[&u];
            let iv = indices[&v];
            if iu < iv {
                {nodes.get_mut(indices[&u]).unwrap().right.insert(v); }
                {nodes.get_mut(indices[&v]).unwrap().left.insert(u); }
            } else {
                {nodes.get_mut(indices[&v]).unwrap().right.insert(u); }
                {nodes.get_mut(indices[&u]).unwrap().left.insert(v); }
            }
        }

        OrdGraph {nodes, indices, m: graph.num_edges()}
    }

    pub fn to_degeneracy_graph(&self) -> DegenGraph {
        let mut builder = DegenGraphBuilder::new();

        for u in self.vertices() {
            let L = self.left_neighbours(u);
            builder.append(u, &L);
        }

        builder.build()
    }    

    pub fn to_wreach_graph<const DEPTH: usize>(&self) -> ReachGraph<DEPTH> {
        let mut builder = ReachGraphBuilder::<DEPTH>::new();
        let wreach_sets = self.wreach_sets(DEPTH as u32);

        for u in self.vertices() {
            let W = &wreach_sets[u];
            builder.append(u, W, &self.indices);
        }

        builder.build()
    }

    pub fn to_sreach_graph<const DEPTH: usize>(&self) -> ReachGraph<DEPTH> {
        let mut builder = ReachGraphBuilder::<DEPTH>::new();

        for u in self.vertices() {
            let S = self.sreach_set(u, DEPTH as u32);
            builder.append(u, &S, &self.indices);
        }

        builder.build()
    }

    /// Swaps the positions of `u` and `v` in the ordering.
    pub fn swap(&mut self, u:&Vertex, v:&Vertex) {
        if u == v {
            return;
        }

        let (iu, iv) = match (self.indices.get(u), self.indices.get(v)) {
            (Some(iu), Some(iv)) => (*iu, *iv),
            _ => return
        };

        // Recompute left/right neighbours of u and v for u when moved to iv
        // and for v when moved to iu.
        for (s,old_i,new_i) in [(u,iu,iv), (v,iv,iu)]{
            let mut needs_update:Vec<usize> = Vec::new();            
            {   // Update vertex itself
                let n = &mut self.nodes[old_i];
                let (mut new_left, mut new_right) = (VertexSet::default(), VertexSet::default());
                for x in n.neighbours() {
                    let ix = self.indices.get(x).unwrap();
                    needs_update.push(*ix);
                    if ix < &new_i {
                        new_left.insert(*x);
                    } else {
                        new_right.insert(*x);
                    }
                }
                (n.left, n.right) = (new_left, new_right);
            }
            // Now update neighbours
            for ix in needs_update {
                let n = &mut self.nodes[ix];
                if new_i > ix && n.left.contains(s) {
                    n.left.remove(s);
                    n.right.insert(*s);
                } else if new_i < ix && n.right.contains(s) {
                    n.right.remove(s);
                    n.left.insert(*s);
                }
            }            
        }

        // Finally, swap u and v
        self.indices.insert(*u, iv);
        self.indices.insert(*v, iu);
        self.nodes.swap(iu, iv);
    }

    /// Returns a copy of `u`'s right neighbourhood. 
    pub fn right_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        let iu = self.indices.get(u).unwrap_or_else(|| panic!("Vertex {u} does not exist")); 
        let node_u = &self.nodes[*iu];

        let mut res:Vec<Vertex> = node_u.right.iter().cloned().collect();
        res.sort_by_cached_key(|v| self.indices.get(v).unwrap());
        
        res
    }    
}

impl Graph for OrdGraph {
    fn num_vertices(&self) -> usize {
        self.nodes.len()
    }

    fn num_edges(&self) -> usize {
        self.m
    }

    fn contains(&self, u:&Vertex) -> bool {
        self.indices.contains_key(u)
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.contains(v) || node_u.right.contains(v)
        } else {
            false
        }
    }

    fn degree(&self, u:&Vertex) -> u32 {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            (node_u.left.len() + node_u.right.len()) as u32
        } else {
            0
        }
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        let it = self.nodes.iter();
        Box::new( it.map(|n| &n.v) )
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.neighbours()
            // Box::new(node_u.left.iter().chain(node_u.right.iter()))
        } else {
            Box::new(iter::empty::<&Vertex>())
        }
    }
}

impl LinearGraph for OrdGraph {
    fn index_of(&self, u:&Vertex) -> usize {
        *self.indices.get(u).unwrap_or_else(|| panic!("Vertex {u} does not exist"))
    }

    fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        let iu = self.index_of(u);
        let node_u = &self.nodes[iu];

        let mut res:Vec<Vertex> = node_u.left.iter().cloned().collect();
        res.sort_by_cached_key(|v| self.indices.get(v).unwrap());
        
        res
    }
    
    fn right_degree(&self, u:&Vertex) -> u32 {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.right.len() as u32
        } else {
            0
        }
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
    use crate::editgraph::EditGraph;
    use itertools::Itertools;    

    #[test]
    fn order_iteration() {
        let G = EditGraph::path(20);
        let order:Vec<_> = (0..20).rev().collect();
        let O = OrdGraph::with_ordering(&G, order.iter());    
        
        assert_eq!(order, O.vertices().copied().collect_vec());
    }

    #[test]
    fn wreach_graph() {
        const R:usize = 5;
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let O = OrdGraph::by_degeneracy(&G);
        let W = O.to_wreach_graph::<R>();
        
        // Ensures that 'reachables' in wreach graph contain the same 
        // information as the wreach sets computed by OrdGraph
        let wreach_sets = O.wreach_sets(R as u32);

        for u in G.vertices() {
            let reachables = W.reachables(&u);
            let wreach_set = wreach_sets.get(&u).unwrap();
            assert_eq!(reachables.len(), wreach_set.len());

            // Verify that the depth of each vertex is correct
            // *and* that the relative order in each depth-group
            // has been maintained.
            for depth in 1..=R as u32 {
                let mut last_index:i64 = -1;
                for v in reachables.at(depth as usize) {
                    assert_eq!(depth, wreach_set[v]);  
                    let index = O.indices[v] as i64;
                    assert!(index > last_index);
                    last_index = index;
                }
            }
        }
    }

    #[test] 
    fn consistency() {
        let G = EditGraph::clique(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());
    
        assert_eq!(O.left_degree(&0), 0);
        assert_eq!(O.left_degree(&1), 1);
        assert_eq!(O.left_degree(&2), 2);
        assert_eq!(O.left_degree(&3), 3);
        assert_eq!(O.left_degree(&4), 4);

        assert_eq!(O.left_neighbours(&0), vec![]);
        assert_eq!(O.left_neighbours(&1), vec![0]);
        assert_eq!(O.left_neighbours(&2), vec![0,1]);
        assert_eq!(O.left_neighbours(&3), vec![0,1,2]);
        assert_eq!(O.left_neighbours(&4), vec![0,1,2,3]);

        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let (lower, upper, order, _) = G.degeneracy();
        assert_eq!(lower, upper);
        let degen = upper;

        let O = OrdGraph::with_ordering(&G, order.iter());

        let left_degs = O.left_degrees();
        assert_eq!(*left_degs.values().max().unwrap(), degen);

        let mut m = 0;
        for u in O.vertices() {
            assert_eq!(O.left_degree(u), O.left_neighbours(u).len() as u32);
            m += O.left_degree(u);
        }
        assert_eq!(m as usize, G.num_edges());

        for (u,v) in O.edges() {
            assert!(G.adjacent(&u, &v));
        }

        for (u,v) in G.edges() {
            assert!(O.adjacent(&u, &v));
        }        
    }

    #[test] 
    fn sreach() {
        let mut G = EditGraph::path(8);
        // 0-1-2-3-(4)-5-6-7
        G.add_edge(&2, &5);
        G.add_edge(&1, &6);
        G.add_edge(&0, &7);

        let ord:Vec<_> = (0..=8).collect();
        let O = OrdGraph::with_ordering(&G, ord.iter());
    
        let S = O.sreach_set(&4, 5);

        assert_eq!(S[&3], 1); // 4-3
        assert_eq!(S[&2], 2); // 4-5-2
        assert_eq!(S[&1], 3); // 4-5-6-1
        assert_eq!(S[&0], 4); // 4-5-6-7-0
    }    

    #[test] 
    fn count_cliques() {
        let G = EditGraph::clique(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());    

        assert_eq!(O.count_max_cliques(), 1);

        let G = EditGraph::complete_kpartite([5,5,5].iter());
        let O = OrdGraph::by_degeneracy(&G);    

        assert_eq!(O.count_max_cliques(), 5*5*5);     
        
        let G = EditGraph::independent(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());    

        assert_eq!(O.count_max_cliques(), 5);         
    }
}
