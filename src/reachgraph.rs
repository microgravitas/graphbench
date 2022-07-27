use std::collections::BTreeSet;

use fxhash::FxHashSet;
use itertools::Itertools;

use crate::graph::*;
use crate::iterators::*;

pub struct ReachGraph<const DEPTH: usize> {
    indices: VertexMap<usize>,
    pub(crate) contents:Vec<u32>,
    right_degrees: VertexMap<u32>,
    edges: EdgeSet
}

#[derive(Debug,PartialEq,Eq)]
pub struct Reachables<'a, const DEPTH: usize> {
    pub(crate) from: Vertex,
    reachables: [&'a [u32]; DEPTH],
    boundaries: [(usize, usize); DEPTH]
}

impl<'a, const DEPTH: usize> Reachables<'a, DEPTH> {
    /// Returns the total number of reachable vertices at all depths.
    pub fn len(&self) -> usize {
        self.reachables.iter().map(|seg| seg.len()).sum()
    }

    /// Returns all vertices that are reachable at `depth` from the
    /// root vertex.
    pub fn at(&self, depth:usize) -> &[Vertex] {
        assert!(depth >= 1 && depth <= DEPTH);
        self.reachables[depth-1]
    }

    /// Returns the boundary indices of vertices that are reachable
    /// at `depth` from the root vertex.
    fn get_boundaries(&self, depth:usize) -> (usize, usize) {
        assert!(depth >=1 && depth < self.reachables.len()+1);
        self.boundaries[depth]
    }
}

impl<const DEPTH: usize> ReachGraph<DEPTH> {
    fn new() -> Self {
        ReachGraph{ indices: VertexMap::default(),
                    edges: EdgeSet::default(),
                    right_degrees: VertexMap::default(),
                    contents: Vec::default() }
    }

    /// Returns the first vertex in the ordering, if the graph is non-empty.
    pub fn first(&self) -> Option<Vertex> {
        if self.contents.is_empty() {
            None
        } else { 
            Some(self.contents[0])
        }
    }

    fn reachables_at(&self, index_u:usize) -> Reachables<DEPTH> {
        let r = DEPTH;
        let u = self.contents[index_u];

        // Layout:
        //    | u | next_vertex | index_2 | index_3 | ... | index_r  | index_end | [dist 1 neighbours] [dist 2 neighbours] ... [dist r neigbhours]
        //  index_u     + 1         +2         + 3          + r          + (r+1)   + (r+2)        
        let mut left = index_u + r + 2; 
        let mut reachables = Vec::with_capacity(DEPTH as usize);
        let mut boundaries = Vec::with_capacity(DEPTH as usize);
        for right in &self.contents[index_u+2..=index_u+r+1] {
            let right = *right as usize;
            reachables.push(&self.contents[left..right]);
            boundaries.push((left,right));
            left = right;
        }

        let reachables:[&[u32]; DEPTH] = reachables.try_into().unwrap();
        let boundaries:[(usize, usize); DEPTH] = boundaries.try_into().unwrap();

        Reachables { from: u, reachables, boundaries }
    }

    pub fn reachables(&self, u:&Vertex) -> Reachables<DEPTH> {
        let index_u = self.index_of(u);
        self.reachables_at(index_u)
    }

    pub fn next_reachables(&self, last:&Vertex) -> Option<Reachables<DEPTH>> {
        let index_last = self.index_of(last);
        debug_assert_eq!(*last, self.contents[index_last]);

        let index_next = self.contents[index_last+1] as usize;
        if index_next == index_last {
            None
        } else {
            Some(self.reachables_at(index_next))
        }
    }

    pub fn reachables_all(&self, u:&Vertex) -> &[u32] {
        let index_u = self.index_of(u);
        let r = DEPTH as usize;
        debug_assert_eq!(*u, self.contents[index_u]);

        let left = index_u + r + 2;
        let right = self.contents[left-1] as usize;

        &self.contents[left..right]
    }

    fn segment(&self, u:&Vertex) -> &[u32] {
        let index_u = self.index_of(u);
        let r = DEPTH as usize;
        debug_assert_eq!(*u, self.contents[index_u]);

        let right = self.contents[index_u + r + 1] as usize;

        &self.contents[index_u..right]
    }    

    pub fn depth(&self) -> usize {
        DEPTH
    }
}

impl<const DEPTH: usize> Graph for ReachGraph<DEPTH> {
    fn num_vertices(&self) -> usize {
        self.indices.len()
    }

    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    fn contains(&self, u:&Vertex) -> bool {
        self.indices.contains_key(u)
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        self.edges.contains(&(*u,*v)) || self.edges.contains(&(*v,*u)) 
    }

    fn degree(&self, u:&Vertex) -> u32 {
        self.left_degree(u) + self.right_degree(u)
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(ReachOrderIterator::new(self))
    }

    fn neighbours<'a>(&'a self, _u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        panic!("ReachGraph::neighbours not supported.")
    }
}

impl<const DEPTH: usize> LinearGraph for ReachGraph<DEPTH> {
    fn index_of(&self, u:&Vertex) -> usize {
        *self.indices.get(u).unwrap_or_else(|| panic!("{u} is not a vertex in this graph."))
    }

    fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        self.reachables(u).at(1).to_vec()
    }

    fn right_degree(&self, u:&Vertex) -> u32 {
        *self.right_degrees.get(u).unwrap_or(&0)
    }
}

pub struct ReachGraphBuilder<const DEPTH: usize> {
    last_index: Option<u32>,
    rgraph: ReachGraph<DEPTH>
}

impl<const DEPTH: usize> ReachGraphBuilder<DEPTH> {
    pub fn new() -> Self {
        ReachGraphBuilder{ last_index: None, rgraph: ReachGraph::<DEPTH>::new() }
    }

    pub fn build(self) -> ReachGraph<DEPTH> {
        self.rgraph
    }

    pub fn append(&mut self, u:&Vertex, reachable:&VertexMap<u32>, order:&VertexMap<usize>) {
        // Let r be the depth of this data structure. Then for each vertex the data is layed out as follows:
        //    
        //                                                                          base_offset
        //    | u | next_vertex | index_2 | index_3 | ... | index_r  | index_end | [dist 1 neighbours] [dist 2 neighbours] ... [dist r neigbhours]
        //  index_u     + 1         +2         + 3          + r          + (r+1)   + (r+2)
        //
        // where index_i points to the first index of the slice [dist i neighbours]. Note that we do not store
        // index_1 as this position is fixed. `next_vertex` points to the next vertex in the sequence, 
        // if `u` is the last vertex then `next_vertex` points to `u`.

        let contents = &mut self.rgraph.contents;
        let indices = &mut self.rgraph.indices;
        let r = DEPTH as u32;

        // Add vertex to contentss
        contents.push(*u);           // | u |
        let index_u = (contents.len()-1) as u32;
        indices.insert(*u, index_u as usize);   
        contents.push(index_u); // | next_vertex |, points to `u` for now
        assert_eq!((contents.len()-1) as u32,  index_u + 1);

        // Link up with previous vertex
        if let Some(last_index) = self.last_index {
            contents[(last_index+1) as usize] = index_u;
        }

        // Compute the local index for reachable neighbours. We first group neighbours by their 
        // reachability distance, which is some value between 1 and r. Second, inside each group of
        // equidistance neighbour, we want to preserve the order of the original OrdGraph.
        let mut neighbour_order:Vec<_> = reachable.iter().map(|(v,dist)| (*dist, order[v], *v)).collect();
        neighbour_order.sort_unstable();
    
        let vertices = neighbour_order.iter().map(|(_,_,v)| *v);
        let dists:Vec<_> = neighbour_order.iter().map(|(dist,_,_)| *dist).collect();

        // Add edges and count right-degrees
        for (v,&dist) in reachable.iter() {
            if dist != 1 {
                continue
            }
            *self.rgraph.right_degrees.entry(*v).or_insert(0) += 1;

            if u < v { 
                self.rgraph.edges.insert((*u,*v));
            } else {
                self.rgraph.edges.insert((*v,*u));
            }
        }
        
        
        // Push | index_2 | ... | index_r |
        let mut curr_dist = 1;
        let base_offset = index_u + r + 2;
        let mut index = 2; // We start at index 2

        // We add a guard element at the end so that positions for 
        // all distances are added to the index. As a result, this 
        // loop adds the elements 
        //     | index_2 | ... | index_r | index_end |
        for (offset, dist) in dists.iter().chain(std::iter::once(&(r + 1))).enumerate() {
            let dist = *dist;
            while curr_dist < dist {
                contents.push(base_offset + offset as u32);
                assert_eq!((contents.len()-1) as u32,  index_u + index);
                curr_dist += 1;
                index += 1;
            }
        }
        assert_eq!((contents.len()-1) as u32,  index_u + r + 1);

        // Finally write all neighbours
        contents.extend(vertices.into_iter());

        self.last_index = Some(index_u);
    }
}

pub struct ReachOrderIterator<'a, const DEPTH: usize> {
    rgraph: &'a ReachGraph<DEPTH>,
    curr_index: Option<usize>
}

impl<'a, const DEPTH: usize> ReachOrderIterator<'a, DEPTH> {
    pub fn new(rgraph: &'a ReachGraph<DEPTH>) -> Self {
        if rgraph.len() == 0 {
            ReachOrderIterator { rgraph, curr_index: None }
        } else {
            ReachOrderIterator { rgraph, curr_index: Some(0) }
        }
    }    
}

impl<'a, const DEPTH: usize> Iterator for ReachOrderIterator<'a, DEPTH> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.curr_index {
            Some(ix) => {
                let contents = &self.rgraph.contents;
                let u = &contents[ix];

                let next_ix = contents[ix+1] as usize;
                if next_ix == ix { 
                    self.curr_index = None;
                } else {
                    self.curr_index = Some(next_ix);
                }

                Some(&u)
            },
            None => None,
        }
    }
}

/// Reachable-set iterator for graphs. At each step, the iterator
/// returns a pair $(v,W^r(v))$.
pub struct ReachIterator<'a, const DEPTH: usize> {
    rgraph: &'a ReachGraph<DEPTH>,
    current: Option<Reachables<'a, DEPTH>>
}

impl<'a, const DEPTH: usize> ReachIterator<'a, DEPTH> {
    pub fn new(rgraph: &'a ReachGraph<DEPTH>) -> Self {
        let current = rgraph.first().map(|u| rgraph.reachables(&u));
        ReachIterator { rgraph, current }
    }    
}

impl<'a, const DEPTH: usize> Iterator for ReachIterator<'a, DEPTH>{
    type Item = (Vertex, Reachables<'a, DEPTH>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(reachables) = &self.current {
            let u = reachables.from;
            let next = self.rgraph.next_reachables(&u);
            let res = std::mem::replace(&mut self.current, next);     
            
            Some((u, res.unwrap()))
        } else {
            None
        }
    }
}

impl<const DEPTH: usize> ReachGraph<DEPTH> {
    pub fn iter(&self) -> ReachIterator<DEPTH> {
        ReachIterator::new(self)
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
    use crate::ordgraph::OrdGraph;
    use crate::algorithms::lineargraph::*;

    #[test]
    fn consistency() {
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let O = OrdGraph::by_degeneracy(&G);
        let W = O.to_wreach_graph::<3>();

        for u in G.vertices() {
            assert_eq!(G.degree(u), W.degree(u));
        }
    }

    #[test]
    fn order_iteration() {
        let G = EditGraph::path(20);
        let order:Vec<_> = (0..20).rev().collect();
        let O = OrdGraph::with_ordering(&G, order.iter());    
        let W = O.to_wreach_graph::<3>();
        
        assert_eq!(order, W.vertices().copied().collect_vec());
    }

    #[test] 
    fn count_cliques() {
        let G = EditGraph::clique(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());    
        let W = O.to_wreach_graph::<3>();

        assert_eq!(W.count_max_cliques(), 1);

        let G = EditGraph::complete_kpartite([5,5,5].iter());
        let O = OrdGraph::by_degeneracy(&G);    
        let W = O.to_wreach_graph::<3>();

        assert_eq!(W.count_max_cliques(), 5*5*5);        
    }

}
