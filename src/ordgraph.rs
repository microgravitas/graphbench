use std::iter;

use crate::algorithms::GraphAlgorithms;
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

    pub fn to_degeneracy_graph(&self) -> ReachGraph {
        let mut builder = ReachGraphBuilder::new(1);

        for u in self.vertices() {
            let L = self.left_neighbours(u);
            let W = L.iter().map(|x| (*x,1) ).collect();
            builder.append(&u, &W, &self.indices);
        }

        builder.build()
    }    

    pub fn to_wreach_graph(&self, r:u32) -> ReachGraph {
        let mut builder = ReachGraphBuilder::new(r);
        let wreach_sets = self.wreach_sets(r);

        for (u, W) in wreach_sets.into_iter() {
            builder.append(&u, &W, &self.indices);
        }

        builder.build()
    }

    pub fn to_sreach_graph(&self, r:u32) -> ReachGraph {
        let mut builder = ReachGraphBuilder::new(r);

        for u in self.vertices() {
            let S = self.sreach_set(u, r);
            builder.append(&u, &S, &self.indices);
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
        for (s,old_i,new_i) in vec![(u,iu,iv), (v,iv,iu)]{
            let mut needs_update:Vec<usize> = Vec::new();            
            {   // Update vertex itself
                let mut n = &mut self.nodes[old_i];
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

    /// Returns the size of `u`'s left neighbourhood
    pub fn left_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.len()
        } else {
            0
        }
    }

    /// Returns a copy of `u`'s left neighbourhood.
    pub fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        let iu = self.indices.get(u).expect(format!("Vertex {u} does not exist").as_str()); 
        let node_u = &self.nodes[*iu];

        let mut res:Vec<Vertex> = node_u.left.iter().cloned().collect();
        res.sort_by_cached_key(|v| self.indices.get(v).unwrap());
        
        res
    }

    /// Returns the sizes of all left neighbourhood as a map.
    pub fn left_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, n.left.len() as u32);
        }
        res
    }
    
    /// Returns the size of `u`'s right neighbourhood.
    pub fn right_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.right.len()
        } else {
            0
        }
    }    

    /// Returns a copy of `u`'s right neighbourhood. 
    pub fn right_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        let iu = self.indices.get(u).expect(format!("Vertex {u} does not exist").as_str()); 
        let node_u = &self.nodes[*iu];

        let mut res:Vec<Vertex> = node_u.right.iter().cloned().collect();
        res.sort_by_cached_key(|v| self.indices.get(v).unwrap());
        
        res
    }    
    
    /// Returns the sizes of all right neighbourhood as a map.
    pub fn right_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, n.right.len() as u32);
        }
        res
    }

    /// Conducts a bfs from `root` for `dist` steps ignoring all vertices
    /// left of `root`.
    /// 
    /// Returns the bfs as a sequence of layers.
    pub fn right_bfs(&self, root:&Vertex, dist:u32) -> Vec<VertexSet> {
        let mut seen:VertexSet = VertexSet::default();
        let iroot = *self.indices.get(root).unwrap();
        let root = *root;

        let mut res = vec![VertexSet::default(); (dist+1) as usize];
        res[0].insert(root);
        seen.insert(root);

        for d in 1..=(dist as usize) {
            let (part1, part2) = res.split_at_mut(d as usize);

            for u in part1[d-1].iter().cloned() {
                let iu = *self.indices.get(&u).unwrap();
                for v in self.nodes[iu].neighbours() {
                    let iv = *self.indices.get(&v).unwrap();
                    if iv > iroot && !seen.contains(v) {
                        part2[0].insert(*v);
                        seen.insert(*v);
                    }
                }
            }
        }

        res
    }

    /// Computes all strongly $r$-reachable vertices to $u$. 
    /// 
    /// A vertex $v$ is strongly $r$-reachable from $u$ if there exists a $u$-$v$-path in the graph
    /// of length at most $r$ where $v$ is the only vertex of the path that comes before $u$ in the
    /// ordering.
    /// 
    /// Returns a map with all vertices that are strongly $r$-reachable
    /// from $u$. For each member $v$ in the map the corresponding values represents
    /// the distance $d \\leq r$ at which $v$ is strongly reachable from $u$.
    pub fn sreach_set(&self, u:&Vertex, r:u32) -> VertexMap<u32> {
        let bfs = self.right_bfs(u, r-1);
        let mut res = VertexMap::default();
        
        let iu = self.indices.get(u).expect(format!("{u} not contained in this graph").as_str()).clone();
        for (d, layer) in bfs.iter().enumerate() {
            for v in layer {
                for x in self.left_neighbours(v) {
                    let ix = self.indices[&x];
                    if ix < iu {
                        // If x is alyread in `res` then it will be for a smaller
                        // distance. Therefore we only insert the current distance if 
                        // no entry exists yet.
                        res.entry(x).or_insert((d+1) as u32);
                    }
                }
            }
        }

        res
    }

    /// Computes all weakly $r$-reachable sets as a map.. 
    /// 
    /// A vertex $v$ is weakly $r$-rechable from $u$ if there exists a $u$-$v$-path in the graph
    /// of length at most $r$ whose leftmost vertex is $v$. In particular, $v$ must be left of
    /// $u$ in the ordering.
    /// 
    /// Returns a [VertexMap] for each vertex. For a vertex $u$ the corresponding [VertexMap] 
    /// contains all vertices that are weakly $r$-reachable from $u$. For each member $v$ 
    /// in this [VertexMap] the corresponding values represents the distance $d \\leq r$ at 
    /// which $v$ is weakly reachable from $u$.
    /// 
    /// If the sizes of the weakly $r$-reachable sets are bounded by a constant the computation 
    /// takes $O(|G|)$ time.      
    pub fn wreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, VertexMap::default());
        }
        for u in self.vertices() {
            for (d, layer) in self.right_bfs(u, r).iter().skip(1).enumerate() {
                for v in layer {
                    assert!(*v != *u);
                    res.get_mut(v).unwrap().insert(*u, (d+1) as u32); 
                }
            }
        }
        res
    }    

    /// Returns for each vertex the size of its $r$-weakly reachable set. 
    /// This method uses less memory than [wreach_sets](OrdGraph::wreach_sets).
    pub fn wreach_sizes(&self, r:u32) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, 0);
        }
        for u in self.vertices() {
            for layer in self.right_bfs(u, r).iter().skip(1) {
                for v in layer {
                    let count = res.entry(*v).or_insert(0);
                    *count += 1;
                }
            }
        }
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


pub struct ReachGraph {
    _depth: u32,
    indices:VertexMap<usize>,
    pub(crate) contents:Vec<u32>,
    edges: EdgeSet
}

#[derive(Debug,PartialEq,Eq)]
pub struct Reachables<'a> {
    pub(crate) from: Vertex,
    reachables: Vec<&'a [u32]>,
    boundaries: Vec<(usize, usize)>
}

impl<'a> Reachables<'a> {
    /// Returns the total number of reachable vertices at all depths.
    pub fn len(&self) -> usize {
        self.reachables.iter().map(|seg| seg.len()).sum()
    }

    /// Returns all vertices that are reachable at `depth` from the
    /// root vertex.
    pub fn at(&self, depth:usize) -> &[Vertex] {
        assert!(depth >= 1 && depth < self.reachables.len()+1);
        self.reachables[(depth-1)]
    }

    /// Returns the boundary indices of vertices that are reachable
    /// at `depth` from the root vertex.
    fn get_boundaries(&self, depth:usize) -> (usize, usize) {
        assert!(depth >=1 && depth < self.reachables.len()+1);
        self.boundaries[depth]
    }
}

impl ReachGraph {
    fn new(depth:u32) -> Self {
        ReachGraph{ _depth: depth,
                    indices: VertexMap::default(),
                    edges: EdgeSet::default(),
                    contents: Vec::default() }
    }

    /// Returns the first vertex in the ordering
    pub fn first(&self) -> Vertex {
        assert!(self.contents.len() > 1);
        self.contents[0]
    }

    fn reachables_at(&self, index_u:usize) -> Reachables {
        let r = self._depth as usize;
        let u = self.contents[index_u];

        // Layout:
        //    | u | next_vertex | index_2 | index_3 | ... | index_r  | index_end | [dist 1 neighbours] [dist 2 neighbours] ... [dist r neigbhours]
        //  index_u     + 1         +2         + 3          + r          + (r+1)   + (r+2)        
        let mut left = index_u + r + 2; 
        let mut reachables = Vec::with_capacity(self._depth as usize);
        let mut boundaries = Vec::with_capacity(self._depth as usize);
        for right in &self.contents[index_u+2..=index_u+r+1] {
            let right = *right as usize;
            reachables.push(&self.contents[left..right]);
            boundaries.push((left,right));
            left = right;
        }

        Reachables { from: u, reachables, boundaries }
    }

    pub fn reachables(&self, u:&Vertex) -> Reachables {
        let index_u = *self.indices.get(u).expect(format!("{u} is not a vertex in this graph.").as_str());
        self.reachables_at(index_u)
    }

    pub fn next_reachables(&self, last:&Vertex) -> Option<Reachables> {
        let index_last = *self.indices.get(last).expect(format!("{last} is not a vertex in this graph.").as_str());
        debug_assert_eq!(*last, self.contents[index_last]);

        let index_next = self.contents[index_last+1] as usize;
        if index_next == index_last {
            None
        } else {
            Some(self.reachables_at(index_next))
        }
    }

    pub fn reachables_all(&self, u:&Vertex) -> &[u32] {
        let index_u = *self.indices.get(u).expect(format!("{u} is not a vertex in this graph.").as_str());
        let r = self._depth as usize;
        debug_assert_eq!(*u, self.contents[index_u]);

        let left = index_u + r + 2;
        let right = self.contents[left-1] as usize;

        &self.contents[left..right]
    }

    fn segment(&self, u:&Vertex) -> &[u32] {
        let index_u = *self.indices.get(u).expect(format!("{u} is not a vertex in this graph.").as_str());
        let r = self._depth as usize;
        debug_assert_eq!(*u, self.contents[index_u]);

        let right = self.contents[index_u + r + 1] as usize;

        &self.contents[index_u..right]
    }    

    pub fn depth(&self) -> u32 {
        self._depth
    }

    pub fn count_max_cliques(&self) -> u64 {
        let mut res = 0;
        for (v,reachables) in self.iter() {
            let neighbours = reachables.at(1);
            let mut include = VertexSet::default();
            include.insert(v);
            let exclude = VertexSet::default();
            let maybe = neighbours.iter().cloned().collect();
            res += self.bk_pivot_count(&neighbours, include, maybe, exclude)
        }
        res
    }

    fn bk_pivot_count(&self, vertices:&[Vertex], include:VertexSet, mut maybe:VertexSet, mut exclude:VertexSet) -> u64 {
        if maybe.len() == 0 && exclude.len() == 0 {
            // `include` is a maximal clique
            return 1
        }

        // Choose the last vertex in ordering which is in either `maybe` or `exclude`
        // as the pivot vertex
        let mut u = None;
        let mut i = vertices.len()-1;
        while i != 0 {
            let cand = vertices[i];
            if maybe.contains(&cand) || exclude.contains(&cand) {
                u = Some(cand);
                break;
            }
            i -= 1;
        }
        let u = u.expect("If this fails there is a bug");

        // Compute u's *left* neighbourhood inside of `vertices`. 
        let left_neighbours:Vec<Vertex> = vertices[0..=(i-1)].iter()
                .filter_map(|v| if self.adjacent(&u, v) {Some(*v)} else {None} ).collect();

        let left_neighbours_set:VertexSet = left_neighbours.iter().cloned().collect();

        let mut res = 0;
        for v in vertices[0..=(i-1)].iter().rev() {
            // We ignore `v` if it is not a maybe-vertex. We also ignore it
            // if it is a neighbour of the pivot `u`.
            if !maybe.contains(&v) || left_neighbours_set.contains(&v) {
                continue
            } 

            // Recursion
            res += self.bk_pivot_count(&left_neighbours, 
                            include.intersection(&left_neighbours_set).cloned().collect(),
                            maybe.intersection(&left_neighbours_set).cloned().collect(),
                            exclude.intersection(&left_neighbours_set).cloned().collect(),
                            );
            
            maybe.remove(&v);
            exclude.insert(*v);
        }

        res
    }
}

impl Graph for ReachGraph {
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
        let reach = self.reachables(u);
        reach.at(1).len() as u32
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(self.indices.keys())
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        let (left, right) = self.reachables(u).get_boundaries(1);
        Box::new(self.contents[left..right].iter())
    }
}

pub struct ReachGraphBuilder {
    last_index: Option<u32>,
    depth: u32,
    rgraph: ReachGraph
}

impl ReachGraphBuilder {
    pub fn new(depth:u32) -> Self {
        ReachGraphBuilder{ last_index: None, depth: depth, rgraph: ReachGraph::new(depth) }
    }

    pub fn build(self) -> ReachGraph {
        self.rgraph
    }

    pub fn append(&mut self, u:&Vertex, reachable:&VertexMap<u32>, order:&VertexMap<usize>) {
        // Let r be the depth of this data structure. Then for each vertex we layout the data as follows:
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
        let r = self.depth;

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
    
        let vertices:Vec<_> = neighbour_order.iter().map(|(_,_,v)| *v).collect();
        let dists:Vec<_> = neighbour_order.iter().map(|(dist,_,_)| *dist).collect();

        // Add edges
        let edges_it = reachable.iter()
            .filter_map(|(v,dist)| if *dist == 1 {Some((*u,*v))} else {None} )
            .map(|(u,v)| if u < v {(u,v)} else {(v,u)} );

        self.rgraph.edges.extend(edges_it);
        
        
        // Push | index_2 | ... | index_r |
        let mut curr_dist = 1;
        let base_offset = index_u + r + 2;
        let mut index = 2; // We start at index 2
        let mut offset = 0;

        // We add a guard element at the end so that positions for 
        // all distances are added to the index. As a result, this 
        // loop adds the elements 
        //     | index_2 | ... | index_r | index_end |
        for dist in dists.iter().chain(iter::once(&(r+1))) {
            let dist = *dist;
            while curr_dist < dist {
                contents.push(base_offset + offset);
                assert_eq!((contents.len()-1) as u32,  index_u + index);
                curr_dist += 1;
                index += 1;
            }
            offset += 1;            
        }
        assert_eq!((contents.len()-1) as u32,  index_u + r + 1);

        // Finally write all neighbours
        contents.extend(vertices.into_iter());

        self.last_index = Some(index_u);
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

    #[test]
    fn wreach_graph() {
        let r = 5;
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let O = OrdGraph::by_degeneracy(&G);
        let W = O.to_wreach_graph(r);
        
        // Ensures that 'reachables' in wreach graph contain the same 
        // information as the wreach sets computed by OrdGraph
        let wreach_sets = O.wreach_sets(r);

        for u in G.vertices() {
            let reachables = W.reachables(&u);
            let wreach_set = wreach_sets.get(&u).unwrap();
            assert_eq!(reachables.len(), wreach_set.len());

            // Verify that the depth of each vertex is correct
            // *and* that the relative order in each depth-group
            // has been maintained.
            for depth in 1..=r {
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
            assert_eq!(O.left_degree(u), O.left_neighbours(u).len());
            m += O.left_degree(u);
        }
        assert_eq!(m, G.num_edges());

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

        println!("Sreach set {S:?}");

        assert_eq!(S[&3], 1); // 4-3
        assert_eq!(S[&2], 2); // 4-5-2
        assert_eq!(S[&1], 3); // 4-5-6-1
        assert_eq!(S[&0], 4); // 4-5-6-7-0
    }    

    #[test] 
    fn count_cliques() {
        let G = EditGraph::clique(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());    
        let W = O.to_degeneracy_graph();

        assert_eq!(W.count_max_cliques(), 1);
    }
}
