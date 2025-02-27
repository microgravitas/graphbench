use std::borrow::Borrow;
use std::collections::{BTreeSet, HashMap};


use fxhash::FxHashSet;
use itertools::Itertools;

use crate::algorithms::GraphAlgorithms;
use crate::graph::*;
use crate::iterators::*;

#[derive(Clone)]
pub struct DegenGraph {
    indices: VertexMap<usize>,
    pub(crate) contents:Vec<u32>,
    right_neighbours: VertexMap<VertexSet>,
    m:usize
}

impl DegenGraph {
    /// Creates a degenerate graph representation from `graph` by computing a degeneracy ordering.
    pub fn from_graph<G: Graph>(graph: &G) -> DegenGraph {
        let (_, _, ord, _) = graph.degeneracy();
        DegenGraph::with_ordering(graph, ord.iter())
    }

    /// Creates a degenerate graph representation from `graph` using `order`.
    pub fn with_ordering<G, I, V>(graph: &G, order:I) -> DegenGraph
        where V: Borrow<Vertex>, G: Graph, I: IntoIterator<Item=V>
    {
        let order:Vec<_> = order.into_iter().map(|u| *u.borrow()).collect();
        let indices:VertexMap<_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (u,i)).collect();

        let mut builder = DegenGraphBuilder::new();

        for u in &order {
            let mut L = Vec::new();
            let iu = indices[u];            
            for v in graph.neighbours(u) {
                let iv = indices[v];
                if iv < iu {
                    L.push((iv, *v));
                }
            }
            L.sort_unstable();

            let L = L.iter().map(|(_,v)| *v).collect();
            builder.append(u, &L);
        }

        builder.build()
    }

    fn new() -> Self {
        DegenGraph{ indices: VertexMap::default(),
                    m: 0,
                    right_neighbours: VertexMap::default(),
                    contents: Vec::default() }
    }

    /// Relabels the vertices of this graph according to the underlying ordering, so
    /// the first vertex is 0, the second 1 etc. 
    /// Additionally, the left neighbourhoods are sorted.
    /// 
    /// Returns a vertex map from the new vertex names (keys) to the old vertex names (values).
    pub fn normalize(&mut self) -> VertexMap<Vertex> {
        let mut i = 0;
        let mut ix = 0;
        let mut new_to_old = VertexMap::default();
        let mut old_to_new = VertexMap::default();
        unsafe {
            loop {
                let u = *self.contents.get_unchecked(i); // contents[i]
                let next = *self.contents.get_unchecked(i+1) as usize; // contents[i+1]
                let num_neighbors = *self.contents.get_unchecked(i+2) as usize; // contents[i+2]

                let left = i+3;
                let right = left+num_neighbors;

                // Relabel u itself
                *self.contents.get_unchecked_mut(i) = ix as u32;
                new_to_old.insert(ix as u32, u);
                old_to_new.insert(u, ix as u32);

                // Relabel left neighbourhood
                let mut N:Vec<Vertex> = self.contents.get_unchecked(left..right).iter().map(|u| old_to_new[u]).collect();
                N.sort_unstable();
                self.contents.get_unchecked_mut(left..right).clone_from_slice(&N);

                if next == i {
                    break
                }
                i = next;
                ix += 1;
            }
        }

        // Change right neighbourhood
        let mut right_Ns = VertexMap::default();
        for (u, N) in self.right_neighbours.iter() {
            let new_N:VertexSet = N.into_iter().map(|x| old_to_new[x]).collect();
            let new_u = old_to_new[u];
            right_Ns.insert(new_u, new_N);
        }
        self.right_neighbours = right_Ns;

        // Change indices
        let mut new_indices = VertexMap::default();
        for (u,ix) in self.indices.iter() {
            new_indices.insert(old_to_new[u], *ix);
        }
        self.indices = new_indices;

        new_to_old
    }

    /// Returns the first vertex in the ordering, if the graph is non-empty.
    pub fn first(&self) -> Option<Vertex> {
        if self.contents.is_empty() {
            None
        } else { 
            Some(self.contents[0])
        }
    }

    pub fn left_neighbours_slice(&self, u:&Vertex) -> &[Vertex] {
        let iu = self.index_of(u);
        debug_assert_eq!(*u, self.contents[iu]);

        //  Layout:
        //    | u | next_vertex | num_neighbors | [neighbours] 
        //  index_u     + 1         +2             + 3        

        let num_neighbours = self.contents[iu+2] as usize;
        // &self.contents[iu+3..iu+3+num_neighbours]
        unsafe{ self.contents.get_unchecked(iu+3..iu+3+num_neighbours) }
    }

    pub fn right_neighbours(&self, u:&Vertex) -> &VertexSet {
        &self.right_neighbours[u]
    }
    
    /// Returns true if `u` is left of `v` and uv is an edge in the graph.
    /// Returns false if `u` is right of `v`, either `u` or `v` is not in the graph
    /// or if the edge uv is not in the graph.
    pub fn adjacent_ordered(&self, u:&Vertex, v:&Vertex) -> bool {
        match self.right_neighbours.get(u) {
            Some(right) => right.contains(v),
            None => false,
        }
    }    
}

impl Graph for DegenGraph {
    fn num_vertices(&self) -> usize {
        self.indices.len()
    }

    fn num_edges(&self) -> usize {
        self.m
    }

    fn contains(&self, u:&Vertex) -> bool {
        self.indices.contains_key(u)
    }

    fn adjacent(&self, u:&Vertex, v:&Vertex) -> bool {
        if !self.right_neighbours.contains_key(u) || !self.right_neighbours.contains_key(v) {
            return false
        }

        self.right_neighbours.get(u).unwrap().contains(v) || self.right_neighbours.get(v).unwrap().contains(u) 
    }

    fn degree(&self, u:&Vertex) -> u32 {
        self.left_degree(u) + self.right_degree(u)
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        Box::new(DegenOrderIterator::new(self))
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&'a Vertex> + 'a> {
        Box::new(self.left_neighbours_slice(u).iter().chain(
            self.right_neighbours.get(u).unwrap().iter()
        ))
    }
}

impl LinearGraph for DegenGraph {    
    fn index_of(&self, u:&Vertex) -> usize {
        *self.indices.get(u).unwrap_or_else(|| panic!("{u} is not a vertex in this graph."))
    }

    fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        self.left_neighbours_slice(u).to_vec()
    }

    fn left_degree(&self, u:&Vertex) -> u32 {
        match self.indices.get(u) {
            Some(iu) => {
                //  Layout:
                //    | u | next_vertex | num_neighbors | [neighbours] 
                //  index_u     + 1         +2             + 3                       
                self.contents[iu+2]
            },
            None => 0,
        }
    }    

    fn right_degree(&self, u:&Vertex) -> u32 {
        match self.right_neighbours.get(u) {
            Some(R) => R.len() as u32,
            None => 0,
        }
    }
}

pub struct DegenGraphBuilder {
    last_index: Option<u32>,
    dgraph: DegenGraph
}

impl Default for DegenGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DegenGraphBuilder {
    pub fn new() -> Self {
        DegenGraphBuilder{ last_index: None, dgraph: DegenGraph::new() }
    }

    pub fn build(self) -> DegenGraph {
        self.dgraph
    }

    pub fn append(&mut self, u:&Vertex, neighbours:&Vec<Vertex>) {
        // Let r be the depth of this data structure. Then for each vertex the data is layed out as follows:
        //                                                                          base_offset
        //    | u | next_vertex | num_neighbors | [neighbours] 
        //  index_u     + 1         +2             + 3

        let contents = &mut self.dgraph.contents;
        let indices = &mut self.dgraph.indices;

        // Ensure that right-neighbourhood entry exist for u
        self.dgraph.right_neighbours.entry(*u).or_default();

        // Add vertex to contents
        contents.push(*u);           // | u |
        let index_u = (contents.len()-1) as u32;
        indices.insert(*u, index_u as usize);   
        contents.push(index_u); // | next_vertex |, points to `u` for now
        assert_eq!((contents.len()-1) as u32,  index_u + 1);

        // Link up with previous vertex
        if let Some(last_index) = self.last_index {
            contents[(last_index+1) as usize] = index_u;
        }

        // Update right neighbours
        for &v in neighbours {
            self.dgraph.right_neighbours.entry(v).or_default();
            self.dgraph.right_neighbours.get_mut(&v).unwrap().insert(*u);
        }

        self.dgraph.m += neighbours.len();
        
        // Push | num_neighbours |
        contents.push(neighbours.len() as u32);

        // Finally write all neighbours
        contents.extend(neighbours.iter());

        self.last_index = Some(index_u);
    }
}

pub struct DegenOrderIterator<'a> {
    dgraph: &'a DegenGraph,
    curr_index: Option<usize>
}

impl<'a> DegenOrderIterator<'a> {
    pub fn new(dgraph: &'a DegenGraph) -> Self {
        if dgraph.len() == 0 {
            DegenOrderIterator { dgraph, curr_index: None }
        } else {
            DegenOrderIterator { dgraph, curr_index: Some(0) }
        }
    }    
}

impl<'a> Iterator for DegenOrderIterator<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.curr_index {
            Some(ix) => {
                let contents = &self.dgraph.contents;
                unsafe {
                    // The following indexing operations are safe if 'contents'
                    // has the intended structure.
                    let u = contents.get_unchecked(ix);

                    let next_ix = *contents.get_unchecked(ix+1) as usize;
                    if next_ix == ix { 
                        self.curr_index = None;
                    } else {
                        self.curr_index = Some(next_ix);
                    }

                    Some(u)
                }
            },
            None => None,
        }
    }
}

/// Reachable-set iterator for graphs. At each step, the iterator
/// returns a pair $(v,W^r(v))$.
pub struct DegenIterator<'a> {
    dgraph: &'a DegenGraph,
    current: Option<usize>
}

impl<'a> DegenIterator<'a> {
    pub fn new(dgraph: &'a DegenGraph) -> Self {
        let current = if dgraph.is_empty() { None } else { Some(0) };
        DegenIterator { dgraph, current }
    }    
}

impl<'a> Iterator for DegenIterator<'a>{
    type Item = (Vertex, &'a [Vertex]);

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            Some(i) => {
                let contents = &self.dgraph.contents;
                unsafe {
                    // The indices in the following are safe assuming that 'contents'
                    // has the propery structure.
                    let u = *contents.get_unchecked(i); // contents[i]
                    let next = *contents.get_unchecked(i+1) as usize; // contents[i+1]
                    let num_neighbors = *contents.get_unchecked(i+2) as usize; // contents[i+2]

                    let left = i+3;
                    let right = left+num_neighbors;

                    if next == i {
                        self.current = None;
                    } else {
                        self.current = Some(next);
                    }

                    Some((u, contents.get_unchecked(left..right)))
                }
            }
            None => None,
        }
    }
}

impl DegenGraph {
    pub fn iter(&self) -> DegenIterator {
        DegenIterator::new(self)
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
    use crate::io::*;

    #[test]
    fn basics() {
        let G = EditGraph::path(3);
        let order:Vec<_> = (0..3).collect();
        let D = DegenGraph::with_ordering(&G, order.iter());

        assert!(D.adjacent(&0, &1));
        assert!(D.adjacent(&1, &0));
        assert!(D.adjacent(&1, &2));
        assert!(D.adjacent(&2, &1));
        assert!(!D.adjacent(&0, &2));
        assert!(!D.adjacent(&2, &0));

        assert!(D.adjacent_ordered(&0, &1));
        assert!(!D.adjacent_ordered(&1, &0));


        println!("{:?}", D.contents);

        for (v,L) in D.iter() {
            println!("{v} {L:?}");
            if v == 0 {
                assert!(L.is_empty());
            } else {
                assert_eq!(L.len(), 1);
                assert_eq!(*L.iter().next().unwrap(), v-1);
            }
        }

        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let D = DegenGraph::from_graph(&G);

        assert_eq!(G.num_vertices(), D.num_vertices());
        assert_eq!(G.num_edges(), D.num_edges());

        for (v,L) in D.iter() {
            assert_eq!(D.left_degree(&v) as usize, L.len());
            for u in L {
                assert!(D.adjacent(u, &v));
            }
        }
    }

    #[test]
    fn consistency() {
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let D = DegenGraph::from_graph(&G);

        for u in G.vertices() {
            assert_eq!(G.degree(u), D.degree(u));
        }

        let O = OrdGraph::with_ordering(&G, D.vertices());
        for u in G.vertices() {
            assert_eq!(O.left_degree(u), D.left_degree(u));
            assert_eq!(O.right_degree(u), D.right_degree(u));

            let mut NG:Vec<_> = G.neighbours(u).collect();
            let mut DG:Vec<_> = D.neighbours(u).collect();
            NG.sort_unstable();
            DG.sort_unstable();
            assert_eq!(NG, DG);
        }


    }

    #[test]
    fn order_iteration() {
        let G = EditGraph::path(20);
        let order:Vec<_> = (0..20).rev().collect();
        let D = DegenGraph::with_ordering(&G, order.iter());
        
        assert_eq!(order, D.vertices().copied().collect_vec());
    }

    #[test] 
    fn count_cliques() {
        let G = EditGraph::clique(5);
        let D = DegenGraph::with_ordering(&G, vec![0,1,2,3,4].iter());

        assert_eq!(D.count_max_cliques(), 1);

        let G = EditGraph::complete_kpartite([5,5,5].iter());
        let D = DegenGraph::from_graph(&G);

        assert_eq!(D.count_max_cliques(), 5*5*5);        
    }

    #[test]
    fn sreach_basics() {
        let mut H = EditGraph::new();
        H.add_vertex(&0);
        H.add_vertex(&1);
        H.add_vertex(&2);
        H.add_vertex(&3);
        H.add_vertex(&4);
        H.add_vertex(&5);
        H.add_vertex(&6);

        //     .---------.
        // 0  1  2--(3)--4  5--6
        // `         `-----'  ' 
        //  `----------------'
        H.add_edges([(0,6),(1,4),(2,3),(3,4),(3,5),(5,6)].into_iter());

        let D = DegenGraph::with_ordering(&H, [0,1,2,3,4,5,6]);
        assert_eq!(D.sreach_set(&3, 1), [(2,1)].into_iter().collect());
        assert_eq!(D.sreach_set(&3, 2), [(2,1),(1,2)].into_iter().collect());
        assert_eq!(D.sreach_set(&3, 3), [(2,1),(1,2),(0,3)].into_iter().collect());

        // Add spurious edges which do not change the strong reachability for (3)
        H.add_edges([(0,1),(1,2),(2,4),(2,5),(2,6)].into_iter());
        let D = DegenGraph::with_ordering(&H, [0,1,2,3,4,5,6]);
        assert_eq!(D.sreach_set(&3, 1), [(2,1)].into_iter().collect());
        assert_eq!(D.sreach_set(&3, 2), [(2,1),(1,2)].into_iter().collect());
        assert_eq!(D.sreach_set(&3, 3), [(2,1),(1,2),(0,3)].into_iter().collect());


        // Larger example to test sreach(2): 
        // 0 ... 100 999 200..300 
        // With edges from i to i + 200 and from 999 to 200...300. 
        let mut H = EditGraph::new();
        let mut S_truth:VertexMap<u32> = VertexMap::default();
        let mut ordering = vec![];

        for i in 0..=100 {
            H.add_vertex(&i);
            S_truth.insert(i, 2);
            ordering.push(i);            
        }
        ordering.push(999);
        H.add_vertex(&999);        

        for i in 200..=300 {
            H.add_vertex(&i);
            ordering.push(i);
        }

        for i in 0..=100 {
            assert!(H.contains(&i));
            assert!(H.contains(&(i+200)));
            H.add_edge(&i,&(i+200));
        }
        for i in 200..=300 {
            assert!(H.contains(&i));
            H.add_edge(&999, &i);
        }

        assert_eq!(H.num_edges(), 2*101 );
        assert_eq!(H.num_vertices(), 2*101+1 );

        println!("{ordering:?}");
        println!("{:?}", H.edges().collect_vec());
        let D = DegenGraph::with_ordering(&H, ordering);

        assert_eq!(D.sreach_set(&999, 2), S_truth);
    }    

    #[test]
    fn sreach_consistency() {
        let mut G = EditGraph::from_file("./resources/Yeast.txt.gz").unwrap();
        G.remove_loops();
        let D = DegenGraph::from_graph(&G);

        let Sreach = D.sreach_sets(3);
        for u in D.vertices() {
            let S_global = &Sreach[u];
            let S_local = D.sreach_set(u, 3);
            println!("{S_global:?}");
            assert_eq!(&S_local, S_global);
        }
    }    

    #[test]
    fn sreach2() {
        let mut G = EditGraph::from_file("./resources/Yeast.txt.gz").unwrap();
        G.remove_loops();
        let D = DegenGraph::from_graph(&G);

        for u in D.vertices() {
            let mut Nu:VertexSet = D.left_neighbours(u).into_iter().collect();

            if Nu.is_empty() {
                continue
            }

            let anchor = Nu.iter().max_by_key(|x| D.index_of(x)).unwrap();
            let S = D.sreach_set(anchor, 2);
            let mut Sverts:VertexSet = S.iter().map(|(x,_)| *x).collect();            
            Sverts.insert(*anchor);

            assert!(Nu.is_subset(&Sverts));

        }
    }     

    #[test]
    fn normalize() {
        let mut G = EditGraph::from_file("./resources/Yeast.txt.gz").unwrap();
        G.remove_loops();
        let D = DegenGraph::from_graph(&G);
        let mut E = D.clone();
        let E_to_D:VertexMap<_> = E.normalize();
        let D_to_E:VertexMap<_> = E_to_D.iter().map(|(k,v)| (*v,*k)).collect();

        for (u,v) in D.edges() {
            assert!(E.adjacent(&D_to_E[&u], &D_to_E[&v]));
        }

        for (u,v) in E.edges() {
            assert!(D.adjacent(&E_to_D[&u], &E_to_D[&v]));
        }

        for u in D.vertices() {
            let eu = &D_to_E[u];
            assert_eq!(D.degree(u), E.degree(eu));
            assert_eq!(D.left_degree(u), E.left_degree(eu));
            assert_eq!(D.right_degree(u), E.right_degree(eu));
        }

    }   
}
