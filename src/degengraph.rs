use std::collections::{BTreeSet, HashMap};


use fxhash::FxHashSet;
use itertools::Itertools;

use crate::algorithms::GraphAlgorithms;
use crate::graph::*;
use crate::iterators::*;

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
    pub fn with_ordering<'a, G, I>(graph: &G, order:I) -> DegenGraph
        where G: Graph, I: Iterator<Item=&'a Vertex>
    {
        let order:Vec<_> = order.collect();
        let indices:VertexMap<_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (*u,i)).collect();

        let mut builder = DegenGraphBuilder::new();

        for u in order {
            let mut L = Vec::new();
            let iu = indices[&u];            
            for v in graph.neighbours(u) {
                let iv = indices[&v];
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
        &self.contents[iu+3..iu+3+num_neighbours]
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

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new(DegenOrderIterator::new(self))
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
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
        self.dgraph.right_neighbours.entry(*u).or_insert_with(VertexSet::default);

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
            self.dgraph.right_neighbours.entry(v).or_insert_with(VertexSet::default);
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
            DegenOrderIterator { dgraph: dgraph, curr_index: None }
        } else {
            DegenOrderIterator { dgraph: dgraph, curr_index: Some(0) }
        }
    }    
}

impl<'a> Iterator for DegenOrderIterator<'a> {
    type Item = &'a Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.curr_index {
            Some(ix) => {
                let contents = &self.dgraph.contents;
                let u = &contents[ix];

                let next_ix = contents[ix+1] as usize;
                if next_ix == ix { 
                    self.curr_index = None;
                } else {
                    self.curr_index = Some(next_ix);
                }

                Some(u)
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
                let u = contents[i];
                let next = contents[i+1] as usize;
                let num_neighbors = contents[i+2] as usize;

                let left = (i+3) as usize;
                let right = left+num_neighbors;

                if next == i {
                    self.current = None;
                } else {
                    self.current = Some(next);
                }

                Some((u, &contents[left..right]))
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

    #[test]
    fn basics() {
        let G = EditGraph::path(3);
        let order:Vec<_> = (0..3).collect();
        let D = DegenGraph::with_ordering(&G, order.iter());

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

}
