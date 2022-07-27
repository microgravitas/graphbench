use std::collections::BTreeSet;

use fxhash::FxHashSet;
use itertools::Itertools;

use crate::graph::*;
use crate::iterators::*;

pub struct DegenGraph {
    indices: VertexMap<usize>,
    pub(crate) contents:Vec<u32>,
    right_degrees: VertexMap<u32>,
    edges: EdgeSet
}

impl DegenGraph {
    fn new() -> Self {
        DegenGraph{ indices: VertexMap::default(),
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
        Box::new(DegenOrderIterator::new(self))
    }

    fn neighbours<'a>(&'a self, _u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        panic!("DegenGraph::neighbours not supported.")
    }
}

impl LinearGraph for DegenGraph {
    fn index_of(&self, u:&Vertex) -> usize {
        *self.indices.get(u).unwrap_or_else(|| panic!("{u} is not a vertex in this graph."))
    }

    fn left_neighbours(&self, u:&Vertex) -> Vec<Vertex> {
        self.left_neighbours_slice(u).to_vec()
    }

    fn right_degree(&self, u:&Vertex) -> u32 {
        *self.right_degrees.get(u).unwrap_or(&0)
    }
}

pub struct DegenGraphBuilder {
    last_index: Option<u32>,
    depth: u32,
    dgraph: DegenGraph
}

impl DegenGraphBuilder {
    pub fn new(depth:u32) -> Self {
        DegenGraphBuilder{ last_index: None, depth, dgraph: DegenGraph::new() }
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

        // Sort neighbours according to order
    
        for &v in neighbours {
            *self.dgraph.right_degrees.entry(v).or_insert(0) += 1;

            if *u < v { 
                self.dgraph.edges.insert((*u,v));
            } else {
                self.dgraph.edges.insert((v,*u));
            }            
        }
        
        // Push | num_neighbours |
        contents.push(neighbours.len() as u32);

        // Finally write all neighbours
        contents.extend(neighbours.iter());

        self.last_index = Some(index_u);
    }
}

pub struct DegenOrderIterator<'a> {
    rgraph: &'a DegenGraph,
    curr_index: Option<usize>
}

impl<'a> DegenOrderIterator<'a> {
    pub fn new(dgraph: &'a DegenGraph) -> Self {
        if dgraph.len() == 0 {
            DegenOrderIterator { rgraph: dgraph, curr_index: None }
        } else {
            DegenOrderIterator { rgraph: dgraph, curr_index: Some(0) }
        }
    }    
}

impl<'a> Iterator for DegenOrderIterator<'a> {
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

                let left = (u+3) as usize;
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
    fn consistency() {
        let G = EditGraph::from_txt("./resources/karate.txt").unwrap();
        let O = OrdGraph::by_degeneracy(&G);
        let D = O.to_wreach_graph::<3>();

        for u in G.vertices() {
            assert_eq!(G.degree(u), D.degree(u));
        }
    }

    #[test]
    fn order_iteration() {
        let G = EditGraph::path(20);
        let order:Vec<_> = (0..20).rev().collect();
        let O = OrdGraph::with_ordering(&G, order.iter());    
        let D = O.to_degeneracy_graph();
        
        assert_eq!(order, D.vertices().copied().collect_vec());
    }

    #[test] 
    fn count_cliques() {
        let G = EditGraph::clique(5);
        let O = OrdGraph::with_ordering(&G, vec![0,1,2,3,4].iter());    
        let D = O.to_degeneracy_graph();

        assert_eq!(D.count_max_cliques(), 1);

        let G = EditGraph::complete_kpartite([5,5,5].iter());
        let O = OrdGraph::by_degeneracy(&G);    
        let D = O.to_degeneracy_graph();

        assert_eq!(D.count_max_cliques(), 5*5*5);        
    }

}
