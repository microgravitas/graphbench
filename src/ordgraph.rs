use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::*;
use crate::iterators::*;

pub struct OrdGraph {
    indices: FnvHashMap<Vertex, usize>,
    nodes: Vec<OrdNode>,
    m: usize
}

pub struct OrdNode {
    left: VertexSet,
    right: VertexSet
}

impl Default for OrdNode {
    fn default() -> Self {
        OrdNode{ left: VertexSet::default(), right: VertexSet::default() }
    }
}

impl OrdGraph {
    pub fn with_degeneracy_order<G>(graph: &G) -> OrdGraph where G: Graph {
        let ord = graph.degeneracy_ordering();
        OrdGraph::with_order(graph, ord.iter())
    }

    pub fn with_order<'a, G, I>(graph: &G, order:I) -> OrdGraph
        where G: Graph, I: Iterator<Item=&'a Vertex>
    {
        let order:Vec<_> = order.collect();
        let indices:FnvHashMap<_,_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (*u,i)).collect();
        let mut nodes:Vec<_> = Vec::new();

        for _ in &order {
            nodes.push(OrdNode::default());
        }

        for (u,v) in graph.edges() {
            // let nU = nodes.get_mut(indices[&u]).unwrap();
            // let nV = nodes.get_mut(indices[&v]).unwrap();
            if u < v {
                {nodes.get_mut(indices[&u]).unwrap().right.insert(v); }
                {nodes.get_mut(indices[&v]).unwrap().left.insert(u); }
            } else {
                {nodes.get_mut(indices[&v]).unwrap().right.insert(u); }
                {nodes.get_mut(indices[&u]).unwrap().left.insert(v); }
            }
        }

        OrdGraph {nodes, indices, m: graph.num_edges()}
    }

    fn left_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.len()
        } else {
            0
        }
    }
    
    fn right_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.right.len()
        } else {
            0
        }
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

    fn degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.len() + node_u.right.len()
        } else {
            0
        }
    }

    fn vertices<'a>(&'a self) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        Box::new( self.indices.keys() )
    }

    fn neighbours<'a>(&'a self, u:&Vertex) -> Box<dyn Iterator<Item=&Vertex> + 'a> {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            Box::new(node_u.left.iter().chain(node_u.right.iter()))
        } else {
            Box::new(std::iter::empty::<&Vertex>())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;


    #[test] 
    fn basic_operations() {

    }
}