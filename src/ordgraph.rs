use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::*;
use crate::iterators::*;

pub struct OrdGraph {
    indices: FnvHashMap<Vertex, usize>,
    nodes: Vec<OrdNode>,
    m: usize
}

pub struct OrdNode {
    v: Vertex,
    left: VertexSet,
    right: VertexSet
}

impl OrdNode {
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
    pub fn by_degeneracy<G>(graph: &G) -> OrdGraph where G: Graph {
        let ord = graph.degeneracy_ordering();
        OrdGraph::with_ordering(graph, ord.iter())
    }

    pub fn with_ordering<'a, G, I>(graph: &G, order:I) -> OrdGraph
        where G: Graph, I: Iterator<Item=&'a Vertex>
    {
        let order:Vec<_> = order.collect();
        let indices:FnvHashMap<_,_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (*u,i)).collect();
        let mut nodes:Vec<_> = Vec::new();

        for v in &order {
            nodes.push(OrdNode::new(v));
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

    fn swap(&mut self, u:&Vertex, v:&Vertex) {
        if (u == v) {
            return;
        }

        let (iu, iv) = match (self.indices.get(u), self.indices.get(v)) {
            (Some(iu), Some(iv)) => (*iu, *iv),
            _ => return
        };

        let (nu, nv) = (&self.nodes[iu], &self.nodes[iv]);

        // Recompute left/right neighbours for u when moved to iv
        {
            let mut nu = &mut self.nodes[iu];
            let (mut new_left, mut new_right) = (VertexSet::default(), VertexSet::default());
            for x in nu.neighbours() {
                let ix = self.indices.get(x).unwrap();
                if ix < &iv {
                    new_left.insert(*x);
                } else {
                    new_right.insert(*x);
                }
            }
            (nu.left, nu.right) = (new_left, new_right);
        }

        // Recompute left/right neighbours for v when moved to iu
        {
            let mut nv = &mut self.nodes[iv];
            let (mut new_left, mut new_right) = (VertexSet::default(), VertexSet::default());
            for x in nv.neighbours() {
                let ix = self.indices.get(x).unwrap();
                if ix < &iu {
                    new_left.insert(*x);
                } else {
                    new_right.insert(*x);
                }
            }
            (nv.left, nv.right) = (new_left, new_right);
        }        

        // Finally, swap u and v
        self.indices.insert(*u, iv);
        self.indices.insert(*v, iu);
        self.nodes.swap(iu, iv);
    }

    fn left_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.len()
        } else {
            0
        }
    }

    // fn left_degrees(&self) -> VertexMap<u32> {
    //     self.nodes.
    // }
    
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

    fn degree(&self, u:&Vertex) -> u32 {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            (node_u.left.len() + node_u.right.len()) as u32
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
            node_u.neighbours()
            // Box::new(node_u.left.iter().chain(node_u.right.iter()))
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