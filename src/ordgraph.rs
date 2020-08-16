use fnv::{FnvHashMap, FnvHashSet};

use crate::graph::*;
use crate::iterators::*;

pub struct OrdGraph {
    indices: FnvHashMap<Vertex, usize>,
    nodes: Vec<OrdNode>,
    m: usize
}

pub struct OrdNode {
    in_arcs: VertexSet,
    out_arcs: VertexSet
}

impl Default for OrdNode {
    fn default() -> Self {
        OrdNode{ in_arcs: VertexSet::default(), out_arcs: VertexSet::default() }
    }
}


impl OrdGraph {
    pub fn with_degeneracy_order<G>(graph: &G) -> OrdGraph where G: Graph<Vertex> {
        let ord = graph.degeneracy_ordering();
        OrdGraph::with_order(graph, ord.iter())
    }

    pub fn with_order<'a, G, I>(graph: &G, order:I) -> OrdGraph
        where G: Graph<Vertex>, I: Iterator<Item=&'a Vertex>
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
                {nodes.get_mut(indices[&u]).unwrap().out_arcs.insert(v); }
                {nodes.get_mut(indices[&v]).unwrap().in_arcs.insert(u); }
            } else {
                {nodes.get_mut(indices[&v]).unwrap().out_arcs.insert(u); }
                {nodes.get_mut(indices[&u]).unwrap().in_arcs.insert(v); }
            }
        }

        OrdGraph {nodes, indices, m: graph.num_edges()}
    }
}
