use crate::algorithms::GraphAlgorithms;
use fxhash::{FxHashMap, FxHashSet};

use crate::graph::*;
use crate::iterators::*;

pub struct OrdGraph {
    indices: FxHashMap<Vertex, usize>,
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
        let (ord, _) = graph.degeneracy();
        OrdGraph::with_ordering(graph, ord.iter())
    }
    
    pub fn with_ordering<'a, G, I>(graph: &G, order:I) -> OrdGraph
        where G: Graph, I: Iterator<Item=&'a Vertex>
    {
        let order:Vec<_> = order.collect();
        let indices:FxHashMap<_,_> = order.iter().cloned()
                .enumerate().map(|(i,u)| (*u,i)).collect();
        let mut nodes:Vec<_> = Vec::with_capacity(order.len());

        for v in &order {
            nodes.push(OrdNode::new(v));
            assert!(indices[v] == nodes.len()-1);
        }

        for (u,v) in graph.edges() {
            assert!(indices.contains_key(&u), "Vertex {} not contained in provided ordering", u);
            assert!(indices.contains_key(&v), "Vertex {} not contained in provided ordering", v);
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

    pub fn swap(&mut self, u:&Vertex, v:&Vertex) {
        if u == v {
            return;
        }

        let (iu, iv) = match (self.indices.get(u), self.indices.get(v)) {
            (Some(iu), Some(iv)) => (*iu, *iv),
            _ => return
        };

        // Recompute left/right neighbours for u when moved to iv
        // and for v when moved to iu.
        for (old_i,new_i) in vec![(iu,iv), (iv,iu)]{
            let mut n = &mut self.nodes[old_i];
            let (mut new_left, mut new_right) = (VertexSet::default(), VertexSet::default());
            for x in n.neighbours() {
                let ix = self.indices.get(x).unwrap();
                if ix < &new_i {
                    new_left.insert(*x);
                } else {
                    new_right.insert(*x);
                }
            }
            (n.left, n.right) = (new_left, new_right);
        }

        // Finally, swap u and v
        self.indices.insert(*u, iv);
        self.indices.insert(*v, iu);
        self.nodes.swap(iu, iv);
    }

    pub fn left_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.left.len()
        } else {
            0
        }
    }

    pub fn left_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, n.left.len() as u32);
        }
        res
    }
    
    pub fn right_degree(&self, u:&Vertex) -> usize {
        if let Some(iu) = self.indices.get(u) {
            let node_u = &self.nodes[*iu];
            node_u.right.len()
        } else {
            0
        }
    }    
    
    pub fn right_degrees(&self) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, n.right.len() as u32);
        }
        res
    }

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

    pub fn wreach_sets(&self, depth:u32) -> VertexMap<VertexMap<u32>> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, VertexMap::default());
        }
        for u in self.vertices() {
            for (d, layer) in self.right_bfs(u, depth).iter().skip(1).enumerate() {
                for v in layer {
                    assert!(*v != *u);
                    res.get_mut(v).unwrap().insert(*u, (d+1) as u32); 
                }
            }
        }
        res
    }    

    pub fn wreach_sizes(&self, depth:u32) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for n in &self.nodes {
            res.insert(n.v, 0);
        }
        for u in self.vertices() {
            for layer in self.right_bfs(u, depth).iter().skip(1) {
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