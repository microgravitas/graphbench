use fnv::{FnvHashMap, FnvHashSet};

use crate::iterators::*;
// use iterators::EdgeIterator

pub type Vertex = u32;
pub type VertexSet = FnvHashSet<Vertex>;

#[derive(Debug)]
pub struct Graph {
    adj: FnvHashMap<Vertex, VertexSet>,
    degs: FnvHashMap<Vertex, u32>,
    _m: u64
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        if self.num_vertices() != other.num_vertices() {
            return false
        }
        if self.num_edges() != other.num_edges() {
            return false
        }
        if self.degs != other.degs {
            return false
        }
        return self.adj == other.adj;
    }
}
impl Eq for Graph {}

impl Graph {
    pub fn new() -> Graph {
        Graph{adj: FnvHashMap::default(),
              degs: FnvHashMap::default(),
              _m: 0}
    }

    /*
        Basic properties and queries
    */
    pub fn num_vertices(&self) -> usize {
        self.adj.len()
    }

    pub fn num_edges(&self) -> u64 {
        self._m
    }

    pub fn adjacent(&self, u:Vertex, v:Vertex) -> bool {
        match self.adj.get(&u) {
            Some(N) => N.contains(&v),
            _ => false
        }
    }

    pub fn degree(&self, u:Vertex) -> u32 {
        *self.degs.get(&u).unwrap_or(&0)
    }

    /*
        Iteration and access
    */
    pub fn contains(&mut self, u:Vertex) -> bool {
        self.adj.contains_key(&u)
    }

    pub fn vertices(&self) -> VertexIterator {
        self.adj.keys()
    }

    pub fn edges(&self) -> EdgeIterator {
        EdgeIterator::new(self)
    }

    pub fn neighbours_iter(&self) -> NIterator {
        NIterator::new(self)
    }

    /*
        Neighbourhood methods
    */
    pub fn neighbours(&self, u:Vertex) -> NVertexIterator {
        match self.adj.get(&u) {
            Some(N) => N.iter(),
            None => panic!("Vertex not contained in graph")
        }
    }

    pub fn neighbourhood<I>(&self, it:I) -> VertexSet where I: Iterator<Item=Vertex>  {
        let mut res = FnvHashSet::default();
        let centers:VertexSet = it.collect();

        for &v in &centers {
            res.extend(self.neighbours(v));
        }

        res.retain(|&u| !centers.contains(&u));
        return res
    }

    pub fn closed_neighbourhood<I>(&self, it:I) -> VertexSet where I: Iterator<Item=Vertex>  {
        let mut res = FnvHashSet::default();
        let centers:VertexSet = it.collect();

        for &v in &centers {
            res.extend(self.neighbours(v));
        }

        return res
    }

    pub fn r_neighbours<I>(&self, u:Vertex, r:u32) -> VertexSet where I: Iterator<Item=Vertex>  {
        return self.r_neighbourhood([u].iter().cloned(), r)
    }

    pub fn r_neighbourhood<I>(&self, it:I, r:u32) -> VertexSet where I: Iterator<Item=Vertex>  {
        let mut res:VertexSet = FnvHashSet::default();
        res.extend(it);
        for _ in 0..r {
            let ext = self.closed_neighbourhood(res.iter().cloned());
            res.extend(ext);
        }
        return res
    }

    /*
        Modification
    */
    pub fn add_vertex(&mut self, u:Vertex) {
        if !self.adj.contains_key(&u) {
            self.adj.insert(u, FnvHashSet::default());
            self.degs.insert(u, 0);
        }
    }

    pub fn add_edge(&mut self, u:Vertex, v:Vertex) -> bool {
        self.add_vertex(u);
        self.add_vertex(v);
        if !self.adjacent(u, v) {
            self.adj.get_mut(&u).unwrap().insert(v);
            self.adj.get_mut(&v).unwrap().insert(u);
            self.degs.insert(u, self.degs[&u] + 1);
            self.degs.insert(v, self.degs[&v] + 1);
            self._m += 1;
            true
        } else {
            false
        }
    }

    pub fn remove_edge(&mut self, u:Vertex, v:Vertex) -> bool {
        if self.adjacent(u, v) {
            self.adj.get_mut(&u).unwrap().remove(&v);
            self.adj.get_mut(&v).unwrap().remove(&u);
            self.degs.insert(u, self.degs[&u] - 1);
            self.degs.insert(v, self.degs[&v] - 1);
            self._m -= 1;
            true
        } else {
            false
        }
    }

    pub fn remove_node(&mut self, u:Vertex) -> bool {
        if !self.contains(u) {
            false
        } else {
            let N = self.adj.get(&u).unwrap().clone();
            for &v in &N {
                self.adj.get_mut(&v).unwrap().remove(&u);
                self.degs.insert(v, self.degs[&v] - 1);
                self._m -= 1;
            }

            self.adj.remove(&u);
            self.degs.remove(&u);

            true
        }
    }

    pub fn remove_loops(&mut self) -> usize {
        let mut cands = Vec::new();
        for u in self.vertices().cloned() {
            if self.adjacent(u, u) {
                cands.push(u)
            }
        }

        let c = cands.len();
        for u in cands.into_iter() {
            self.remove_edge(u, u);
        }

        return c
    }

    pub fn remove_isolates(&mut self) -> usize {
        let cands:Vec<Vertex> = self.vertices().filter(|&u| self.degree(*u) == 0).cloned().collect();
        let c = cands.len();
        for u in cands.into_iter() {
            self.remove_node(u);
        }

        return c
    }

    /*
        Subgraphs and components
    */
    pub fn copy(&self) -> Graph {
        let mut G = Graph::new();
        for v in self.vertices() {
            G.add_vertex(*v);
            for u in self.neighbours(*v) {
                G.add_edge(*u, *v);
            }
        }

        return G
    }


    pub fn subgraph<I>(&self, vertices:I) -> Graph where I: Iterator<Item=Vertex> {
        let mut G = Graph::new();
        let selected:VertexSet = vertices.collect();
        for &v in &selected {
            G.add_vertex(v);
            let Nv:VertexSet = self.neighbours(v).cloned().collect();
            for u in Nv.intersection(&selected) {
                G.add_edge(*u, v);
            }
        }

        return G
    }

    pub fn components(&self) -> Vec<VertexSet> {
        let mut vertices:VertexSet = self.vertices().cloned().collect();
        let mut comps = Vec::new();
        while vertices.len() > 0 {
            let mut comp = VertexSet::default();
            let u = vertices.iter().cloned().next().unwrap();
            vertices.remove(&u);
            comp.insert(u);

            loop {
                let ext = self.neighbourhood(comp.iter().cloned());
                if ext.is_empty() {
                    break;
                }
                comp.extend(ext);
            }
            vertices.retain(|&u| !comp.contains(&u));

            comps.push(comp);
        }
        comps
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn components() {
        let mut G = Graph::new();
        let n:u32 = 10;
        for i in 0..(n/2) {
            G.add_edge(i, 5+i);
        }

        assert_eq!(G.components().len(), G.edges().count());

        for comp in G.components() {
            println!("{:?}", comp);
        }
    }

    #[test]
    fn equality() {
        let mut G = Graph::new();
        G.add_edge(0, 1);
        G.add_edge(1, 2);
        G.add_edge(2, 3);
        G.add_edge(3, 0);

        let mut H = G.subgraph([0,1,2,3].iter().cloned());
        assert_eq!(G, H);
        H.add_edge(0, 2);
        assert_ne!(G, H);
        H.remove_edge(0, 2);
        assert_eq!(G, H);
    }

    #[test]
    fn isolates_and_loops() {
        let mut G = Graph::new();
        G.add_edge(0, 1);
        G.add_edge(0, 0);
        G.add_edge(2, 2);

        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 3);

        G.remove_loops();

        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 1);

        G.remove_isolates();
        assert_eq!(G.num_vertices(), 2);
        assert_eq!(G.num_edges(), 1);

        G.remove_edge(0, 1);
        G.remove_isolates();

        assert_eq!(G, Graph::new());
    }

    #[test]
    fn basic_iteration() {
        let mut G = Graph::new();
        let n:u32 = 10;
        for i in 0..(n/2) {
            G.add_edge(i, 5+i);
        }

        assert_eq!(G.edges().count(), (n/2) as usize);
        assert_eq!(G.edges().count(), G.num_edges() as usize);
    }

    #[test]
    fn basic_operations() {
        let mut G = Graph::new();
        G.add_vertex(0);
        G.add_vertex(1);
        G.add_vertex(2);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(0, 1);
        assert_eq!(G.degree(0), 1);
        assert_eq!(G.degree(1), 1);
        assert_eq!(G.degree(2), 0);
        assert_eq!(G.num_vertices(), 3);
        assert_eq!(G.num_edges(), 1);

        G.remove_edge(0, 1);
        assert_eq!(G.degree(0), 0);
        assert_eq!(G.degree(1), 0);
        assert_eq!(G.num_edges(), 0);

        G.add_edge(0, 1);
        G.add_edge(0, 2);
        G.add_edge(1, 2);
        assert_eq!(G.degree(0), 2);
        assert_eq!(G.num_edges(), 3);

        G.remove_node(2);
        assert_eq!(G.degree(0), 1);
        assert_eq!(G.num_vertices(), 2);
        assert_eq!(G.num_edges(), 1);

        G.remove_node(1);
        assert_eq!(G.degree(0), 0);
        assert_eq!(G.num_vertices(), 1);
        assert_eq!(G.num_edges(), 0);

        G.remove_node(0);
        assert_eq!(G.num_vertices(), 0);
        assert_eq!(G.num_edges(), 0);
    }
}
