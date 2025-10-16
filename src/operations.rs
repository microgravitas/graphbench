use fxhash::FxHashMap;
use itertools::Itertools;

use crate::editgraph::EditGraph;
use crate::graph::*;
use crate::iterators::*;


pub trait GraphOperations<G> where G: Graph {
    fn invert(&self) -> EditGraph;
    fn merge(&self, other:&G) -> EditGraph;
    fn intersect(&self, other:&G) -> EditGraph;
    fn subdivide(&self, times:u32) -> EditGraph;

    fn lexicographic_product(&self, other:&G) -> (EditGraph, VertexMap<(Vertex,Vertex)>);
    fn cartesian_product(&self, other:&G) -> (EditGraph, VertexMap<(Vertex,Vertex)>);
}

impl<G> GraphOperations<G> for G where G: Graph {
    fn invert(&self) -> EditGraph {
        let mut res = EditGraph::with_capacity(self.len());
        res.add_vertices(self.vertices());
        for pair in self.vertices().combinations(2) {
            let (u,v) = (pair[0], pair[1]);
            if !self.adjacent(u, v) {
                res.add_edge(u, v);
            }
        }

        res
    }

    fn merge(&self, other:&G) -> EditGraph {
        let mut res = EditGraph::with_capacity(std::cmp::max(self.len(), other.len()));
        res.add_vertices(self.vertices());
        res.add_vertices(other.vertices());
        res.add_edges(self.edges());
        res.add_edges(other.edges());
        res
    }

    fn intersect(&self, other:&G) -> EditGraph {
        let mut res = EditGraph::new();
        res.add_vertices(self.vertices().filter(|u| other.contains(u)));
        res.add_edges(self.edges().filter(|(u,v)| other.adjacent(u,v)));
        res
    }

    fn subdivide(&self, times:u32) -> EditGraph {
        let mut res = EditGraph::with_capacity(self.num_vertices() + self.num_edges());
        let mut max_id:Vertex = 0;
        for u in self.vertices() {
            res.add_vertex(u);
            max_id = Vertex::max(max_id, *u);
        }
        let mut free_id = max_id + 1;

        for (u,v) in self.edges() {
            let mut last = u;
            for s in free_id..free_id+times {
                debug_assert!(!self.contains(&s));
                res.add_edge(&last, &s);
                last = s;
            }
            res.add_edge(&last, &v);
            free_id += times;
        }

        res
    }

    fn lexicographic_product(&self, other:&G) -> (EditGraph, VertexMap<(Vertex,Vertex)>) {
        let mut res = EditGraph::with_capacity(self.num_vertices()*other.num_edges());
        let mut map:FxHashMap<(Vertex,Vertex), Vertex> = FxHashMap::default();
        let mut inverse:VertexMap<(Vertex,Vertex)> = VertexMap::default();
        let mut free_id:Vertex = 0;
        for u in self.vertices() {
            for v in other.vertices() {
                map.insert((*u,*v), free_id);
                inverse.insert(free_id, (*u,*v));
                free_id += 1;
            }
        }

        /*
              a ---- b

            1,a     1,b
             |       |
            2,a  X  2,b
             |       |
            3,a     3,b

         */
        for ((u,v), x) in map.iter() {
            // This loop takes care of the edges inside each copy of `self`
            for w in self.neighbours(u) {
                let y = map.get(&(*w,*v)).unwrap();
                res.add_edge(x, y);
            }

            // This loop connects vertices between copies for each edge that
            // exists in the second graph
            for w in other.neighbours(v) {
                for uu in self.vertices() {
                    let y = map.get(&(*uu,*w)).unwrap();
                    res.add_edge(x, y);
                }
            }
        }

        (res, inverse)
    }

    fn cartesian_product(&self, other:&G) -> (EditGraph, VertexMap<(Vertex,Vertex)>) {
        let mut res = EditGraph::with_capacity(self.num_vertices()*other.num_edges());
        let mut map:FxHashMap<(Vertex,Vertex), Vertex> = FxHashMap::default();
        let mut inverse:VertexMap<(Vertex,Vertex)> = VertexMap::default();
        let mut free_id:Vertex = 0;
        for u in self.vertices() {
            for v in other.vertices() {
                map.insert((*u,*v), free_id);
                inverse.insert(free_id, (*u,*v));
                free_id += 1;
            }
        }

        for ((u,v), x) in map.iter() {
            // Fix u, find neighbours of v
            for w in other.neighbours(v) {
                // (u,v) -- (u,w)
                let y = map.get(&(*u,*w)).unwrap();
                res.add_edge(x, y);
            }

            // Find neighbours of u, fix v
            for w in self.neighbours(u) {
                // (u,v) -- (w,v)
                let y = map.get(&(*w,*v)).unwrap();
                res.add_edge(x, y);
            }
        }

        (res, inverse)
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

    #[test]
    fn invert() {
        let K5 = EditGraph::clique(5);
        let I5 = K5.invert();
        assert_eq!(I5.num_vertices(), 5);
        assert_eq!(I5.num_edges(), 0);

        let K5 = I5.invert();
        assert_eq!(K5.num_vertices(), 5);
        assert_eq!(K5.num_edges(), 10);
    }

    #[test]
    fn merge() {
        let G = EditGraph::from_iter([(0,1),(2,3)].into_iter());
        let H = EditGraph::from_iter([(1,2),(3,0)].into_iter());
        let C4 = EditGraph::cycle(4);

        assert_eq!(G.merge(&H), C4);
        assert_eq!(H.merge(&G), C4);

        let G = EditGraph::from_iter([(0,1),(1,2)].into_iter());
        let H = EditGraph::from_iter([(2,3),(3,4)].into_iter());
        let P5 = EditGraph::path(5);

        assert_eq!(G.merge(&H), P5);
        assert_eq!(H.merge(&G), P5);
    }

    #[test]
    fn intersect() {
        let G = EditGraph::from_iter([(0,1),(1,2)].into_iter());
        let H = EditGraph::from_iter([(0,1),(1,3)].into_iter());
        let R = EditGraph::from_iter([(0,1)].into_iter());

        assert_eq!(G.intersect(&H), R);
        assert_eq!(H.intersect(&G), R);

        let G = EditGraph::from_iter([(0,1)].into_iter());
        let H = EditGraph::from_iter([(0,2)].into_iter());
        let mut R = EditGraph::new();
        R.add_vertex(&0);

        assert_eq!(G.intersect(&H), R);
        assert_eq!(H.intersect(&G), R);
    }

    #[test]
    fn subdivide() {
        let G = EditGraph::path(2); // 0 -- 1
        let R = EditGraph::from_iter([(0,2),(2,1)].into_iter()); // 0 -- 2 -- 1
        assert_eq!(G.subdivide(1), R);

        let R = EditGraph::from_iter([(0,2),(2,3),(3,1)].into_iter()); // 0 -- 2 -- 3 -- 1
        assert_eq!(G.subdivide(2), R);

        // Ensure zero-subdivision is equal to copy
        let G = EditGraph::cycle(3); // 0 -- 1 -- 2 -- 0
        assert_eq!(G.subdivide(0), G);
    }

    #[test]
    fn cartesian_product() {
        let G = EditGraph::path(2); // 0 -- 1
        let H = EditGraph::path(2); // 0 -- 1
        let (R,_) = G.cartesian_product(&H); // Isomorphic to C4

        assert_eq!(R.num_vertices(), 4);
        assert_eq!(R.num_edges(), 4);
    }

    #[test]
    fn lexicographic_product() {
        let G = EditGraph::path(2); // 0 -- 1
        let H = EditGraph::path(2); // 0 -- 1
        let (R,_) = G.lexicographic_product(&H); // Isomorphic to K4

        assert_eq!(R.num_vertices(), 4);
        assert_eq!(R.num_edges(), 6);
    }
}
