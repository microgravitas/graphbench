use std::collections::BTreeSet;

use fxhash::{FxHashMap, FxHashSet};

use crate::graph::*;
use crate::iterators::*;

pub trait LinearGraphAlgorithms {
    fn right_bfs(&self, root:&Vertex, dist:u32) -> Vec<VertexSet>;
    fn sreach_set(&self, u:&Vertex, r:u32) -> VertexMap<u32>;
    fn wreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>>;
    fn wreach_sizes(&self, r:u32) -> VertexMap<u32>;
    fn count_max_cliques(&self) -> u64;
}

impl<L> LinearGraphAlgorithms for L where L: LinearGraph {
    /// Conducts a bfs from `root` for `dist` steps ignoring all vertices
    /// left of `root`.
    /// 
    /// Returns the bfs as a sequence of layers.
    fn right_bfs(&self, root:&Vertex, dist:u32) -> Vec<VertexSet> {
        let mut seen:VertexSet = VertexSet::default();
        let iroot = self.index_of(root);
        let root = *root;

        let mut res = vec![VertexSet::default(); (dist+1) as usize];
        res[0].insert(root);
        seen.insert(root);

        for d in 1..=(dist as usize) {
            let (part1, part2) = res.split_at_mut(d as usize);

            for u in part1[d-1].iter() {
                for v in self.neighbours(u) {
                    let iv = self.index_of(v);
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
    fn sreach_set(&self, u:&Vertex, r:u32) -> VertexMap<u32> {
        let bfs = self.right_bfs(u, r-1);
        let mut res = VertexMap::default();
        
        let iu = self.index_of(u);
        for (d, layer) in bfs.iter().enumerate() {
            for v in layer {
                for x in self.left_neighbours(v) {
                    let ix = self.index_of(&x);
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
    fn wreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            res.insert(*u, VertexMap::default());
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
    fn wreach_sizes(&self, r:u32) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            res.insert(*u, 0);
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

    /// Computes the total number of maximal cliques in the graph. 
    fn count_max_cliques(&self) -> u64 {
        let mut results = FxHashSet::<BTreeSet<Vertex>>::default(); 

        for (v,neighbours) in self.left_neighbourhoods() {
            let mut include = VertexSet::default();
            include.insert(v);
            let exclude = VertexSet::default();
            let maybe = neighbours.iter().cloned().collect();
            println!("Anchor {v} with left-neighbours {neighbours:?}");
            println!("I = {include:?}, M = {maybe:?}, X = {exclude:?}");
            bk_pivot_count(self, &v, &neighbours, &mut include, maybe, exclude, &mut results, 0);
        }
        results.len() as u64
    }
}

fn bk_pivot_count<'a, L: LinearGraph>(graph:&'a L, v:&Vertex, vertices:&[Vertex], include:&mut VertexSet, mut maybe:VertexSet, mut exclude:VertexSet, results:&mut FxHashSet<BTreeSet<Vertex>>, depth:usize) {
    let indent = format!("{:indent$}", "", indent=2*(depth+1));

    println!("{indent}I = {include:?}, M = {maybe:?}, X = {exclude:?}");
    if maybe.is_empty() && exclude.is_empty() {
        // `include` is a maximal clique
        
        if include.len() > 1 {
            println!("{indent}M and X empty: I = {include:?} is a (local) maxm. clique");

            // Add new maximal clique
            let clique:BTreeSet<Vertex> = include.iter().copied().collect();
            results.insert(clique);
        }

        if include.len() >= 3 { 
            // Remove prefix of clique. While we know that it must have been added to `results`
            // at some point, it could have been removed in the meantime.
            let mut clique:BTreeSet<Vertex> = include.iter().copied().collect();
            clique.remove(v);
            results.remove(&clique);
        }

        return ;
    }

    // Choose the last vertex in ordering which is in either `maybe` or `exclude`
    // as the pivot vertex
    let mut u = None;
    let mut iu = 0;
    for i in (0..vertices.len()).rev() {
        let cand = vertices[i];
        if maybe.contains(&cand) || exclude.contains(&cand) {
            u = Some(cand);
            iu = i;
            break;
        }
    }
    let u = u.expect("If this fails there is a bug");

    // Compute u's *left* neighbourhood inside of `vertices`. 
    let left_neighbours:Vec<Vertex> = vertices[0..iu].iter()
            .filter_map(|v| if graph.adjacent(&u, v) {Some(*v)} else {None} ).collect();

    let left_neighbours_set:VertexSet = left_neighbours.iter().cloned().collect();
    println!("{indent}Chose pivot {u:?} with left-neighbours {left_neighbours:?} ");

    for w in vertices[0..=iu].iter().rev() {
        // We ignore `w` if it is not a maybe-vertex. We also ignore it
        // if it is a neighbour of the pivot `u`.
        if !maybe.contains(w) || left_neighbours_set.contains(w) {
            continue
        } 

        // Recursion
        include.insert(*w);
        bk_pivot_count(graph, 
                v,
                &left_neighbours, 
                include,
                maybe.intersection(&left_neighbours_set).cloned().collect(),
                exclude.intersection(&left_neighbours_set).cloned().collect(),
                results,
                depth+1 );
        include.remove(w);

        maybe.remove(w);
        exclude.insert(*w);
    }
}    