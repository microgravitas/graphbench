use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::u32;

use fxhash::{FxHashMap, FxHashSet};
use union_find_rs::prelude::*;

use crate::editgraph::EditGraph;
use crate::ordgraph::OrdGraph;
use crate::graph::*;
use crate::iterators::*;

use itertools::*;

/// Implements various algorithms for the [LinearGraph](crate::graph::LinearGraph) trait.
pub trait LinearGraphAlgorithms {
    /// Conducts a bfs from `root` for `dist` steps ignoring all vertices
    /// left of `root`.
    /// 
    /// Returns the bfs as a sequence of layers.    
    fn right_bfs(&self, root:&Vertex, dist:u32) -> Vec<VertexSet>;


    /// Computes all strongly $r$-reachable vertices to $u$. 
    /// 
    /// A vertex $v$ is strongly $r$-reachable from $u$ if there exists a $u$-$v$-path in the graph
    /// of length at most $r$ where $v$ is the only vertex of the path that comes before $u$ in the
    /// ordering.
    /// 
    /// Returns a map with all vertices that are strongly $r$-reachable
    /// from $u$. For each member $v$ in the map the corresponding values represents
    /// the distance $d \\leq r$ at which $v$ is strongly reachable from $u$.
    fn sreach_set(&self, u:&Vertex, r:u32) -> VertexMap<u32>;

    /// Compute all strongly $r$-reachable sets as a map.
    /// 
    /// For each vertex, the return value contains a map whose keys are the strongly $r$-reachable
    /// vertices and the values are the respective distances at which those vertices can be
    /// strongly reached.
    fn sreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>>;

  /// Returns for each vertex the size of its $r$-weakly reachable set. 
    /// This method uses less memory than [sreach_sets](LinearGraphAlgorithms::sreach_sets).   
    fn sreach_sizes(&self, r:u32) -> VertexMap<u32>;

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
    fn wreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>>;

    /// Returns for each vertex the size of its $r$-weakly reachable set. 
    /// This method uses less memory than [wreach_sets](LinearGraphAlgorithms::wreach_sets).    
    fn wreach_sizes(&self, r:u32) -> VertexMap<u32>;

    /// Computes the total number of maximal cliques in the graph. 
    /// 
    /// This count includes vertices of degree zero and singles edges which 
    /// cannot be extended into a triangle.
    fn count_max_cliques(&self) -> u64;


    // Computes a greedy colouring of this graph.
    fn colour_greedy(&self) -> VertexColouring<u32>;

    /// Computes an approximate r-dominating set of the graph. If `witness` is set
    /// to `true`, the algorithm also computes an r-scatterd subset of the dominating set,
    /// e.g. a set in which all vertices have pairwise distance at least 2r+1 (their r-neighbourhoods
    /// are pairswise disjoint).
    /// 
    /// Computing the witness is more expensive than computing the dominating set.
    /// The method returns the r-dominating set, a vertex map containting the minimum distance of
    /// every vertex to the r-dominating set, and optionally a vertex colouring of the r-dominating set
    /// with the property that every colour class is 2r-scattered.
    /// 
    /// To retrieve the largest r-scattered set from the colouring, use [graph::VertexColouring::majority_set]
    fn domset(&self, radius:u32, witness:bool) -> (VertexSet, VertexMap<u32>, Option<VertexColouring<u32>>);

    // Computes an approximate r-domainting set for a specific subset of vertices `target`, similar to 
    // [domset](LinearGraphAlgorithms::domset). If `witness` is set to `true`, the algorithm also
    // computes and r-scattered subset of the `target` set. 
    //
    /// Computing the witness is more expensive than computing the dominating set.
    /// The method returns the r-dominating set, a vertex map containting the minimum distance of
    /// every vertex to the r-dominating set, and optionally a vertex colouring of the r-dominating set
    /// with the property that every colour class is 2r-scattered.
    /// 
    fn domset_with_target<V,I>(&self, radius:u32, witness:bool, target:I) -> (VertexSet, VertexMap<u32>, Option<VertexColouring<u32>>) 
    where V: Borrow<Vertex>, I:IntoIterator<Item=V>;    


    /// Computes a vertex colouring such that every pair of vertices with the same colour
    /// have distance at least `distance` to each other.
    fn scattered_colouring<V,I>(&self, distance:u32, target:I) -> VertexColouring<u32>
    where V: Borrow<Vertex>, I:IntoIterator<Item=V>;    
}

impl<L> LinearGraphAlgorithms for L where L: LinearGraph {
    fn right_bfs(&self, root:&Vertex, dist:u32) -> Vec<VertexSet> {
        let mut seen:VertexSet = VertexSet::default();
        let iroot = self.index_of(root);
        let root = *root;

        let mut res = vec![VertexSet::default(); (dist+1) as usize];
        res[0].insert(root);
        seen.insert(root);

        for d in 1..=(dist as usize) {
            let (part1, part2) = res.split_at_mut(d);

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

    fn sreach_sets(&self, r:u32) -> VertexMap<VertexMap<u32>> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            let sreach = self.sreach_set(u, r);
            res.insert(*u, sreach);
        }
        res
    }
    
    fn sreach_sizes(&self, r:u32) -> VertexMap<u32> {
        let mut res = VertexMap::default();
        for u in self.vertices() {
            let sreach = self.sreach_set(u, r);
            res.insert(*u, sreach.len() as u32);
        }
        res
    }       

    fn count_max_cliques(&self) -> u64 {
        let mut results = FxHashSet::<BTreeSet<Vertex>>::default(); 

        for (v,neighbours) in self.left_neighbourhoods() {
            let mut include = VertexSet::default();
            include.insert(v);
            let exclude = VertexSet::default();
            let maybe = neighbours.iter().cloned().collect();
            bk_pivot_count(self, &v, &neighbours, &mut include, maybe, exclude, &mut results);
        }
        results.len() as u64
    }

    fn colour_greedy(&self) -> VertexColouring<u32> {
        let mut colours:VertexColouring<u32> = VertexColouring::default();
        let mut used_colours:Vec<u32> = vec![];
        for (v, N) in self.left_neighbourhoods() {
            let Ncols:Vec<u32> = N.iter().map(|x| colours[x]).collect();
            if Ncols.len() == used_colours.len() {
                let new_colour = used_colours.len() as u32;
                used_colours.push(new_colour); 
                colours.insert(v, new_colour);
                continue
            }

            // Find unused colour
            for c in used_colours.iter() {
                if !Ncols.contains(c) {
                    colours.insert(v, *c);
                    break
                }
            }
        }

        colours
    }

    // pub fn contract<V, I>(&mut self, mut vertices:I) -> Vertex 
        // where V: Borrow<Vertex>,  I: Iterator<Item=V> {
    fn domset(&self, radius:u32, witness:bool) -> (VertexSet, VertexMap<u32>, Option<VertexColouring<u32>>) {
        self.domset_with_target(radius, witness, self.vertices())
    }

    fn domset_with_target<V,I>(&self, radius:u32, witness:bool, target:I) -> (VertexSet, VertexMap<u32>, Option<VertexColouring<u32>>) 
    where V: Borrow<Vertex>, I:IntoIterator<Item=V> {
        let target:VertexSet = target.into_iter().map(|u| *u.borrow()).collect();
        if self.is_empty() || target.is_empty() {
            if witness {
                return (VertexSet::default(), VertexMap::default(), Some(VertexColouring::default()))
            } else {
                return (VertexSet::default(), VertexMap::default(), None)
            }
        }

        let mut domset = VertexSet::default();
        let mut scattered = VertexSet::default();
        let mut dom_distance = FxHashMap::<Vertex, u32>::default();
        let mut dom_counter = FxHashMap::<Vertex, u32>::default();

        let cutoff = (2*radius).pow(2);
        let n = self.num_vertices() as i64;

        // Sort by _decreasing_ in-degree, tie-break by
        // total degree.
        let order:Vec<Vertex> = self.vertices()
                .cloned()
                .sorted_by_key(|u| -(self.left_degree(u) as i64)*n - (self.degree(u) as i64))
                .collect();
        let undominated = radius+1;

        let wreach = self.wreach_sets(radius);

        for v in order.iter() {
            if target.contains(v) {
                dom_distance.insert(*v, undominated);
                dom_counter.insert(*v, 0);
            } else {
                // Mark as already dominated. We use a radius `radius` here so that 
                // no neighbour is marked as dominated because of this trick.
                dom_distance.insert(*v, radius);
                dom_counter.insert(*v, 0);
            }
        }

        for v in order {
            // Update domination distance of v via its in-neighbours
            for (u,dist) in wreach.get(&v).unwrap().iter() {
                *dom_distance.get_mut(&v).unwrap() = u32::min(dom_distance[&v],  dist+dom_distance[u]);
            }

            // If v is a already dominated we have nothing else to do
            if dom_distance[&v] <= radius {
                continue
            }

            // Otherwise, we add v to the dominating set
            assert!(target.contains(&v));
            domset.insert(v);
            scattered.insert(v);
            dom_distance.insert(v, 0);

            // Update dominating distance for v's in-neighbours
            for (u,dist) in wreach.get(&v).unwrap().iter() {
                *dom_counter.get_mut(u).unwrap() += 1;
                *dom_distance.get_mut(u).unwrap() = u32::min(dom_distance[u], *dist);

                // If a vertex has been an in-neigbhour of a domination node for
                // too many time, we include it in the domset.
                if dom_counter[u] > cutoff && !domset.contains(u) {
                    domset.insert(*u);
                    dom_distance.insert(*u, 0);

                    for (x,xdist) in wreach.get(u).unwrap().iter() {
                        *dom_distance.get_mut(x).unwrap() = u32::min(dom_distance[x], *xdist);
                    }
                }
            }
        }

        if !witness {
            return (domset, dom_distance, None)
        }

        // We need to construct an auxilliary graph in with `scattered` as nodes and
        // connect each pair with distance < 2*radius by an edge.
        let wreach = self.wreach_sets(2*radius);

        // Collect 'out-neighbours' so we can compute auxilliary graph
        let mut out_neigbhours:VertexMap<VertexMap<u32>> = VertexMap::default();
        for x in &scattered { 
            for (w,dist) in wreach.get(x).unwrap().iter() {
                out_neigbhours.entry(*w).or_default().insert(*x, *dist);
            }
        }

        let mut H = EditGraph::new();
        H.add_vertices(scattered.iter().cloned());
        for (w, O) in out_neigbhours {
            for pair in O.iter().combinations(2) {
                let (x, dx) = pair[0];
                let (y, dy) = pair[1];
                if dx + dy <= 2*radius {
                    H.add_edge(x, y);
                }
            }

            if !scattered.contains(&w) {
                continue
            }

            for (x,dx) in O.iter() {
                assert!(*dx <= 2*radius);
                H.add_edge(&w,x);
            }
        }
        assert_eq!(H.num_vertices(), scattered.len());

        // Compute greedy colouring and return largest monochromatic
        // subset. This set is guaranteed to be 
        let OH = OrdGraph::by_degeneracy(&H);
        let colours = OH.colour_greedy();

        (domset, dom_distance, Some(colours.into()))
    }    

    fn scattered_colouring<V,I>(&self, distance:u32, target:I) -> VertexColouring<u32>
    where V: Borrow<Vertex>, I:IntoIterator<Item=V> {
        let mut target:VertexSet = target.into_iter().map(|u| *u.borrow()).collect();
        if self.is_empty() || target.is_empty() {
            return VertexColouring::default();
        }

        let mut colouring = VertexColouring::default();

        let radius = distance.div_ceil(2);
        let wreach = self.wreach_sets(distance);
        let undominated = radius+1;

        // Sort by _decreasing_ in-degree, tie-break by
        // total degree.
        let cutoff = distance.pow(2);        
        let n = self.num_vertices() as i64;        
        let order:Vec<Vertex> = self.vertices()
                .cloned()
                .sorted_by_key(|u| -(self.left_degree(u) as i64)*n - (self.degree(u) as i64))
                .collect();        

        while !target.is_empty() {
            let mut domset = VertexSet::default();
            let mut scattered = VertexSet::default();
            let mut dom_distance:VertexMap<u32> = VertexMap::default();
            let mut dom_counter:VertexMap<u32> = VertexMap::default();

            for v in &order {
                if target.contains(v) {
                    dom_distance.insert(*v, undominated);
                    dom_counter.insert(*v, 0);
                } else {
                    // Mark as already dominated. We use a radius `radius` here so that 
                    // no neighbour is marked as dominated because of this trick.
                    dom_distance.insert(*v, radius);
                    dom_counter.insert(*v, 0);
                }
            }

            for &v in &order {
                // Update domination distance of v via its in-neighbours
                for (u,dist) in wreach.get(&v).unwrap().iter() {
                    *dom_distance.get_mut(&v).unwrap() = u32::min(dom_distance[&v],  dist+dom_distance[u]);
                }

                // If v is a already dominated we have nothing else to do
                if dom_distance[&v] <= radius {
                    continue
                }

                // Otherwise, we add v to the dominating set
                assert!(target.contains(&v), "{v} is not a target vertex");
                domset.insert(v);
                scattered.insert(v);
                dom_distance.insert(v, 0);

                // Update dominating distance for v's in-neighbours
                for (u,dist) in wreach.get(&v).unwrap().iter() {
                    *dom_counter.get_mut(u).unwrap() += 1;
                    *dom_distance.get_mut(u).unwrap() = u32::min(dom_distance[u], *dist);

                    // If a vertex has been an in-neigbhour of a domination node for
                    // too many time, we include it in the domset.
                    if dom_counter[u] > cutoff && !domset.contains(u) {
                        domset.insert(*u);
                        dom_distance.insert(*u, 0);

                        for (x,xdist) in wreach.get(u).unwrap().iter() {
                            *dom_distance.get_mut(x).unwrap() = u32::min(dom_distance[x], *xdist);
                        }
                    }
                }
            }

            // Collect 'out-neighbours' so we can compute auxilliary graph
            let mut out_neigbhours:VertexMap<VertexMap<u32>> = VertexMap::default();
            for x in &scattered { 
                for (w,dist) in wreach.get(x).unwrap().iter() {
                    out_neigbhours.entry(*w).or_default().insert(*x, *dist);
                }
            }

            let mut H = EditGraph::new();
            H.add_vertices(scattered.iter().cloned());
            for (w, O) in out_neigbhours {
                for pair in O.iter().combinations(2) {
                    let (x, dx) = pair[0];
                    let (y, dy) = pair[1];
                    if dx + dy <= 2*radius {
                        H.add_edge(x, y);
                    }
                }

                if !scattered.contains(&w) {
                    continue
                }

                for (x,dx) in O.iter() {
                    if *dx > 2*radius {
                        continue
                    }
                    H.add_edge(&w,x);
                }
            }
            assert_eq!(H.num_vertices(), scattered.len());

            // Compute greedy colouring and return largest monochromatic
            // subset. This set is guaranteed to be 
            let OH = OrdGraph::by_degeneracy(&H);
            let colours = OH.colour_greedy();
            colouring.disjoint_extend(&colours);

            assert_eq!(colours.len(), scattered.len());

            // Remove coloured vertices from target
            target.retain(|u| !colouring.contains(&u));
        }

        // Attempt to improve colouring
        let mut improved = true;
        while improved {
            improved = false;

            let classes = colouring.invert();
            let mut order = classes.iter().collect_vec();
            order.sort_by_key(|(_,set)| set.len());

            // Set of 'used' colours for this round
            let mut used:FxHashSet<u32> = FxHashSet::default();
            for ((col1,set1),(col2,set2)) in order.iter().tuple_combinations() {
                if used.contains(col1) || used.contains(col2) {
                    continue
                }

                let dist = small_distance_sets(&wreach, distance, set1.iter(), set2.iter());
                if let Some(dist) = dist {
                    if dist < 2*radius {
                        // Cannot merge these two classes
                        continue;
                    }
                }

                // Merge these two classes, mark as used up for this round
                for x in set1.iter() {
                    colouring.insert(*x, **col2);
                }
                used.insert(**col1);
                used.insert(**col2);
                improved = true;
            }
        };

        colouring
    }    
}

// TODO: These methods should probably go into its own dedicated 'WReach' struct
fn small_distance_sets<V,I1,I2>(wreach:&VertexMap<VertexMap<u32>>, distance:u32, X:I1, Y:I2) -> Option<u32>
    where V: Borrow<Vertex>, I1: IntoIterator<Item=V>, I2: IntoIterator<Item=V> {
    let X:VertexSet = X.into_iter().map(|u| *u.borrow()).collect();
    let Y:VertexSet = Y.into_iter().map(|u| *u.borrow()).collect();

    let mut dist = u32::MAX;
    for x in &X {
        for y in &Y {
            if x == y {
                return Some(0)
            }

            if let Some(d) = small_distance(wreach, distance, x, y) {
                dist = u32::min(dist, d);
            }
        }
    }
    if dist == u32::MAX {
        return None
    }
    Some(dist)
}

fn small_distance(wreach:&VertexMap<VertexMap<u32>>, distance:u32, x:&Vertex, y:&Vertex) -> Option<u32> {
    if x == y {
        return Some(0)
    }

    let Wx = &wreach[x];
    let Wy = &wreach[y];

    let mut dist = u32::MAX;
    if Wx.contains_key(y) {
        dist = Wx[y];
    }
    if Wy.contains_key(x) {
        dist = dist.min(Wy[x]);
    }

    for w in Wx.keys() {
        if !Wy.contains_key(w) {
            continue
        }
        dist = dist.min(Wx[&w] + Wy[&w]);
    }

    if dist > distance {
        return None
    }
    Some(dist)
}

fn bk_pivot_count<L: LinearGraph>(graph:&L, v:&Vertex, vertices:&[Vertex], include:&mut VertexSet, mut maybe:VertexSet, mut exclude:VertexSet, results:&mut FxHashSet<BTreeSet<Vertex>>) {
    if maybe.is_empty() && exclude.is_empty() {
        // `include` is a maximal clique
        
        // Add new maximal clique
        let clique:BTreeSet<Vertex> = include.iter().copied().collect();
        results.insert(clique);

        if include.len() > 1 { 
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
                results);
        include.remove(w);

        maybe.remove(w);
        exclude.insert(*w);
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
    use num::Integer;

    use super::*;
    use crate::editgraph::EditGraph;
    use crate::ordgraph::OrdGraph;
    use crate::algorithms::lineargraph::*;
    use crate::io::*;

    #[test]
    fn colouring() {
        let G = EditGraph::clique(5);
        let D = OrdGraph::by_degeneracy(&G);

        let colouring = D.colour_greedy();
        let colours:FxHashSet<u32> = colouring.colours().cloned().collect();
        assert_eq!(colours.len(), 5 );

        let mut G = EditGraph::new();
        G.add_vertices(vec![1,2,3,4,5].into_iter());
        let D = OrdGraph::by_degeneracy(&G);

        let colouring = D.colour_greedy();
        let colours:FxHashSet<u32> = colouring.colours().cloned().collect();

        assert_eq!(colours.len(), 1 );        
    }

    #[test]
    fn domset() {
        let G = EditGraph::from_txt("resources/karate.txt").unwrap();
        let OG = OrdGraph::by_degeneracy(&G);

        for r in 1..5 {
            let (D, _, Some(C)) = OG.domset(r, true) else { panic!("No witness returned") };
            let (S, _) = C.majority_set().unwrap();
            println!("Found Karate {}-domset of size {} with scattered set of size {}", r, D.len(), S.len());
            assert_eq!(G.r_neighbourhood(D.iter(), r as usize).len(), G.num_vertices());
        }

        let G = EditGraph::grid(50, 50);
        let OG = OrdGraph::by_degeneracy(&G);
        let r = 3;
        let (D, _, Some(C)) = OG.domset(r, true) else { panic!("No witness returned") };
        let (S, _) = C.majority_set().unwrap();
        println!("Found {}-domset of size {} with scattered set of size {} in 50x50 grid", r, D.len(), S.len());
        assert_eq!(G.r_neighbourhood(D.iter(), r as usize).len(), G.num_vertices());

        // Check distances of scattered set
        let mut DTFG = crate::dtfgraph::DTFGraph::orient(&G);
        DTFG.augment(2*r as usize, 1);

        for tup in S.iter().combinations(2) {
            let (x,y) = (tup[0], tup[1]);
            let dist = DTFG.small_distance(x,y);
            if let Some(d) = dist  {
                assert!(d >= 2*r+1);
            }
        }
    }

    #[test]
    fn domset_partial() {
        let G = EditGraph::grid(10, 10);
        let OG = OrdGraph::by_degeneracy(&G);
        let T:VertexSet = (0..50).collect();
        let r = 2;
        let (D, _, Some(C)) = OG.domset_with_target(r, true, &T) else { panic!("No witness returned") };

        let (S, _) = C.majority_set().unwrap();

        println!("Found {}-domset of size {} with scattered set of size {} in 10x10 grid", r, D.len(), S.len());
        println!("Target = {T:?}");
        println!("Domset = {D:?}");
        println!("Scattered = {S:?}");        
        let dominated = G.r_neighbourhood(D.iter(), r as usize);
        assert!(T.is_subset(&dominated));

        // Check distances of scattered set
        let mut DTFG = crate::dtfgraph::DTFGraph::orient(&G);
        DTFG.augment(2*r as usize, 1);

        for tup in S.iter().combinations(2) {
            let (x,y) = (tup[0], tup[1]);
            let dist = DTFG.small_distance(x,y);
            if let Some(d) = dist  {
                assert!(d >= 2*r+1);
            }
        }
    }

    #[test]
    fn test_small_distance() {
        let G = EditGraph::grid(20, 20);
        let r = 5;
        let OG = OrdGraph::by_degeneracy(&G);

        let wreach = OG.wreach_sets(2*r);

        let mut DTFG = crate::dtfgraph::DTFGraph::orient(&G);        
        DTFG.augment(5 as usize, 1);        

        for pair in G.vertices().combinations(2) {
            let (u,v) = (pair[0], pair[1]);
            let dist_dtf = DTFG.small_distance(u, v);
            let dist_wreach = small_distance(&wreach, r, u, v);

            if let Some(d1) = dist_dtf {
                if let Some(d2) = dist_wreach {
                    assert_eq!(d1, d2);
                }
            }
        }
    }

    #[test]
    fn test_small_distance_sets() {
        let G = EditGraph::grid(5, 6);
        let r = 12; // Twice the distances we care about
        // 00 -- 01 -- 02 -- 03 -- 04 -- 05
        // |     |     |     |     |     |
        // 06 -- 07 -- 08 -- 09 -- 10 -- 11
        // |     |     |     |     |     |
        // 12 -- 13 -- 14 -- 15 -- 16 -- 17
        // |     |     |     |     |     |
        // 18 -- 19 -- 20 -- 21 -- 22 -- 23
        // |     |     |     |     |     |
        // 24 -- 25 -- 26 -- 27 -- 28 -- 29
        let OG = OrdGraph::by_degeneracy(&G);
        let wreach = OG.wreach_sets(r+1);

        assert_eq!(small_distance(&wreach, r, &0, &0), Some(0));
        assert_eq!(small_distance(&wreach, r, &0, &1), Some(1));
        assert_eq!(small_distance(&wreach, r, &0, &2), Some(2));
        assert_eq!(small_distance(&wreach, r, &0, &3), Some(3));
        assert_eq!(small_distance(&wreach, r, &0, &4), Some(4));
        assert_eq!(small_distance(&wreach, r, &0, &5), Some(5));

        let A = vec![0,1,2,3,4,5];
        let B = vec![24,25,26,27,28,29];
        assert_eq!(small_distance_sets(&wreach, r, A, B), Some(4));

        let A = vec![24];
        let B = vec![15, 10, 05];
        assert_eq!(small_distance_sets(&wreach, r, A, B), Some(5));
    }

    #[test]
    fn scattered_colouring() {
        let G = EditGraph::grid(30, 30);
        let OG = OrdGraph::by_degeneracy(&G);
        let distance = 4;

        let targets = G.vertices().filter(|x| *x % 3 == 0).collect_vec();

        let colouring = OG.scattered_colouring(distance, targets.clone());
        assert_eq!(colouring.len(), targets.len());

        let colours = colouring.invert();

        // Check distances of scattered colours
        let mut DTFG = crate::dtfgraph::DTFGraph::orient(&G);
        DTFG.augment(2*distance as usize, 1); // Larger distance than needed, for good measure

        for (col, set) in colours.iter() {
            for (x,y) in set.iter().tuple_combinations() {
                assert!(x != y);
                let dist = DTFG.small_distance(x,y); 
                if let Some(d) = dist  {
                    assert!(d >= distance);
                }
            }
        }
    }

    /*
        Fixed. The bug was in _all_ dominating set algorithms when updating the neighbours
        of a vertex that was added because of the `domcounter`` trick. In the inner loop on 
        the marked line below the `=` was accidentally a `+=`. This was rare enough to almost
        never matter.
        ```
            if dom_counter[u] > cutoff && !domset.contains(u) {
                domset.insert(*u);
                dom_distance.insert(*u, 0);

                for (x,xdist) in wreach.get(u).unwrap().iter() {
            --->   *dom_distance.get_mut(x).unwrap() = u32::min(dom_distance[x], *xdist);
                }
            }
        ```
     */
    // #[test]
    // fn bughunt() {
    //     let mut G = EditGraph::from_file("resources/Yeast.txt.gz").unwrap();
    //     G.remove_loops();
    //     let target = vec![1952, 753, 1509, 2138, 499, 688, 2073, 874, 369, 1187, 304, 1249, 1122, 1311, 428, 806, 995, 301, 1435, 425, 992, 1181, 549, 422, 1618, 168, 1680, 292, 1615, 165, 543, 227, 416, 478, 162, 35, 537, 283, 661, 850, 156, 153, 150, 23, 212, 463, 147, 209, 398, 584, 2096, 268, 203, 1337, 1021, 1966, 138, 767, 135, 1647, 1520, 826, 132, 699, 2022, 1517, 2273, 64, 253, 442, 126, 882, 566, 61, 628, 1068, 185, 58, 1570, 120, 1443, 2010, 811, 492, 176, 238, 46, 1053, 170, 737, 988, 294, 167, 985, 1614, 1487, 285, 1041, 31, 2299, 849, 1038, 1983, 155, 1100, 595, 784, 214, 1726, 1915, 276, 1599, 149, 1912, 146, 586, 775, 648, 1714, 1587, 199, 388, 1900, 1962, 134, 1646, 69, 131, 509, 382, 1767, 252, 627, 122, 500, 184, 1129, 57, 748, 243, 1377, 872, 1817, 2257, 302, 237, 299, 1055, 172, 1808, 169, 925, 1114, 231, 1554, 166, 544, 1678, 39, 290, 163, 352, 854, 727, 721, 1666, 972, 1350, 1034, 591, 210, 399, 588, 777, 966, 83, 272, 1028, 334, 585, 1025, 266, 644, 833, 390, 263, 1208, 136, 765, 1521, 71, 827, 1016, 133, 6, 1896, 68, 1202, 508, 697, 254, 1010, 505];
    //     let OG = OrdGraph::by_degeneracy(&G);

    //     let colouring = OG.scattered_colouring(3, &target);
    // }
}