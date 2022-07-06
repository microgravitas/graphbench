# Graphbench

This library provides various graph datastructures tailored to specific styles of algorithms, as well as a generic graph
data structure for manipulating graphs.


## Basic usage

The primary struct to load and modify graphs is [`EditGraph`](https://docs.rs/graphbench/latest/graphbench/editgraph/struct.EditGraph.html) which provides common graph operations.

```rust
use graphbench::graph::*;
use graphbench::editgraph::EditGraph;

fn main() {
    let mut graph = EditGraph::new();
    graph.add_vertex(&0);   
    graph.add_edge(&1, &2); // Vertices 1,2 are added implicitly
    graph.add_edge(&1, &1); // Loops are allowed

    println!("Graph has {} vertices and {} edges", graph.num_vertices(), graph.num_edges());

    // Use .contains(..) to query vertices
    assert_eq!(graph.contains(&0), true);
    assert_eq!(graph.contains(&3), false);

    // Use .adjacent(..) to query edges
    assert_eq!(graph.adjacent(&1, &1), true);
    assert_eq!(graph.adjacent(&1, &2), true);
    assert_eq!(graph.adjacent(&0, &2), false);

    // Use .degree(...) to query vertex degres
    assert_eq!(graph.degree(&0), 0);
    assert_eq!(graph.degree(&1), 3);
    assert_eq!(graph.degree(&2), 1);
}
```

## File I/O

Graphbench currently supports only one basic file format in which every edge is defined on a single line.
Each edge must consist of two integers separated by a space. For example, we can create a file `edges.txt` 
with the following content:
```text
0 1
0 2
0 3
```
We can then load the file as follows:

```rust,no_run
use graphbench::graph::*;
use graphbench::editgraph::EditGraph;
use graphbench::iterators::EdgeIterable;

fn main() {
    let graph = EditGraph::from_txt("edges.txt").expect("Could not open edges.txt");
    println!("Vertices: {:?}", graph.vertices().collect::<Vec<&Vertex>>());
    println!("Edges: {:?}", graph.edges().collect::<Vec<Edge>>());
}
```

## Iteration

Several iterators over the graph contents are provided by the [`graphbench::iterators`](https://docs.rs/graphbench/latest/graphbench/iterators/index.html) module, we recommend `use graphbench::iterators::*` for simplicity. 

```rust
use graphbench::graph::*;
use graphbench::editgraph::EditGraph;
use graphbench::iterators::*;

fn main() {
    let graph = EditGraph::path(5); // Path on 5 vertices 0...4

    for u in graph.vertices() {
        let degree = graph.degree(u);
        let neighs:Vec<&Vertex> = graph.neighbours(u).collect();
        println!("Vertex {} has {} neighbour(s): {:?}", u, degree, neighs);
    }

    // Needs traits graphbench::iterators::NeighIterable
    for (u,neighs_it) in graph.neighbourhoods() {
        let neighs:Vec<&Vertex> = neighs_it.collect();
        println!("Vertex {} has neighbour(s) {:?}", u, neighs);
    }

    // Needs traits graphbench::iterators::EdgeIterable
    for (u,v) in graph.edges() {
        println!("Edge {} {}", u, v);
    }
}
```

