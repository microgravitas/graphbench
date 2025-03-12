//! Various basic graph algorithms which operate on the [Graph]() trait.
//! 
//! ## Testing bipartiteness
//! ```
//! use graphbench::editgraph::EditGraph;
//! use graphbench::graph::*;
//! use graphbench::algorithms::*;
//! use std::matches;
//! fn main() {
//!     let mut graph = EditGraph::biclique(2,4); // Bipartite graph
//!     let witness = graph.is_bipartite();
//!     assert!(matches!(witness, BipartiteWitness::Bipartition(_, _)));
//! 
//!     graph.add_edge(&0, &1); // Graph is not bipartite anymore
//!     let witness = graph.is_bipartite(); 
//!     assert!(matches!(witness, BipartiteWitness::OddCycle(_)));
//! }
//! ```

pub mod graph;
pub use graph::*;

pub mod lineargraph;
pub use lineargraph::*;