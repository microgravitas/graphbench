#![doc = include_str!("../README.md")]


#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

use std::iter::Map;

/// Different graph traits.
pub mod graph;

/// Graph data structure with various editing operations.
pub mod editgraph;

/// Graph data structure for fast short-distance queries.
pub mod dtfgraph;

/// Data structure for graphs with an ordered vertex set.
pub mod ordgraph;

/// Data structures for graphs with a fixed vertex ordering which makes
/// stores certain 'reachable' sets for each vertex.
pub mod reachgraph;

/// Various iterators for graph traits.
pub mod iterators;

/// Graph algorithms.
pub mod algorithms;

/// File I/O.
pub mod io;
