//! Sparse graph analysis library.
//! 
//! This library provides various graph datastructures taylored for specific
//! styles of algorithms.

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

/// Various iterators for graph traits.
pub mod iterators;


pub mod algorithms;
pub mod io;
