use crate::editgraph::EditGraph;
use crate::graph::Vertex;
use crate::iterators::*;

use std::io;
use std::io::{BufRead,BufReader,BufWriter,Write};
use std::fs::File;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use crate::graph::*;


pub trait WriteGraph {
    fn write_txt(&self, filename:&str) -> io::Result<()>;
    fn write_gzipped(&self, filename:&str) -> io::Result<()>;
}

impl<G:Graph> WriteGraph for G {
    fn write_txt(&self, filename:&str) -> io::Result<()> {
        let file = File::create(filename)?;
        let mut buf = BufWriter::new(file);

        for (u,v) in self.edges() {
            buf.write(format!("{u} {v}\n").as_bytes())?;
        }
        buf.flush()?;

        Ok(())
    }

    fn write_gzipped(&self, filename:&str) -> io::Result<()> {
        let file = File::create(filename)?;
        let gz = GzEncoder::new(file, Compression::default());
        let mut buf = BufWriter::new(gz);

        for (u,v) in self.edges() {
            buf.write(format!("{u} {v}\n").as_bytes())?;
        }
        buf.flush()?;
        
        Ok(())
    }
}

/// I/O operations for [EditGraph] defined in [crate::io]
impl EditGraph {
    /// Loads the graph from a text file which contains edges separated by line breaks.
    /// Edges must be pairs of integers separated by a space.
    /// 
    /// For example, assume the file `edges.txt` contains the following:
    /// ```text
    /// 0 1
    /// 0 2
    /// 0 3
    /// ```
    /// We can then load the file as follows:
    /// 
    /// ```rust,no_run
    /// use graphbench::graph::*;
    /// use graphbench::editgraph::EditGraph;
    /// use graphbench::iterators::EdgeIterable;
    /// 
    /// fn main() {
    ///     let graph = EditGraph::from_txt("edges.txt").expect("Could not open edges.txt");
    ///     println!("Vertices: {:?}", graph.vertices().collect::<Vec<&Vertex>>());
    ///     println!("Edges: {:?}", graph.edges().collect::<Vec<Edge>>());
    /// }
    /// ```
    pub fn from_txt(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len(); // Size in bytes

        // This is a rought estimate of n given the number of bytes,
        // obtained from the my network collection.
        let n_estimate = 0.007*(size as f32);

        EditGraph::parse(&mut BufReader::new(file), n_estimate as usize)
    }

    /// Loads the graph from a gzipped text file which otherwise follows the format
    /// described in [EditGraph::from_txt].
    pub fn from_gzipped(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len();
        let gz = GzDecoder::new(file);

        // This is a rought estimate of n given the number of bytes in
        // the compresed file, obtained from data in my network collection.
        let n_estimate = 0.025*(size as f32);

        EditGraph::parse(&mut BufReader::new(gz), n_estimate as usize)
    }

    fn parse<R: BufRead>(reader: &mut R, n_estimate:usize) -> io::Result<EditGraph> {
        let mut G = EditGraph::with_capacity(n_estimate);
        for (i, line) in reader.lines().enumerate() {
            let l = line.unwrap();
            let tokens:Vec<&str> = l.split_whitespace().collect();
            if tokens.len() != 2 {
                let err = io::Error::new(io::ErrorKind::InvalidData,
                        format!("Line {} does not contain two tokens", i));
                return Err(err)
            }
            let u = EditGraph::parse_vertex(tokens[0])?;
            let v = EditGraph::parse_vertex(tokens[1])?;

            G.add_edge(&u,&v);
        }

        Ok(G)
    }

    fn parse_vertex(s: &str) -> io::Result<Vertex> {
        match s.parse::<Vertex>() {
            Ok(x) => Ok(x),
            Err(_) => Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("Cannot parse vertex id {}", s)))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn read_graph() {
        let G = EditGraph::from_gzipped("resources/karate.txt.gz").unwrap();

        assert_eq!(G.num_vertices(), 34);
        assert_eq!(G.num_edges(), 78);

        let H = EditGraph::from_txt("resources/karate.txt").unwrap();

        assert_eq!(G, H);
    }

    #[test]
    fn write_graph() {
        let G = EditGraph::biclique(3, 3);
        G.write_txt("resources/temp.txt");
        G.write_gzipped("resources/temp.txt.gz");
        
        let H1 = EditGraph::from_txt("resources/temp.txt").unwrap();
        let H2 = EditGraph::from_gzipped("resources/temp.txt.gz").unwrap();

        assert_eq!(H1, H2);
    }
}
