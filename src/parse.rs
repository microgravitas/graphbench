use crate::editgraph::EditGraph;
use crate::graph::Vertex;

use std::io;
use std::io::BufRead;
use std::fs::File;
use flate2::read::GzDecoder;

use crate::graph::*;

impl EditGraph {
    pub fn from_txt(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len(); // Size in bytes

        // This is a rought estimate of n given the number of bytes,
        // obtained from the my network collection.
        let n_estimate = 0.007*(size as f32);

        EditGraph::parse(&mut io::BufReader::new(file), n_estimate as usize)
    }

    pub fn from_gzipped(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len();
        let gz = GzDecoder::new(file);

        // This is a rought estimate of n given the number of bytes in
        // the compresed file, obtained from the my network collection.
        let n_estimate = 0.025*(size as f32);

        EditGraph::parse(&mut io::BufReader::new(gz), n_estimate as usize)
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
    fn import() {
        let G = EditGraph::from_gzipped("resources/karate.txt.gz").unwrap();

        assert_eq!(G.num_vertices(), 34);
        assert_eq!(G.num_edges(), 78);

        let H = EditGraph::from_txt("resources/karate.txt").unwrap();

        assert_eq!(G, H);
    }
}
