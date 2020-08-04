use crate::graph::Graph;
use crate::graph::Vertex;

use std::io;
use std::io::BufRead;
use std::fs::File;
use flate2::read::GzDecoder;

impl Graph {
    pub fn from_txt(filename: &str) -> io::Result<Graph> {
        let file = File::open(filename)?;
        Graph::parse(&mut io::BufReader::new(file))
    }

    pub fn from_gzipped(filename: &str) -> io::Result<Graph> {
        let file = File::open(filename)?;
        let gz = GzDecoder::new(file);

        Graph::parse(&mut io::BufReader::new(gz))
    }

    fn parse<R: BufRead>(reader: &mut R) -> io::Result<Graph> {
        let mut G = Graph::new();
        let mut i = 0;
        for line in reader.lines() {
            let l = line.unwrap();
            let tokens:Vec<&str> = l.split_whitespace().collect();
            if tokens.len() != 2 {
                let err = io::Error::new(io::ErrorKind::InvalidData,
                        format!("Line {} does not contain two tokens", i));
                return Err(err)
            }
            let u = Graph::parse_vertex(tokens[0])?;
            let v = Graph::parse_vertex(tokens[1])?;

            G.add_edge(u,v);
            i += 1;
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
        let G = Graph::from_gzipped("resources/karate.txt.gz").unwrap();

        assert_eq!(G.num_vertices(), 34);
        assert_eq!(G.num_edges(), 78);

        let H = Graph::from_txt("resources/karate.txt").unwrap();

        assert_eq!(G, H);
    }
}
