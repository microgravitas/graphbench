use crate::editgraph::EditGraph;
use crate::graph::Vertex;
use crate::iterators::*;

use std::ffi::OsStr;
use std::io;
use std::io::{BufRead,BufReader,BufWriter,Write};
use std::fs::File;
use std::path::Path;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use crate::graph::*;

pub trait WriteToFile {
    fn write_txt(&self, filename:&str) -> io::Result<()> {
        let file = File::create(filename)?;
        let buf = BufWriter::new(file);
        self.write_buf(Box::new(buf))
    }

    fn write_gzipped(&self, filename:&str) -> io::Result<()> {
        let file = File::create(filename)?;
        let gz = GzEncoder::new(file, Compression::default());
        let buf = BufWriter::new(gz);
        self.write_buf(Box::new(buf))
    }

    fn write_buf(&self, buf:Box<dyn Write>) -> io::Result<()>;
}

pub trait LoadFromFile {
    fn from_txt(filename:&str) -> io::Result<Self> where Self: Sized{
        let buf = open_reader_txt(filename)?;
        Self::from_buf(buf)
    }

    fn from_gzipped(filename:&str) -> io::Result<Self> where Self: Sized {
        let buf = open_reader_gzip(filename)?;
        Self::from_buf(buf)
    }

    fn from_file(filename:&str) -> io::Result<Self> where Self: Sized {
        let buf = open_reader(filename)?;
        Self::from_buf(buf)
    }

    fn from_buf(buf:Box<dyn BufRead>) -> io::Result<Self> where Self: Sized;
}

impl LoadFromFile for VertexSet {
    fn from_buf(buf:Box<dyn BufRead>) -> io::Result<Self> where Self: Sized {
        let mut res = VertexSet::default();
        for (i, line) in buf.lines().enumerate() {
            let l = line.unwrap();
            let u = parse_vertex(l.trim(), i)?;
            res.insert(u);
        }
    
        Ok(res)
    }
}

impl WriteToFile for VertexSet {
    fn write_buf(&self, mut buf:Box<dyn Write>) -> io::Result<()> {
        for x in self {
            buf.write_all(format!("{x}\n").as_bytes())?;
        }
        buf.flush()?;
        Ok(())
    }
}

impl LoadFromFile for Vec<Vertex> {
    fn from_buf(buf:Box<dyn BufRead>) -> io::Result<Self> where Self: Sized {
        let mut res = Vec::default();
        for (i, line) in buf.lines().enumerate() {
            let l = line.unwrap();
            let u = parse_vertex(l.trim(), i)?;
            res.push(u);
        }
    
        Ok(res)
    }
}

impl WriteToFile for Vec<Vertex> {
    fn write_buf(&self, mut buf:Box<dyn Write>) -> io::Result<()> {
        for u in self {
            buf.write_all(format!("{u}\n").as_bytes())?;
        }
        buf.flush()?;
        
        Ok(())
    }
}

impl<G:Graph> WriteToFile for G {  
    fn write_buf(&self, mut buf:Box<dyn Write>) -> io::Result<()> {
        for (u,v) in self.edges() {
            buf.write_all(format!("{u} {v}\n").as_bytes())?;
        }
        buf.flush()?;
        
        Ok(())
    }
}

impl EditGraph {
    /// Since we assume graphs to be quite big and EditGraphs are the 'work horse' of
    /// this library, this method provides a way to read a graph without too many reallocations
    /// by reserving a suitable amount of memory upfront.
    fn from_buf_with_estimate<R: BufRead>(reader: &mut R, n_estimate:usize) -> io::Result<EditGraph> {
        let mut G = EditGraph::with_capacity(n_estimate);
        for (lineno, line) in reader.lines().enumerate() {
            let l = line.unwrap();
            let tokens:Vec<&str> = l.split_whitespace().collect();
            if tokens.len() != 2 {
                let err = io::Error::new(io::ErrorKind::InvalidData,
                        format!("Line {} does not contain two tokens", lineno));
                return Err(err)
            }
            let u = parse_vertex(tokens[0], lineno)?;
            let v = parse_vertex(tokens[1], lineno)?;

            G.add_edge(&u,&v);
        }

        Ok(G)
    }
}

/// I/O operations for [EditGraph] defined in [crate::io]
impl LoadFromFile for EditGraph {
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
    /// use graphbench::io::*;
    /// use graphbench::editgraph::EditGraph;
    /// use graphbench::iterators::EdgeIterable;
    /// 
    /// let graph = EditGraph::from_txt("edges.txt").expect("Could not open edges.txt");
    /// println!("Vertices: {:?}", graph.vertices().collect::<Vec<&Vertex>>());
    /// println!("Edges: {:?}", graph.edges().collect::<Vec<Edge>>());
    /// ```
    fn from_txt(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len(); // Size in bytes

        // This is a rought estimate of n given the number of bytes,
        // obtained from the my network collection.
        let n_estimate = 0.007*(size as f32);

        EditGraph::from_buf_with_estimate(&mut BufReader::new(file), n_estimate as usize)
    }

    /// Loads the graph from a gzipped text file which otherwise follows the format
    /// described in [EditGraph::from_txt].
    fn from_gzipped(filename:&str) -> io::Result<EditGraph> {
        let file = File::open(filename)?;
        let metadata = file.metadata()?;
        let size = metadata.len();
        let gz = GzDecoder::new(file);

        // This is a rought estimate of n given the number of bytes in
        // the compresed file, obtained from data in my network collection.
        let n_estimate = 0.025*(size as f32);

        EditGraph::from_buf_with_estimate(&mut BufReader::new(gz), n_estimate as usize)
    }

    fn from_buf(buf:Box<dyn BufRead>) -> io::Result<Self> where Self: Sized {
        let mut G = EditGraph::with_capacity(500);
        for (lineno, line) in buf.lines().enumerate() {
            let l = line.unwrap();
            let tokens:Vec<&str> = l.split_whitespace().collect();
            if tokens.len() != 2 {
                let err = io::Error::new(io::ErrorKind::InvalidData,
                        format!("Line {} does not contain two tokens", lineno));
                return Err(err)
            }
            let u = parse_vertex(tokens[0], lineno)?;
            let v = parse_vertex(tokens[1], lineno)?;

            G.add_edge(&u,&v);
        }

        Ok(G)
    }
}

fn open_reader(filename:&str) -> io::Result<Box<dyn BufRead>> {
    let path = Path::new(&filename);
    let extension = path.extension().and_then(OsStr::to_str);
    let reader:Box<dyn BufRead> = match extension {
        Some("txt") => {
            let file = File::open(path)?;
            Box::new(BufReader::new(file))
        }
        Some("gz") => {
            let file = File::open(path)?;
            let gz = GzDecoder::new(file);
            Box::new(BufReader::new(gz))
        }
        _ => {
            let error = std::io::Error::new(std::io::ErrorKind::InvalidInput, 
                format!("Invalid file `{filename:?}`. The supported formats are `.txt.gz` and `.txt`."));
            return Err(error);
        }        
    };
    Ok(reader)
}

fn open_reader_txt(filename:&str) -> io::Result<Box<dyn BufRead>> {
    let path = Path::new(&filename);
    let file = File::open(path)?;
    Ok(Box::new(BufReader::new(file)))
}

fn open_reader_gzip(filename:&str) -> io::Result<Box<dyn BufRead>> {
    let path = Path::new(&filename);
    let file = File::open(path)?;
    let gz = GzDecoder::new(file);
    Ok(Box::new(BufReader::new(gz)))
}

fn parse_vertex(s: &str, lineno:usize) -> io::Result<Vertex> {
    match s.parse::<Vertex>() {
        Ok(x) => Ok(x),
        Err(_) => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Cannot parse vertex id {} at input line {}", s, lineno)))
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
    fn read_graph() {
        let G = EditGraph::from_gzipped("resources/karate.txt.gz").unwrap();

        assert_eq!(G.num_vertices(), 34);
        assert_eq!(G.num_edges(), 78);

        let H = EditGraph::from_txt("resources/karate.txt").unwrap();

        assert_eq!(G, H);
    }

    #[test]
    #[allow(unused_must_use)]
    fn write_graph() {
        let G = EditGraph::biclique(3, 3);
        G.write_txt("resources/temp.txt");
        G.write_gzipped("resources/temp.txt.gz");
        
        let H1 = EditGraph::from_txt("resources/temp.txt").unwrap();
        let H2 = EditGraph::from_gzipped("resources/temp.txt.gz").unwrap();

        assert_eq!(H1, H2);
    }

    #[test]
    #[allow(unused_must_use)]
    fn read_write_set() {
        let mut S = VertexSet::default();

        S.insert(1);
        S.insert(2);
        S.insert(999);

        S.write_txt("resources/set.txt");
        S.write_gzipped("resources/set.txt.gz");
        
        let S1 = VertexSet::from_txt("resources/set.txt").unwrap();
        let S2 = VertexSet::from_gzipped("resources/set.txt.gz").unwrap();

        assert_eq!(S1, S);
        assert_eq!(S2, S);

        let S1 = VertexSet::from_file("resources/set.txt").unwrap();
        let S2 = VertexSet::from_file("resources/set.txt.gz").unwrap();

        assert_eq!(S1, S);
        assert_eq!(S2, S);        
    }

    #[test]
    #[allow(unused_must_use)]
    fn read_write_vec() {
        let mut S = Vec::<Vertex>::default();

        S.push(1);
        S.push(2);
        S.push(999);

        S.write_txt("resources/vec.txt");
        S.write_gzipped("resources/vec.txt.gz");
        
        let S1 = Vec::<Vertex>::from_txt("resources/vec.txt").unwrap();
        let S2 = Vec::<Vertex>::from_gzipped("resources/vec.txt.gz").unwrap();

        assert_eq!(S1, S);
        assert_eq!(S2, S);

        let S1 = Vec::<Vertex>::from_file("resources/vec.txt").unwrap();
        let S2 = Vec::<Vertex>::from_file("resources/vec.txt.gz").unwrap();        

        assert_eq!(S1, S);
        assert_eq!(S2, S);        
    }    
}
