use crate::graph::*;
use crate::iterators::*;

/// The possible results of comparing two graphs under the subgraph relation.
#[derive(Debug, PartialEq)]
pub enum SubgraphRel {
    Sub,
    Eq,
    Sup,
    Incomp
}

pub trait SubgraphComparable<G> where G: Graph {
    fn compare_subgraph(&self, other:&G) -> SubgraphRel;
}

impl<G> SubgraphComparable<G> for G where G: Graph {
    fn compare_subgraph(&self, other:&G) -> SubgraphRel {
        let mut is_sub: bool = self.vertices().all(|x| other.contains(x));
        is_sub &= self.edges().all(|(x,y)| other.adjacent(&x, &y));
        let mut is_super: bool = other.vertices().all(|x| self.contains(x));
        is_super &= other.edges().all(|(x,y)| self.adjacent(&x, &y));

        match (is_sub, is_super) {
            (true, true) => SubgraphRel::Eq,
            (true, false) => SubgraphRel::Sub,
            (false, true) => SubgraphRel::Sup,
            (false, false) => SubgraphRel::Incomp,
        }
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
    use crate::{editdigraph::EditDigraph, editgraph::*, io::LoadFromFile, ordgraph::OrdGraph};

    #[test]
    fn subgraph() {
        let K5 = EditGraph::clique(5);
        let P5 = EditGraph::path(5);
        let C5 = EditGraph::cycle(5);
        assert_eq!(K5.compare_subgraph( &EditGraph::clique(5) ), SubgraphRel::Eq );
        assert_eq!(K5.compare_subgraph( &EditGraph::clique(4) ), SubgraphRel::Sup );
        assert_eq!(K5.compare_subgraph( &EditGraph::clique(6) ), SubgraphRel::Sub );

        assert_eq!(P5.compare_subgraph( &C5 ), SubgraphRel::Sub );
        assert_eq!(P5.compare_subgraph( &K5 ), SubgraphRel::Sub );

        assert_eq!(C5.compare_subgraph( &K5 ), SubgraphRel::Sub );
        assert_eq!(C5.compare_subgraph( &EditGraph::cycle(4)  ), SubgraphRel::Incomp );
    }
}
