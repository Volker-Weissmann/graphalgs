//! Implementation of [this
//! algorithm](https://www.cs.tufts.edu/comp/150GA/homeworks/hw1/Johnson%2075.PDF)
//! to find all elementary circuits in a directed graph
//! There is a Java Implementation: https://github.com/1123/johnson

// todo: enable these warnings
#![allow(unused_imports)]
#![allow(soft_unstable)]
// #![allow(unused_variables)]
// #![allow(dead_code)]
// #![allow(unused_mut)]
// #![allow(unreachable_code)]
#![allow(clippy::let_and_return)]
#![allow(clippy::too_many_arguments)]
use petgraph::algo::tarjan_scc;
use petgraph::stable_graph::IndexType;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::visit::{IntoEdges, NodeCount, NodeIndexable};
use petgraph::Directed;
use petgraph::Graph;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::hash::Hash;

// fn unblock<NI: Copy + Eq + Hash>(
//     b: &HashMap<NI, HashMap<NI, bool>>,
//     blocked: &mut HashMap<NI, bool>,
//     u: NI,
// ) {
//     blocked.insert(u, false);
//     for (&w, &contains) in &b.get(&u).unwrap_or(false) {
//         if contains {

//         }
//     }
// }

fn unblock(n: usize, b: &mut Vec<Vec<bool>>, blocked: &mut [bool], u: usize) {
    blocked[u] = false;
    for w in 0..n {
        if b[u][w] {
            b[u][w] = false;
            if blocked[w] {
                unblock(n, b, blocked, w);
            }
        }
    }
}

fn circuit<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
    n: usize,
    output: &mut Vec<Vec<usize>>,
    stack: &mut Vec<usize>,
    s: usize,
    b: &mut Vec<Vec<bool>>,
    blocked: &mut [bool],
    v: usize,
) -> bool
where
    Ix: IndexType,
{
    let mut f = false;
    stack.push(v);
    blocked[v] = true;
    for e in graph.edges(NodeIndex::<Ix>::new(v)) {
        if e.target().index() < s {
            continue;
        }
        let w = e.target().index();
        if w == s {
            output.push(stack.clone());
            f = true;
        } else if !blocked[w] && circuit(graph, n, output, stack, s, b, blocked, w) {
            f = true;
        }
    }
    if f {
        unblock(n, b, blocked, v)
    } else {
        for e in graph.edges(NodeIndex::<Ix>::new(v)) {
            if e.target().index() < s {
                continue;
            }
            let w = e.target().index();
            b[w][v] = true;
        }
    }
    assert!(stack.pop() == Some(v));
    f
}

// -> Option<&Vec<NodeIndex>>
fn scc<N, E, Ix>(graph: &Graph<N, E, Directed, Ix>, s: usize) -> Option<Vec<usize>>
where
    Ix: IndexType,
{
    let subgraph = graph.filter_map(
        |ni, _| {
            if ni.index() >= s {
                Some(ni.index())
            } else {
                None
            }
        },
        |_, edge| Some(edge),
    );
    let new_indices = tarjan_scc(&subgraph)
        .into_iter()
        .min_by_key(|x| x.iter().min().copied());
    let old_indices = new_indices.map(|v| {
        v.into_iter()
            .map(|ni| *subgraph.node_weight(ni).unwrap())
            .collect::<Vec<_>>()
    });
    old_indices
}

// Two elementary circuits are distinct if one is not a cyclic permutation of the other.

// todo: test with undirected graph
// todo: check which traits are needed
// todo: should we use i32?
// todo: look at graph.filter_map
// todo: assert or debug_assert

// todo: usize or NodeIndex?

// todo: check if we get a nice error message if we pass an undirected graph

// todo: Graph<N, E, Ty, Ix>
// Ty: EdgeType,
//     Ix: IndexType,

// impl<N, E, Ty, Ix> Graph<N, E, Ty, Ix>
// where
//     Ty: EdgeType,
//     Ix: IndexType,
pub fn johnson_elementary_circuits<N, E, Ix>(graph: &Graph<N, E, Directed, Ix>) -> Vec<Vec<usize>>
where
    Ix: IndexType,
{
    // let a = HashMap::<G::NodeId, Vec<G::NodeId>>::new();
    // let b = HashMap::<G::NodeId, HashMap<G::NodeId, bool>>::new();
    // let blocked = HashMap::<G::NodeId, bool>::new();
    //let s: G::NodeId = 0;
    let n = graph.node_count();
    let mut blocked = vec![false; n];
    let mut b = vec![vec![false; n]; n];
    let mut s = 0;
    let mut output = Vec::new();
    while s < n {
        match scc(graph, s) {
            Some(least) => {
                let mut stack = Vec::<usize>::new();
                // The paper says s := least vertex in V_K, i.e. we should
                // assign s = *least.iter().min().unwrap(),  but unless I'm
                // mistaken, this is always already true
                assert_eq!(s, *least.iter().min().unwrap());
                for i in least {
                    blocked[i] = false;
                    b[i][0..n].fill(false);
                }
                // todo: we don't need n as an argument, we can use graph.node_count()
                circuit(
                    graph,
                    n,
                    &mut output,
                    &mut stack,
                    s,
                    &mut b,
                    &mut blocked,
                    s,
                );
                assert!(stack.is_empty());
                s += 1;
            }
            None => {
                s = n;
            }
        };
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_test_johnson_elementary_circuits() {
        let mut all_outputs = Vec::new();
        if true {
            all_outputs =
                serde_json::from_reader(std::fs::File::open("known_good").unwrap()).unwrap();
        }
        for seed in 0..100 {
            let graph: Graph<(), ()> =
                Graph::from_edges(volker_random_digraph(5, 7, seed).unwrap());
            // let n = 13;
            // let graph: Graph<(), ()> =
            //     Graph::from_edges(volker_random_digraph(n, n * n / 2, seed).unwrap());
            let output = johnson_elementary_circuits(&graph);
            // dbg!(output.len());
            // dbg!(output.iter().map(|x| x.len()).sum::<usize>());
            assert_eq!(output, all_outputs[seed as usize]);
            if false {
                all_outputs.push(output);
            }
        }
        if false {
            serde_json::to_writer_pretty(
                std::fs::File::create("known_good").unwrap(),
                &all_outputs,
            )
            .unwrap();
        }
    }

    #[test]
    fn test_johnson_elementary_circuits() {
        let input_output_pairs = [
            // Tests from the Java Implementation:
            // https://github.com/1123/johnson/blob/master/src/test/java/jgraphalgos/johnson/TestJohnson.java
            (
                vec![(1, 3), (3, 1), (3, 2), (2, 3)],
                vec![vec![1, 3], vec![2, 3]],
            ),
            (
                vec![(1, 2), (2, 1), (2, 3), (4, 5), (5, 6), (6, 4)],
                vec![vec![1, 2], vec![4, 5, 6]],
            ),
            (vec![(2, 1), (1, 2), (3, 2), (3, 1)], vec![vec![1, 2]]),
            (
                vec![(1, 3), (3, 2), (3, 1), (2, 1)],
                vec![vec![1, 3], vec![1, 3, 2]],
            ),
            (
                vec![(1, 2), (2, 1), (2, 3), (2, 4), (4, 2)],
                vec![vec![1, 2], vec![2, 4]],
            ),
            // Tests not from the Java Implementation
            (
                vec![(10, 30), (30, 10), (30, 20), (20, 30)],
                vec![vec![10, 30], vec![20, 30]],
            ),
            (
                vec![(3, 1), (1, 3), (3, 2), (2, 3)],
                vec![vec![1, 3], vec![2, 3]],
            ),
            (vec![(1, 2), (2, 3), (3, 4), (5, 6)], vec![]),
            (vec![], vec![]),
            (vec![(1, 1)], vec![vec![1]]),
            (vec![(1, 1), (1, 2), (2, 1)], vec![vec![1, 2], vec![1]]),
            (
                vec![(1, 2), (2, 1), (1, 3), (3, 1)],
                vec![vec![1, 3], vec![1, 2]],
            ),
            (
                vec![
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                    (3, 1),
                    (3, 2),
                    (3, 3),
                ],
                vec![
                    vec![1, 3, 2],
                    vec![1, 3],
                    vec![1, 2, 3],
                    vec![1, 2],
                    vec![1],
                    vec![2, 3],
                    vec![2],
                    vec![3],
                ],
            ),
        ];
        for (input, expected_output) in input_output_pairs {
            let graph = Graph::<(), i32>::from_edges(input);
            let actual_output = johnson_elementary_circuits(&graph);
            assert_eq!(actual_output, expected_output);
        }
    }
    #[test]
    fn johnson_profile() {
        let nodes = 14;
        let nedges = 14 * 14 / 2;
        let graph: Graph<(), ()> =
            Graph::from_edges(volker_random_digraph(nodes, nedges, 0).unwrap());
        let output = johnson_elementary_circuits(&graph);
        std::hint::black_box(output);
    }
}

#[allow(dead_code)]
fn volker_random_digraph(
    nodes: u32,
    nedges: u32,
    seed: u64,
) -> Result<Vec<(u32, u32)>, &'static str> {
    if nodes == 0 {
        return Ok(Vec::new());
    }

    if nedges > nodes * (nodes - 1) {
        return Err("Too many edges, too few nodes");
    }

    //let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(nedges as usize);
    let mut count = 0;

    while count < nedges {
        let i = rng.gen_range(0..nodes);
        let j = rng.gen_range(0..nodes);

        if i != j && !edges.contains(&(i, j)) {
            edges.push((i, j));
            count += 1;
        }
    }

    Ok(edges)
}

#[allow(dead_code)]
fn main() {
    let nodes = 14;
    let nedges = 15 * 14 / 2;
    let graph: Graph<(), ()> = Graph::from_edges(volker_random_digraph(nodes, nedges, 0).unwrap());
    let output = johnson_elementary_circuits(&graph);
    std::hint::black_box(output);
}
