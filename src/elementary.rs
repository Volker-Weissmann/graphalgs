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
use petgraph::visit::GetAdjacencyMatrix;
use petgraph::visit::{IntoEdges, NodeCount, NodeIndexable};
use petgraph::Directed;
use petgraph::Graph;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

pub fn elementary_circuits<N, E, Ix>(graph: &Graph<N, E, Directed, Ix>) -> Vec<Vec<usize>>
where
    Ix: IndexType,
{
    let n = graph.node_count();
    let mut possible_paths = vec![vec![HashSet::new(); n]; n];

    for e in graph.edge_references() {
        let source = e.source().index();
        let target = e.target().index();
        let mut used = vec![false; n];
        used[source] = true;
        //used[target] = true;
        if source == target {
            continue;
        }
        possible_paths[source][target].insert(used);
    }
    loop {
        let mut new_paths = vec![vec![Vec::new(); n]; n];
        let mut stale: bool = true;
        for source in 0..n {
            for mid in 0..n {
                if source == mid {
                    continue;
                }
                for target in 0..n {
                    if target == mid || target == source {
                        continue;
                    }
                    for p1 in possible_paths[source][mid].iter() {
                        for p2 in possible_paths[mid][target].iter() {
                            if p1.iter().zip(p2.iter()).any(|(&a, &b)| a && b) {
                                continue;
                            }
                            let combined = p1
                                .iter()
                                .zip(p2.iter())
                                .map(|(&a, &b)| a || b)
                                .collect::<Vec<_>>();
                            if !possible_paths[source][target].contains(&combined) {
                                new_paths[source][target].push(combined);
                                stale = false;
                            }
                        }
                    }
                }
            }
        }
        if stale {
            break;
        }
        for source in 0..n {
            for target in 0..n {
                while let Some(path) = new_paths[source][target].pop() {
                    possible_paths[source][target].insert(path);
                }
            }
        }
    }
    std::hint::black_box(possible_paths);

    vec![vec![]]
}

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

// enum Task {
//     Search(usize),
//     Unblock(usize),
// }

// fn circuit<N, E, Ix>(
//     graph: &Graph<N, E, Directed, Ix>,
//     n: usize,
//     perfgraph: &Vec<Vec<usize>>,
//     is_precessor_of_s: &[bool],
//     output: &mut Vec<Vec<usize>>,
//     s: usize,
// ) where
//     Ix: IndexType,
// {
//     let mut stack = Vec::<usize>::new();
//     let mut blocked = vec![false; n];
//     let mut tasks = vec![Task::Search(s)];
//     while let Some(task) = tasks.pop() {
//         match task {
//             Task::Search(v) => {
//                 stack.push(v);
//                 blocked[v] = true;
//                 if is_precessor_of_s[v] {
//                     output.push(stack.clone());
//                 }
//                 tasks.push(Task::Unblock(v));
//                 for &w in &perfgraph[v] {
//                     debug_assert!(w > s);
//                     if !blocked[w] {
//                         tasks.push(Task::Search(w));
//                     }
//                 }
//             }
//             Task::Unblock(v) => {
//                 blocked[v] = false;
//                 let popped = stack.pop();
//                 debug_assert!(popped == Some(v));
//             }
//         }
//     }
//     debug_assert!(stack.is_empty());
// }

fn construct_vec_from_tree(tree: &[(usize, usize)], mut index: usize) -> Vec<usize> {
    let mut ret = Vec::new();
    while index != 0 {
        ret.push(tree[index].1);
        index = tree[index].0
    }
    ret.reverse();
    ret
}

fn circuit<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
    n: usize,
    perfgraph: &Vec<Vec<usize>>,
    is_precessor_of_s: &[bool],
    output: &mut Vec<Vec<usize>>,
    s: usize,
) where
    Ix: IndexType,
{
    let mut tasks = vec![(s, vec![false; n], 0)];
    let mut tree = vec![(0, s)];
    while let Some((v, blocked, index)) = tasks.pop() {
        let mut new_blocked = blocked.clone();
        new_blocked[v] = true;
        let new_index = tree.len();
        tree.push((index, v));
        if is_precessor_of_s[v] {
            output.push(construct_vec_from_tree(&tree, new_index));
        }
        for &w in &perfgraph[v] {
            debug_assert!(w > s);
            if !new_blocked[w] {
                tasks.push((w, new_blocked.clone(), new_index));
            }
        }

        // match task {
        //     Task::Search(v) => {
        //         stack.push(v);
        //         blocked[v] = true;
        //         if is_precessor_of_s[v] {
        //             output.push(stack.clone());
        //         }
        //         tasks.push(Task::Unblock(v));
        //         for &w in &perfgraph[v] {
        //             debug_assert!(w > s);
        //             if !blocked[w] {
        //                 tasks.push(Task::Search(w));
        //             }
        //         }
        //     }
        //     Task::Unblock(v) => {
        //         blocked[v] = false;
        //         let popped = stack.pop();
        //         debug_assert!(popped == Some(v));
        //     }
        // }
    }
}

// Two elementary circuits are distinct if one is not a cyclic permutation of the other.

// todo: test with undirected graph
// todo: check which traits are needed
// todo: should we use i32?
// todo: look at graph.filter_map
// todo: assert or debug_assert

// todo: usize or NodeIndex?

// todo: what if there are multiple edges connecting a and b?

// todo: does the performance change if the node weights or edge weights are not ()

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

    let mut perfgraph = vec![Vec::new(); n];
    for e in graph.edge_references() {
        perfgraph[e.source().index()].push(e.target().index());
    }

    let mut output = Vec::new();
    for s in 0..n {
        for vec in perfgraph.iter_mut() {
            if let Some(pos) = vec.iter().position(|x| *x <= s) {
                vec.remove(pos);
            }
        }
        let is_precessor_of_s = (0..n)
            .map(|x| {
                graph
                    .edges_connecting(NodeIndex::<Ix>::new(x), NodeIndex::<Ix>::new(s))
                    .next()
                    .is_some()
            })
            .collect::<Vec<_>>();

        // todo: we don't need n as an argument, we can use graph.node_count()
        circuit(graph, n, &perfgraph, &is_precessor_of_s, &mut output, s);
        //debug_assert!(stack.is_empty());
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
            let mut output = johnson_elementary_circuits(&graph);
            if false {
                all_outputs.push(output.clone());
            }
            // dbg!(output.len());
            // dbg!(output.iter().map(|x| x.len()).sum::<usize>());
            let mut expected = all_outputs[seed as usize].clone();
            expected.sort();
            output.sort();
            assert_eq!(output, expected);
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
            let mut actual_output = johnson_elementary_circuits(&graph);
            let mut expected_output = expected_output.clone();
            expected_output.sort();
            actual_output.sort();
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
    // let graph = Graph::<(), i32>::from_edges(vec![(0, 1), (0, 2), (1, 1), (2, 1)]);
    // let mut actual_output = johnson_elementary_circuits(&graph);
    // dbg!(actual_output);
    // return;

    let nodes = 14;
    let nedges = 15 * 14 / 2;

    let nodes = 12;
    let nedges = nodes * nodes / 2;

    let graph: Graph<(), ()> = Graph::from_edges(volker_random_digraph(nodes, nedges, 0).unwrap());
    dbg!(&graph);
    let output = elementary_circuits(&graph);
    std::hint::black_box(output);
}
