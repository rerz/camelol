use itertools::Itertools;
use petgraph::algo::dijkstra;
use petgraph::prelude::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::{Display, Formatter};
use std::iter;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum ScaleKind {
    Major,
    Minor,
}

impl ScaleKind {
    fn swap(self) -> Self {
        match self {
            ScaleKind::Minor => ScaleKind::Major,
            ScaleKind::Major => ScaleKind::Minor,
        }
    }
}

impl Display for ScaleKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minor => write!(f, "A"),
            Self::Major => write!(f, "B"),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
struct Scale {
    index: usize,
    kind: ScaleKind,
}

fn mod_cyclic(num: isize, modulus: usize) -> isize {
    let modulus = modulus as isize;
    ((num % modulus) + modulus) % modulus
}

impl Scale {
    pub fn swap_kind(self) -> Self {
        Self {
            kind: self.kind.swap(),
            ..self
        }
    }

    pub fn change_index(self, amount: isize) -> Self {
        let index = mod_cyclic((self.index as isize) + amount, 12);
        Self {
            index: index as usize,
            ..self
        }
    }
}

impl Display for Scale {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.index + 1, self.kind)
    }
}

fn scale(index: usize, kind: ScaleKind) -> Scale {
    Scale { index, kind }
}

fn make_nodes() -> Vec<Scale> {
    (0..=11)
        .flat_map(|i| [scale(i, ScaleKind::Minor), scale(i, ScaleKind::Major)])
        .collect::<Vec<_>>()
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ScaleTransition {
    Vertical,
    Diagonal,
    ChangeIndex(isize),
    MajorToMinor,
    FlatToMinor,
}

fn possible_transitions() -> Vec<ScaleTransition> {
    vec![
        ScaleTransition::Vertical,
        ScaleTransition::Diagonal,
        ScaleTransition::MajorToMinor,
        ScaleTransition::FlatToMinor,
        ScaleTransition::ChangeIndex(1),
        ScaleTransition::ChangeIndex(2),
        ScaleTransition::ChangeIndex(7),
        ScaleTransition::ChangeIndex(-1),
        ScaleTransition::ChangeIndex(-2),
        ScaleTransition::ChangeIndex(-7),
    ]
}

fn make_transition(scale: Scale, transition: ScaleTransition) -> Scale {
    match transition {
        ScaleTransition::Vertical => scale.swap_kind(),
        ScaleTransition::ChangeIndex(amount) => scale.change_index(amount),
        ScaleTransition::Diagonal if matches!(scale.kind, ScaleKind::Major) => {
            scale.swap_kind().change_index(1)
        }
        ScaleTransition::Diagonal if matches!(scale.kind, ScaleKind::Minor) => {
            scale.swap_kind().change_index(-1)
        }
        ScaleTransition::FlatToMinor if matches!(scale.kind, ScaleKind::Minor) => {
            scale.swap_kind().change_index(-4)
        }
        ScaleTransition::FlatToMinor if matches!(scale.kind, ScaleKind::Major) => {
            scale.swap_kind().change_index(4)
        }
        ScaleTransition::MajorToMinor if matches!(scale.kind, ScaleKind::Minor) => {
            scale.swap_kind().change_index(3)
        }
        ScaleTransition::MajorToMinor if matches!(scale.kind, ScaleKind::Major) => {
            scale.swap_kind().change_index(-3)
        }
        _ => unreachable!(),
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct NodeDistance {
    node: NodeIndex,
    distance: usize,
}

impl Ord for NodeDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for NodeDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct Path {
    cost: i32,
    node: NodeIndex<u32>,
    transition: Option<ScaleTransition>,
    path: Vec<NodeIndex<u32>>,
    transition_path: Vec<ScaleTransition>,
}

impl Eq for Path {}

impl PartialEq for Path {
    fn eq(&self, other: &Path) -> bool {
        self.cost == other.cost
    }
}

impl Ord for Path {
    fn cmp(&self, other: &Path) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for Path {
    fn partial_cmp(&self, other: &Path) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn multi_path_dijkstra(
    graph: &Graph<Scale, ScaleTransition>,
    source: NodeIndex<u32>,
    target: NodeIndex<u32>,
    n: usize,
) -> Vec<Path> {
    let mut min_heap = BinaryHeap::new();
    let mut paths = Vec::new();

    min_heap.push(Path {
        cost: 0,
        node: source,
        transition: None,
        path: vec![],
        transition_path: vec![],
    });

    while let Some(mut path) = min_heap.pop() {
        path.path.push(path.node);

        if let Some(transition) = path.transition {
            path.transition_path.push(transition);
        }

        if path.node == target {
            paths.push(path.clone());
            if paths.len() >= n {
                break;
            }
        }

        for edge in graph.edges(path.node) {
            let neighbor = edge.target();
            let weight = graph.edge_weight(edge.id()).unwrap();
            min_heap.push(Path {
                cost: path.cost + 1,
                node: neighbor,
                transition: Some(*weight),
                transition_path: path.transition_path.clone(),
                path: path.path.clone(),
            });
        }
    }

    paths
}

fn main() {
    let mut graph = petgraph::Graph::new();

    let nodes = make_nodes();
    let transitions = possible_transitions();

    let scale_to_index = nodes
        .iter()
        .map(|scale| (*scale, graph.add_node(*scale)))
        .collect::<HashMap<_, _>>();

    for scale in &nodes {
        let source_scale_node = *scale_to_index.get(scale).unwrap();
        for transition in &transitions {
            let target_scale = make_transition(*scale, *transition);
            let target_scale_node = *scale_to_index.get(&target_scale).unwrap();
            graph.add_edge(source_scale_node, target_scale_node, *transition);
        }
    }

    let a_minor = *scale_to_index
        .get(&Scale {
            index: 11,
            kind: ScaleKind::Minor,
        })
        .unwrap();
    let d_flat_major = *scale_to_index
        .get(&Scale {
            index: 0,
            kind: ScaleKind::Major,
        })
        .unwrap();

    let paths = multi_path_dijkstra(&graph, a_minor, d_flat_major, 10);

    for path in paths {
        let transitions = path
            .transition_path
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None));

        let path = path
            .path
            .into_iter()
            .map(|node| graph.node_weight(node).unwrap())
            .map(|scale| scale.to_string())
            .zip(transitions)
            .flat_map(|(scale, transition)| match transition {
                Some(transition) => vec![scale.to_string(), format!("{transition:?}")],
                None => vec![scale.to_string()],
            })
            .collect::<Vec<_>>();

        let path = path
            .into_iter()
            .intersperse(" -> ".into())
            .collect::<String>();
        dbg!(path);
    }
}
