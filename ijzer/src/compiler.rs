use std::collections::{HashMap, HashSet};

use crate::node::Node;

struct CompilerContext {
    node_map: HashMap<usize, Node>,
    parseable: Vec<usize>,
    parsed: HashSet<usize>,
    parent: HashMap<usize, usize>,
}

impl CompilerContext {
    fn new(root: Node) -> Self {
        let mut node_map = HashMap::new();
        let mut leaves = vec![];
        let mut parent = HashMap::new();
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            node_map.insert(node.id, node.clone());
            if node.operands.is_empty() {
                leaves.push(node.id);
            }
            for child in node.operands {
                parent.insert(child.id, node.id);
                stack.push(child);
            }
        }
        let parsed = HashSet::new();

        Self {
            node_map,
            parseable: leaves,
            parsed,
            parent,
        }
    }

    fn submit_as_parsed(&mut self, node_id: usize) {
        self.parsed.insert(node_id);
        let parent = *self.parent.get(&node_id).unwrap();
        let siblings = self
            .node_map
            .get(&parent)
            .unwrap()
            .operands
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();
        if siblings.iter().all(|s| self.parsed.contains(s)) {
            self.parseable.push(parent);
        }
    }
}

// pub fn parse_node()
