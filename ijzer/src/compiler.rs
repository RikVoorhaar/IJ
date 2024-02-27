use anyhow::Result;
use std::collections::{HashMap, HashSet};
use syn::Ident;

use proc_macro2::{Span, TokenStream};
use quote::{quote, IdentFragment};

use crate::{node::Node, operations::Operation, tokens::Token};

pub struct CompilerContext {
    pub node_map: HashMap<usize, Node>,
    parseable: Vec<usize>,
    parsed: HashSet<usize>,
    pub parent: HashMap<usize, usize>,
}

impl CompilerContext {
    pub fn new(root: Node) -> Self {
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

    pub fn submit_as_parsed(&mut self, node_id: usize) {
        self.parsed.insert(node_id);

        if let Some(&parent) = self.parent.get(&node_id) {
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

    pub fn pop_next_parseable(&mut self) -> Option<usize> {
        self.parseable.pop()
    }

    pub fn compile_node(&mut self, node_id: usize) -> Result<TokenStream> {
        let node = self.node_map.get(&node_id).unwrap();

        let stream = match node.op {
            Operation::Number(_) => Number::compile(node, self)?,
            _ => NotImplemented::compile(node, self)?,
        };

        Ok(stream)
    }

    pub fn get_varname(&self, id: usize) -> Ident {
        Ident::new(&format!("_{}", id), Span::call_site())
    }
}

pub trait CompileNode {
    fn compile(node: &Node, compiler: &CompilerContext) -> Result<TokenStream>;
}

struct Number;
impl CompileNode for Number {
    fn compile(node: &Node, compiler: &CompilerContext) -> Result<TokenStream> {
        if let Operation::Number(number) = &node.op {
            let varname = compiler.get_varname(node.id);
            let val = number.value;

            let res = quote! {
                let #varname = #val;
            };
            Ok(res)
        } else {
            panic!("Expected number node, found {:?}", node);
        }
    }
}

struct NotImplemented;
impl CompileNode for NotImplemented {
    fn compile(_node: &Node, _compiler: &CompilerContext) -> Result<TokenStream> {
        Ok(quote! {
        panic!("Operation not implemented");
        })
    }
}
