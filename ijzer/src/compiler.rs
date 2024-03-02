use anyhow::Result;
use std::collections::HashMap;
use syn::Ident;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use crate::{node::Node, operations::Operation};

pub struct CompilerContext {
    pub node_map: HashMap<usize, Node>,
    parseable: Vec<usize>,
    parsed: HashMap<usize, TokenStream>,
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
        let parsed = HashMap::new();

        Self {
            node_map,
            parseable: leaves,
            parsed,
            parent,
        }
    }

    pub fn submit_as_parsed(&mut self, node_id: usize, token_stream: TokenStream) {
        self.parsed.insert(node_id, token_stream);

        if let Some(&parent) = self.parent.get(&node_id) {
            let siblings = self
                .node_map
                .get(&parent)
                .unwrap()
                .operands
                .iter()
                .map(|n| n.id)
                .collect::<Vec<_>>();
            if siblings.iter().all(|s| self.parsed.contains_key(s)) {
                self.parseable.push(parent);
            }
        }
    }

    pub fn pop_next_parseable(&mut self) -> Option<usize> {
        self.parseable.pop()
    }

    pub fn compile_node(&mut self, node_id: usize) -> Result<TokenStream> {
        let node = self.node_map.get(&node_id).unwrap();
        let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let child_streams: HashMap<usize, TokenStream> = children
            .iter()
            .map(|id| (*id, self.parsed.remove(id).unwrap()))
            .collect();

        let stream = match node.op {
            Operation::Number(_) => Number::compile(node, self, child_streams)?,
            Operation::Symbol(_) => Symbol::compile(node, self, child_streams)?,
            Operation::Group => Group::compile(node, self, child_streams)?,
            Operation::Assign => Assign::compile(node, self, child_streams)?,
            Operation::Add => Add::compile(node, self, child_streams)?,
            Operation::Multiply => Multiply::compile(node, self, child_streams)?,

            _ => NotImplemented::compile(node, self, child_streams)?,
        };

        Ok(stream)
    }

    pub fn get_varname(&self, id: usize) -> Ident {
        Ident::new(&format!("_{}", id), Span::call_site())
    }
}

pub trait CompileNode {
    fn compile(
        node: &Node,
        compiler: &CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream>;
}

struct Number;
impl CompileNode for Number {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Number(number) = &node.op {
            let val = number.value;

            let res = quote! {
                array![#val]
            };
            Ok(res)
        } else {
            panic!("Expected number node, found {:?}", node);
        }
    }
}

struct Symbol;
impl CompileNode for Symbol {
    fn compile(
        node: &Node,
        _: &CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Symbol(symbol) = &node.op {
            let varname = Ident::new(symbol.as_str(), Span::call_site());
            Ok(quote!(#varname))
        } else {
            panic!("Expected symbol node, found {:?}", node);
        }
    }
}

struct Assign;
impl CompileNode for Assign {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Assign = &node.op {
        } else {
            panic!("Expected assign node, found {:?}", node);
        }
        let left_member = &node.operands[0];
        let left_stream = child_streams[&left_member.id].clone();
        let right_member = &node.operands[1];
        let right_stream = child_streams[&right_member.id].clone();

        Ok(quote!(
            let #left_stream = #right_stream;
        ))
    }
}

struct Add;
impl CompileNode for Add {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Add = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if !children.len() == 2 {
                panic!("Expected 2 children, found {:?}", children.len());
            }

            let childstream1 = child_streams.get(&children[0]).unwrap();
            let childstream2 = child_streams.get(&children[1]).unwrap();

            let res = quote! {
                arrays::Add::call(&[&#childstream1, &#childstream2])
            };
            Ok(res)
        } else {
            panic!("Expected add node, found {:?}", node);
        }
    }
}
struct Multiply;
impl CompileNode for Multiply {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Multiply = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if !children.len() == 2 {
                panic!("Expected 2 children, found {:?}", children.len());
            }

            let childstream1 = child_streams.get(&children[0]).unwrap();
            let childstream2 = child_streams.get(&children[1]).unwrap();

            let res = quote! {
                arrays::Multiply::call(&[&#childstream1, &#childstream2])
            };
            Ok(res)
        } else {
            panic!("Expected multiply node, found {:?}", node);
        }
    }
}

struct Group;
impl CompileNode for Group {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Group = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let child_streams: Vec<TokenStream> = children
                .iter()
                .map(|id| child_streams[id].clone())
                .collect();

            // let varname = compiler.get_varname(node.id);
            let res = quote! {
                (#(#child_streams),*)
            };
            Ok(res)
        } else {
            panic!("Expected group node, found {:?}", node);
        }
    }
}

struct NotImplemented;
impl CompileNode for NotImplemented {
    fn compile(
        node: &Node,
        _compiler: &CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let error_msg = format!("Compilation for operation '{:?}' not implemented", node.op);
        Ok(quote! {
            panic!(#error_msg);
        })
    }
}
