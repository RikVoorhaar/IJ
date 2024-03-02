use anyhow::Result;
use std::{collections::HashMap, rc::Rc};
use syn::Ident;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use crate::{node::Node, operations::Operation};

pub struct CompilerContext {
    pub node_map: HashMap<usize, Rc<Node>>,
    parseable: Vec<usize>,
    parsed: HashMap<usize, TokenStream>,
    pub parent: HashMap<usize, usize>,
    inputs: Vec<usize>,
}

impl CompilerContext {
    pub fn new(root: Node) -> Self {
        let mut node_map = HashMap::new();
        let mut leaves = vec![];
        let mut parent = HashMap::new();
        let mut stack = vec![Rc::new(root)];

        while let Some(node_rc) = stack.pop() {
            node_map.insert(node_rc.id, node_rc.clone());
            if node_rc.operands.is_empty() {
                leaves.push(node_rc.id);
            }
            for child in &node_rc.operands {
                parent.insert(child.id, node_rc.id);
                stack.push(Rc::new(child.clone()));
            }
        }
        let parsed = HashMap::new();

        Self {
            node_map,
            parseable: leaves,
            parsed,
            parent,
            inputs: vec![],
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
        let node = self.node_map.get(&node_id).unwrap().clone();

        let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let child_streams: HashMap<usize, TokenStream> = children
            .iter()
            .map(|id| (*id, self.parsed.remove(id).unwrap()))
            .collect();
        let node_op = &node.op;

        let stream = match node_op {
            Operation::Number(_) => Number::compile(node, self, child_streams)?,
            Operation::Symbol(_) => Symbol::compile(node, self, child_streams)?,
            Operation::Group => Group::compile(node, self, child_streams)?,
            Operation::Assign => Assign::compile(node, self, child_streams)?,
            Operation::Function => Function::compile(node, self, child_streams)?,
            Operation::Add => Add::compile(node, self, child_streams)?,
            Operation::Multiply => Multiply::compile(node, self, child_streams)?,
            Operation::Identity => Identity::compile(node, self, child_streams)?,
            

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
        node: Rc<Node>,
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream>;
}

struct Number;
impl CompileNode for Number {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
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
        node: Rc<Node>,
        _: &mut CompilerContext,
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

struct Identity;
impl CompileNode for Identity {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        _child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Identity = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if !children.len() == 0 {
                panic!("Expected 0 child, found {:?}", children.len());
            }
            _compiler.inputs.push(node.id);
            let varname = _compiler.get_varname(node.id);

            let res = quote! {
                #varname
            };
            Ok(res)
        } else {
            panic!("Expected identity node, found {:?}", node);
        }
    }
}

struct Assign;
impl CompileNode for Assign {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
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

struct Function;
impl CompileNode for Function {
    fn compile(
        node: Rc<Node>,
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match &node.op {
            Operation::Function => {}
            _ => unreachable!("Expected function node, found {:?}", node),
        }

        let left_member = &node.operands[0];
        let left_stream = child_streams[&left_member.id].clone();
        let right_member = &node.operands[1];
        let right_stream = child_streams[&right_member.id].clone();

        let input_varnames = compiler
            .inputs
            .iter()
            .map(|id| compiler.get_varname(*id))
            .collect::<Vec<_>>();

        Ok(quote!(
            let #left_stream = {|#(#input_varnames),*| #right_stream};
        ))
    }
}

struct Add;
impl CompileNode for Add {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
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
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
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
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
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
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let error_msg = format!("Compilation for operation '{:?}' not implemented", node.op);
        Ok(quote! {
            panic!(#error_msg);
        })
    }
}
