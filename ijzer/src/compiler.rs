use anyhow::Result;
use std::{collections::HashMap, rc::Rc};
use syn::Ident;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use crate::{
    ast_node::{IJType, LineHasSemicolon, Node},
    operations::Operation,
};

pub fn compile_line_from_node(
    node: Rc<Node>,
    has_semicolon: LineHasSemicolon,
) -> Result<TokenStream> {
    let mut compiler = CompilerContext::new(node.clone());
    let mut line_stream: TokenStream = TokenStream::new();

    while let Some(node_id) = compiler.pop_next_parseable() {
        line_stream = compiler.compile(node_id)?;

        compiler.submit_as_parsed(node_id, line_stream.clone())
    }
    if has_semicolon == LineHasSemicolon::Yes {
        line_stream = quote! {#line_stream;};
    }

    Ok(line_stream)
}

pub struct CompilerContext {
    pub node_map: HashMap<usize, Rc<Node>>,
    parseable: Vec<usize>,
    parsed: HashMap<usize, TokenStream>,
    pub parent: HashMap<usize, usize>,
    inputs: Vec<usize>,
}

impl CompilerContext {
    pub fn new(root: Rc<Node>) -> Self {
        let mut node_map = HashMap::new();
        let mut leaves = vec![];
        let mut parent = HashMap::new();
        let mut stack = vec![root];

        while let Some(node_rc) = stack.pop() {
            node_map.insert(node_rc.id, node_rc.clone());
            if node_rc.operands.is_empty() {
                leaves.push(node_rc.id);
            }
            for child in &node_rc.operands {
                parent.insert(child.id, node_rc.id);
                stack.push(child.clone());
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

    pub fn compile(&mut self, node_id: usize) -> Result<TokenStream> {
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
            Operation::Function(_) => Function::compile(node, self, child_streams)?,
            Operation::Group => Group::compile(node, self, child_streams)?,
            Operation::Assign => Assign::compile(node, self, child_streams)?,
            Operation::Add => Add::compile(node, self, child_streams)?,
            Operation::Multiply => Multiply::compile(node, self, child_streams)?,
            Operation::Identity => Identity::compile(node, self, child_streams)?,
            Operation::Nothing => Nothing::compile(node, self, child_streams)?,
            Operation::Subtract => Subtract::compile(node, self, child_streams)?,
            Operation::Negate => Negate::compile(node, self, child_streams)?,
            Operation::Array(_) => Array::compile(node, self, child_streams)?,
            Operation::Reduce => Reduce::compile(node, self, child_streams)?,
            Operation::Scalar => NotImplemented::compile(node, self, child_streams)?,
            // _ => NotImplemented::compile(node, self, child_streams)?,
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
            let val = number.value.clone();

            let parsed_val = syn::parse_str::<proc_macro2::TokenStream>(&val).map_err(|_| {
                syn::Error::new_spanned(&val, "Failed to parse value into a Rust TokenStream")
            })?;
            let res = quote! {
                ijzer::tensor::Tensor::scalar(#parsed_val)
            };
            Ok(res)
        } else {
            panic!("Expected number node, found {:?}", node);
        }
    }
}

struct Array;
impl CompileNode for Array {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Array(s) = &node.op {
            let val = s.clone();

            let parsed_val = syn::parse_str::<proc_macro2::TokenStream>(&val).map_err(|_| {
                syn::Error::new_spanned(&val, "Failed to parse value into a Rust TokenStream")
            })?;
            let res = quote! {
                ijzer::tensor::Tensor::from_vec(vec![#parsed_val], None)
            };
            Ok(res)
        } else {
            panic!("Expected array node, found {:?}", node);
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

struct Function;
impl CompileNode for Function {
    fn compile(
        node: Rc<Node>,
        _: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Function(function_name) = &node.op {
            let varname = Ident::new(function_name.as_str(), Span::call_site());
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();

            let child_streams: Vec<TokenStream> = children
                .iter()
                .map(|id| child_streams[id].clone())
                .collect();

            if child_streams.is_empty() {
                Ok(quote! {#varname})
            } else {
                Ok(quote! {
                    #varname(#(#child_streams),*)
                })
            }
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
            if !children.is_empty() {
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
        compiler: &mut CompilerContext,
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

        match node.input_types.first() {
            Some(IJType::Function(_)) => {
                let input_varnames = compiler
                    .inputs
                    .iter()
                    .map(|id| compiler.get_varname(*id))
                    .collect::<Vec<_>>();

                Ok(quote!(
                    let #left_stream = {|#(#input_varnames),*| #right_stream};
                ))
            }
            Some(_) => Ok(quote!(
                let #left_stream = #right_stream;
            )),
            None => {
                panic!("Expected function node, found {:?}", node);
            }
        }
    }
}

struct Add;
impl CompileNode for Add {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();

        let res = match node.output_type {
            IJType::Tensor => {
                if children.len() != 2 {
                    panic!("Expected 2 children for add, found {:?}", children.len());
                }

                let childstream1 = child_streams.get(&children[0]).unwrap();
                let childstream2 = child_streams.get(&children[1]).unwrap();

                quote! {
                    #childstream1.apply_binary_op(&#childstream2, |a, b| a + b).unwrap()
                }
            }
            IJType::Function(_) => {
                quote! { |a, b| a + b }
            }
            _ => panic!(
                "Found add node with unimplemented output type: {:?}",
                node.output_type
            ),
        };
        Ok(res)
    }
}
struct Subtract;
impl CompileNode for Subtract {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Subtract = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if children.len() != 2 {
                return Ok(quote! {|a,b| a-b});
            }

            let childstream1 = child_streams.get(&children[0]).unwrap();
            let childstream2 = child_streams.get(&children[1]).unwrap();

            let res = quote! {
                #childstream1.apply_binary_op(&#childstream2, |a, b| a - b).unwrap()
            };
            Ok(res)
        } else {
            panic!("Expected subtract node, found {:?}", node);
        }
    }
}
struct Negate;
impl CompileNode for Negate {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Negate = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if children.len() != 1 {
                return Ok(quote! {|a| -a});
            }

            let childstream = child_streams.get(&children[0]).unwrap();

            let res = quote! {
               #childstream.map(|x| -x)
            };
            Ok(res)
        } else {
            panic!("Expected negate node, found {:?}", node);
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
            if children.len() != 2 {
                return Ok(quote! {|a,b| a*b});
            }

            let childstream1 = child_streams.get(&children[0]).unwrap();
            let childstream2 = child_streams.get(&children[1]).unwrap();

            let res = quote! {
                #childstream1.apply_binary_op(&#childstream2, |a, b| a * b).unwrap()
            };
            Ok(res)
        } else {
            Ok(quote! {|a,b| a * b})
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

            match child_streams.len() {
                1 => {
                    let stream = child_streams.first().unwrap();
                    Ok(quote! {#stream})
                }
                _ => Ok(quote! {
                    (#(#child_streams),*)
                }),
            }
        } else {
            panic!("Expected group node, found {:?}", node);
        }
    }
}

struct Reduce;
impl CompileNode for Reduce {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Reduce = &node.op {
        } else {
            panic!("Expected reduce node, found {:?}", node.op);
        }
        let operands = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        if operands.len() != 2 {
            panic!(
                "Expected 2 operands for reduce operation, found {}",
                operands.len()
            );
        }

        let functional_operand = &operands[0];
        let functional_operand_stream = child_streams.get(functional_operand).unwrap();

        let data_stream = child_streams.get(&operands[1]).unwrap();

        Ok(quote! {
            #data_stream.reduce(#functional_operand_stream)
        })
    }
}

struct Nothing;
impl CompileNode for Nothing {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Nothing = &node.op {
            let res = quote! {};
            Ok(res)
        } else {
            panic!("Expected nothing node, found {:?}", node);
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

/// TODO tests to write. Check output for statements:
/// empty
/// var
/// x: T = [1]
/// x = [1]
/// x: S = 1
/// x = + [1] I
/// + x 1 (error; undefined)
/// x = 1; + x 1
/// var x: Fn(S->T); x 1
/// + [1] [2]
/// + [1] 2
/// + 1 2
/// * [1] [2]
/// * [1] 2
/// * 1 2
/// - 1.0
/// - [1]
/// - [1] [2]
/// - [1] 2
/// - 1 2
/// (- (+ (1) 2))
/// /+ [1,2]
/// var f: Fn(S,S->S); /f [1,2]
/// Some more complicated multiline statement
#[cfg(test)]
mod tests {}
