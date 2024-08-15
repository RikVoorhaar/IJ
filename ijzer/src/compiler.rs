use anyhow::Result;
use std::{collections::HashMap, rc::Rc};
use syn::Ident;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};

use crate::ast_node::{LineHasSemicolon, Node};
use crate::operations::Operation;
use crate::types::IJType;

pub fn number_type_from_string(number_type: &str) -> Result<TokenStream, syn::Error> {
    let parsed_val = syn::parse_str::<proc_macro2::TokenStream>(number_type).map_err(|err| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("Failed to parse value into a Rust TokenStream: {}", err),
        )
    })?;
    Ok(parsed_val)
}

pub fn compile_line_from_node(
    node: Rc<Node>,
    has_semicolon: LineHasSemicolon,
    number_type: TokenStream,
) -> Result<TokenStream> {
    let mut compiler = CompilerContext::new(node.clone(), number_type);
    let mut line_stream: TokenStream = TokenStream::new();

    while let Some(node_id) = compiler.pop_next_parseable() {
        line_stream = compiler.compile(node_id)?;

        compiler.submit_as_parsed(node_id, line_stream.clone())
    }

    // Add semicolon if necessary; avoid putting double semicolons
    if has_semicolon == LineHasSemicolon::Yes {
        let mut ts_iter = line_stream.clone().into_iter().collect::<Vec<_>>();
        let last_token = ts_iter.pop();
        let needs_semicolon = !matches!(last_token, Some(proc_macro2::TokenTree::Punct(ref p)) if p.as_char() == ';' && p.spacing() == proc_macro2::Spacing::Alone)
            && last_token.is_some();

        if needs_semicolon {
            line_stream = quote! {#line_stream;};
        }
    }

    Ok(line_stream)
}

fn generate_identifiers(n: usize, name: &str) -> Vec<Ident> {
    (1..=n).map(|i| format_ident!("{}{}", name, i)).collect()
}

pub struct CompilerContext {
    pub node_map: HashMap<usize, Rc<Node>>,
    parseable: Vec<usize>,
    parsed: HashMap<usize, TokenStream>,
    pub parent: HashMap<usize, usize>,
    inputs: Vec<(usize, String)>,
    number_type: TokenStream,
}

impl CompilerContext {
    pub fn new(root: Rc<Node>, number_type: TokenStream) -> Self {
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
            number_type,
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
            Operation::LambdaVariable(_) => LambdaVariable::compile(node, self, child_streams)?,
            Operation::FunctionComposition(_) => {
                FunctionComposition::compile(node, self, child_streams)?
            }
            Operation::Apply => Apply::compile(node, self, child_streams)?,
            Operation::TypeConversion => TypeConversion::compile(node, self, child_streams)?,
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
        compiler: &mut CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Number(number) = &node.op {
            let val = number.value.clone();

            let parsed_val = syn::parse_str::<proc_macro2::TokenStream>(&val).map_err(|_| {
                syn::Error::new_spanned(&val, "Failed to parse value into a Rust TokenStream")
            })?;
            let number_type = compiler.number_type.clone();
            let res = quote! {
                ijzer::tensor::Tensor::<#number_type>::scalar(#parsed_val)
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
        compiler: &mut CompilerContext,
        _: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Array(s) = &node.op {
            let val = s.clone();

            let parsed_val = syn::parse_str::<proc_macro2::TokenStream>(&val).map_err(|_| {
                syn::Error::new_spanned(&val, "Failed to parse value into a Rust TokenStream")
            })?;
            let number_type = compiler.number_type.clone();
            let res = quote! {
                ijzer::tensor::Tensor::<#number_type> ::from_vec(vec![#parsed_val], None)
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
            _compiler.inputs.push((node.id, format!("_{}", node.id)));
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

struct LambdaVariable;
impl CompileNode for LambdaVariable {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        _child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::LambdaVariable(name) = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if !children.is_empty() {
                panic!("Expected 0 child, found {:?}", children.len());
            }
            let name_with_underscore = format!("_{}", name);
            _compiler
                .inputs
                .push((node.id, name_with_underscore.clone()));
            let varname = Ident::new(name_with_underscore.as_str(), Span::call_site());

            let res = quote! {
                #varname
            };
            Ok(res)
        } else {
            panic!("Expected lambda variable node, found {:?}", node);
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
                    .map(|(_, name)| Ident::new(name, Span::call_site()))
                    .collect::<Vec<_>>();

                let number_type = compiler.number_type.clone();
                Ok(quote!(
                    let #left_stream = {|#(#input_varnames: ijzer::tensor::Tensor::<#number_type>),*| #right_stream};
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
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let number_type = compiler.number_type.clone();

        let res = match node.output_type.clone() {
            IJType::Tensor | IJType::Scalar => {
                if children.len() != 2 {
                    panic!("Expected 2 children for add, found {:?}", children.len());
                }

                let childstream1 = child_streams.get(&children[0]).unwrap();
                let childstream2 = child_streams.get(&children[1]).unwrap();

                quote! {
                    #childstream1.apply_binary_op(&#childstream2, |a: #number_type, b: #number_type| a + b).unwrap()
                }
            }
            IJType::Function(f) => {
                let output_type = f.output.first().unwrap();
                match output_type {
                    IJType::Number => {
                        quote! { |a: #number_type, b: #number_type| a + b }
                    }
                    IJType::Scalar | IJType::Tensor => {
                        quote! {
                            |x1: ijzer::tensor::Tensor::<#number_type>, x2: ijzer::tensor::Tensor::<#number_type>|
                            x1.apply_binary_op(&x2, |a: #number_type, b: #number_type| a + b).unwrap()
                        }
                    }
                    _ => panic!(
                        "Found add node with unimplemented output type: {:?}",
                        node.output_type
                    ),
                }
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
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Subtract = &node.op {
            let number_type = compiler.number_type.clone();
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor | IJType::Scalar => {
                    let childstream1 = child_streams.get(&children[0]).unwrap();
                    let childstream2 = child_streams.get(&children[1]).unwrap();

                    quote! {
                       #childstream1.apply_binary_op(&#childstream2, |a: #number_type, b: #number_type| a - b).unwrap()
                    }
                }
                IJType::Function(f) => {
                    let output_type = f.output.first().unwrap();
                    match output_type {
                        IJType::Number => {
                            quote! { |a: #number_type, b: #number_type| a - b }
                        }
                        IJType::Scalar | IJType::Tensor => {
                            quote! {
                                |x1: ijzer::tensor::Tensor::<#number_type>, x2: ijzer::tensor::Tensor::<#number_type>|
                                x1.apply_binary_op(&x2, |a: #number_type, b: #number_type| a - b).unwrap()
                            }
                        }
                        _ => panic!(
                            "Found subtract node with unimplemented output type: {:?}",
                            node.output_type
                        ),
                    }
                }
                _ => panic!(
                    "Found subtract node with unimplemented output type: {:?}",
                    node.output_type
                ),
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
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Negate = &node.op {
            let number_type = compiler.number_type.clone();
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor | IJType::Scalar => {
                    let childstream = child_streams.get(&children[0]).unwrap();
                    quote! {
                        #childstream.map(|a: #number_type| -a)
                    }
                }
                IJType::Function(f) => {
                    let output_type = f.output.first().unwrap();
                    match output_type {
                        IJType::Number => {
                            quote! { |a: #number_type| -a }
                        }
                        IJType::Scalar | IJType::Tensor => {
                            quote! {
                                |x: ijzer::tensor::Tensor::<#number_type>| x.map(|a: #number_type| -a)
                            }
                        }
                        _ => panic!(
                            "Found negate node with unimplemented output type: {:?}",
                            node.output_type
                        ),
                    }
                }
                _ => panic!(
                    "Found negate node with unimplemented output type: {:?}",
                    node.output_type
                ),
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
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Multiply = &node.op {
            let number_type = compiler.number_type.clone();
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor | IJType::Scalar => {
                    let childstream1 = child_streams.get(&children[0]).unwrap();
                    let childstream2 = child_streams.get(&children[1]).unwrap();

                    quote! {
                        #childstream1.apply_binary_op(&#childstream2, |a: #number_type, b: #number_type| a * b).unwrap()
                    }
                }
                IJType::Function(f) => {
                    let output_type = f.output.first().unwrap();
                    match output_type {
                        IJType::Number => {
                            quote! { |a: #number_type, b: #number_type| a * b }
                        }
                        IJType::Scalar | IJType::Tensor => {
                            quote! {
                                |x1: ijzer::tensor::Tensor::<#number_type>, x2: ijzer::tensor::Tensor::<#number_type>|
                                x1.apply_binary_op(&x2, |a: #number_type, b: #number_type| a * b).unwrap()
                            }
                        }
                        _ => panic!(
                            "Found multiply node with unimplemented output type: {:?}",
                            node.output_type
                        ),
                    }
                }
                _ => panic!(
                    "Found multiply node with unimplemented output type: {:?}",
                    node.output_type
                ),
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

struct FunctionComposition;
impl FunctionComposition {
    fn apply_stream_to_stream(stream1: TokenStream, stream2: TokenStream) -> TokenStream {
        quote! {
            (#stream1)(#stream2)
        }
    }
}
impl CompileNode for FunctionComposition {
    fn compile(
        node: Rc<Node>,
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let number_type = compiler.number_type.clone();
        let num_functions = if let Operation::FunctionComposition(n) = &node.op {
            *n
        } else {
            panic!(
                "Expected FunctionComposition operation, found {:?}",
                node.op
            );
        };
        let num_data_operands = node.operands.len() - num_functions;
        let operands = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let functional_operands = operands.iter().take(num_functions);
        let data_operands = operands.iter().skip(num_functions);
        let data_streams = data_operands
            .map(|id| child_streams[id].clone())
            .collect::<Vec<_>>();

        let identifiers = generate_identifiers(num_data_operands, &format!("_{}_", node.id));
        let closure = functional_operands.rev().fold(
            quote! {
                #(#identifiers),*
            },
            |acc, id| FunctionComposition::apply_stream_to_stream(child_streams[id].clone(), acc),
        );
        let closure = quote! {|#(#identifiers: ijzer::tensor::Tensor::<#number_type>),*| #closure};

        Ok(quote! {
            (#closure)(#(#data_streams),*)
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

/// Apply nodes have as first argument a function, and the other arguments are the operands to apply the function to.
struct Apply;
impl CompileNode for Apply {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Apply = &node.op {
        } else {
            panic!("Expected apply node, found {:?}", node.op);
        }

        let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let child_streams: Vec<TokenStream> = children
            .iter()
            .map(|id| child_streams[id].clone())
            .collect();

        if child_streams.is_empty() {
            panic!("Expected at least one child for apply node, found 0");
        }

        let function_stream = child_streams.first().unwrap();
        let operand_streams = child_streams.iter().skip(1);

        Ok(quote! {
            #function_stream(#(#operand_streams),*)
        })
    }
}

struct TypeConversion;
impl TypeConversion {
    fn convert_type(
        from: &IJType,
        to: &IJType,
        child_stream: TokenStream,
        number_type: TokenStream,
        id: usize,
    ) -> Result<TokenStream> {
        let res = match (from, to) {
            (IJType::Tensor, IJType::Tensor) => quote! {#child_stream},
            (IJType::Scalar, IJType::Scalar) => quote! {#child_stream},
            (IJType::Scalar, IJType::Tensor) => quote! {#child_stream},
            (IJType::Scalar, IJType::Number) => quote! {#child_stream.extract_scalar()},
            (IJType::Number, IJType::Number) => quote! {#child_stream},
            (IJType::Number, IJType::Scalar) => {
                quote! {ijzer::tensor::Tensor::<#number_type>::scalar(#child_stream)}
            }
            (IJType::Number, IJType::Tensor) => {
                quote! {ijzer::tensor::Tensor::<#number_type>::scalar(#child_stream)}
            }
            (IJType::Function(ref signature_from), IJType::Function(ref signature_to)) => {
                let num_ops = signature_from.input.len();
                let idents = generate_identifiers(num_ops, &format!("_{}_", id));
                let input_conversions = signature_from
                    .input
                    .iter()
                    .zip(signature_to.input.clone())
                    .zip(idents.clone())
                    .map(|((from, to), ident)| {
                        Self::convert_type(&to, from, quote!(#ident), number_type.clone(), id)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let input_stream = quote! {(#child_stream)(#(#input_conversions),*)};
                let converted_stream = Self::convert_type(
                    signature_from.output.first().unwrap(),
                    signature_to.output.first().unwrap(),
                    input_stream,
                    number_type.clone(),
                    id,
                )?;
                quote! {
                    (|#(#idents: ijzer::tensor::Tensor::<#number_type>),*| (#converted_stream))
                }
            }
            _ => {
                panic!("Type conversion from {} to {} not implemented", from, to);
            }
        };
        Ok(res)
    }
}
impl CompileNode for TypeConversion {
    fn compile(
        node: Rc<Node>,
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let number_type = compiler.number_type.clone();
        if node.operands.len() != 1 {
            panic!("Expected 1 operands, found {}", node.operands.len());
        }
        let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
        Self::convert_type(
            &node.input_types[0],
            &node.output_type,
            child_stream,
            number_type,
            node.id,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile;

    fn compiler_compare(input: &str, expected: &str, number_type: &str) {
        let compiled_stream_result = compile(input, number_type);
        if let Err(e) = &compiled_stream_result {
            println!("Error: {}", e);
            panic!("Compilation failed");
        }
        let compiled_stream = compiled_stream_result.unwrap();
        println!("------------------------");
        println!("Input:");
        println!("{}", input);
        println!("Compiled stream (string):");
        println!("'{}'", compiled_stream);
        // println!("Compiled stream (TokenStream):");
        // println!("{:?}", compiled_stream);

        let expected_stream_result = syn::parse_str(expected);
        assert!(expected_stream_result.is_ok());
        let expected_stream: TokenStream = expected_stream_result.unwrap();

        println!("------------------------");
        println!("Expected stream (string):");
        println!("'{}'", expected_stream);
        // println!("Expected stream (TokenStream):");
        // println!("{:?}", expected_stream);
        println!("------------------------");

        assert_eq!(
            compiled_stream.to_string().replace(" ", ""),
            expected_stream.to_string().replace(" ", "")
        );
    }

    #[test]
    fn test_empty_input() {
        compiler_compare("", "", "f64");
    }

    #[test]
    fn test_var() {
        compiler_compare("var x: T", "", "f64");
    }

    #[test]
    fn test_assign_tensor() {
        let input1 = "x: T = [1]";
        let input2 = "x = [1]";
        let expexted = "let x = ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None);";
        compiler_compare(input1, expexted, "i64");
        compiler_compare(input2, expexted, "i64");
    }

    #[test]
    fn test_assign_scalar() {
        let input1 = "x: S = 1";
        let input2 = "x = 1";
        let expexted = "let x = ijzer::tensor::Tensor::<i64>::scalar(1);";
        compiler_compare(input1, expexted, "i64");
        compiler_compare(input2, expexted, "i64");
    }

    #[test]
    fn test_multiline() {
        let input1 = "x = [1]
        y = + x x
        ";
        let input2 = "x = [1]; y= + x x";
        let input3 = "
        x = [1];
        y= + x x;";
        let expexted = "let x = ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None); let y = x.apply_binary_op(&x, |a: i64, b: i64| a + b).unwrap();";
        compiler_compare(input1, expexted, "i64");
        compiler_compare(input2, expexted, "i64");
        compiler_compare(input3, expexted, "i64");
    }

    #[test]
    fn test_simple_function() {
        let input1 = "x: Fn(T->T) = + [1] I";
        let input2 = "x = + [1] I";
        let expexted = "let x = { | _1: ijzer::tensor::Tensor::<i64> | ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&_1, |a: i64, b: i64| a + b).unwrap() } ;";
        compiler_compare(input1, expexted, "i64");
        compiler_compare(input2, expexted, "i64");
    }

    #[test]
    fn test_function_apply() {
        let input1 = "var x: Fn(S->T); x 1";
        let input2 = "var x: Fn(S->T)
         x 1";
        let expexted = "x (ijzer::tensor::Tensor::<i64>::scalar(1))";
        compiler_compare(input1, expexted, "i64");
        compiler_compare(input2, expexted, "i64");
    }

    #[test]
    fn test_add() {
        compiler_compare("+ [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a + b).unwrap()", "i64");
        compiler_compare("+ [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a + b).unwrap()", "i64");
        compiler_compare("+ 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a + b).unwrap()", "i64");
        compiler_compare("+ 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a + b).unwrap()", "i64");
    }
    #[test]
    fn test_subtract() {
        compiler_compare("- [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a - b).unwrap()", "i64");
        compiler_compare("- [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a - b).unwrap()", "i64");
        compiler_compare("- 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a - b).unwrap()", "i64");
        compiler_compare("- 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a - b).unwrap()", "i64");
    }
    #[test]
    fn test_multiply() {
        compiler_compare("* [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a * b).unwrap()", "i64");
        compiler_compare("* [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a * b).unwrap()", "i64");
        compiler_compare("* 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a * b).unwrap()", "i64");
        compiler_compare("* 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a * b).unwrap()", "i64");
    }

    #[test]
    fn test_negate() {
        compiler_compare(
            "-[1]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).map(|a: i64| -a)",
            "i64",
        );
        compiler_compare("var x: S; -x", "x.map(|a: i64| -a)", "i64");
        compiler_compare("- I", "_0 . map (| a: i64 | -a)", "i64");
    }

    #[test]
    fn test_reduction() {
        compiler_compare(
            "var f: Fn(N,N->N); /f [1,2]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None).reduce(f)",
            "i64",
        );
        compiler_compare(
            "/+ [1,2]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None).reduce(|a: i64, b: i64| a + b)",
            "i64",
        );
    }

    #[test]
    fn test_lambda_variable() {
        let expected1 =
            "let f = { | _x: ijzer::tensor::Tensor::<i64> , _x: ijzer::tensor::Tensor::<i64> | _x . apply_binary_op (& _x , | a: i64 , b: i64 | a + b) . unwrap () } ;";
        compiler_compare("f = + $x $x", expected1, "i64");
        compiler_compare("f: Fn(T,T->T) = + $x $x", expected1, "i64");

        let expected2 =
            "let f = { | _x: ijzer::tensor::Tensor::<i64> , _y: ijzer::tensor::Tensor::<i64> | _x . apply_binary_op (& _y , | a: i64 , b: i64 | a + b) . unwrap () } ;";
        compiler_compare("f = + $x $y", expected2, "i64");
        compiler_compare("f: Fn(T,T->T) = + $x $y", expected2, "i64");

        let expected3="let h = { | _x: ijzer::tensor::Tensor::<i64> | k (_x , ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None)) } ;";
        compiler_compare(
            "var k: Fn(T,T->T); h: Fn(T->T) = k $x [1,2]",
            expected3,
            "i64",
        );
        compiler_compare("var k: Fn(T,T->T); h = k $x [1,2]", expected3, "i64");
    }

    #[test]
    fn test_function_composition() {
        let input = "@(-,+) 1 2";
        let expected = "(| _15_1 : ijzer :: tensor :: Tensor :: < i64 > , _15_2 : ijzer :: tensor :: Tensor :: < i64 > | (| x : ijzer :: tensor :: Tensor :: < i64 > | x . map (| a : i64 | - a)) ((| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_15_1 , _15_2))) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected, "i64");

        let input = "@(+) 1 2";
        let expected = "(| _7_1 : ijzer :: tensor :: Tensor :: < i64 > , _7_2 : ijzer :: tensor :: Tensor :: < i64 > | (| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_7_1 , _7_2)) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected, "i64");

        let input = "var f: Fn(T->T); @(f,-) [1]";
        let expected = "(|_11_1:ijzer::tensor::Tensor::<i64>|(f)((|x:ijzer::tensor::Tensor::<i64>|x.map(|a:i64|-a))(_11_1)))(ijzer::tensor::Tensor::<i64>::from_vec(vec![1],None))";
        compiler_compare(input, expected, "i64");
    }

    #[test]
    fn test_number_type_from_string() {
        let tokens = number_type_from_string("i64").unwrap();
        assert_eq!(tokens.to_string(), (quote! {i64}).to_string());

        let tokens = number_type_from_string("f64").unwrap();
        assert_eq!(tokens.to_string(), (quote! {f64}).to_string());
    }

    #[test]
    fn test_type_conersion() {
        let input = "var x: S; <-T x";
        let expected = "x";
        compiler_compare(input, expected, "i64");

        let input = "var f: Fn(T->T); <-Fn(S->T) f 1";
        let expected = "(| _2_1 : ijzer :: tensor :: Tensor :: < i64 > | ((f) (_2_1))) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1))";
        compiler_compare(input, expected, "i64");

        let input = "var f: Fn(T->N); <-Fn(S->T) f 1";
        let expected = "(|_2_1:ijzer::tensor::Tensor::<i64>|(ijzer::tensor::Tensor::<i64>::scalar((f)(_2_1))))(ijzer::tensor::Tensor::<i64>::scalar(1))";
        compiler_compare(input, expected, "i64");

        let input = "var f: Fn(S,S->S); /<-Fn(N,N->N) f [1]";
        let expected = "ijzer::tensor::Tensor::<i64>::from_vec(vec![1],None).reduce((|_2_1:ijzer::tensor::Tensor::<i64>,_2_2:ijzer::tensor::Tensor::<i64>|((f)(ijzer::tensor::Tensor::<i64>::scalar(_2_1),ijzer::tensor::Tensor::<i64>::scalar(_2_2)).extract_scalar())))";
        compiler_compare(input, expected, "i64");
    }

    #[test]
    fn test_type_implicit_function() {
        let input = "<-Fn(T,T->T) + $x $y";
        let expected = "(|_5_1:ijzer::tensor::Tensor::<i64>,_5_2:ijzer::tensor::Tensor::<i64>|((|x1:ijzer::tensor::Tensor::<i64>,x2:ijzer::tensor::Tensor::<i64>|x1.apply_binary_op(&x2,|a:i64,b:i64|a+b).unwrap())(_5_1,_5_2)))(_x,_y)";
        compiler_compare(input, expected, "i64");
    }
}
