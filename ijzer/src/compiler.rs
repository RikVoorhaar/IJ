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
pub fn annotation_from_type(var_type: &IJType) -> Result<TokenStream> {
    let number_type_annot = number_type_from_string(
        &var_type
            .extract_number_type()
            .unwrap_or_default()
            .unwrap_or("_".to_string()),
    )?;
    let res = match var_type {
        IJType::Number(_) => quote!(#number_type_annot),
        IJType::Tensor(_) => quote!(ijzer::tensor::Tensor::<#number_type_annot>),
        IJType::Scalar(_) => quote!(ijzer::tensor::Tensor::<#number_type_annot>),
        IJType::Function(signature) => {
            let input_types = signature
                .input
                .iter()
                .map(annotation_from_type)
                .collect::<Result<Vec<_>>>()?;
            let output_type = annotation_from_type(&signature.output)?;
            quote! {
                fn(#(#input_types),*) -> #output_type
            }
        }
        _ => panic!("Unsupported type: {:?}", var_type),
    };
    Ok(res)
}

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
            Operation::LambdaVariable(_) => LambdaVariable::compile(node, self, child_streams)?,
            Operation::FunctionComposition(_) => {
                FunctionComposition::compile(node, self, child_streams)?
            }
            Operation::Apply => Apply::compile(node, self, child_streams)?,
            Operation::TypeConversion => TypeConversion::compile(node, self, child_streams)?,
            Operation::AsFunction => AsFunction::compile(node, self, child_streams)?,
            Operation::GeneralizedContraction => {
                GeneralizedContraction::compile(node, self, child_streams)?
            } // _ => NotImplemented::compile(node, self, child_streams)?,
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
            let number_type = node.output_type.extract_number_type().unwrap_or_default();
            let t = annotation_from_type(&IJType::Scalar(number_type))?;
            let res = quote! {
                #t::scalar(#parsed_val)
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
            let number_type = node.output_type.extract_number_type().unwrap_or_default();
            let t = annotation_from_type(&IJType::Scalar(number_type))?;
            let res = quote! {
                #t::from_vec(vec![#parsed_val], None)
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
                |#varname| #varname
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
        compiler: &mut CompilerContext,
        _child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::LambdaVariable(name) = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            if !children.is_empty() {
                panic!("Expected 0 child, found {:?}", children.len());
            }
            let name_with_underscore = format!("_{}", name);
            compiler
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Assign = &node.op {
        } else {
            panic!("Expected assign node, found {:?}", node);
        }
        let lhs_stream = child_streams[&node.operands[0].id].clone();
        let rhs_stream = child_streams[&node.operands[1].id].clone();
        let args_streams = node
            .operands
            .iter()
            .skip(2)
            .map(|n| {
                let s = child_streams[&n.id].clone();
                let t = annotation_from_type(&n.output_type)?;
                Ok(quote!(#s: #t))
            })
            .collect::<Result<Vec<_>>>()?;

        if args_streams.is_empty() {
            Ok(quote!(
                let #lhs_stream = #rhs_stream;
            ))
        } else {
            Ok(quote!(
                let #lhs_stream = {|#(#args_streams),*| #rhs_stream};
            ))
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

        let res = match node.output_type.clone() {
            IJType::Tensor(number_type) | IJType::Scalar(number_type) => {
                let number_annot = annotation_from_type(&IJType::Number(number_type))?;
                if children.len() != 2 {
                    panic!("Expected 2 children for add, found {:?}", children.len());
                }

                let childstream1 = child_streams.get(&children[0]).unwrap();
                let childstream2 = child_streams.get(&children[1]).unwrap();

                quote! {
                    #childstream1.apply_binary_op(&#childstream2, |a: #number_annot, b: #number_annot| a + b).unwrap()
                }
            }
            IJType::Function(f) => {
                let output_type = *f.output;
                let number_type = output_type.extract_number_type().unwrap_or_default();
                let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
                let tensor_annot = annotation_from_type(&IJType::Tensor(number_type))?;
                match output_type {
                    IJType::Number(_) => {
                        quote! { |a: #number_annot, b: #number_annot| a + b }
                    }
                    IJType::Scalar(_) | IJType::Tensor(_) => {
                        quote! {
                            |x1: #tensor_annot, x2: #tensor_annot|
                            x1.apply_binary_op(&x2, |a: #number_annot, b: #number_annot| a + b).unwrap()
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Subtract = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor(_) | IJType::Scalar(_) => {
                    let number_type = node.output_type.extract_number_type().unwrap_or_default();
                    let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
                    let childstream1 = child_streams.get(&children[0]).unwrap();
                    let childstream2 = child_streams.get(&children[1]).unwrap();

                    quote! {
                       #childstream1.apply_binary_op(&#childstream2, |a: #number_annot, b: #number_annot| a - b).unwrap()
                    }
                }
                IJType::Function(f) => {
                    let output_type = *f.output;
                    let number_type = output_type.extract_number_type().unwrap_or_default();
                    let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
                    let tensor_annot = annotation_from_type(&IJType::Tensor(number_type))?;
                    match output_type {
                        IJType::Number(_) => {
                            quote! { |a: #number_annot, b: #number_annot| a - b }
                        }
                        IJType::Scalar(_) | IJType::Tensor(_) => {
                            quote! {
                                |x1: #tensor_annot, x2: #tensor_annot|
                                x1.apply_binary_op(&x2, |a: #number_annot, b: #number_annot| a - b).unwrap()
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Negate = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor(number_type) | IJType::Scalar(number_type) => {
                    let number_annot = annotation_from_type(&IJType::Number(number_type))?;
                    let childstream = child_streams.get(&children[0]).unwrap();
                    quote! {
                        #childstream.map(|a: #number_annot| -a)
                    }
                }
                IJType::Function(f) => {
                    let output_type = *f.output;
                    let number_type = output_type.extract_number_type().unwrap_or_default();
                    let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
                    let tensor_annot = annotation_from_type(&IJType::Tensor(number_type))?;
                    match output_type {
                        IJType::Number(_) => {
                            quote! { |a: #number_annot| -a }
                        }
                        IJType::Scalar(_) | IJType::Tensor(_) => {
                            quote! {
                                |x: #tensor_annot| x.map(|a: #number_annot| -a)
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Multiply = &node.op {
            let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
            let res = match node.output_type.clone() {
                IJType::Tensor(number_type) | IJType::Scalar(number_type) => {
                    let number_annot = annotation_from_type(&IJType::Number(number_type))?;
                    let childstream1 = child_streams.get(&children[0]).unwrap();
                    let childstream2 = child_streams.get(&children[1]).unwrap();

                    quote! {
                        #childstream1.apply_binary_op(&#childstream2, |a: #number_annot, b: #number_annot| a * b).unwrap()
                    }
                }
                IJType::Function(f) => {
                    let output_type = *f.output;
                    let number_type = output_type.extract_number_type().unwrap_or_default();
                    let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
                    let tensor_annot = annotation_from_type(&IJType::Tensor(number_type))?;
                    match output_type {
                        IJType::Number(_) => {
                            quote! { |a: #number_annot, b: #number_annot| a * b }
                        }
                        IJType::Scalar(_) | IJType::Tensor(_) => {
                            quote! {
                                |x1: #tensor_annot, x2: #tensor_annot|
                                x1.apply_binary_op(&x2, |a: #number_annot, b: #number_annot| a * b).unwrap()
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
        compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Reduce = &node.op {
        } else {
            panic!("Expected reduce node, found {:?}", node.op);
        }
        let operand_ids = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        match operand_ids.len() {
            1 => {
                let functional_operand = &operand_ids[0];
                let number_type = node
                    .output_type
                    .extract_signature()
                    .unwrap()
                    .output
                    .extract_number_type()
                    .unwrap_or_default();
                let tensor_t = annotation_from_type(&IJType::Tensor(number_type))?;
                let functional_operand_stream = child_streams.get(functional_operand).unwrap();
                let ident = compiler.get_varname(node.id);
                Ok(quote! {
                    |#ident: #tensor_t| #ident.reduce(#functional_operand_stream)
                })
            }
            2 => {
                let functional_operand = &operand_ids[0];
                let functional_operand_stream = child_streams.get(functional_operand).unwrap();

                let data_stream = child_streams.get(&operand_ids[1]).unwrap();

                Ok(quote! {
                    #data_stream.reduce(#functional_operand_stream)
                })
            }
            _ => {
                panic!(
                    "Expected 1 or 2 operands for reduce operation, found {}",
                    operand_ids.len()
                );
            }
        }
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let num_functions = if let Operation::FunctionComposition(n) = &node.op {
            *n
        } else {
            panic!(
                "Expected FunctionComposition operation, found {:?}",
                node.op
            );
        };

        // let num_data_operands = node.operands.len() - num_functions;
        let operand_ids = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let functional_operand_ids = operand_ids.iter().take(num_functions);
        let data_operand_ids = operand_ids.iter().skip(num_functions);
        let data_streams = data_operand_ids
            .map(|id| child_streams[id].clone())
            .collect::<Vec<_>>();

        let last_functional_operand = node.operands[num_functions - 1].clone();
        let mut args = Vec::new();
        let mut args_with_type = Vec::new();
        for (i, arg_type) in last_functional_operand
            .output_type
            .extract_signature()
            .unwrap()
            .input
            .iter()
            .enumerate()
        {
            let ident = Ident::new(&format!("_{}_{}", node.id, i + 1), Span::call_site());
            let type_annotation = annotation_from_type(arg_type)?;
            args_with_type.push(quote! { #ident: #type_annotation });
            args.push(quote! { #ident });
        }
        let closure = functional_operand_ids.rev().fold(
            quote! {
                #(#args),*
            },
            |acc, id| FunctionComposition::apply_stream_to_stream(child_streams[id].clone(), acc),
        );
        let closure = quote! {|#(#args_with_type),*| #closure};

        if data_streams.is_empty() {
            Ok(closure)
        } else {
            Ok(quote! {
                (#closure)(#(#data_streams),*)
            })
        }
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
        id: usize,
    ) -> Result<TokenStream> {
        let res = match (from, to) {
            (IJType::Tensor(_), IJType::Tensor(_)) => quote! {#child_stream},
            (IJType::Scalar(_), IJType::Scalar(_)) => quote! {#child_stream},
            (IJType::Scalar(_), IJType::Tensor(_)) => quote! {#child_stream},
            (IJType::Scalar(_), IJType::Number(_)) => {
                quote! {#child_stream.extract_scalar().unwrap()}
            }
            (IJType::Number(_), IJType::Number(_)) => quote! {#child_stream},
            (IJType::Number(_), IJType::Scalar(number_type)) => {
                let tensor_t = annotation_from_type(&IJType::Tensor(number_type.clone()))?;
                quote! {#tensor_t::scalar(#child_stream)}
            }
            (IJType::Number(_), IJType::Tensor(number_type)) => {
                let tensor_t = annotation_from_type(&IJType::Tensor(number_type.clone()))?;
                quote! {#tensor_t::scalar(#child_stream)}
            }
            (IJType::Function(ref signature_from), IJType::Function(ref signature_to)) => {
                let num_ops = signature_from.input.len();
                let idents = generate_identifiers(num_ops, &format!("_{}_", id));

                let mut input_conversions = Vec::new();
                let mut args = Vec::new();
                for ((from, to), ident) in signature_from
                    .input
                    .iter()
                    .zip(signature_to.input.clone())
                    .zip(idents.clone())
                {
                    input_conversions.push(Self::convert_type(&to, from, quote!(#ident), id)?);
                    let arg_type = annotation_from_type(&to)?;
                    args.push(quote!(#ident: #arg_type));
                }
                let input_stream = quote! {(#child_stream)(#(#input_conversions),*)};
                let converted_stream = Self::convert_type(
                    &signature_from.output,
                    &signature_to.output,
                    input_stream,
                    id,
                )?;
                quote! {
                    (|#(#args),*| (#converted_stream))
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
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if node.operands.len() != 1 {
            panic!("Expected 1 operands, found {}", node.operands.len());
        }
        let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
        Self::convert_type(
            &node.input_types[0],
            &node.output_type,
            child_stream,
            node.id,
        )
    }
}

struct AsFunction;
impl CompileNode for AsFunction {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if node.operands.len() != 1 {
            panic!(
                "Expected 1 operand for AsFunction, found {}",
                node.operands.len()
            );
        }
        let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
        Ok(child_stream)
    }
}

struct GeneralizedContraction;
impl CompileNode for GeneralizedContraction {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let child_streams = node
            .operands
            .iter()
            .map(|n| child_streams[&n.id].clone())
            .collect::<Vec<_>>();
        let f_stream = child_streams[0].clone();
        let f_node = &node.operands[0];
        let f_arg_annot = annotation_from_type(&f_node.input_types[0])?;
        let f_stream_extract = quote! {
            (|z: &#f_arg_annot| (#f_stream)(z.clone()).extract_scalar().unwrap())
        };
        let g_stream = child_streams[1].clone();
        let g_node = &node.operands[1];
        let input_number_type = g_node.input_types[0]
            .extract_number_type()
            .unwrap_or_default();
        let tensor_t = annotation_from_type(&IJType::Tensor(input_number_type))?;
        match node.operands.len() {
            2 => Ok(quote! {
                |x: #tensor_t, y: #tensor_t| x.generalized_contraction(&y, #f_stream_extract, #g_stream).unwrap()
            }),
            4 => {
                let op1 = child_streams[2].clone();
                let op2 = child_streams[3].clone();
                Ok(quote! {
                    #op1.generalized_contraction(&#op2, #f_stream_extract, #g_stream).unwrap()
                })
            }
            _ => {
                panic!(
                    "Expected 2 or 4 operands for GeneralizedContraction, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile;

    fn compiler_compare(input: &str, expected: &str) {
        let compiled_stream_result = compile(input);
        if let Err(e) = &compiled_stream_result {
            println!("Error: {:?}", e);
            panic!("Compilation failed");
        }
        let compiled_stream = compiled_stream_result.unwrap();
        println!("------------------------");
        println!("Input:");
        println!("{}", input);
        println!("Compiled stream (string):");
        println!("'{}'", compiled_stream);

        let expected_stream_result = syn::parse_str(expected);
        assert!(expected_stream_result.is_ok());
        let expected_stream: TokenStream = expected_stream_result.unwrap();

        println!("------------------------");
        println!("Expected stream (string):");
        println!("'{}'", expected_stream);
        println!("------------------------");

        assert_eq!(
            compiled_stream.to_string().replace(" ", ""),
            expected_stream.to_string().replace(" ", "")
        );
    }

    #[test]
    fn test_empty_input() {
        compiler_compare("", "");
    }

    #[test]
    fn test_var() {
        compiler_compare("var x: T", "");
    }

    #[test]
    fn test_assign_tensor() {
        let input1 = "x: T<i64> = [1]<i64>";
        let input2 = "x = [1]<i64>";
        let expected = "let x = ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None);";
        compiler_compare(input1, expected);
        compiler_compare(input2, expected);
    }

    #[test]
    fn test_assign_scalar() {
        let input1 = "x: S = 1";
        let input2 = "x = 1";
        let expected = "let x = ijzer::tensor::Tensor::<i64>::scalar(1);";
        compiler_compare(input1, expected);
        compiler_compare(input2, expected);
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
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
        compiler_compare(input3, expexted);
    }

    #[test]
    fn test_simple_function() {
        let input1 = "f($x) -> T = + [1] $x";
        let input2 = "f($x) = + [1] $x";
        let expexted = "let f = { | _x: ijzer::tensor::Tensor::<i64> | ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&_x, |a: i64, b: i64| a + b).unwrap() } ;";
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
    }

    #[test]
    fn test_function_apply() {
        let input1 = "var x: Fn(S->T); x 1";
        let input2 = "var x: Fn(S->T)
         x 1";
        let expexted = "x (ijzer::tensor::Tensor::<i64>::scalar(1))";
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
    }

    #[test]
    fn test_add() {
        compiler_compare("+ [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a + b).unwrap()");
        compiler_compare("+ [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a + b).unwrap()");
        compiler_compare("+ 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a + b).unwrap()");
        compiler_compare("+ 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a + b).unwrap()");
    }
    #[test]
    fn test_subtract() {
        compiler_compare("- [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a - b).unwrap()");
        compiler_compare("- [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a - b).unwrap()");
        compiler_compare("- 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a - b).unwrap()");
        compiler_compare("- 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a - b).unwrap()");
    }
    #[test]
    fn test_multiply() {
        compiler_compare("* [1] [2]", "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a * b).unwrap()");
        compiler_compare("* [1] 2","ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a * b).unwrap()");
        compiler_compare("* 1 [2]","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2], None), |a: i64, b: i64| a * b).unwrap()");
        compiler_compare("* 1 2","ijzer::tensor::Tensor::<i64>::scalar(1).apply_binary_op(&ijzer::tensor::Tensor::<i64>::scalar(2), |a: i64, b: i64| a * b).unwrap()");
    }

    #[test]
    fn test_negate() {
        compiler_compare(
            "-[1]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None).map(|a: i64| -a)",
        );
        compiler_compare("var x: S; -x", "x.map(|a: i64| -a)");
    }

    #[test]
    fn test_reduction() {
        compiler_compare(
            "var f: Fn(N,N->N); /f [1,2]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None).reduce(f)",
        );
        compiler_compare(
            "/+ [1,2]",
            "ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None).reduce(|a: i64, b: i64| a + b)",
        );
    }

    #[test]
    fn test_lambda_variable() {
        let expected1 =
            "let f = { | _x: ijzer::tensor::Tensor::<i64>| _x . apply_binary_op (& _x , | a: i64 , b: i64 | a + b) . unwrap () } ;";
        compiler_compare("f($x) = + $x $x", expected1);

        let expected2 =
            "let f = { | _x: ijzer::tensor::Tensor::<i64> , _y: ijzer::tensor::Tensor::<i64> | _x . apply_binary_op (& _y , | a: i64 , b: i64 | a + b) . unwrap () } ;";
        compiler_compare("f($x, $y) = + $x $y", expected2);

        let expected3 = "let h = { | _x: ijzer::tensor::Tensor::<i64> | k (_x , ijzer::tensor::Tensor::<i64>::from_vec(vec![1,2], None)) } ;";
        compiler_compare("var k: Fn(T,T->T); h($x) = k $x [1,2]", expected3);
    }

    #[test]
    fn test_function_composition() {
        let input = "@(-,+) 1 2";
        let expected = "(| _15_1 : ijzer :: tensor :: Tensor :: < i64 > , _15_2 : ijzer :: tensor :: Tensor :: < i64 > | (| x : ijzer :: tensor :: Tensor :: < i64 > | x . map (| a : i64 | - a)) ((| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_15_1 , _15_2))) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected);

        let input = "@(+) 1 2";
        let expected = "(| _7_1 : ijzer :: tensor :: Tensor :: < i64 > , _7_2 : ijzer :: tensor :: Tensor :: < i64 > | (| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_7_1 , _7_2)) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->T); @(f,-) [1]";
        let expected = "(|_11_1:ijzer::tensor::Tensor::<i64>|(f)((|x:ijzer::tensor::Tensor::<i64>|x.map(|a:i64|-a))(_11_1)))(ijzer::tensor::Tensor::<i64>::from_vec(vec![1],None))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_number_type_from_string() {
        let tokens = number_type_from_string("i64").unwrap();
        assert_eq!(tokens.to_string(), (quote! {i64}).to_string());

        let tokens = number_type_from_string("f64").unwrap();
        assert_eq!(tokens.to_string(), (quote! {f64}).to_string());
    }

    #[test]
    fn test_type_conversion() {
        let input = "var x: S; <-T x";
        let expected = "x";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->T); <-Fn(S->T) f 1";
        let expected = "(| _2_1 : ijzer :: tensor :: Tensor :: < i64 > | ((f) (_2_1))) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1))";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->N); <-Fn(S->T) f 1";
        let expected = "(|_2_1:ijzer::tensor::Tensor::<i64>|(ijzer::tensor::Tensor::<i64>::scalar((f)(_2_1))))(ijzer::tensor::Tensor::<i64>::scalar(1))";
        compiler_compare(input, expected);

        let input = "var f: Fn(S,S->S); /<-Fn(N,N->N) f [1]";
        let expected = "ijzer :: tensor :: Tensor :: < i64 > :: from_vec (vec ! [1] , None) . reduce ((| _2_1 : i64 , _2_2 : i64 | ((f) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (_2_1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (_2_2)) . extract_scalar () . unwrap ())))";
        compiler_compare(input, expected);

        let input = "var x: N; <-S x";
        let expected = "ijzer::tensor::Tensor::<i64>::scalar(x)";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_as_function() {
        let input = "var f: Fn(T->T); ~f";
        let expected = "f";
        compiler_compare(input, expected);

        let input = "~+: Fn(S,S->S)";
        let expected = "| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_type_conversion_with_as_function() {
        let input = "~(<-Fn(S,S->T) +)";
        let expected = "(| _5_1 : ijzer :: tensor :: Tensor :: < i64 > , _5_2 : ijzer :: tensor :: Tensor :: < i64 > | ((| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_5_1 , _5_2)))";
        compiler_compare(input, expected);
    }
    #[test]
    fn test_function_composition_functional() {
        let input = "~@(-,+):Fn(S,S->S)";
        let expected = "| _13_1 : ijzer :: tensor :: Tensor :: < i64 > , _13_2 : ijzer :: tensor :: Tensor :: < i64 > | (| x : ijzer :: tensor :: Tensor :: < i64 > | x . map (| a : i64 | - a)) ((| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_13_1 , _13_2))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_reduction_in_composition() {
        let input = "~@(/+,+):Fn(T,T->S)";
        let expected = "| _7_1 : ijzer :: tensor :: Tensor :: < i64 > , _7_2 : ijzer :: tensor :: Tensor :: < i64 > | (| _1 : ijzer :: tensor :: Tensor :: < i64 > | _1 . reduce (| a : i64 , b : i64 | a + b)) ((| x1 : ijzer :: tensor :: Tensor :: < i64 > , x2 : ijzer :: tensor :: Tensor :: < i64 > | x1 . apply_binary_op (& x2 , | a : i64 , b : i64 | a + b) . unwrap ()) (_7_1 , _7_2))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_apply() -> Result<()> {
        let input = "var f: Fn(S,S->S); .~f 1 2";
        let expected = "f (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected);

        let input = "var f: Fn(S->Fn(S->S)); .(f 1) 2";
        let expected = "f (ijzer :: tensor :: Tensor :: < i64 > :: scalar (1)) (ijzer :: tensor :: Tensor :: < i64 > :: scalar (2))";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_lambda_variable_functional() -> Result<()> {
        let input = "g($x:Fn(N,N->N)) -> S = /$x [1]";
        let expected = "let g = { | _x : fn (i64 , i64) -> i64 | ijzer :: tensor :: Tensor :: < i64 > :: from_vec (vec ! [1] , None) . reduce (_x) } ;";
        compiler_compare(input, expected);

        let input = "g($x:Fn(S->T), $y:Fn(T->S)) -> Fn(T->T) = ~@($x,$y)";
        let expected = "let g = { | _x : fn (ijzer :: tensor :: Tensor :: < i64 >) -> ijzer :: tensor :: Tensor :: < i64 > , _y : fn (ijzer :: tensor :: Tensor :: < i64 >) -> ijzer :: tensor :: Tensor :: < i64 > | | _4_1 : ijzer :: tensor :: Tensor :: < i64 > | (_x) ((_y) (_4_1)) } ;";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_generalized_contraction() -> Result<()> {
        let input = "?/+* [1] [2]";
        let expected = "ijzer::tensor::Tensor::<i64>::from_vec(vec![1],None).generalized_contraction(&ijzer::tensor::Tensor::<i64>::from_vec(vec![2],None),(|z:&ijzer::tensor::Tensor::<i64>|(|_1:ijzer::tensor::Tensor::<i64>|_1.reduce(|a:i64,b:i64|a+b))(z.clone()).extract_scalar().unwrap()),|a:i64,b:i64|a*b).unwrap()";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functional() -> Result<()> {
        let input = "~?/+*";
        let expected = "| x : ijzer :: tensor :: Tensor :: < i64 > , y : ijzer :: tensor :: Tensor :: < i64 > | x . generalized_contraction (& y , (| z : & ijzer :: tensor :: Tensor :: < i64 > | (| _1 : ijzer :: tensor :: Tensor :: < i64 > | _1 . reduce (| a : i64 , b : i64 | a + b)) (z . clone ()) . extract_scalar () . unwrap ()) , | a : i64 , b : i64 | a * b) . unwrap ()";
        compiler_compare(input, expected);

        Ok(())
    }
}
