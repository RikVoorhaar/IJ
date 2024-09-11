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
        IJType::Group(types) => {
            let group_types = types
                .iter()
                .map(annotation_from_type)
                .collect::<Result<Vec<_>>>()?;
            quote!((#(#group_types),*))
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
            Operation::Array => Array::compile(node, self, child_streams)?,
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
            }
            Operation::TensorBuilder(_) => TensorBuilder::compile(node, self, child_streams)?,
            Operation::Transpose => Transpose::compile(node, self, child_streams)?,
            Operation::Shape => Shape::compile(node, self, child_streams)?,
            Operation::QR => QR::compile(node, self, child_streams)?,
            Operation::Svd => Svd::compile(node, self, child_streams)?,
            Operation::Solve => Solve::compile(node, self, child_streams)?,
            Operation::Diag => Diag::compile(node, self, child_streams)?,
            Operation::Index => Index::compile(node, self, child_streams)?,
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
            let number_type = node.output_type.extract_number_type().unwrap_or_default();
            let res = match number_type {
                Some(_) => {
                    let t = annotation_from_type(&IJType::Number(number_type))?;
                    quote! {
                        #parsed_val as #t
                    }
                }
                None => quote! {
                    #parsed_val
                },
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
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::Array = &node.op {
        } else {
            panic!("Expected array node, found {:?}", node);
        }
        let child_ids = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();
        let child_streams = child_ids
            .iter()
            .map(|id| child_streams[id].clone())
            .collect::<Vec<_>>();

        let is_all_tensors = node
            .operands
            .iter()
            .all(|n| n.output_type.type_match(&IJType::Tensor(None)));
        let is_all_scalars = node
            .operands
            .iter()
            .all(|n| n.output_type.type_match(&IJType::Number(None)));

        let number_type = node.output_type.extract_number_type().unwrap_or_default();
        let tensor_t = annotation_from_type(&IJType::Tensor(number_type))?;
        if is_all_scalars {
            let extracted_streams = child_streams
                .into_iter()
                .map(|s| {
                    quote! {#s}
                })
                .collect::<Vec<_>>();
            let res = quote! {
                #tensor_t::from_vec(vec![#(#extracted_streams),*], None)
            };
            Ok(res)
        } else if is_all_tensors {
            let res = quote! {
                #tensor_t::from_tensors(&[#(#child_streams),*]).unwrap()
            };
            Ok(res)
        } else {
            panic!("Array elements must be all tensors or all scalars");
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
        let lhs_annot = annotation_from_type(&node.operands[0].output_type)?;
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
                let #lhs_stream: #lhs_annot = #rhs_stream;
            ))
        } else {
            Ok(quote!(
                let #lhs_stream = {|#(#args_streams),*| #rhs_stream};
            ))
        }
    }
}

fn _compile_binary_op(
    node: Rc<Node>,
    child_streams: HashMap<usize, TokenStream>,
    binary_op: impl Fn(TokenStream, TokenStream) -> TokenStream,
) -> Result<TokenStream> {
    let children = node.operands.iter().map(|n| n.id).collect::<Vec<_>>();

    let res = match node.output_type.clone() {
        IJType::Tensor(number_type) | IJType::Number(number_type) => {
            if children.len() != 2 {
                panic!("Expected 2 children for add, found {:?}", children.len());
            }
            let childstream1 = child_streams.get(&children[0]).unwrap();
            let childstream2 = child_streams.get(&children[1]).unwrap();
            let number_annot = annotation_from_type(&IJType::Number(number_type))?;
            let operand1_type = node.operands[0].output_type.clone();
            let operand2_type = node.operands[1].output_type.clone();

            match (operand1_type, operand2_type) {
                (IJType::Tensor(_), IJType::Tensor(_)) => {
                    let binop_stream = binary_op(quote! {a}, quote! {b});
                    quote! {
                        #childstream1.apply_binary_op(&#childstream2, |a: #number_annot, b: #number_annot| #binop_stream).unwrap()
                    }
                }
                (IJType::Tensor(_), IJType::Number(_)) => {
                    let binop_stream = binary_op(quote! {x}, quote! {#childstream2});
                    quote! {
                        #childstream1.map(|x: #number_annot| #binop_stream)
                    }
                }
                (IJType::Number(_), IJType::Tensor(_)) => {
                    let binop_stream = binary_op(quote! {#childstream1}, quote! {x});
                    quote! {
                        #childstream2.map(|x: #number_annot| #binop_stream)
                    }
                }
                (IJType::Number(_), IJType::Number(_)) => {
                    binary_op(quote! {#childstream1}, quote! {#childstream2})
                }
                _ => unreachable!(),
            }
        }
        IJType::Function(f) => {
            let number_type = f.output.extract_number_type().unwrap_or_default();
            let number_annot = annotation_from_type(&IJType::Number(number_type.clone()))?;
            let tensor_annot = annotation_from_type(&IJType::Tensor(number_type))?;
            let input_types = f.input.clone();
            if input_types.len() != 2 {
                panic!(
                    "Expected 2 input types for add, found {:?}",
                    input_types.len()
                );
            }
            let (input1_type, input2_type) = (input_types[0].clone(), input_types[1].clone());
            match (input1_type, input2_type) {
                (IJType::Number(_), IJType::Number(_)) => {
                    let binop_stream = binary_op(quote! {a}, quote! {b});
                    quote! { |a: #number_annot, b: #number_annot| #binop_stream }
                }
                (IJType::Tensor(_), IJType::Tensor(_)) => {
                    let binop_stream = binary_op(quote! {a}, quote! {b});
                    quote! {
                        |x1: #tensor_annot, x2: #tensor_annot|
                        x1.apply_binary_op(&x2, |a: #number_annot, b: #number_annot| #binop_stream).unwrap()
                    }
                }
                (IJType::Tensor(_), IJType::Number(_)) => {
                    let binop_stream = binary_op(quote! {a}, quote! {y});
                    quote! {
                        |x: #tensor_annot, y: #number_annot|
                        x.map(|a: #number_annot| #binop_stream)
                    }
                }
                (IJType::Number(_), IJType::Tensor(_)) => {
                    let binop_stream = binary_op(quote! {y}, quote! {a});
                    quote! {
                        |y: #number_annot, x: #tensor_annot|
                        x.map(|a: #number_annot| #binop_stream)
                    }
                }
                _ => panic!(
                    "Found add node with unimplemented input types: {:?}",
                    input_types
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

struct Add;
impl CompileNode for Add {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let binary_op = |a: TokenStream, b: TokenStream| quote! {#a + #b};
        _compile_binary_op(node, child_streams, binary_op)
    }
}
struct Subtract;
impl CompileNode for Subtract {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let binary_op = |a: TokenStream, b: TokenStream| quote! {#a - #b};
        _compile_binary_op(node, child_streams, binary_op)
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
                IJType::Tensor(number_type) => {
                    let number_annot = annotation_from_type(&IJType::Number(number_type))?;
                    let childstream = child_streams.get(&children[0]).unwrap();
                    quote! {
                        #childstream.map(|a: #number_annot| -a)
                    }
                }
                IJType::Number(_) => {
                    let childstream = child_streams.get(&children[0]).unwrap();
                    quote! {
                        -#childstream
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
                        IJType::Tensor(_) => {
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
        let binary_op = |a: TokenStream, b: TokenStream| quote! {#a * #b};
        _compile_binary_op(node, child_streams, binary_op)
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
            (#function_stream)(#(#operand_streams),*)
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
            (IJType::Number(_), IJType::Number(_)) => quote! {#child_stream},
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
        let f_arg_annot =
            annotation_from_type(&f_node.output_type.extract_signature().unwrap().input[0])?;
        let f_stream_extract = quote! {
            (|z: &#f_arg_annot| (#f_stream)(z.clone()))
        };
        let g_stream = child_streams[1].clone();
        let g_node = &node.operands[1];
        let input_number_type = g_node.output_type.extract_signature().unwrap().input[0]
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

struct TensorBuilder;
impl CompileNode for TensorBuilder {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        if let Operation::TensorBuilder(builder_name) = &node.op {
            let builder_name_stream = syn::parse_str::<proc_macro2::TokenStream>(builder_name)
                .map_err(|_| {
                    syn::Error::new_spanned(
                        builder_name,
                        "Failed to parse builder name into a Rust TokenStream",
                    )
                })?;

            match node.operands.len() {
                1 => {
                    let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                    let number_type = node.output_type.extract_number_type().unwrap();
                    let tensor_t = annotation_from_type(&IJType::Tensor(number_type))?;
                    Ok(quote! {
                        #tensor_t::#builder_name_stream(#child_stream.to_vec().as_slice())
                    })
                }
                0 => {
                    let tensor_type = node.output_type.extract_signature().unwrap().output;
                    let tensor_t = annotation_from_type(&tensor_type)?;
                    Ok(quote! {
                        |_x: #tensor_t| #tensor_t::#builder_name_stream(_x.to_vec().as_slice())
                    })
                }

                _ => {
                    panic!(
                        "Expected 0 or 1 operand for TensorBuilder, found {}",
                        node.operands.len()
                    );
                }
            }
        } else {
            unreachable!()
        }
    }
}

struct Transpose;
impl CompileNode for Transpose {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            1 => {
                let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                Ok(quote! {
                    #child_stream.transpose()
                })
            }
            0 => {
                let tensor_type = node.output_type.extract_signature().unwrap().output;
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    |_x: #tensor_t| _x.transpose()
                })
            }
            _ => {
                panic!(
                    "Expected 1 operand for Transpose, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

struct Shape;
impl CompileNode for Shape {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            1 => {
                let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                Ok(quote! {
                    ijzer::tensor::Tensor::<usize>::from_vec(#child_stream.shape().to_vec(), None)
                })
            }
            0 => {
                let tensor_type = node.output_type.extract_signature().unwrap().input[0].clone();
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    |_x: #tensor_t| ijzer::tensor::Tensor::<usize>::from_vec(_x.shape().to_vec(), None)
                })
            }
            _ => {
                panic!(
                    "Expected 1 operand for Shape, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

struct QR;
impl CompileNode for QR {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            1 => {
                let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                Ok(quote! {
                    #child_stream.qr().unwrap()
                })
            }
            0 => {
                let tensor_type = node.output_type.extract_signature().unwrap().input[0].clone();
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    |_x: #tensor_t| _x.qr().unwrap()
                })
            }
            _ => {
                panic!(
                    "Expected 0 or 1 operand for QR, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

struct Svd;
impl CompileNode for Svd {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            1 => {
                let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                Ok(quote! {
                    #child_stream.svd().unwrap()
                })
            }
            0 => {
                let tensor_type = node.output_type.extract_signature().unwrap().input[0].clone();
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    |_x: #tensor_t| _x.svd().unwrap()
                })
            }
            _ => {
                panic!(
                    "Expected 0 or 1 operand for SVD, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

struct Solve;
impl CompileNode for Solve {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            2 => {
                let lhs_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                let rhs_stream = child_streams.get(&node.operands[1].id).unwrap().clone();
                Ok(quote! {
                    #lhs_stream.solve(&#rhs_stream).unwrap()
                })
            }
            0 => {
                let tensor_type0 = node.output_type.extract_signature().unwrap().input[0].clone();
                let tensor_type1 = node.output_type.extract_signature().unwrap().input[1].clone();
                let tensor_t0 = annotation_from_type(&tensor_type0)?;
                let tensor_t1 = annotation_from_type(&tensor_type1)?;
                Ok(quote! {
                    |_x: #tensor_t0, _y: #tensor_t1| _x.solve(&_y).unwrap()
                })
            }
            _ => {
                panic!(
                    "Expected 0 or 2 operands for Solve, found {}",
                    node.operands.len()
                );
            }
        }
    }
}

struct Diag;
impl CompileNode for Diag {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        match node.operands.len() {
            1 => {
                let child_stream = child_streams.get(&node.operands[0].id).unwrap().clone();
                let tensor_type = node.output_type.clone();
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    #tensor_t::diag(&#child_stream)
                })
            }
            0 => {
                let tensor_type = node.output_type.extract_signature().unwrap().input[0].clone();
                let tensor_t = annotation_from_type(&tensor_type)?;
                Ok(quote! {
                    |_x: #tensor_t| #tensor_t::diag(&_x)
                })
            }
            _ => {
                panic!("Expected 1 operand for Diag, found {}", node.operands.len());
            }
        }
    }
}

struct Index;
impl CompileNode for Index {
    fn compile(
        node: Rc<Node>,
        _compiler: &mut CompilerContext,
        child_streams: HashMap<usize, TokenStream>,
    ) -> Result<TokenStream> {
        let tensor_operand = node.operands[0].clone();
        let tensor_stream = child_streams.get(&tensor_operand.id).unwrap().clone();

        let index_operands = &node.operands[1..];
        let all_numbers = index_operands.iter().all(|n| {
            n.output_type
                .type_match(&IJType::Number(Some("usize".to_string())))
        });
        let contains_colon = index_operands
            .iter()
            .any(|n| n.output_type.type_match(&IJType::Void));

        match (all_numbers, contains_colon) {
            (true, _) => {
                let index_operands_stream = index_operands
                    .iter()
                    .map(|n| child_streams.get(&n.id).unwrap().clone())
                    .collect::<Vec<TokenStream>>();
                Ok(quote! {
                    #tensor_stream[&vec![#(#index_operands_stream),*]].clone()
                })
            }
            (false, true) => {
                let mut index_operands_streams = vec![];
                for index_operand in index_operands {
                    match index_operand.output_type {
                        IJType::Number(_) => {
                            let stream = child_streams.get(&index_operand.id).unwrap().clone();
                            index_operands_streams.push(quote! {Some(#stream)});
                        }
                        IJType::Void => index_operands_streams.push(quote! {None}),
                        _ => unreachable!(),
                    }
                }
                Ok(quote! {
                    #tensor_stream.sub_tensor(vec![#(#index_operands_streams),*]).unwrap()
                })
            }
            (false, false) => {
                let index_operands_streams = index_operands
                    .iter()
                    .map(|n| child_streams.get(&n.id).unwrap().clone())
                    .collect::<Vec<TokenStream>>();
                Ok(quote! {
                    #tensor_stream.multi_index(vec![#(#index_operands_streams),*]).unwrap()
                })
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
        let input1 = "x: T<_> = [1]<_>";
        let input2 = "x = [1]<_>";
        let expected = "let x : ijzer :: tensor :: Tensor :: < _ > = ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1 ] , None) ;";
        compiler_compare(input1, expected);
        compiler_compare(input2, expected);
    }

    #[test]
    fn test_assign_scalar() {
        let input1 = "x: N = 1";
        let input2 = "x = 1";
        let expected = "let x: _ = 1;";
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
        let expexted = "let x : ijzer :: tensor :: Tensor :: < _ > = ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) ; let y : ijzer :: tensor :: Tensor :: < _ > = x . apply_binary_op (& x , | a : _ , b : _ | a + b) . unwrap () ;";
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
        compiler_compare(input3, expexted);
    }

    #[test]
    fn test_simple_function() {
        let input1 = "var y: T; f($x) -> T = + y $x";
        let input2 = "var y: T; f($x) = + y $x";
        let expexted = "let f = { | _x: ijzer::tensor::Tensor::<_> | y.apply_binary_op(&_x, |a: _, b: _| a + b).unwrap() } ;";
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
    }

    #[test]
    fn test_function_apply() {
        let input1 = "var x: Fn(N->T); x 1";
        let input2 = "var x: Fn(N->T)
         x 1";
        let expexted = "x (1)";
        compiler_compare(input1, expexted);
        compiler_compare(input2, expexted);
    }

    #[test]
    fn test_add() {
        compiler_compare("+ [1] [2]", "ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) . apply_binary_op (& ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [2] , None) , | a : _ , b : _ | a + b) . unwrap ()");
        compiler_compare("+ [1] 2","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) . map (| x : _ | x + 2)");
        compiler_compare("+ 1 [2]","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [2] , None) . map (| x : _ | 1 + x)");
        compiler_compare("+ 1 2", "1+2");
    }
    #[test]
    fn test_subtract() {
        compiler_compare("- [1] [2]", "ijzer::tensor::Tensor::<_>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<_>::from_vec(vec![2], None), |a: _, b: _| a - b).unwrap()");
        compiler_compare("- [1] 2","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) . map (| x : _ | x - 2)");
        compiler_compare("- 1 [2]","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [2] , None) . map (| x : _ | 1-x)");
        compiler_compare("- 1 2", "1-2");
    }
    #[test]
    fn test_multiply() {
        compiler_compare("* [1] [2]", "ijzer::tensor::Tensor::<_>::from_vec(vec![1], None).apply_binary_op(&ijzer::tensor::Tensor::<_>::from_vec(vec![2], None), |a: _, b: _| a * b).unwrap()");
        compiler_compare("* [1] 2","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) . map (| x : _ | x * 2)");
        compiler_compare("* 1 [2]","ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [2] , None) . map (| x : _ | 1 * x)");
        compiler_compare("* 1 2", "1*2");
    }

    #[test]
    fn test_negate() {
        compiler_compare(
            "-[1]",
            "ijzer::tensor::Tensor::<_>::from_vec(vec![1], None).map(|a: _| -a)",
        );
        compiler_compare("var x: T; -x", "x.map(|a: _| -a)");
        compiler_compare("var x: N<f64>; -x", "-x");
    }

    #[test]
    fn test_reduction() {
        compiler_compare(
            "var f: Fn(N,N->N); /f [1,2]",
            "ijzer::tensor::Tensor::<_>::from_vec(vec![1,2], None).reduce(f)",
        );
        compiler_compare(
            "/+ [1,2]",
            "ijzer::tensor::Tensor::<_>::from_vec(vec![1,2], None).reduce(|a: _, b: _| a + b)",
        );

        compiler_compare(
            "/+ [1,2]<f64>",
            "ijzer::tensor::Tensor::<f64>::from_vec(vec![1,2], None).reduce(|a: _, b: _| a + b)",
        );
    }

    #[test]
    fn test_lambda_variable() {
        let expected1 =
            "let f = { | _x: ijzer::tensor::Tensor::<_>| _x . apply_binary_op (& _x , | a: _ , b: _ | a + b) . unwrap () } ;";
        compiler_compare("f($x) = + $x $x", expected1);

        let expected2 =
            "let f = { | _x: ijzer::tensor::Tensor::<_> , _y: ijzer::tensor::Tensor::<_> | _x . apply_binary_op (& _y , | a: _ , b: _ | a + b) . unwrap () } ;";
        compiler_compare("f($x, $y) = + $x $y", expected2);

        let expected3 = "let h = { | _x: ijzer::tensor::Tensor::<_> | k (_x , ijzer::tensor::Tensor::<_>::from_vec(vec![1,2], None)) } ;";
        compiler_compare("var k: Fn(T,T->T); h($x) = k $x [1,2]", expected3);
    }

    #[test]
    fn test_function_composition() {
        let input = "@(-,+) 1 2";
        let expected = "(| _12_1 : _ , _12_2 : _ | (| a : _ | - a) ((| a : _ , b : _ | a + b) (_12_1 , _12_2))) (1 , 2)";
        compiler_compare(input, expected);

        let input = "@(+) 1 2";
        let expected = "(| _6_1 : _ , _6_2 : _ | (| a : _ , b : _ | a + b) (_6_1 , _6_2)) (1 , 2)";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->T); @(f,-) [1]";
        let expected = "(| _10_1 : ijzer :: tensor :: Tensor :: < _ > | (f) ((| x : ijzer :: tensor :: Tensor :: < _ > | x . map (| a : _ | - a)) (_10_1))) (ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_number_type_from_string() {
        let tokens = number_type_from_string("_").unwrap();
        assert_eq!(tokens.to_string(), (quote! {_}).to_string());

        let tokens = number_type_from_string("f64").unwrap();
        assert_eq!(tokens.to_string(), (quote! {f64}).to_string());
    }

    #[test]
    fn test_type_conversion() {
        let input = "var x: N; <-T x";
        let expected = "ijzer :: tensor :: Tensor :: < _ > :: scalar (x)";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->T); <-Fn(N->T) f 1";
        let expected =
            "((| _2_1 : _ | ((f) (ijzer :: tensor :: Tensor :: < _ > :: scalar (_2_1))))) (1)";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->N); <-Fn(N->T) f 1";
        let expected = "((| _2_1 : _ | (ijzer :: tensor :: Tensor :: < _ > :: scalar ((f) (ijzer :: tensor :: Tensor :: < _ > :: scalar (_2_1)))))) (1)";
        compiler_compare(input, expected);

        let input = "var x: N; <-N x";
        let expected = "x";
        compiler_compare(input, expected);

        let input = "var x: N<f64>; <-N<f64> x";
        let expected = "x";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_as_function() {
        let input = "var f: Fn(T->T); ~f";
        let expected = "f";
        compiler_compare(input, expected);

        let input = "var f: Fn(T<f64>->T<f64>); ~f";
        let expected = "f";
        compiler_compare(input, expected);

        let input = "~+: Fn(S,S->S)";
        let expected = "| x1 : ijzer :: tensor :: Tensor :: < _ > , x2 : ijzer :: tensor :: Tensor :: < _ > | x1 . apply_binary_op (& x2 , | a : _ , b : _ | a + b) . unwrap ()";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_type_conversion_with_as_function() {
        let input = "~(<-Fn(S,S->T) +)";
        let expected = "(| _5_1 : ijzer :: tensor :: Tensor :: < _ > , _5_2 : ijzer :: tensor :: Tensor :: < _ > | ((| x1 : ijzer :: tensor :: Tensor :: < _ > , x2 : ijzer :: tensor :: Tensor :: < _ > | x1 . apply_binary_op (& x2 , | a : _ , b : _ | a + b) . unwrap ()) (_5_1 , _5_2)))";
        compiler_compare(input, expected);
    }
    #[test]
    fn test_function_composition_functional() {
        let input = "~@(-,+):Fn(S,S->S)";
        let expected = "| _13_1 : ijzer :: tensor :: Tensor :: < _ > , _13_2 : ijzer :: tensor :: Tensor :: < _ > | (| x : ijzer :: tensor :: Tensor :: < _ > | x . map (| a : _ | - a)) ((| x1 : ijzer :: tensor :: Tensor :: < _ > , x2 : ijzer :: tensor :: Tensor :: < _ > | x1 . apply_binary_op (& x2 , | a : _ , b : _ | a + b) . unwrap ()) (_13_1 , _13_2))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_reduction_in_composition() {
        let input = "~@(/+,+):Fn(T,T->S)";
        let expected = "| _7_1 : ijzer :: tensor :: Tensor :: < _ > , _7_2 : ijzer :: tensor :: Tensor :: < _ > | (| _1 : ijzer :: tensor :: Tensor :: < _ > | _1 . reduce (| a : _ , b : _ | a + b)) ((| x1 : ijzer :: tensor :: Tensor :: < _ > , x2 : ijzer :: tensor :: Tensor :: < _ > | x1 . apply_binary_op (& x2 , | a : _ , b : _ | a + b) . unwrap ()) (_7_1 , _7_2))";
        compiler_compare(input, expected);
    }

    #[test]
    fn test_apply() -> Result<()> {
        let input = "var f: Fn(S,S->S); .~f 1 2";
        let expected = "(f) (ijzer :: tensor :: Tensor :: < _ > :: scalar (1) , ijzer :: tensor :: Tensor :: < _ > :: scalar (2))";
        compiler_compare(input, expected);

        let input = "var f: Fn(S<f64>,S<f64>->S<f64>); .~f 1<f64> 2<f64>";
        let expected = "(f) (ijzer :: tensor :: Tensor :: < f64 > :: scalar (1) , ijzer :: tensor :: Tensor :: < f64 > :: scalar (2))";
        compiler_compare(input, expected);

        let input = "var f: Fn(S->Fn(S->S)); .(f 1) 2";
        let expected = "(f (ijzer :: tensor :: Tensor :: < _ > :: scalar (1))) (ijzer :: tensor :: Tensor :: < _ > :: scalar (2))";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_lambda_variable_functional() -> Result<()> {
        let input = "g($x:Fn(N,N->N)) -> S = /$x [1]";
        let expected = "let g = { | _x : fn (_ , _) -> _ | ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None) . reduce (_x) } ;";
        compiler_compare(input, expected);

        let input = "g($x:Fn(S->T), $y:Fn(T->S)) -> Fn(T->T) = ~@($x,$y)";
        let expected = "let g = { | _x : fn (ijzer :: tensor :: Tensor :: < _ >) -> ijzer :: tensor :: Tensor :: < _ > , _y : fn (ijzer :: tensor :: Tensor :: < _ >) -> ijzer :: tensor :: Tensor :: < _ > | | _4_1 : ijzer :: tensor :: Tensor :: < _ > | (_x) ((_y) (_4_1)) } ;";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_generalized_contraction() -> Result<()> {
        let input = "?/+* [1] [2]";
        let expected = "ijzer::tensor::Tensor::<_>::from_vec(vec![1],None).generalized_contraction(&ijzer::tensor::Tensor::<_>::from_vec(vec![2],None),(|z:&ijzer::tensor::Tensor::<_>|(|_1:ijzer::tensor::Tensor::<_>|_1.reduce(|a:_,b:_|a+b))(z.clone()).extract_scalar().unwrap()),|a:_,b:_|a*b).unwrap()";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functional() -> Result<()> {
        let input = "~?/+*";
        let expected = "| x : ijzer :: tensor :: Tensor :: < _ > , y : ijzer :: tensor :: Tensor :: < _ > | x . generalized_contraction (& y , (| z : & ijzer :: tensor :: Tensor :: < _ > | (| _1 : ijzer :: tensor :: Tensor :: < _ > | _1 . reduce (| a : _ , b : _ | a + b)) (z . clone ()) . extract_scalar () . unwrap ()) , | a : _ , b : _ | a * b) . unwrap ()";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_builder() -> Result<()> {
        let input = "zeros [1,2]";
        let expected = "ijzer::tensor::Tensor::<_>::zeros(ijzer::tensor::Tensor::<_>::from_vec(vec![1,2],None).to_vec().as_slice())";
        compiler_compare(input, expected);

        let input = "ones<f64> [1]";
        let expected = "ijzer::tensor::Tensor::<f64>::ones(ijzer::tensor::Tensor::<_>::from_vec(vec![1],None).to_vec().as_slice())";
        compiler_compare(input, expected);

        let input = "var x: T<usize>; randu<f32> x";
        let expected = "ijzer::tensor::Tensor::<f32>::randu(x.to_vec().as_slice())";
        compiler_compare(input, expected);

        let input = "var x: T<usize>; randn<f32> x";
        let expected = "ijzer::tensor::Tensor::<f32>::randn(x.to_vec().as_slice())";
        compiler_compare(input, expected);

        let input = "eye<f32> [3,3]<usize>";
        let expected = "ijzer::tensor::Tensor::<f32>::eye(ijzer::tensor::Tensor::<usize>::from_vec(vec![3,3],None).to_vec().as_slice())";
        compiler_compare(input, expected);

        let input = "var x: T<usize>; .~eye x";
        let expected = "(|_x: ijzer::tensor::Tensor::<_>| ijzer::tensor::Tensor::<_>::eye(_x.to_vec().as_slice()))(x)";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let input = "var x: T<f64>; |x";
        let expected = "x.transpose()";
        compiler_compare(input, expected);

        let input = "var x: T<f64>; .~|x";
        let expected = "(|_x: ijzer::tensor::Tensor::<_>| _x.transpose())(x)";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_shape() -> Result<()> {
        let input = "var x: T; %x";
        let expected = "ijzer::tensor::Tensor::<usize>::from_vec(x.shape().to_vec(), None)";
        compiler_compare(input, expected);

        let input = "var x: T;.~%x";
        let expected = "(|_x: ijzer::tensor::Tensor::<_>| ijzer::tensor::Tensor::<usize>::from_vec(_x.shape().to_vec(), None))(x)";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_group_assign() -> Result<()> {
        let input = "(x: S<a>, y: S) = (1<a>,2)";
        let expected = "let (x , y) : (ijzer :: tensor :: Tensor :: < a > , ijzer :: tensor :: Tensor :: < _ >) = (ijzer :: tensor :: Tensor :: < a > :: scalar (1) , ijzer :: tensor :: Tensor :: < _ > :: scalar (2)) ;";
        compiler_compare(input, expected);

        let input = "var f: Fn(T->(T,T)); (x: T, y: T) = f [1]";
        let expected = "let (x , y) : (ijzer :: tensor :: Tensor :: < _ > , ijzer :: tensor :: Tensor :: < _ >) = f (ijzer :: tensor :: Tensor :: < _ > :: from_vec (vec ! [1] , None)) ;";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_solve() -> Result<()> {
        let input = r"var x: T; var y: T; \ x y";
        let expected = "x.solve(&y).unwrap()";
        compiler_compare(input, expected);

        let input = r"~\";
        let expected = "| _x: ijzer :: tensor :: Tensor :: < _ > , _y: ijzer :: tensor :: Tensor :: < _ > | _x.solve(&_y).unwrap()";
        compiler_compare(input, expected);

        let input = r"var x: T; var y: T; z = \ x y";
        let expected = "let z: ijzer :: tensor :: Tensor :: < _ > = x.solve(&y).unwrap();";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_qr() -> Result<()> {
        let input = r"var x: T; qr x";
        let expected = "x.qr().unwrap()";
        compiler_compare(input, expected);

        let input = r"~qr ";
        let expected = "| _x: ijzer :: tensor :: Tensor :: < _ > | _x.qr().unwrap()";
        compiler_compare(input, expected);

        let input = r"var x: T; (q,r) = qr x";
        let expected = "let (q , r) : (ijzer :: tensor :: Tensor :: < _ > , ijzer :: tensor :: Tensor :: < _ >) = x.qr () . unwrap () ;";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_svd() -> Result<()> {
        let input = r"var x: T; svd x";
        let expected = "x.svd().unwrap()";
        compiler_compare(input, expected);

        let input = r"~svd";
        let expected = "| _x: ijzer :: tensor :: Tensor :: < _ > | _x.svd().unwrap()";
        compiler_compare(input, expected);

        let input = r"var x: T; (u,s,v) = svd x";
        let expected = "let (u , s , v) : (ijzer :: tensor :: Tensor :: < _ > , ijzer :: tensor :: Tensor :: < _ > , ijzer :: tensor :: Tensor :: < _ >) = x.svd () . unwrap () ;";
        compiler_compare(input, expected);
        Ok(())
    }

    #[test]
    fn test_diag() -> Result<()> {
        let input = r"var x: T; diag x";
        let expected = "ijzer::tensor::Tensor::<_>::diag(&x)";
        compiler_compare(input, expected);

        let input = r"~diag";
        let expected =
            "| _x: ijzer :: tensor :: Tensor :: < _ > | ijzer::tensor::Tensor::<_>::diag(&_x)";
        compiler_compare(input, expected);

        let input = r"var x: T; y = diag x";
        let expected =
            "let y: ijzer :: tensor :: Tensor :: < _ > = ijzer::tensor::Tensor::<_>::diag(&x);";
        compiler_compare(input, expected);

        Ok(())
    }

    #[test]
    fn test_multi_index() -> Result<()> {
        let input = r"var x: T; <| x [1,2]";
        let expected = "ijzer::tensor::Tensor::<_>::scalar(x[&vec![ijzer::tensor::Tensor::<_>::scalar(1).extract_scalar().unwrap(),ijzer::tensor::Tensor::<_>::scalar(2).extract_scalar().unwrap()]].clone())";
        compiler_compare(input, expected);

        let input = r"var x: T; <| x [1,:]";
        let expected = "x.sub_tensor (vec ! [Some (ijzer :: tensor :: Tensor :: < _ > :: scalar (1) . extract_scalar () . unwrap ()) , None]).unwrap()";
        compiler_compare(input, expected);

        let input = r"var x: T; var s1: T; var s2: T; <| x [s1,s2]";
        let expected = "x.multi_index (vec ! [s1 , s2]).unwrap()";
        compiler_compare(input, expected);

        Ok(())
    }
}
