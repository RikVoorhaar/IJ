use super::{next_node, next_node_functional};
use crate::{
    ast_node::{ASTContext, Node, TokenSlice},
    operations::Operation,
    parser::utils::gather_operands,
    parser::{ParseNode, ParseNodeFunctional},
    syntax_error::SyntaxError,
    tokens::Token,
    types::{FunctionSignature, IJType},
};
use anyhow::Result;
use itertools::{iproduct, Itertools};
use std::rc::Rc;

/// List all possible types a type can be converted to
fn list_conversions_to(from: &IJType) -> Vec<IJType> {
    match from {
        IJType::Number(None) | IJType::Scalar(None) => vec![IJType::Number(None), IJType::Scalar(None), IJType::Tensor(None)],
        IJType::Function(signature) => {
            let input_conversions: Vec<Vec<IJType>> = signature
                .input
                .clone()
                .iter()
                .map(list_conversions_from) // from because of contravariance
                .multi_cartesian_product()
                .collect();
            let output_conversion: Vec<IJType> = list_conversions_to(&signature.output);

            let all_conversions: Vec<IJType> = iproduct!(input_conversions, output_conversion)
                .map(|(input_conversion, output_conversion)| {
                    IJType::Function(FunctionSignature::new(input_conversion, output_conversion))
                })
                .collect();
            all_conversions
        }
        IJType::Tensor(None) => vec![IJType::Tensor(None)],
        _ => vec![],
    }
}

/// list all possible conversions a type could have been converted from
fn list_conversions_from(to: &IJType) -> Vec<IJType> {
    match to {
        IJType::Number(None) | IJType::Scalar(None) => vec![IJType::Number(None), IJType::Scalar(None)],
        IJType::Tensor(None) => vec![IJType::Tensor(None), IJType::Number(None), IJType::Scalar(None)],
        IJType::Function(signature) => {
            let input_conversions: Vec<Vec<IJType>> = signature
                .input
                .clone()
                .iter()
                .map(list_conversions_to) // to because of contravariance
                .multi_cartesian_product()
                .collect();
            let output_conversions: Vec<IJType> = list_conversions_from(&signature.output);

            let all_conversions: Vec<IJType> = iproduct!(input_conversions, output_conversions)
                .map(|(input_conversion, output_conversion)| {
                    IJType::Function(FunctionSignature::new(input_conversion, output_conversion))
                })
                .collect();
            all_conversions
        }
        _ => vec![],
    }
}

fn conversion_possible(from: &IJType, to: &IJType) -> bool {
    list_conversions_from(to).contains(from)
}

/// given the token slice and the desired type, find a function matching the desired type.
fn type_conversion_functional_part(
    slice: TokenSlice,
    context: &mut ASTContext,
    desired_signature: FunctionSignature,
) -> Result<(Rc<Node>, TokenSlice)> {
    let needed_outputs = list_conversions_from(&desired_signature.output);
    let (function_candidates, rest) =
        next_node_functional(slice, context, Some(needed_outputs.as_slice()))?;

    let desired_type = IJType::Function(desired_signature);
    match function_candidates
        .into_iter()
        .find(|candidate| conversion_possible(&candidate.output_type, &desired_type))
    {
        Some(function_node) => {
            let conversion_node = Node::new(
                Operation::TypeConversion,
                vec![function_node.output_type.clone()],
                desired_type,
                vec![function_node],
                context,
            )?;
            Ok((Rc::new(conversion_node), rest))
        }
        None => Err(SyntaxError::TypeConversionNotPossible(
            desired_type.to_string(),
            context.token_slice_to_string(rest),
        )
        .into()),
    }
}

pub struct TypeConversion;
impl ParseNode for TypeConversion {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        // get the type from the slice
        let (desired_type, type_end) = IJType::parse_tokens(&context.get_tokens_from_slice(slice))?;
        let rest = slice.move_start(type_end)?;

        match desired_type {
            IJType::Scalar(None) | IJType::Tensor(None) => {
                let (operand, rest) = next_node(rest, context)?;
                let possible_conversions = list_conversions_from(&desired_type);
                let current_type = operand.output_type.clone();
                if !possible_conversions.contains(&current_type) {
                    return Err(SyntaxError::TypeConversionNotPossible(
                        current_type.to_string(),
                        desired_type.clone().to_string(),
                    )
                    .into());
                }

                let conversion_node = Node::new(
                    Operation::TypeConversion,
                    vec![current_type],
                    desired_type,
                    vec![operand],
                    context,
                )?;
                Ok((Rc::new(conversion_node), rest))
            }
            IJType::Function(ref signature) => {
                let (convered_function, rest) =
                    type_conversion_functional_part(rest, context, signature.clone())?;
                let input_types = signature.input.clone();
                let (operands, rest) = gather_operands(vec![input_types.clone()], rest, context)?;
                Ok((
                    Rc::new(Node::new(
                        Operation::Apply,
                        vec![convered_function.output_type.clone()]
                            .into_iter()
                            .chain(input_types)
                            .collect(),
                        *signature.output.clone(),
                        vec![convered_function]
                            .into_iter()
                            .chain(operands)
                            .collect(),
                        context,
                    )?),
                    rest,
                ))
            }
            _ => Err(SyntaxError::TypeConversionNotPossible(
                context.token_slice_to_string(rest),
                desired_type.to_string(),
            )
            .into()),
        }
    }
}

impl ParseNodeFunctional for TypeConversion {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let slice = slice.move_start(1)?;
        // get the type from the slice
        let (desired_type, type_end) = IJType::parse_tokens(&context.get_tokens_from_slice(slice))?;
        let rest = slice.move_start(type_end)?;

        match desired_type {
            IJType::Function(ref signature) => {
                if let Some(outputs) = needed_outputs {
                    if !outputs.contains(&signature.output) {
                        return Err(SyntaxError::TypeConversionNotPossible(
                            desired_type.to_string(),
                            context.token_slice_to_string(slice),
                        )
                        .into());
                    }
                }
                let (converted_function, rest) =
                    type_conversion_functional_part(rest, context, signature.clone())?;
                Ok((vec![converted_function], rest))
            }
            _ => Err(SyntaxError::ExpectedFunction(context.token_slice_to_string(rest)).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_str, parse_str_no_context};
    use crate::tokens::lexer;

    #[test]
    fn test_list_conversions_to() {
        let types = list_conversions_to(&IJType::Number(None));
        assert_eq!(types.len(), 3);

        let types = list_conversions_to(&IJType::Function(FunctionSignature::new(
            vec![IJType::Number(None)],
            IJType::Scalar(None),
        )));
        assert_eq!(types.len(), 6);

        let types = list_conversions_to(&IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor(None)],
            IJType::Function(FunctionSignature::new(vec![IJType::Number(None)], IJType::Scalar(None))),
        )));
        assert_eq!(types.len(), 18);
    }

    #[test]
    fn test_list_conversions_from() {
        let types = list_conversions_from(&IJType::Number(None));
        assert_eq!(types.len(), 2);

        let types = list_conversions_from(&IJType::Function(FunctionSignature::new(
            vec![IJType::Number(None)],
            IJType::Scalar(None),
        )));
        assert_eq!(types.len(), 6);

        let types = list_conversions_from(&IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor(None)],
            IJType::Function(FunctionSignature::new(vec![IJType::Number(None)], IJType::Scalar(None))),
        )));
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_type_conversion_functional_part() {
        let tokens = lexer("+ -").unwrap();
        let mut context = ASTContext::from_tokens(tokens);
        let slice = context.full_slice();
        let desired_signature =
            FunctionSignature::new(vec![IJType::Number(None), IJType::Number(None)], IJType::Scalar(None));
        let maybe_nodes = type_conversion_functional_part(slice, &mut context, desired_signature);
        assert!(maybe_nodes.is_ok());
        let (node, rest) = maybe_nodes.unwrap();
        println!("{:?}", node);
        assert_eq!(node.op, Operation::TypeConversion);
        assert_eq!(rest.len(), 1);

        let tokens = lexer("*").unwrap();
        let mut context = ASTContext::from_tokens(tokens);
        let slice = context.full_slice();
        let desired_signature =
            FunctionSignature::new(vec![IJType::Tensor(None), IJType::Tensor(None)], IJType::Scalar(None));
        let maybe_nodes = type_conversion_functional_part(slice, &mut context, desired_signature);
        assert!(maybe_nodes.is_err());
    }

    #[test]
    fn test_type_conversion_node_simple() {
        let result = parse_str_no_context("<-T 1.0");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::TypeConversion);

        let result = parse_str_no_context("<-T [1.0]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::TypeConversion);

        let result = parse_str_no_context("<-S [1.0]");
        assert!(result.is_err());
    }

    #[test]
    fn test_type_conversion_node_function1() {
        let result = parse_str_no_context("<-Fn(S,S->T) + 1 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Apply);
    }

    #[test]
    fn test_type_conversion_node_function2() {
        let mut context = ASTContext::new();
        let result = parse_str("var f: Fn(S->S)", &mut context);
        assert!(result.is_ok());
        let maybe_node = parse_str("<-Fn(S->S) f 1", &mut context);
        assert!(maybe_node.is_ok());
        let node = maybe_node.unwrap();
        assert_eq!(node.op, Operation::Apply);
    }

    #[test]
    fn test_type_conversion_node_function3() {
        let mut context = ASTContext::new();
        let result = parse_str("var f: Fn(T->T)", &mut context);
        assert!(result.is_ok());
        let maybe_node = parse_str("<-Fn(S->T) f 1", &mut context);
        assert!(maybe_node.is_ok());
        let node = maybe_node.unwrap();
        assert_eq!(node.op, Operation::Apply);
        println!("{:?}", node);
    }

    #[test]
    fn test_type_conversion_node_function4() {
        let mut context = ASTContext::new();
        let result = parse_str("var f: Fn(T->T)", &mut context);
        assert!(result.is_ok());
        let maybe_node = parse_str("<-Fn(T->S) f [1]", &mut context);
        assert!(maybe_node.is_err());
    }

    #[test]
    fn test_type_conversion_functional() {
        let mut context = ASTContext::new();
        let result = parse_str("var f: Fn(S,S->S)", &mut context);
        assert!(result.is_ok());
        let maybe_node = parse_str("/<-Fn(N,N->N) f [1]", &mut context);
        assert!(maybe_node.is_ok());
        let node = maybe_node.unwrap();
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Reduce);
    }
}
