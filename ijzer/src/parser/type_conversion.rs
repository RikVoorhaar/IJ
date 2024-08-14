use super::{next_node, next_node_functional};
use crate::{
    ast_node::{ASTContext, Node, TokenSlice},
    operations::Operation,
    parser::utils::{gather_operands, match_operand_types},
    parser::ParseNode,
    syntax_error::SyntaxError,
    tokens::Token,
    types::{FunctionSignature, IJType},
};
use anyhow::Result;
use itertools::{iproduct, Itertools};
use std::rc::Rc;

/// List all possible types a type can be converted to
fn list_conversions_to(from: IJType) -> Vec<IJType> {
    match from {
        IJType::Number | IJType::Scalar => vec![IJType::Number, IJType::Scalar, IJType::Tensor],
        IJType::Function(signature) => {
            let input_conversions: Vec<Vec<IJType>> = signature
                .input
                .clone()
                .into_iter()
                .map(list_conversions_from) // from because of contravariance
                .multi_cartesian_product()
                .collect();
            let output_conversions: Vec<Vec<IJType>> = signature
                .output
                .clone()
                .into_iter()
                .map(list_conversions_to)
                .multi_cartesian_product()
                .collect();

            let all_conversions: Vec<IJType> = iproduct!(input_conversions, output_conversions)
                .map(|(input_conversion, output_conversion)| {
                    let mut new_signature = signature.clone();
                    new_signature.input = input_conversion.clone();
                    new_signature.output = output_conversion.clone();
                    IJType::Function(new_signature)
                })
                .collect();
            all_conversions
        }
        IJType::Tensor => vec![IJType::Tensor],
        _ => vec![],
    }
}

/// list all possible conversions a type could have been converted from
fn list_conversions_from(to: IJType) -> Vec<IJType> {
    match to {
        IJType::Number | IJType::Scalar => vec![IJType::Number, IJType::Scalar],
        IJType::Tensor => vec![IJType::Tensor, IJType::Number, IJType::Scalar],
        IJType::Function(signature) => {
            let input_conversions: Vec<Vec<IJType>> = signature
                .input
                .clone()
                .into_iter()
                .map(list_conversions_to) // to because of contravariance
                .multi_cartesian_product()
                .collect();
            let output_conversions: Vec<Vec<IJType>> = signature
                .output
                .clone()
                .into_iter()
                .map(list_conversions_from)
                .multi_cartesian_product()
                .collect();

            let all_conversions: Vec<IJType> = iproduct!(input_conversions, output_conversions)
                .map(|(input_conversion, output_conversion)| {
                    let mut new_signature = signature.clone();
                    new_signature.input = input_conversion.clone();
                    new_signature.output = output_conversion.clone();
                    IJType::Function(new_signature)
                })
                .collect();
            all_conversions
        }
        _ => vec![],
    }
}

fn conversion_possible(from: IJType, to: IJType) -> bool {
    list_conversions_from(to).contains(&from)
}

/// given the token slice and the desired type, find a function matching the desired type.
fn type_conversion_functional_part(
    slice: TokenSlice,
    context: &mut ASTContext,
    desired_signature: FunctionSignature,
) -> Result<(Rc<Node>, TokenSlice)> {
    let needed_outputs: Vec<Vec<IJType>> = desired_signature
        .output
        .clone()
        .into_iter()
        .map(list_conversions_from)
        .multi_cartesian_product()
        .collect();
    let (function_candidates, rest) =
        next_node_functional(slice, context, Some(needed_outputs.as_slice()))?;

    let desired_type = IJType::Function(desired_signature);
    match function_candidates
        .into_iter()
        .find(|candidate| conversion_possible(candidate.output_type.clone(), desired_type.clone()))
    {
        Some(function_node) => {
            let conversion_node = Node::new(
                Operation::TypeConversion,
                vec![function_node.output_type.clone()],
                desired_type,
                vec![function_node],
                context.get_increment_id(),
            );
            Ok((Rc::new(conversion_node), rest))
        }
        None => Err(SyntaxError::TypeConversionNotPossible(
            desired_type.to_string(),
            context.tokens_to_string(rest),
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
        let tokens = context.get_tokens_from_slice(slice);
        let (desired_type, type_end) = IJType::parse_tokens(&tokens)?;
        let rest = slice.move_start(type_end)?;

        match desired_type {
            IJType::Scalar | IJType::Tensor => {
                let (operand, rest) = next_node(rest, context)?;
                let possible_conversions = list_conversions_from(desired_type.clone());
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
                    context.get_increment_id(),
                );
                Ok((Rc::new(conversion_node), rest))
            }
            IJType::Function(ref signature) => {
                if signature.output.len() != 1 {
                    return Err(context
                        .add_context_to_syntax_error(SyntaxError::MultipleOutputs.into(), slice));
                }
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
                        signature.output[0].clone(),
                        vec![convered_function]
                            .into_iter()
                            .chain(operands)
                            .collect(),
                        context.get_increment_id(),
                    )),
                    rest,
                ))
            }
            _ => Err(SyntaxError::TypeConversionNotPossible(
                context.tokens_to_string(rest),
                desired_type.to_string(),
            )
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::lexer;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_list_conversions_to() {
        let types = list_conversions_to(IJType::Number);
        assert_eq!(types.len(), 3);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Number],
            vec![IJType::Scalar],
        )));
        assert_eq!(types.len(), 6);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Number, IJType::Scalar],
            vec![IJType::Number, IJType::Scalar],
        )));
        assert_eq!(types.len(), 36);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor],
            vec![IJType::Function(FunctionSignature::new(
                vec![IJType::Number],
                vec![IJType::Scalar],
            ))],
        )));
        assert_eq!(types.len(), 18);
    }

    #[test]
    fn test_list_conversions_from() {
        let types = list_conversions_from(IJType::Number);
        assert_eq!(types.len(), 2);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Number],
            vec![IJType::Scalar],
        )));
        assert_eq!(types.len(), 6);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Number, IJType::Scalar],
            vec![IJType::Number, IJType::Tensor],
        )));
        assert_eq!(types.len(), 54);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor],
            vec![IJType::Function(FunctionSignature::new(
                vec![IJType::Number],
                vec![IJType::Scalar],
            ))],
        )));
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_type_conversion_functional_part() {
        let tokens = lexer("+ -").unwrap();
        let mut context = ASTContext::from_tokens(tokens);
        let slice = context.full_slice();
        let desired_signature =
            FunctionSignature::new(vec![IJType::Number, IJType::Number], vec![IJType::Scalar]);
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
            FunctionSignature::new(vec![IJType::Tensor, IJType::Tensor], vec![IJType::Scalar]);
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
    fn test_type_conversion_node_function() {
        let result = parse_str_no_context("<-Fn(S,S->T) + 1 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Apply);
    }

}
