use super::{next_node, next_node_functional};
use crate::{
    ast_node::{ASTContext, Node, TokenSlice},
    operations::Operation,
    parser::ParseNode,
    syntax_error::SyntaxError,
    tokens::Token,
    types::{FunctionSignature, IJType},
};
use anyhow::{Error, Result};
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
                .map(list_conversions_to)
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
                .map(list_conversions_from)
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
) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
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
    let filtered_candidates: Vec<Rc<Node>> = function_candidates
        .into_iter()
        .filter(|candidate| {
            conversion_possible(candidate.output_type.clone(), desired_type.clone())
        })
        .collect();
    if filtered_candidates.is_empty() {
        return Err(SyntaxError::TypeConversionNotPossible(
            desired_type.to_string(),
            context.tokens_to_string(rest),
        )
        .into());
    }

    let possible_nodes: Vec<Rc<Node>> = filtered_candidates
        .into_iter()
        .map(|candidate| {
            Rc::new(Node::new(
                Operation::TypeConversion,
                vec![candidate.output_type.clone()],
                desired_type.clone(),
                vec![candidate],
                context.get_increment_id(),
            ))
        })
        .collect();

    Ok((possible_nodes, rest))
}

struct TypeConversion;
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
                return Ok((Rc::new(conversion_node), rest));
            }
            IJType::Function(signature) => {
                let (possible_nodes, rest) =
                    type_conversion_functional_part(rest, context, signature)?;
                let possible_input_types: Vec<Vec<IJType>> = possible_nodes
                    .iter()
                    .filter_map(|node| {
                        if let IJType::Function(signature) = node.output_type.clone() {
                            Some(signature.input)
                        } else {
                            None
                        }
                    })
                    .collect();
                let operands: Vec<Rc<Node>> = possible_nodes.iter().map(|node| node.clone()).collect();
                /// TODO: figure out which nodes are compatible with these operands, and use them in the final output.
            }
            _ => {
                todo!()
            }
        }

        // return the correct node

        Err(anyhow::anyhow!("TypeConversion::next_node not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_conversions_to() {
        let types = list_conversions_to(IJType::Number);
        assert_eq!(types.len(), 3);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Number],
            vec![IJType::Scalar],
        )));
        assert_eq!(types.len(), 9);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Number, IJType::Scalar],
            vec![IJType::Number, IJType::Scalar],
        )));
        assert_eq!(types.len(), 81);

        let types = list_conversions_to(IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor],
            vec![IJType::Function(FunctionSignature::new(
                vec![IJType::Number],
                vec![IJType::Scalar],
            ))],
        )));
        assert_eq!(types.len(), 9);
    }

    #[test]
    fn test_list_conversions_from() {
        let types = list_conversions_from(IJType::Number);
        assert_eq!(types.len(), 2);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Number],
            vec![IJType::Scalar],
        )));
        assert_eq!(types.len(), 4);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Number, IJType::Scalar],
            vec![IJType::Number, IJType::Tensor],
        )));
        assert_eq!(types.len(), 24);

        let types = list_conversions_from(IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor],
            vec![IJType::Function(FunctionSignature::new(
                vec![IJType::Number],
                vec![IJType::Scalar],
            ))],
        )));
        assert_eq!(types.len(), 12);
    }
}
