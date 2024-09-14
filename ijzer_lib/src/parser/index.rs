use super::{comma_separate, find_matching_parenthesis, gather_operands, ParseNode};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

pub struct Index;
impl ParseNode for Index {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (tensor_operands, rest) =
            gather_operands(vec![vec![IJType::Tensor(None)]], slice, context)?;
        let tensor_operand = tensor_operands[0].clone();

        let next_token = context.get_token_at_index(rest.start)?;
        let rest = rest.move_start(1)?;
        if !matches!(next_token, Token::LSqBracket) {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::ExpectedLSqBracketAfterIndex.into(),
                slice,
            ));
        }
        let inside_size =
            find_matching_parenthesis(context, rest, &Token::LSqBracket, &Token::RSqBracket)?;
        let inside_slice = rest.move_end(inside_size)?;
        let rest = rest.move_start(inside_size + 1)?;

        let slices = comma_separate(inside_slice, context)?;
        let mut index_operands: Vec<Rc<Node>> = vec![];
        for operand_slice in slices {
            if operand_slice.is_empty() {
                return Err(context
                    .add_context_to_syntax_error(SyntaxError::EmptySlice.into(), inside_slice));
            }
            let first_token = context.get_token_at_index(operand_slice.start)?;
            if matches!(first_token, Token::TypeDeclaration) {
                let node = Rc::new(Node::new(
                    Operation::Nothing,
                    vec![],
                    IJType::Void,
                    vec![],
                    context,
                )?);
                index_operands.push(node);
            } else {
                let (node, _) = gather_operands(
                    vec![
                        vec![IJType::Number(Some("usize".to_string()))],
                        vec![IJType::Tensor(Some("usize".to_string()))],
                    ],
                    operand_slice,
                    context,
                )?;
                index_operands.push(node[0].clone());
            }
        }
        let is_all_numbers = index_operands.iter().all(|n| {
            n.output_type
                .type_match(&IJType::Number(Some("usize".to_string())))
        });
        let contains_colon = index_operands
            .iter()
            .any(|n| n.output_type.type_match(&IJType::Void));
        let contains_tensor = index_operands
            .iter()
            .any(|n| n.output_type.type_match(&IJType::Tensor(None)));
        if contains_tensor && contains_colon {
            return Err(context
                .add_context_to_syntax_error(SyntaxError::CannotMixTensorAndColon.into(), slice));
        }
        let all_operands = [vec![tensor_operand.clone()], index_operands.clone()].concat();
        let node = if is_all_numbers {
            Rc::new(Node::new(
                Operation::Index,
                all_operands.iter().map(|n| n.output_type.clone()).collect(),
                IJType::Number(
                    tensor_operand
                        .output_type
                        .extract_number_type()
                        .unwrap_or_default(),
                ),
                all_operands,
                context,
            )?)
        } else {
            Rc::new(Node::new(
                Operation::Index,
                all_operands.iter().map(|n| n.output_type.clone()).collect(),
                tensor_operand.output_type.clone(),
                all_operands,
                context,
            )?)
        };
        Ok((node, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_all_numbers() -> Result<()> {
        let (node, _) = parse_str_no_context("<| [[0,1],[2,3]] [0,1]")?;
        assert_eq!(node.op, Operation::Index);
        assert_eq!(
            node.input_types,
            vec![
                IJType::Tensor(None),
                IJType::Number(None),
                IJType::Number(None)
            ]
        );
        assert_eq!(node.output_type, IJType::Number(None));
        Ok(())
    }

    #[test]
    fn test_numbers_and_colon() -> Result<()> {
        let (node, _) = parse_str_no_context("<| [[0,1],[2,3]] [0,:]")?;
        assert_eq!(node.op, Operation::Index);
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Number(None), IJType::Void]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
        Ok(())
    }

    #[test]
    fn test_scalar_tensor() -> Result<()> {
        let (node, _) = parse_str_no_context("<| [[0,1],[2,3]] [0,[0,1]]")?;
        assert_eq!(node.op, Operation::Index);
        assert_eq!(
            node.input_types,
            vec![
                IJType::Tensor(None),
                IJType::Number(None),
                IJType::Tensor(None)
            ]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
        Ok(())
    }

    #[test]
    fn test_tensor_colon() -> Result<()> {
        let res = parse_str_no_context("<| [[0,1],[2,3]] [[0,1],:]");
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn test_with_types() -> Result<()> {
        let (node, _) = parse_str_no_context("<| [[0,1],[2,3]]<i64> [0<usize>,1<usize>]")?;
        assert_eq!(node.op, Operation::Index);
        assert_eq!(
            node.input_types,
            vec![
                IJType::Tensor(Some("i64".to_string())),
                IJType::Number(Some("usize".to_string())),
                IJType::Number(Some("usize".to_string()))
            ]
        );
        assert_eq!(node.output_type, IJType::Number(Some("i64".to_string())));
        Ok(())
    }
}
