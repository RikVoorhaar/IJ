use super::{
    check_ok_needed_outputs, comma_separate, find_matching_parenthesis, gather_operands, ParseNode,
    ParseNodeFunctional,
};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
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

        let rest = rest.move_start(1)?;
        let next_token = context.get_token_at_index(rest.start)?;
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
        let index_operands = match slices.len() {
            1 => {
                gather_operands(
                    vec![
                        vec![IJType::Tensor(Some("usize".to_string()))],
                        vec![IJType::Number(Some("usize".to_string()))],
                    ],
                    slices[0],
                    context,
                )?
                .0
            }
            _ => slices
                .into_iter()
                .map(|slice| {
                    gather_operands(
                        vec![vec![IJType::Number(Some("usize".to_string()))]],
                        slice,
                        context,
                    )
                    .map(|(operands, _)| operands)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        };
        let is_all_numbers = index_operands
            .iter()
            .all(|n| n.output_type.type_match(&IJType::Number(None)));
        let all_operands = vec![vec![tensor_operand.clone()], index_operands.clone()].concat();
        let node = if is_all_numbers {
            Rc::new(Node::new(
                Operation::Index,
                index_operands
                    .iter()
                    .map(|n| n.output_type.clone())
                    .collect(),
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
            if index_operands.len() != 1 {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::IndexingMustBeTensorOrNumbers.into(),
                    slice,
                ));
            }
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
