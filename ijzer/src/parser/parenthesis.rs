use super::{find_matching_parenthesis, gather_all, ParseNode};
use crate::ast_node::{ASTContext, IJType, Node, TokenSlice};
use crate::operations::Operation;
use crate::tokens::Token;
use anyhow::{anyhow, Context, Result};
use std::fmt::Debug;
use std::rc::Rc;
/// Left parenthesis denotes the start of a group.
/// This node is therefore a function with operands as inputs and a group as output.
/// A group with a single operand is however just the identity function.
#[derive(Debug, PartialEq)]
pub struct LParen;
impl ParseNode for LParen {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;

        let operands = gather_all(slice, context)
            .with_context(|| anyhow!("Error created by bracket parsing"))?;

        match operands.len() {
            0 => unreachable!(),
            1 => Ok((operands[0].clone(), remainder)),
            _ => {
                let operands_types = operands
                    .iter()
                    .map(|n| n.output_type.clone())
                    .collect::<Vec<IJType>>();
                Ok((
                    Rc::new(Node::new(
                        Operation::Group,
                        operands_types.clone(),
                        IJType::Group(operands_types),
                        operands,
                        context.get_increment_id(),
                    )),
                    remainder,
                ))
            }
        }
    }
}
