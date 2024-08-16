use super::{find_matching_parenthesis, ParseNode};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::types::IJType;
use crate::operations::Operation;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

/// Square brackets denote the start of an array.
#[derive(Debug, PartialEq)]
pub struct LSqBracket;
impl ParseNode for LSqBracket {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LSqBracket, &Token::RSqBracket)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;
        let contents = context.token_slice_to_string(slice);

        Ok((
            Rc::new(Node::new(
                Operation::Array(contents),
                vec![],
                IJType::Tensor,
                vec![],
                context.get_increment_id(),
            )),
            remainder,
        ))
    }
}
