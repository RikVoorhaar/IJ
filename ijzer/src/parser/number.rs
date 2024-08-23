use super::ParseNode;

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

pub struct NumberNode;
impl ParseNode for NumberNode {
    fn next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };

        Ok((
            Rc::new(Node::new(
                node_op,
                vec![],
                IJType::Scalar(None),
                vec![],
                context.get_increment_id(),
            )),
            tokens,
        ))
    }
}
