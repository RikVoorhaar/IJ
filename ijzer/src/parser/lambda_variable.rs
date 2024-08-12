use super::ParseNode;

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::types::IJType;
use crate::operations::Operation;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

pub struct LambdaVariable;
impl ParseNode for LambdaVariable {
    fn next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let name = match op {
            Token::LambdaVariable(v) => v.name,
            _ => unreachable!(),
        };
        let node = Node::new(
            Operation::LambdaVariable(name),
            vec![IJType::Tensor],
            IJType::Tensor,
            vec![],
            context.get_increment_id(),
        );
        Ok((Rc::new(node), tokens))
    }
}
