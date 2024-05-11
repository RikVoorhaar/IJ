use super::{gather_operands, ParseNode};

use crate::ast_node::{ASTContext, IJType, Node, TokenSlice};
use crate::operations::Operation;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

pub struct IdentityNode;
impl ParseNode for IdentityNode {
    fn next_node(
        _: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (mut operands, rest) =
            gather_operands(vec![vec![], vec![IJType::Tensor]], tokens, context)?;
        if !operands.is_empty() {
            Ok((operands.remove(0), rest))
        } else {
            let node = Node::new(
                Operation::Identity,
                vec![IJType::Tensor],
                IJType::Tensor,
                operands,
                context.get_increment_id(),
            );
            Ok((Rc::new(node), rest))
        }
    }
}
