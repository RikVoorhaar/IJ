use super::ParseNode;

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

pub struct LambdaVariable;
impl ParseNode for LambdaVariable {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let name = match op {
            Token::LambdaVariable(v) => v.name,
            _ => unreachable!(),
        };
        let (node_type, rest) = if !slice.is_empty()
            && context.get_token_at_index(slice.start)? == &Token::TypeDeclaration
        {
            let mut rest = slice.move_start(1)?;
            let (parsed_type, type_end) =
                IJType::parse_tokens(&context.get_tokens_from_slice(rest))?;
            rest = rest.move_start(type_end)?;
            (parsed_type, rest)
        } else {
            (IJType::Tensor, slice)
        };

        let node = Node::new(
            Operation::LambdaVariable(name),
            vec![node_type.clone()],
            node_type,
            vec![],
            context.get_increment_id(),
        );
        Ok((Rc::new(node), rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;
    use crate::types::IJType;
    use anyhow::Result;

    #[test]
    fn test_lambda_variable() -> Result<()> {
        let (node, _) = parse_str_no_context("+($x, $y)")?;
        assert_eq!(node.operands.len(), 2);
        assert_eq!(
            node.operands[0].op,
            Operation::LambdaVariable("x".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Tensor
        );
        assert_eq!(
            node.operands[1].op,
            Operation::LambdaVariable("y".to_string())
        );
        assert_eq!(
            node.operands[1].output_type,
            IJType::Tensor
        );

        Ok(())
    }

    #[test]
    fn test_lambda_variable_with_type() -> Result<()> {
        let (node, _) = parse_str_no_context("+($x: S, $y: S)")?;
        assert_eq!(node.operands.len(), 2);
        assert_eq!(
            node.operands[0].op,
            Operation::LambdaVariable("x".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Scalar
        );
        assert_eq!(
            node.operands[1].op,
            Operation::LambdaVariable("y".to_string())
        );
        assert_eq!(
            node.operands[1].output_type,
            IJType::Scalar
        );

        Ok(())
    }

    #[test]
    fn test_assign_with_lambda_variable() -> Result<()> {
        let (node, _) = parse_str_no_context("g: Fn(S,S->S) = +($x:S, $y:S)")?;
        assert_eq!(node.operands.len(), 2);
        let lhs = node.operands[0].clone();
        let rhs = node.operands[1].clone();
        assert_eq!(lhs.op, Operation::Symbol("g".to_string()));
        assert_eq!(lhs.output_type, IJType::scalar_function(2,1));
        assert_eq!(rhs.op, Operation::Add);

        Ok(())
    }
}
