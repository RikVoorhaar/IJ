use super::{
    find_matching_parenthesis, gather_all, next_node_functional, ParseNode, ParseNodeFunctional,
};
use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
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
            0 => Err(SyntaxError::EmptyGroup.into()),
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

impl ParseNodeFunctional for LParen {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let slice = slice.move_start(1)?;
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;

        let (operand_variants, rest) = next_node_functional(slice, context, needed_outputs)?;
        if !rest.is_empty() {
            return Err(
                SyntaxError::GroupReturnsMultipleFunctions(context.token_slice_to_string(rest)).into(),
            );
        }
        Ok((operand_variants, remainder))
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use crate::parser::parse_str_no_context;
    use crate::tokens::Number;
    use anyhow::Error;

    #[test]
    fn test_empty_group() {
        let result = parse_str_no_context("()");
        assert!(result.is_err());
    }

    #[test]
    fn test_single_group() -> Result<(), Error> {
        let (node, _) = parse_str_no_context("(1)")?;
        assert_eq!(node.op, Operation::Number(Number::from_str("1")?));

        let (node, _) = parse_str_no_context("/+([1])")?;
        assert_eq!(node.op, Operation::Reduce);

        let (node, _) = parse_str_no_context("-(1)")?;
        assert_eq!(node.op, Operation::Number(Number::from_str("-1")?));
        Ok(())
    }

    #[test]
    fn test_single_group_function() -> Result<(), Error> {
        let (node, _) = parse_str_no_context("/(+)([1])")?;
        assert_eq!(node.op, Operation::Reduce);

        Ok(())
    }

    #[test]
    fn test_double_group_function() -> Result<(), Error> {
        let result = parse_str_no_context("/(+ -)");
        assert!(result.is_err());
        Ok(())
    }
    
    #[test]
    fn test_group_with_comma() -> Result<(), Error> {
        let (node, _) = parse_str_no_context("(1, 2)")?;
        assert_eq!(node.op, Operation::Group);
        assert_eq!(node.operands.len(), 2);

        let (node, _) = parse_str_no_context("(1, 2 3)")?;
        assert_eq!(node.op, Operation::Group);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_add_group() -> Result<(), Error> {
        let (node, _) = parse_str_no_context("+(1, 2)")?;
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.operands.len(), 2);

        let (node, _) = parse_str_no_context("+(1 2)")?;
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.operands.len(), 2);

        let (node, _) = parse_str_no_context("+([1] [1])")?;
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.operands.len(), 2);
        Ok(())
    }
}
