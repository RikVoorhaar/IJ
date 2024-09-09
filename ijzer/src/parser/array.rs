use super::utils::{comma_separate, gather_operands};
use super::{find_matching_parenthesis, ParseNode};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
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
        let rest = slice.move_start(rparen_index + 1)?;
        let inner_slice = slice.move_end(rparen_index)?;
        let content_slices = comma_separate(inner_slice, context)?;
        let content_operands = content_slices
            .into_iter()
            .map(|s| {
                let (operands, remainder) = gather_operands(
                    vec![vec![IJType::Tensor(None)], vec![IJType::Scalar(None)]],
                    s,
                    context,
                )?;
                match remainder.is_empty() {
                    true => Ok(operands[0].clone()),
                    false => Err(SyntaxError::UnhandledTokens(
                        context.token_slice_to_string(remainder),
                    )
                    .into()),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let (output_type, rest) = match context.get_token_at_index(rest.start) {
            Ok(Token::NumberType(n)) => (IJType::Tensor(Some(n.clone())), rest.move_start(1)?),
            _ => (IJType::Tensor(None), rest),
        };

        Ok((
            Rc::new(Node::new(
                Operation::Array,
                content_operands
                    .iter()
                    .map(|n| n.output_type.clone())
                    .collect(),
                output_type,
                content_operands,
                context,
            )?),
            rest,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_array_node() -> Result<()> {
        let (node, _) = parse_str_no_context("[1, 2, 3]")?;
        assert_eq!(node.op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context("[[1.0]]")?;
        assert_eq!(node.op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(None));

        Ok(())
    }

    #[test]
    fn test_array_node_with_type() -> Result<()> {
        let (node, _) = parse_str_no_context("[1, 2, 3]<i64>")?;
        assert_eq!(node.op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(Some("i64".to_string())));

        let (node, _) = parse_str_no_context("[1, 2, 3]<_>")?;
        assert_eq!(node.op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(Some("_".to_string())));

        let (node, _) = parse_str_no_context("[[1.0]]<f64>")?;
        assert_eq!(node.op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(Some("f64".to_string())));

        Ok(())
    }
}
