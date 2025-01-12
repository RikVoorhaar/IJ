//! Parses numbers and infers their type.
//! 
//! For example `1` is a number, and so is `1.0<f64>`. 
//! If no type is specified, the type is inferred to be `_`.
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
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };
        match context.get_token_at_index(slice.start) {
            Ok(Token::NumberType(n)) => {
                let slice = slice.move_start(1)?;
                Ok((
                    Rc::new(Node::new(
                        node_op,
                        vec![],
                        IJType::Number(Some(n.clone())),
                        vec![],
                        context,
                    )?),
                    slice,
                ))
            }
            _ => Ok((
                Rc::new(Node::new(
                    node_op,
                    vec![],
                    IJType::Number(None),
                    vec![],
                    context,
                )?),
                slice,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;
    use crate::tokens::Number;

    #[test]
    fn test_number_node() -> Result<()> {
        let (node, _) = parse_str_no_context("1.0")?;
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1.0".to_string()
            })
        );
        assert_eq!(node.output_type, IJType::Number(None));
        let (node, _) = parse_str_no_context("1")?;
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1".to_string()
            })
        );
        assert_eq!(node.output_type, IJType::Number(None));
        Ok(())
    }

    #[test]
    fn test_number_node_type() -> Result<()> {
        let (node, _) = parse_str_no_context("1.0<_>")?;
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1.0".to_string()
            })
        );
        assert_eq!(node.output_type, IJType::Number(Some("_".to_string())));
        let (node, _) = parse_str_no_context("1.0<f64>")?;
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1.0".to_string()
            })
        );
        assert_eq!(node.output_type, IJType::Number(Some("f64".to_string())));
        let (node, _) = parse_str_no_context("1<i32>")?;
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1".to_string()
            })
        );
        assert_eq!(node.output_type, IJType::Number(Some("i32".to_string())));
        Ok(())
    }
}
