//! Parses the apply operation `.`
//! 
//! This operations takes a function and applies it to a list of arguments.
//! Example: if `f: Fn(N,N->N)` then `.f 1 2` is equivalent to `f(1, 2)` in Rust.
use super::{gather_operands, next_node, Operation, ParseNode};
use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct Apply;

impl ParseNode for Apply {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (function_node, rest) = next_node(slice, context)?;
        let signature = function_node.output_type.extract_signature().ok_or(
            context.add_context_to_syntax_error(
                SyntaxError::ExpectedFunction(function_node.to_string()).into(),
                slice,
            ),
        )?;
        let (operands, rest) = gather_operands(vec![signature.input.clone()], rest, context)?;
        let operands_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();

        let node_input_types = [vec![function_node.output_type.clone()], operands_types].concat();
        let output_type = *signature.output;
        let operands = [vec![function_node.clone()], operands].concat();

        Ok((
            Rc::new(Node::new(
                Operation::Apply,
                node_input_types,
                output_type.clone(),
                operands,
                context,
            )?),
            rest,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::ASTContext;
    use crate::parser::test_utils::{parse_str, parse_str_no_context};
    use anyhow::Result;

    #[test]
    fn test_apply() -> Result<()> {
        let (node, _) = parse_str_no_context(".~+:Fn(N,N->N) 1 2")?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_with_var() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N,N->N)", &mut context)?;

        let node = parse_str(".~f 1 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_with_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N,N->N)", &mut context)?;

        let node = parse_str(".~f 1<i64> 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.output_type, IJType::Number(Some("i64".to_string())));
        assert_eq!(node.operands.len(), 3);

        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<b>->N<c>)", &mut context)?;

        let node = parse_str(".~f 1<a> 2<b>", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.output_type, IJType::Number(Some("c".to_string())));
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_with_composition() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N,N->N)", &mut context)?;

        let node = parse_str(".~@(-,f) 1 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_function() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N->Fn(N->N))", &mut context)?;

        let node = parse_str(".(f 1) 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 2);
        Ok(())
    }
}
