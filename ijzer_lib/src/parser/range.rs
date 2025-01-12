//! Range operator. 
//! 
//! Produces a range from a start to an end value. 
//! Example:
//! ```ijzer
//! .. 1 10
//! ```
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Range;
impl ParseNode for Range {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(
            vec![vec![IJType::Number(None), IJType::Number(None)]],
            slice,
            context,
        )?;
        let input_type = operands
            .iter()
            .map(|operand| operand.output_type.clone())
            .collect::<Vec<_>>();
        Ok((
            Rc::new(Node::new(
                Operation::Range,
                input_type,
                IJType::Tensor(None),
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Range {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let rest = slice.move_start(1)?;
        if !check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
            return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                format!("{:?}", needed_outputs),
                IJType::Tensor(None).to_string(),
            )
            .into());
        }
        let node = Rc::new(Node::new(
            Operation::Range,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None), IJType::Number(None)],
                IJType::Tensor(None),
            )),
            vec![],
            context,
        )?);
        Ok((vec![node], rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_range() -> Result<()> {
        let (node, _) = parse_str_no_context(".. 1 10")?;
        assert_eq!(node.op, Operation::Range);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context(".. 1 10<usize>")?;
        assert_eq!(node.op, Operation::Range);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.output_type, IJType::Tensor(Some("usize".to_string())));

        let (node, _) = parse_str_no_context(".. 1<usize> 10")?;
        assert_eq!(node.op, Operation::Range);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.output_type, IJType::Tensor(Some("usize".to_string())));

        let (node, _) = parse_str_no_context(".. 1<usize> 10<usize>")?;
        assert_eq!(node.op, Operation::Range);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.output_type, IJType::Tensor(Some("usize".to_string())));

        let result = parse_str_no_context(".. 1<a> 10<b>");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_range_functional() -> Result<()> {
        let (node, _) = parse_str_no_context(".~.. 1 10")?;
        assert_eq!(node.op, Operation::Apply);
        let node = node.operands[0].clone();
        assert_eq!(node.op, Operation::Range);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None), IJType::Number(None)],
                IJType::Tensor(None),
            ))
        );
        Ok(())
    }
}
