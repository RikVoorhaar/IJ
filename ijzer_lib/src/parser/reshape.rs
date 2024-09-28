//! Reshape operator.
//! 
//! Example:
//! ```ijzer
//! >% [2,1] [1,2]
//! ```
//! 
//! This will produce a 2x1 tensor. 
//! 
//! The reshape operator takes in a tensor and a shape, and produces a tensor.
//! The shape tensor must be castable to a usize. 
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Reshape;
impl ParseNode for Reshape {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(
            vec![vec![
                IJType::Tensor(None),
                IJType::Tensor(Some("usize".to_string())),
            ]],
            slice,
            context,
        )?;
        let input_type = operands
            .iter()
            .map(|operand| operand.output_type.clone())
            .collect::<Vec<_>>();
        let output_type = operands[0].output_type.clone();
        Ok((
            Rc::new(Node::new_skip_number_inference(
                Operation::Reshape,
                input_type,
                output_type,
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Reshape {
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
            Operation::Reshape,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Tensor(None),
                    IJType::Tensor(Some("usize".to_string())),
                ],
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
    fn test_reshape() -> Result<()> {
        let (node, _) = parse_str_no_context(">% [1,2] [2,1]")?;
        assert_eq!(node.op, Operation::Reshape);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Array);
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context(">% [1,2] [2,1]<usize>")?;
        assert_eq!(node.op, Operation::Reshape);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Array);
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context(">% [1,2]<a> [2,1]<usize>")?;
        assert_eq!(node.op, Operation::Reshape);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Array);
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));

        let (node, _) = parse_str_no_context(">% [1,2]<a> [2,1]")?;
        assert_eq!(node.op, Operation::Reshape);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Array);
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));
        Ok(())
    }

    #[test]
    fn test_reshape_functional() -> Result<()> {
        let (node, _) = parse_str_no_context(".~>% [1,2] [2,1]")?;
        assert_eq!(node.op, Operation::Apply);
        let node = node.operands[0].clone();
        assert_eq!(node.op, Operation::Reshape);
        assert_eq!(node.output_type, IJType::Function(FunctionSignature::new(vec![
                    IJType::Tensor(None),
                    IJType::Tensor(Some("usize".to_string())),
                ],
                IJType::Tensor(None),
            )));
        Ok(())
    }
}
