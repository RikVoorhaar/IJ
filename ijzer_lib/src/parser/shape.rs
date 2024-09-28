//! Shape operator.
//! 
//! Example:
//! ```ijzer
//! %[1,2]
//! ```
//! 
//! This will produce a scalar `2`.
//! 
//! The shape operator takes in a tensor and produces a scalar.
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Shape;
impl ParseNode for Shape {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(vec![vec![IJType::Tensor(None)]], slice, context)?;
        let input_type = operands[0].output_type.clone();
        Ok((
            Rc::new(Node::new(
                Operation::Shape,
                vec![input_type.clone()],
                IJType::Tensor(Some("usize".to_string())),
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Shape {
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
            Operation::Shape,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(Some("usize".to_string())),
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
    fn test_shape() -> Result<()> {
        let (node, _) = parse_str_no_context("%[1,2]")?;
        assert_eq!(node.op, Operation::Shape);
        Ok(())
    }

    #[test]
    fn test_shape_functional() -> Result<()> {
        let (node, _) = parse_str_no_context(".~%[1,2]")?;
        assert_eq!(node.op, Operation::Apply);
        let node = node.operands[0].clone();
        assert_eq!(node.op, Operation::Shape);
        Ok(())
    }
}
