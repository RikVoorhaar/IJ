//! Parses the diag operation `diag`.
//! 
//! This operation is used to create a diagonal matrix from a tensor.
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Diag;
impl ParseNode for Diag {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(vec![vec![IJType::Tensor(None)]], slice, context)?;
        let number_type = operands[0]
            .output_type
            .extract_number_type()
            .unwrap_or_default();

        Ok((
            Rc::new(Node::new(
                Operation::Diag,
                vec![operands[0].output_type.clone()],
                IJType::Tensor(number_type),
                operands,
                context,
            )?),
            rest,
        ))
    }
}

impl ParseNodeFunctional for Diag {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let rest = slice.move_start(1)?;
        let output_type = IJType::Tensor(None);
        if !check_ok_needed_outputs(needed_outputs, &output_type) {
            return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                format!("{:?}", needed_outputs),
                output_type.to_string(),
            )
            .into());
        }

        let node = Rc::new(Node::new(
            Operation::Diag,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                output_type.clone(),
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
    fn test_diag() -> Result<()> {
        let (node, _) = parse_str_no_context("diag [1,2]")?;
        assert_eq!(node.op, Operation::Diag);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context(".~diag [1,2]")?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context("x = diag [1,2]")?;
        assert_eq!(node.op, Operation::Assign);
        let lhs = node.operands[0].clone();
        assert_eq!(lhs.op, Operation::AssignSymbol("x".to_string()));
        let rhs = node.operands[1].clone();
        assert_eq!(rhs.op, Operation::Diag);
        Ok(())
    }
}
