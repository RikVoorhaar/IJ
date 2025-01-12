//! Solve operator. Solves the system AX=B. 
//! 
//! Example:
//! ```ijzer
//! \ [1,2] [1,2]
//! ```
//! 
//! This will produce a tensor `[1,2]`.
//! 
//! The solve operator takes in two tensors and produces a tensor.
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Solve;
impl ParseNode for Solve {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(
            vec![vec![IJType::Tensor(None), IJType::Tensor(None)]],
            slice,
            context,
        )?;

        Ok((
            Rc::new(Node::new(
                Operation::Solve,
                vec![
                    operands[0].output_type.clone(),
                    operands[1].output_type.clone(),
                ],
                IJType::Tensor(None),
                operands,
                context,
            )?),
            rest,
        ))
    }
}

impl ParseNodeFunctional for Solve {
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
            Operation::Solve,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Tensor(None)],
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
    fn test_solve() -> Result<()> {
        let (node, _) = parse_str_no_context(r"\ [1,2] [1,2]")?;
        assert_eq!(node.op, Operation::Solve);
        assert_eq!(
            node.output_type,
            IJType::Tensor(None)
        );


        let (node, _) = parse_str_no_context(r".~\ [1,2] [1,2]")?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(
            node.output_type,
            IJType::Tensor(None)
        );

        Ok(())
    }
}
