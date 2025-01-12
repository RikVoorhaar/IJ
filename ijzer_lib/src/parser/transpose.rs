//! Transpose operator.
//! 
//! Example:
//! ```ijzer
//! |[[1,2],[3,4]]
//! ```
//! 
//! This will produce a tensor `[[1,3],[2,4]]`.
use super::{gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Transpose;
impl ParseNode for Transpose {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(vec![vec![IJType::Tensor(None)]], slice, context)?;
        let tensor_type = operands[0].output_type.clone();
        Ok((
            Rc::new(Node::new(
                Operation::Transpose,
                vec![tensor_type.clone()],
                tensor_type,
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Transpose {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let rest = slice.move_start(1)?;
        let number_type = match needed_outputs {
            Some(slice) => match slice.iter().find(|&item| matches!(item, IJType::Tensor(_))) {
                Some(IJType::Tensor(number_type)) => number_type.clone(),
                _ => {
                    return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                        format!("{:?}", needed_outputs),
                        IJType::Tensor(None).to_string(),
                    )
                    .into())
                }
            },
            None => None,
        };
        let tensor_type = IJType::Tensor(number_type);
        let node = Rc::new(Node::new(
            Operation::Transpose,
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![tensor_type.clone()],
                tensor_type,
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
    fn test_transpose() -> Result<()> {
        let (node, _) = parse_str_no_context("|[1,2,3]")?;
        assert_eq!(node.op, Operation::Transpose);
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context("|[1,2,3]<f64>")?;
        assert_eq!(node.op, Operation::Transpose);
        assert_eq!(node.output_type, IJType::Tensor(Some("f64".to_string())));
        Ok(())
    }

    #[test]
    fn test_transpose_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~|")?;
        assert_eq!(node.op, Operation::Transpose);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(None)
            ))
        );
        Ok(())
    }
}
