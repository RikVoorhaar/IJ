use super::{gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct IdentityNode;
impl ParseNode for IdentityNode {
    fn next_node(
        _: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (operands, rest) = gather_operands(
            vec![vec![IJType::Scalar], vec![IJType::Tensor]],
            tokens,
            context,
        )?;
        if operands.len() != 1 {
            return Err(anyhow::anyhow!("Identity operation expects one operand"));
        }
        Ok((operands[0].clone(), rest))
    }
}

impl ParseNodeFunctional for IdentityNode {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let slice = slice.move_start(1)?;
        let output_types = match needed_outputs {
            Some(outputs) => outputs,
            None => &[vec![IJType::Scalar], vec![IJType::Tensor]],
        };

        let nodes = output_types
            .iter()
            .map(|output_type| {
                Rc::new(Node::new(
                    Operation::Identity,
                    vec![],
                    IJType::Function(FunctionSignature {
                        input: output_type.clone(),
                        output: output_type.clone(),
                    }),
                    vec![],
                    context.get_increment_id(),
                ))
            })
            .collect();
        Ok((nodes, slice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_identity() -> Result<()> {
        let (node, _) = parse_str_no_context("I 1")?;
        assert!(matches!(node.op, Operation::Number(..)));

        let (node, _) = parse_str_no_context("I [1,2,3]")?;
        assert!(matches!(node.op, Operation::Array(..)));

        let (node, _) = parse_str_no_context("I I I I I 1")?;
        assert!(matches!(node.op, Operation::Number(..)));

        Ok(())
    }

    #[test]
    fn test_identity_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~I: Fn(S->S)")?;
        assert_eq!(node.op, Operation::Identity);
        assert_eq!(node.input_types.len(), 0);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature {
                input: vec![IJType::Scalar],
                output: vec![IJType::Scalar],
            })
        );
        let (node, _) = parse_str_no_context("~I: Fn(T->T)")?;
        assert_eq!(node.op, Operation::Identity);
        assert_eq!(node.input_types.len(), 0);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor],
                output: vec![IJType::Tensor],
            })
        );
        let (node, _) = parse_str_no_context("@(~-:Fn(T->T), I) [1]")?;
        assert_eq!(node.op, Operation::FunctionComposition(2, 1));
        assert_eq!(node.output_type, IJType::Tensor);

        Ok(())
    }
}
