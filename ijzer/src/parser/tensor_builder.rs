use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct TensorBuilder;
impl ParseNode for TensorBuilder {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let Token::TensorBuilder(builder_name) = op else {
            unreachable!()
        };
        let mut rest = slice;
        let number_type = if rest.is_empty() {
            None
        } else if let Token::NumberType(number_type) = context.get_token_at_index(rest.start)? {
            rest = rest.move_start(1)?;
            Some(number_type.clone())
        } else {
            None
        };
        let (operands, rest) = gather_operands(
            vec![vec![IJType::Tensor(Some("usize".to_string()))]],
            rest,
            context,
        )?;

        Ok((
            Rc::new(Node::new_skip_number_inference(
                Operation::TensorBuilder(builder_name),
                vec![operands[0].output_type.clone()],
                IJType::Tensor(number_type),
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for TensorBuilder {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let Token::TensorBuilder(builder_name) = op else {
            unreachable!()
        };
        let mut rest = slice.move_start(1)?;
        if !check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
            return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                format!("{:?}", needed_outputs),
                IJType::Tensor(None).to_string(),
            )
            .into());
        }
        let number_type = if rest.is_empty() {
            None
        } else if let Token::NumberType(number_type) = context.get_token_at_index(rest.start)? {
            rest = rest.move_start(1)?;
            Some(number_type.clone())
        } else {
            None
        };
        let node = Rc::new(Node::new(
            Operation::TensorBuilder(builder_name),
            vec![],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(Some("usize".to_string()))],
                IJType::Tensor(number_type),
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
    fn test_tensor_builder() -> Result<()> {
        let (node, _) = parse_str_no_context("randu [3]")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(node.output_type, IJType::Tensor(None));

        let (node, _) = parse_str_no_context("randu<f64> [3]")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(node.output_type, IJType::Tensor(Some("f64".to_string())));

        let (node, _) = parse_str_no_context("randu<f64> [3]<usize>")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(node.output_type, IJType::Tensor(Some("f64".to_string())));

        let (node, _) = parse_str_no_context("randu [3]<usize>")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(node.output_type, IJType::Tensor(None));

        Ok(())
    }

    #[test]
    fn test_tensor_builder_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~randu<f64>")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(Some("usize".to_string()))],
                IJType::Tensor(Some("f64".to_string())),
            ))
        );

        let (node, _) = parse_str_no_context("~randu")?;
        assert_eq!(node.op, Operation::TensorBuilder("randu".to_string()));
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(Some("usize".to_string()))],
                IJType::Tensor(None),
            ))
        );

        Ok(())
    }
}
