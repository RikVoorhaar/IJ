use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct UnaryFunction;
impl ParseNode for UnaryFunction {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let Token::UnaryFunction(function_name) = op else {
            unreachable!()
        };
        let (operands, rest) = gather_operands(
            vec![vec![IJType::Tensor(None)], vec![IJType::Number(None)]],
            slice,
            context,
        )?;
        let operand = operands[0].clone();

        Ok((
            Rc::new(Node::new_skip_number_inference(
                Operation::UnaryFunction(function_name),
                vec![operand.output_type.clone()],
                operand.output_type.clone(),
                operands,
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for UnaryFunction {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let Token::UnaryFunction(function_name) = op else {
            unreachable!()
        };
        let rest = slice.move_start(1)?;
        let mut output_tokens = vec![];
        if check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
            output_tokens.push(Rc::new(Node::new(
                Operation::UnaryFunction(function_name.clone()),
                vec![],
                IJType::Function(FunctionSignature::new(
                    vec![IJType::Tensor(None)],
                    IJType::Tensor(None),
                )),
                vec![],
                context,
            )?));
        }
        if check_ok_needed_outputs(needed_outputs, &IJType::Number(None)) {
            output_tokens.push(Rc::new(Node::new(
                Operation::UnaryFunction(function_name),
                vec![],
                IJType::Function(FunctionSignature::new(
                    vec![IJType::Number(None)],
                    IJType::Number(None),
                )),
                vec![],
                context,
            )?));
        }
        if output_tokens.is_empty() {
            return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                format!("{:?}", needed_outputs),
                IJType::Tensor(None).to_string(),
            )
            .into());
        }
        Ok((output_tokens, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_unary_function() -> Result<()> {
        let (node, _) = parse_str_no_context("sin 1.0")?;
        assert_eq!(node.op, Operation::UnaryFunction("sin".to_string()));
        assert_eq!(node.operands.len(), 1);

        let (node, _) = parse_str_no_context("sin cos 1.0")?;
        assert_eq!(node.op, Operation::UnaryFunction("sin".to_string()));
        assert_eq!(node.operands.len(), 1);
        Ok(())
    }

    #[test]
    fn test_unary_function_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~sin: Fn(T->T)")?;
        assert_eq!(node.op, Operation::UnaryFunction("sin".to_string()));
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(None)
            ))
        );

        let (node, _) = parse_str_no_context("~exp: Fn(N->N)")?;
        assert_eq!(node.op, Operation::UnaryFunction("exp".to_string()));
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None)],
                IJType::Number(None)
            ))
        );
        Ok(())
    }

    #[test]
    fn test_unary_function_composition() -> Result<()> {
        let (node, _) = parse_str_no_context("@(ln, sin, cos) [1]")?;
        assert_eq!(node.op, Operation::FunctionComposition(3));
        assert_eq!(node.output_type, IJType::Tensor(None));

        Ok(())
    }
}
