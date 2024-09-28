//! Parses binary operations.
//! 
//! Supported operations:
//! - max
//! - min
//! - ^ (power)
//! - +
//! - *
//! - /: (division)
//! - == (equal)
//! - != (not equal)
//! - >. (greater than)
//! - <. (less than)
//! - >= (greater than or equal to)
//! - <= (less than or equal to)
//! - && (logical and)
//! - || (logical or)
//! 
//! If either operand is a tensor, the output is a tensor. Only if both operands are scalars, the output is a scalar. 
//! 
//! If the number type is specified for at least one operand, the output type is inferred.
//! If no number type is specified for an operand, the output type is `_`.
use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::{anyhow, Context, Result};
use std::rc::Rc;

/// Parses a simple tensor function, such as addition or multiplication. These all have
/// a range of input aritys. All inputs tensors or scalars.
pub fn next_node_simple_binary_op(
    op: Operation,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let types = vec![
        vec![IJType::Tensor(None), IJType::Tensor(None)],
        vec![IJType::Number(None), IJType::Tensor(None)],
        vec![IJType::Tensor(None), IJType::Number(None)],
        vec![IJType::Number(None), IJType::Number(None)],
    ];
    let (operands, rest) = gather_operands(types, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    let input_types = operands
        .iter()
        .map(|n| n.output_type.clone())
        .collect::<Vec<IJType>>();
    let output_type = match input_types.as_slice() {
        [IJType::Tensor(_), IJType::Tensor(_)] => IJType::Tensor(None),
        [IJType::Number(_), IJType::Tensor(_)] => IJType::Tensor(None),
        [IJType::Tensor(_), IJType::Number(_)] => IJType::Tensor(None),
        [IJType::Number(_), IJType::Number(_)] => IJType::Number(None),
        _ => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::TypeError(
                    input_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<String>(),
                    "Tensor -> Tensor or Scalar -> Scalar".to_string(),
                )
                .into(),
                slice,
            ))
        }
    };
    Ok((
        Rc::new(Node::new(op, input_types, output_type, operands, context)?),
        rest,
    ))
}

fn _next_node_functional_binary(
    operation: Operation,
    slice: TokenSlice,
    context: &mut ASTContext,
    needed_outputs: Option<&[IJType]>,
) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
    let rest = slice.move_start(1)?;
    let mut nodes = vec![];
    if check_ok_needed_outputs(needed_outputs, &IJType::Number(None)) {
        let output_type = IJType::Number(None);
        let input_type = vec![IJType::Number(None), IJType::Number(None)];
        nodes.push(Rc::new(Node::new(
            operation.clone(),
            vec![],
            IJType::Function(FunctionSignature::new(input_type, output_type)),
            vec![],
            context,
        )?));
    }
    if check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
        let output_type = IJType::Tensor(None);
        let input_types = vec![
            vec![IJType::Tensor(None), IJType::Tensor(None)],
            vec![IJType::Number(None), IJType::Tensor(None)],
            vec![IJType::Tensor(None), IJType::Number(None)],
        ];
        for input_type in input_types {
            nodes.push(Rc::new(Node::new(
                operation.clone(),
                vec![],
                IJType::Function(FunctionSignature::new(input_type, output_type.clone())),
                vec![],
                context,
            )?));
        }
    }
    if nodes.is_empty() {
        return Err(SyntaxError::FunctionSignatureMismatch(
            format!("{:?}", needed_outputs),
            "Fn(T,T->T) or Fn(S,S->S) or Fn(T,S->T) or Fn(S,T->S) or Fn(N,N->N)".to_string(),
        )
        .into());
    }
    Ok((nodes, rest))
}

pub struct BinaryOp;
impl ParseNodeFunctional for BinaryOp {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let function = match op {
            Token::BinaryFunction(f) => f,
            _ => unreachable!(
                "BinaryOp::next_node_functional_impl called with non-binary function token"
            ),
        };
        _next_node_functional_binary(
            Operation::BinaryFunction(function),
            slice,
            context,
            needed_outputs,
        )
    }
}
impl ParseNode for BinaryOp {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let function = match op {
            Token::BinaryFunction(f) => f,
            _ => unreachable!("BinaryOp::next_node called with non-binary function token"),
        };
        next_node_simple_binary_op(Operation::BinaryFunction(function), slice, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_enums::BinaryMathFunctionEnum;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_add_with_tensors() {
        let result = parse_str_no_context("+ [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Add)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Tensor(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_add_with_tensors_and_scalar() {
        let result = parse_str_no_context("+ [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Add)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Number(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_add_with_scalars() {
        let result = parse_str_no_context("+ 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Add)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Number(None), IJType::Number(None)]
        );
        assert_eq!(node.output_type, IJType::Number(None));
    }

    #[test]
    fn test_multiply_with_tensors() {
        let result = parse_str_no_context("* [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Multiply)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Tensor(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_multiply_with_tensors_and_scalar() {
        let result = parse_str_no_context("* [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Multiply)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Number(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_multiply_with_scalars() {
        let result = parse_str_no_context("* 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Multiply)
        );
        assert_eq!(
            node.input_types,
            vec![IJType::Number(None), IJType::Number(None)]
        );
        assert_eq!(node.output_type, IJType::Number(None));
    }

    #[test]
    fn test_with_type_inference() -> Result<()> {
        let (node, _) = parse_str_no_context("+ 1 2<i64>")?;
        assert_eq!(node.output_type, IJType::Number(Some("i64".to_string())));

        let (node, _) = parse_str_no_context("+ 1 2<_>")?;
        assert_eq!(node.output_type, IJType::Number(Some("_".to_string())));

        let (node, _) = parse_str_no_context("+ 1<_> 2<_>")?;
        assert_eq!(node.output_type, IJType::Number(Some("_".to_string())));

        let result = parse_str_no_context("+ 1<usize> 2<i64>");
        assert!(result.is_err());

        Ok(())
    }
}
