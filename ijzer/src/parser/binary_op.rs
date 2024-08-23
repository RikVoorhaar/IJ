use super::{check_ok_needed_outputs, gather_operands, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::types::{FunctionSignature, IJType};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
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
        vec![IJType::Scalar(None), IJType::Tensor(None)],
        vec![IJType::Tensor(None), IJType::Scalar(None)],
        vec![IJType::Scalar(None), IJType::Scalar(None)],
    ];
    let (operands, rest) = gather_operands(types, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    let input_types = operands
        .iter()
        .map(|n| n.output_type.clone())
        .collect::<Vec<IJType>>();
    let output_type = match input_types.as_slice() {
        [IJType::Tensor(None), IJType::Tensor(None)] => IJType::Tensor(None),
        [IJType::Scalar(None), IJType::Tensor(None)] => IJType::Tensor(None),
        [IJType::Tensor(None), IJType::Scalar(None)] => IJType::Tensor(None),
        [IJType::Scalar(None), IJType::Scalar(None)] => IJType::Scalar(None),
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
        Rc::new(Node::new(
            op,
            input_types,
            output_type,
            operands,
            context.get_increment_id(),
        )),
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
    if check_ok_needed_outputs(needed_outputs, &IJType::Scalar(None)) {
        let output_type = IJType::Scalar(None);
        let input_type = vec![IJType::Scalar(None), IJType::Scalar(None)];
        nodes.push(Rc::new(Node::new(
            operation.clone(),
            vec![],
            IJType::Function(FunctionSignature::new(input_type, output_type)),
            vec![],
            context.get_increment_id(),
        )));
    }
    if check_ok_needed_outputs(needed_outputs, &IJType::Number(None)) {
        let output_type = IJType::Number(None);
        let input_type = vec![IJType::Number(None), IJType::Number(None)];
        nodes.push(Rc::new(Node::new(
            operation.clone(),
            vec![],
            IJType::Function(FunctionSignature::new(input_type, output_type)),
            vec![],
            context.get_increment_id(),
        )));
    }
    if check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
        let output_type = IJType::Tensor(None);
        let input_types = vec![
            vec![IJType::Tensor(None), IJType::Tensor(None)],
            vec![IJType::Scalar(None), IJType::Tensor(None)],
            vec![IJType::Tensor(None), IJType::Scalar(None)],
        ];
        for input_type in input_types {
            nodes.push(Rc::new(Node::new(
                operation.clone(),
                vec![],
                IJType::Function(FunctionSignature::new(input_type, output_type.clone())),
                vec![],
                context.get_increment_id(),
            )));
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

pub struct Add;
impl ParseNodeFunctional for Add {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        _next_node_functional_binary(Operation::Add, slice, context, needed_outputs)
    }
}
pub struct Multiply;
impl ParseNodeFunctional for Multiply {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        _next_node_functional_binary(Operation::Multiply, slice, context, needed_outputs)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_add_with_tensors() {
        let result = parse_str_no_context("+ [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor(None), IJType::Tensor(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_add_with_tensors_and_scalar() {
        let result = parse_str_no_context("+ [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor(None), IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_add_with_scalars() {
        let result = parse_str_no_context("+ 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Scalar(None), IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_multiply_with_tensors() {
        let result = parse_str_no_context("* [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Tensor(None), IJType::Tensor(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_multiply_with_tensors_and_scalar() {
        let result = parse_str_no_context("* [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Tensor(None), IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_multiply_with_scalars() {
        let result = parse_str_no_context("* 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Scalar(None), IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Scalar(None));
    }
}
