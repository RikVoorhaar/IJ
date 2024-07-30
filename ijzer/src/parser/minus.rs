use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, IJType, Node, TokenSlice, FunctionSignature};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, Token};
use anyhow::Result;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub struct MinusOp;
impl ParseNode for MinusOp {
    fn next_node(
        _: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        // let single_tensor = vec![IJType::Tensor];
        // let two_tensors = vec![IJType::Tensor; 2];
        let allowed_types = vec![
            vec![IJType::Tensor],
            vec![IJType::Scalar],
            vec![IJType::Tensor, IJType::Tensor],
            vec![IJType::Scalar, IJType::Scalar],
            vec![IJType::Tensor, IJType::Scalar],
            vec![IJType::Scalar, IJType::Tensor],
        ];
        let (operands, rest) = gather_operands(allowed_types, slice, context)?;
        let input_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();
        let output_type = if input_types.iter().all(|t| t == &IJType::Scalar) {
            IJType::Scalar
        } else {
            IJType::Tensor
        };
        if operands.len() == 1 {
            if let Operation::Number(x) = operands[0].op.clone() {
                let x_val = x.value;
                let modified_x_val = if x_val.starts_with('-') {
                    x_val.trim_start_matches('-').to_string()
                } else {
                    format!("-{}", x_val)
                };

                let node = Node::new(
                    Operation::Number(Number {
                        value: modified_x_val,
                    }),
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                );
                return Ok((Rc::new(node), rest));
            }
        }
        match operands.len() {
            1 => Ok((
                Rc::new(Node::new(
                    Operation::Negate,
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            _ => unreachable!(),
        }
    }
}

impl ParseNodeFunctional for MinusOp {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let rest = slice.move_start(1)?;
        let mut nodes = vec![];
        if check_ok_needed_outputs(needed_outputs, &[IJType::Scalar]) {
            let output_type = vec![IJType::Scalar];
            let input_types = vec![vec![IJType::Scalar, IJType::Scalar], vec![IJType::Scalar]];
            for input_type in input_types {
                nodes.push(Rc::new(Node::new(
                    Operation::Subtract,
                    vec![],
                    IJType::Function(FunctionSignature::new(input_type, output_type.clone())),
                    vec![],
                    context.get_increment_id(),
                )));
            }
        }
        if check_ok_needed_outputs(needed_outputs, &[IJType::Tensor]) {
            let output_type = vec![IJType::Tensor];
            let input_types = vec![
                vec![IJType::Tensor, IJType::Tensor],
                vec![IJType::Scalar, IJType::Tensor],
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor],
            ];
            for input_type in input_types {
                nodes.push(Rc::new(Node::new(
                    Operation::Subtract,
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
                "Fn(T,T->T) or Fn(S,S->S) or Fn(T,S->T) or Fn(S,T->S)".to_string(),
            )
            .into());
        }
        Ok((nodes, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

    #[test]
    fn test_negate_with_scalar() {
        let result = parse_str_no_context("- 1.0");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "-1.0".to_string()
            })
        );
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }
    #[test]
    fn test_double_negate_with_scalar() {
        let result = parse_str_no_context("-- 1.0");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1.0".to_string()
            })
        );
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_negate_with_tensor() {
        let result = parse_str_no_context("- [1]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Negate);
        assert_eq!(node.input_types, vec![IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_tensors() {
        let result = parse_str_no_context("- [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_tensor_and_scalar() {
        let result = parse_str_no_context("- [1] 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_scalars() {
        let result = parse_str_no_context("- 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Scalar, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }
}
