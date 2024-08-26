use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, Token};
use crate::types::{FunctionSignature, IJType};
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
            vec![IJType::Tensor(None)],
            vec![IJType::Scalar(None)],
            vec![IJType::Tensor(None), IJType::Tensor(None)],
            vec![IJType::Scalar(None), IJType::Scalar(None)],
            vec![IJType::Tensor(None), IJType::Scalar(None)],
            vec![IJType::Scalar(None), IJType::Tensor(None)],
        ];
        let (operands, rest) = gather_operands(allowed_types, slice, context)?;
        let input_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();
        let output_type = if input_types
            .iter()
            .all(|t| t.type_match(&IJType::Scalar(None)))
        {
            IJType::Scalar(None)
        } else if input_types
            .iter()
            .any(|t| t.type_match(&IJType::Tensor(None)))
        {
            IJType::Tensor(None)
        } else {
            return Err(SyntaxError::TypeError(
                "S,S or S,T, or T,S or T,T".to_string(),
                format!("{:?}", input_types),
            )
            .into());
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
                    vec![],
                    output_type,
                    vec![],
                    context,
                )?;
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
                    context,
                )?),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    input_types,
                    output_type,
                    operands,
                    context,
                )?),
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
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let rest = slice.move_start(1)?;
        let mut nodes = vec![];
        if check_ok_needed_outputs(needed_outputs, &IJType::Scalar(None)) {
            nodes.push(Rc::new(Node::new(
                Operation::Subtract,
                vec![],
                IJType::scalar_function(2),
                vec![],
                context,
            )?));
            nodes.push(Rc::new(Node::new(
                Operation::Negate,
                vec![],
                IJType::scalar_function(1),
                vec![],
                context,
            )?));
        } else if check_ok_needed_outputs(needed_outputs, &IJType::Number(None)) {
            nodes.push(Rc::new(Node::new(
                Operation::Subtract,
                vec![],
                IJType::number_function(2),
                vec![],
                context,
            )?));
            nodes.push(Rc::new(Node::new(
                Operation::Negate,
                vec![],
                IJType::number_function(1),
                vec![],
                context,
            )?));
        } else if check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
            let output_type = IJType::Tensor(None);
            let input_types = vec![
                vec![IJType::Tensor(None), IJType::Tensor(None)],
                vec![IJType::Scalar(None), IJType::Tensor(None)],
                vec![IJType::Tensor(None), IJType::Scalar(None)],
                vec![IJType::Tensor(None)],
            ];
            for input_type in input_types {
                let op = if input_type.len() == 1 {
                    Operation::Negate
                } else {
                    Operation::Subtract
                };
                nodes.push(Rc::new(Node::new(
                    op,
                    vec![],
                    IJType::Function(FunctionSignature::new(input_type, output_type.clone())),
                    vec![],
                    context,
                )?));
            }
        } else {
            return Err(SyntaxError::FunctionSignatureMismatch(
                format!("{:?}", needed_outputs),
                "Fn(T,T->T) or Fn(S,S->S) or Fn(T,S->T) or Fn(S,T->S) or Fn(N,N->N) or Fn(N->N) or Fn(S->S) or Fn(T->T)".to_string(),
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
    use crate::tokens::Number;
    use std::str::FromStr;

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
        assert_eq!(node.input_types, vec![]);
        assert_eq!(node.output_type, IJType::Scalar(None));
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
        assert_eq!(node.input_types, vec![]);
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_negate_with_tensor() {
        let result = parse_str_no_context("- [1]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Negate);
        assert_eq!(node.input_types, vec![IJType::Tensor(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_subtract_with_tensors() {
        let result = parse_str_no_context("- [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Tensor(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_subtract_with_tensor_and_scalar() {
        let result = parse_str_no_context("- [1] 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![IJType::Tensor(None), IJType::Scalar(None)]
        );
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_subtract_with_scalars() {
        let result = parse_str_no_context("- 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![IJType::Scalar(None), IJType::Scalar(None)]
        );
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_negate_single_group() -> Result<()> {
        let (node, _) = parse_str_no_context("-(1)")?;
        assert_eq!(node.op, Operation::Number(Number::from_str("-1")?));
        assert_eq!(node.operands.len(), 0);
        Ok(())
    }

    #[test]
    fn test_subtract_with_number_types() -> Result<()> {
        let (node, _) = parse_str_no_context("- 1<a> 2")?;
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![IJType::Scalar(Some("a".to_string())), IJType::Scalar(None)]
        );
        assert_eq!(node.output_type, IJType::Scalar(Some("a".to_string())));

        let (node, _) = parse_str_no_context("- 1 2<a>")?;
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![IJType::Scalar(None), IJType::Scalar(Some("a".to_string()))]
        );
        assert_eq!(node.output_type, IJType::Scalar(Some("a".to_string())));

        let (node, _) = parse_str_no_context("- 1<a> 2<a>")?;
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(
            node.input_types,
            vec![
                IJType::Scalar(Some("a".to_string())),
                IJType::Scalar(Some("a".to_string()))
            ]
        );
        assert_eq!(node.output_type, IJType::Scalar(Some("a".to_string())));

        let result = parse_str_no_context("- 1<a> 2<b>");
        assert!(result.is_err());

        Ok(())
    }
}
