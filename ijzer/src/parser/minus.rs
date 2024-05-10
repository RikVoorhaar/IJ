use super::{gather_operands, ParseNode};

use crate::ast_node::{ASTContext, IJType, Node, TokenSlice};
use crate::operations::Operation;
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
