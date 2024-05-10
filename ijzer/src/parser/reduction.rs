use super::{
    check_ok_needed_outputs, gather_operands, next_node_functional, ParseNode, ParseNodeFunctional,
};

use crate::ast_node::{ASTContext, FunctionSignature, IJType, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

fn reduction_get_functional_part(
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let function_signature = FunctionSignature::new(vec![IJType::Scalar; 2], vec![IJType::Scalar]);
    let function_type = IJType::Function(function_signature.clone());
    let (function_options, rest) =
        next_node_functional(slice, context, Some(&[function_signature.output]))?;
    let function = function_options
        .into_iter()
        .find(|n| n.output_type == function_type);

    if function.is_none() {
        return Err(SyntaxError::TypeError(
            function_type.to_string(),
            "Function type not found".to_string(),
        )
        .into());
    }
    Ok((function.unwrap(), rest))
}

pub struct Reduction;
impl ParseNode for Reduction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (function, rest) = reduction_get_functional_part(slice, context)?;
        let (operands, rest) = gather_operands(vec![vec![IJType::Tensor]], rest, context)?;
        let operand = operands.into_iter().next().unwrap();
        Ok((
            Rc::new(Node::new(
                Operation::Reduce,
                vec![function.output_type.clone(), IJType::Tensor],
                IJType::Scalar,
                vec![function, operand],
                context.get_increment_id(),
            )),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Reduction {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        if !check_ok_needed_outputs(needed_outputs, &[IJType::Tensor]) {
            return Err(SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into());
        }
        let (function, slice) = reduction_get_functional_part(slice, context)?;
        let node = Rc::new(Node::new(
            Operation::Reduce,
            vec![function.output_type.clone(), IJType::Tensor],
            IJType::Scalar,
            vec![function],
            context.get_increment_id(),
        ));
        Ok((vec![node], slice))
    }
}
