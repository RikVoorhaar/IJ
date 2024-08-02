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
        let actual_outputs = vec![IJType::Scalar];
        if !check_ok_needed_outputs(needed_outputs, &actual_outputs) {
            return Err(SyntaxError::FunctionSignatureMismatch(
                format!("{:?}", needed_outputs),
                format!("{:?}", actual_outputs),
            )
            .into());
        }
        let (function, rest) = reduction_get_functional_part(slice.move_start(1)?, context)?;
        let node = Rc::new(Node::new(
            Operation::Reduce,
            vec![IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar, IJType::Scalar],
                vec![IJType::Scalar],
            ))],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Scalar],
            )),
            vec![function],
            context.get_increment_id(),
        ));
        Ok((vec![node], rest))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::test_utils::tokenize_str_no_context;
    use crate::parser::{parse_str, parse_str_no_context};

    #[test]
    fn test_reduce_plus_with_tensor() {
        let result = parse_str_no_context("/+ [1,2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Reduce);

        assert_eq!(
            node.input_types,
            vec![IJType::scalar_function(2, 1), IJType::Tensor]
        );
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_reduce_with_function_and_tensor() {
        let mut context = ASTContext::new();
        let var_declaration = parse_str("var f: Fn(S,S->S)", &mut context);
        assert!(var_declaration.is_ok());
        let result = parse_str("/f [1,2]", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(
            node.input_types,
            vec![IJType::scalar_function(2, 1), IJType::Tensor]
        );
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_next_node_functional() {
        let (mut context, slice) = tokenize_str_no_context("/+ [1,2]").unwrap();
        let result = Reduction::next_node_functional_impl(
            Token::Reduction,
            slice,
            &mut context,
            Some(&[vec![IJType::Scalar]]),
        );
        assert!(result.is_ok());
    }
}
