use super::{
    check_ok_needed_outputs, gather_operands, next_node_specific_function, ParseNode,
    ParseNodeFunctional,
};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub struct Reduction;
impl ParseNode for Reduction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (function, rest) =
            next_node_specific_function(slice, context, FunctionSignature::number_function(2))?;
        let signature = function.output_type.extract_signature().unwrap();
        let output_number_type = signature.output.extract_number_type().unwrap_or_default();
        let input_number_type = signature.input[0].extract_number_type().unwrap_or_default();
        let (operands, rest) =
            gather_operands(vec![vec![IJType::Tensor(input_number_type)]], rest, context)?;
        let operand = operands.into_iter().next().unwrap();

        Ok((
            Rc::new(Node::new(
                Operation::Reduce,
                vec![function.output_type.clone(), operand.output_type.clone()],
                IJType::Scalar(output_number_type),
                vec![function, operand],
                context,
            )?),
            rest,
        ))
    }
}
impl ParseNodeFunctional for Reduction {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let actual_outputs = IJType::Scalar(None);
        if !check_ok_needed_outputs(needed_outputs, &actual_outputs) {
            return Err(SyntaxError::FunctionSignatureMismatch(
                format!("{:?}", needed_outputs),
                format!("{:?}", actual_outputs),
            )
            .into());
        }
        let (function, rest) = next_node_specific_function(
            slice.move_start(1)?,
            context,
            FunctionSignature::number_function(2),
        )?;
        let signature = function.output_type.extract_signature().unwrap();
        let output_number_type = signature.output.extract_number_type().unwrap_or_default();
        let input_number_type = signature.input[0].extract_number_type().unwrap_or_default();
        let node = Rc::new(Node::new(
            Operation::Reduce,
            vec![function.output_type.clone()],
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(input_number_type)],
                IJType::Scalar(output_number_type),
            )),
            vec![function],
            context,
        )?);
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
            vec![IJType::number_function(2), IJType::Tensor(None)]
        );
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_reduce_with_function_and_tensor() {
        let mut context = ASTContext::new();
        let var_declaration = parse_str("var f: Fn(N,N->N)", &mut context);
        assert!(var_declaration.is_ok());
        let result = parse_str("/f [1,2]", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(
            node.input_types,
            vec![IJType::number_function(2), IJType::Tensor(None)]
        );
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_next_node_functional() -> Result<()> {
        let (mut context, slice) = tokenize_str_no_context("/+ [1,2]")?;
        let result = Reduction::next_node_functional_impl(
            Token::Reduction,
            slice,
            &mut context,
            Some(&[IJType::Scalar(None)]),
        )?;
        println!("{:?}", result);
        Ok(())
    }

    #[test]
    fn test_reduction_number_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<a>->N<a>)", &mut context)?;
        let node = parse_str("/f [1,2]<a>", &mut context)?;
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(node.output_type, IJType::Scalar(Some("a".to_string())));

        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<a>->N<b>)", &mut context)?;
        let node = parse_str("/f [1,2]<a>", &mut context)?;
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(node.output_type, IJType::Scalar(Some("b".to_string())));
        Ok(())
    }

    #[test]
    fn test_reduction_functional_number_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<a>->N<a>)", &mut context)?;
        let node = parse_str("~/f", &mut context)?;
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(Some("a".to_string()))],
                IJType::Scalar(Some("a".to_string()))
            ))
        );

        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<a>->N<b>)", &mut context)?;
        let node = parse_str("~/f", &mut context)?;
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(Some("a".to_string()))],
                IJType::Scalar(Some("b".to_string()))
            ))
        );
        Ok(())
    }

    #[test]
    fn test_with_composition() -> Result<()> {
        let (node, _) = parse_str_no_context("@(/+, -, +) [1,2] [3,4]")?;
        assert_eq!(node.op, Operation::FunctionComposition(3));
        assert_eq!(node.output_type, IJType::Scalar(None));
        Ok(())
    }
}
