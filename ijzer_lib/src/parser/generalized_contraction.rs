//! Parses the generalized contraction operation `?`.
//! 
//! This operation is used to contract two tensors using two functions.
//! 
//! This is mainly used to define matrix multiplicatio ?/+*
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

pub struct GeneralizedContraction;
impl GeneralizedContraction {
    fn first_signature() -> FunctionSignature {
        FunctionSignature::new(vec![IJType::Tensor(None)], IJType::Number(None))
    }
    fn second_signature() -> FunctionSignature {
        FunctionSignature::number_function(2)
    }
    fn get_functional_part(
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, Rc<Node>, TokenSlice)> {
        let (f_node, rest) = next_node_specific_function(slice, context, Self::first_signature())?;
        let (g_node, rest) = next_node_specific_function(rest, context, Self::second_signature())?;
        Ok((f_node, g_node, rest))
    }
    fn get_number_type_input_output(
        f_node: &Rc<Node>,
        g_node: &Rc<Node>,
    ) -> Result<(Option<String>, Option<String>)> {
        let f_number_type_output = f_node
            .output_type
            .extract_signature()
            .unwrap()
            .output
            .extract_number_type()
            .unwrap_or_default();
        let f_number_type_input = f_node.output_type.extract_signature().unwrap().input[0]
            .extract_number_type()
            .unwrap_or_default();
        let g_number_type = g_node
            .output_type
            .extract_signature()
            .unwrap()
            .output
            .extract_number_type()
            .unwrap_or_default();
        if let (Some(n), Some(m)) = (f_number_type_input.clone(), g_number_type.clone()) {
            if n != m {
                return Err(SyntaxError::NumberTypeMismatch(n, m).into());
            }
        }
        match (g_number_type, f_number_type_input, f_number_type_output) {
            (Some(n1), _, Some(n2)) => Ok((Some(n1.clone()), Some(n2.clone()))),
            (Some(n1), _, None) => Ok((Some(n1.clone()), Some(n1.clone()))),
            (None, Some(n1), Some(n2)) => Ok((Some(n1.clone()), Some(n2.clone()))),
            (None, None, Some(n2)) => Ok((None, Some(n2.clone()))),
            _ => Ok((None, None)),
        }
    }
}
impl ParseNode for GeneralizedContraction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (f_node, g_node, rest) = Self::get_functional_part(slice, context)?;
        let (number_type_input, number_type_output) =
            Self::get_number_type_input_output(&f_node, &g_node)?;
        let (operands, rest) = gather_operands(
            vec![vec![
                IJType::Tensor(number_type_input.clone()),
                IJType::Tensor(number_type_input.clone()),
            ]],
            rest,
            context,
        )?;
        let output_type = IJType::Tensor(number_type_output.clone());
        let node = Node::new(
            Operation::GeneralizedContraction,
            vec![
                f_node.output_type.clone(),
                g_node.output_type.clone(),
                operands[0].output_type.clone(),
                operands[1].output_type.clone(),
            ],
            output_type,
            vec![f_node, g_node, operands[0].clone(), operands[1].clone()],
            context,
        )?;
        Ok((Rc::new(node), rest))
    }
}
impl ParseNodeFunctional for GeneralizedContraction {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let slice = slice.move_start(1)?;
        if !check_ok_needed_outputs(needed_outputs, &IJType::Tensor(None)) {
            return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                format!("{:?}", needed_outputs),
                IJType::Tensor(None).to_string(),
            )
            .into());
        }
        let (f_node, g_node, rest) = Self::get_functional_part(slice, context)?;
        let (number_type_input, number_type_output) =
            Self::get_number_type_input_output(&f_node, &g_node)?;
        let node = Node::new(
            Operation::GeneralizedContraction,
            vec![f_node.output_type.clone(), g_node.output_type.clone()],
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Tensor(number_type_input.clone()),
                    IJType::Tensor(number_type_input),
                ],
                IJType::Tensor(number_type_output),
            )),
            vec![f_node, g_node],
            context,
        )?;
        Ok((vec![Rc::new(node)], rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_str, parse_str_no_context};

    #[test]
    fn test_generalized_contraction() -> Result<()> {
        let (node, _) = parse_str_no_context("?/+* [1] [2]")?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~?/+*")?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        Ok(())
    }
    #[test]
    fn test_generalized_contraction_functional_apply() -> Result<()> {
        let (node, _) = parse_str_no_context(".~?/+* [1] [2]")?;
        assert_eq!(node.op, Operation::Apply);
        let node = node.operands[0].clone();
        assert_eq!(node.op, Operation::GeneralizedContraction);

        Ok(())
    }

    #[test]
    fn test_generalized_contraction_number_type() -> Result<()> {
        let (node, _) = parse_str_no_context("?/+* [1]<a> [2]")?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));

        let (node, _) = parse_str_no_context("?/+* [1] [2]<a>")?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));

        let (node, _) = parse_str_no_context("?/+* [1]<a> [2]<a>")?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));

        let result = parse_str_no_context("?/+* [1]<a> [2]<b>");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functions() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(T->N)", &mut context)?;
        parse_str("var g: Fn(N,N->N)", &mut context)?;
        let node = parse_str("~? f g", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Tensor(None)],
                IJType::Tensor(None)
            ))
        );
        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functions_number_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(T->N)", &mut context)?;
        parse_str("var g: Fn(N<f64>,N<f64>->N<f64>)", &mut context)?;
        let node = parse_str("? f g [1] [2]", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("f64".to_string())));

        let mut context = ASTContext::new();
        parse_str("var f: Fn(T<a>->N<b>)", &mut context)?;
        parse_str("var g: Fn(N<a>,N<a>->N<a>)", &mut context)?;
        let node = parse_str("? f g [1]<a> [2]<a>", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("b".to_string())));

        let mut context = ASTContext::new();
        parse_str("var f: Fn(T<a>->N<b>)", &mut context)?;
        parse_str("var g: Fn(N,N->N)", &mut context)?;
        let node = parse_str("? f g [1]<a> [2]<a>", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(node.output_type, IJType::Tensor(Some("b".to_string())));
        Ok(())
    }

    #[test]
    fn test_generalized_contraction_functional_number_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(T->N)", &mut context)?;
        parse_str("var g: Fn(N<f64>,N<f64>->N<f64>)", &mut context)?;
        let node = parse_str("~? f g", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Tensor(Some("f64".to_string())),
                    IJType::Tensor(Some("f64".to_string()))
                ],
                IJType::Tensor(Some("f64".to_string()))
            ))
        );

        let mut context = ASTContext::new();
        parse_str("var f: Fn(T<f64>->N<f32>)", &mut context)?;
        parse_str("var g: Fn(N<f64>,N<f64>->N<f64>)", &mut context)?;
        let node = parse_str("~? f g", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Tensor(Some("f64".to_string())),
                    IJType::Tensor(Some("f64".to_string()))
                ],
                IJType::Tensor(Some("f32".to_string()))
            ))
        );

        let mut context = ASTContext::new();
        parse_str("var f: Fn(T<a>->N<b>)", &mut context)?;
        parse_str("var g: Fn(N,N->N)", &mut context)?;
        let node = parse_str("~? f g", &mut context)?;
        assert_eq!(node.op, Operation::GeneralizedContraction);
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Tensor(Some("a".to_string())),
                    IJType::Tensor(Some("a".to_string()))
                ],
                IJType::Tensor(Some("b".to_string()))
            ))
        );
        Ok(())
    }
}
