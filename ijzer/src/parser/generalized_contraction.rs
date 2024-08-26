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
        FunctionSignature::new(vec![IJType::Tensor(None)], IJType::Scalar(None))
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
}
impl ParseNode for GeneralizedContraction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (f_node, g_node, rest) = Self::get_functional_part(slice, context)?;
        let (operands, rest) =
            gather_operands(vec![vec![IJType::Tensor(None), IJType::Tensor(None)]], rest, context)?;
        let node = Node::new(
            Operation::GeneralizedContraction,
            vec![
                IJType::Function(Self::first_signature()),
                IJType::Function(Self::second_signature()),
                operands[0].output_type.clone(),
                operands[1].output_type.clone(),
            ],
            IJType::Tensor(None),
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
        let node = Node::new(
            Operation::GeneralizedContraction,
            vec![
                IJType::Function(Self::first_signature()),
                IJType::Function(Self::second_signature()),
            ],
            IJType::tensor_function(2),
            vec![f_node, g_node],
            context,
        )?;
        Ok((vec![Rc::new(node)], rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str_no_context;

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
}
