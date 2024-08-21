use super::{
    check_ok_needed_outputs, gather_operands, next_node_functional, next_node_specific_function,
    ParseNode, ParseNodeFunctional,
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
        FunctionSignature::new(vec![IJType::Tensor], IJType::Scalar)
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
            gather_operands(vec![vec![IJType::Tensor, IJType::Tensor]], rest, context)?;
        let node = Node::new(
            Operation::GeneralizedContraction,
            vec![
                IJType::Function(Self::first_signature()),
                IJType::Function(Self::second_signature()),
                IJType::Tensor,
                IJType::Tensor,
            ],
            IJType::Tensor,
            vec![f_node, g_node, operands[0].clone(), operands[1].clone()],
            context.get_increment_id(),
        );
        Ok((Rc::new(node), rest))
    }
}
