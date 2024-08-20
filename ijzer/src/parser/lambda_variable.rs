use super::{ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

pub fn parse_lambda_assign_lhs(
    op: Token,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let name = match op {
        Token::LambdaVariable(v) => v.name,
        _ => unreachable!(),
    };
    let (node_type, rest) = if !slice.is_empty()
        && context.get_token_at_index(slice.start)? == &Token::TypeDeclaration
    {
        let mut rest = slice.move_start(1)?;
        let (parsed_type, type_end) = IJType::parse_tokens(&context.get_tokens_from_slice(rest))?;
        rest = rest.move_start(type_end)?;
        (parsed_type, rest)
    } else {
        (IJType::Tensor, slice)
    };

    let node = Node::new(
        Operation::LambdaVariable(name),
        vec![node_type.clone()],
        node_type,
        vec![],
        context.get_increment_id(),
    );
    Ok((Rc::new(node), rest))
}

pub struct LambdaVariable;
impl ParseNode for LambdaVariable {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let name = match op {
            Token::LambdaVariable(v) => v.name,
            _ => unreachable!(),
        };
        let var_type = match context.get_lambda_var_type(name.clone()) {
            Some(var_type) => var_type,
            None => return Err(SyntaxError::UnknownSymbol(name.clone()).into()),
        };
        let node = Node::new(
            Operation::LambdaVariable(name),
            vec![],
            var_type,
            vec![],
            context.get_increment_id(),
        );
        Ok((Rc::new(node), slice))
    }
}

impl ParseNodeFunctional for LambdaVariable {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let name = match op {
            Token::LambdaVariable(v) => v.name,
            _ => unreachable!(),
        };
        let signature = match context.get_lambda_var_type(name.clone()) {
            Some(IJType::Function(signature)) => signature,
            Some(_) => return Err(SyntaxError::ExpectedFunction(name.clone()).into()),
            _ => return Err(SyntaxError::UnknownSymbol(name.clone()).into()),
        };

        if let Some(outputs) = needed_outputs {
            if outputs.iter().any(|output| output != &signature.output) {
                return Err(SyntaxError::RequiredOutputsDoNotMatchFunctionOutputs(
                    needed_outputs
                        .unwrap()
                        .iter()
                        .map(|output| format!("{:?}", output))
                        .collect::<Vec<String>>()
                        .join(", "),
                    signature
                        .output
                        .iter()
                        .map(|output| output.to_string())
                        .collect::<Vec<String>>()
                        .join(", "),
                )
                .into());
            }
        }

        let node = Node::new(
            Operation::LambdaVariable(name),
            vec![],
            IJType::Function(signature),
            vec![],
            context.get_increment_id(),
        );
        Ok((vec![Rc::new(node)], slice.move_start(1)?))
    }
}
