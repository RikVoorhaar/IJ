use super::{next_node_functional, ParseNode};
use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct AsFunction;

impl ParseNode for AsFunction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (nodes, rest) = next_node_functional(slice, context, None)?;

        if !rest.is_empty() && context.get_token_at_index(rest.start)? == &Token::TypeDeclaration {
            let rest = rest.move_start(1)?;
            let (fn_type, type_end) = IJType::parse_tokens(&context.get_tokens_from_slice(rest))?;
            let rest = rest.move_start(type_end)?;
            if let IJType::Function(_) = fn_type {
            } else {
                return Err(SyntaxError::ExpectedFunction(fn_type.to_string()).into());
            }

            let node = nodes
                .iter()
                .find(|n| n.output_type == fn_type)
                .ok_or_else(|| {
                    SyntaxError::TypeError(
                        fn_type.to_string(),
                        nodes
                            .iter()
                            .map(|n| n.output_type.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    )
                })?;

            return Ok((node.clone(), rest));
        }
        if nodes.len() != 1 {
            Err(SyntaxError::FunctionTypeAmbiguous(
                context.token_slice_to_string(slice),
                nodes
                    .iter()
                    .map(|n| n.output_type.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            )
            .into())
        } else {
            Ok((nodes[0].clone(), rest))
        }
    }
}
