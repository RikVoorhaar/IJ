use crate::operations::Operation;
use crate::parser::utils::{comma_separate, find_matching_parenthesis};
use crate::parser::{next_node, next_node_functional, ASTContext, LambdaVariable, Node, ParseNode};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::{anyhow, Result};
use std::rc::Rc;

/// Parses the left hand side of an assignment statement.
/// Returns a Symbol node for the assignment target, a list of strings for each argument, and an optional type in case it is annotated.
fn parse_assign_lhs(
    context: &mut ASTContext,
    tokens: &[Token],
) -> Result<(String, Vec<Rc<Node>>, Option<IJType>)> {
    let symbol_name = if let [Token::Symbol(symbol_token), ..] = tokens {
        symbol_token.name.clone()
    } else {
        return Err(SyntaxError::InvalidAssignmentStatement(format!(
            "First token should be a symbol but got '{}'",
            context.token_slice_to_string(context.full_slice())
        ))
        .into());
    };
    let mut args = vec![];
    let mut slice = context.full_slice().move_start(2)?;

    if context.get_token_at_index(slice.start)? == &Token::LParen {
        let close_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)?;
        let paren_slice = slice.move_end(close_index)?;
        slice = paren_slice.move_start_saturating(close_index + 1);
        let arg_slices = comma_separate(paren_slice, context)?;
        for arg_slice in arg_slices {
            let op = context.get_token_at_index(arg_slice.start)?;
            match op {
                Token::LambdaVariable(arg_name) => {
                    let (node, rest) = LambdaVariable::next_node(
                        op.clone(),
                        arg_slice.move_start_saturating(1),
                        context,
                    )?;
                    if !rest.is_empty() {
                        return Err(SyntaxError::InvalidAssignmentStatement(
                            format!("Expected comma separated list of lambda variables but one of the slices was this: '{}'", context.token_slice_to_string(rest))
                        )
                        .into());
                    }
                    args.push(node);
                }
                _ => {
                    return Err(SyntaxError::InvalidAssignmentStatement(
                        format!("Expected comma separated list of lambda variables but one of the slices was this: '{}'", context.token_slice_to_string(arg_slice))
                    )
                    .into());
                }
            }
        }
    }
    for arg in args.iter() {
        let arg_name = match &arg.op {
            Operation::LambdaVariable(name) => name.clone(),
            _ => unreachable!(),
        };
        context.insert_lambda_variable(arg_name, arg.output_type.clone());
    }
    let output_type = if context.get_token_at_index(slice.start)? == &Token::TypeDeclaration {
        let (output_type, num_tokens) =
            IJType::parse_tokens(&context.get_tokens_from_slice(slice.move_start(1)?))?;
        if slice.move_start(num_tokens)?.is_empty() {
            return Err(SyntaxError::InvalidAssignmentStatement(format!(
                "Got invalid tokens after type declaration: '{}'",
                context.token_slice_to_string(slice)
            ))
            .into());
        }
        Some(output_type)
    } else {
        None
    };
    Ok((symbol_name, args, output_type))
}

pub fn parse_assign(context: &mut ASTContext) -> Result<Rc<Node>> {
    let tokens = context.get_tokens();

    let (lhs, rhs) = if let Some(pos) = tokens.iter().position(|t| matches!(t, Token::Assign)) {
        let (lhs, rhs) = tokens.split_at(pos);
        let rhs = &rhs[1..]; // Skip the `Assign` token
        (lhs.to_vec(), rhs.to_vec())
    } else {
        return Err(SyntaxError::InvalidAssignmentStatement(
            context.token_slice_to_string(context.full_slice()),
        )
        .into());
    };
    let (symbol_name, args, output_type) = parse_assign_lhs(context, &lhs)?;

    Err(anyhow::anyhow!("Not implemented"))
}
