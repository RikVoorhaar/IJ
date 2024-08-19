use crate::operations::Operation;
use crate::parser::utils::{comma_separate, find_matching_parenthesis};
use crate::parser::{next_node, next_node_functional, ASTContext, LambdaVariable, Node, ParseNode};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
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
    let mut slice = context.full_slice().move_start(1)?;

    if context.get_token_at_index(slice.start)? == &Token::LParen {
        slice = slice.move_start(1)?;
        let close_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)?;
        let paren_slice = slice.move_end(close_index)?;
        slice = slice.move_start_saturating(close_index + 1);
        let arg_slices = comma_separate(paren_slice, context)?;
        for arg_slice in arg_slices {
            let op = context.get_token_at_index(arg_slice.start)?;
            match op {
                Token::LambdaVariable(_) => {
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
    let output_type = match context.get_token_at_index(slice.start)? {
        Token::TypeDeclaration => {
            let slice = slice.move_start(1)?;
            let (output_type, num_tokens) =
                IJType::parse_tokens(&context.get_tokens_from_slice(slice))?;
            if !slice.move_start(num_tokens)?.is_empty() {
                return Err(SyntaxError::InvalidAssignmentStatement(format!(
                    "Got unparsed tokens '{}' after type declaration",
                    context.token_slice_to_string(slice)
                ))
                .into());
            }
            Some(output_type)
        }
        Token::Arrow => {
            let slice = slice.move_start(1)?;
            let parts = comma_separate(slice, context)?;

            let args_types = args.iter().map(|arg| arg.output_type.clone()).collect();
            let parts_types = parts
                .into_iter()
                .map(|part| IJType::parse_tokens(&context.get_tokens_from_slice(part)).map(|x| x.0))
                .collect::<Result<Vec<_>, _>>()?;
            Some(IJType::Function(FunctionSignature::new(
                args_types,
                parts_types,
            )))
        }
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::test_utils::parse_str;
    use crate::parser::ASTContext;
    use crate::tokens::lexer;

    #[test]
    fn test_parse_assign_lhs_simple() -> Result<()> {
        let tokens = lexer("g")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert!(args.is_empty());
        assert_eq!(output_type, None);

        Ok(())
    }

    #[test]
    fn test_parse_assign_type() -> Result<()> {
        let tokens = lexer("g: T")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert!(args.is_empty());
        assert_eq!(output_type, Some(IJType::Tensor));

        Ok(())
    }

    #[test]
    fn test_parse_assign_args() -> Result<()> {
        let tokens = lexer("g($x, $y)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        println!("{:?}", args);
        assert_eq!(output_type, None);

        Ok(())
    }

    #[test]
    fn test_parse_assign_args_type() -> Result<()> {
        let tokens = lexer("g($x: T, $y: S)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor);
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Scalar);
        assert_eq!(output_type, None);

        Ok(())
    }

    #[test]
    fn test_parse_assign_args_type_output_arrow() -> Result<()> {
        let tokens = lexer("g($x: T, $y: S) -> T")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor);
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Scalar);
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor]
            )))
        );

        let tokens = lexer("g($x: T, $y: S) -> T,S")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor);
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Scalar);
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor, IJType::Scalar]
            )))
        );

        Ok(())
    }

    #[test]
    fn test_parse_assign_args_type_output_declaration() -> Result<()> {
        let tokens = lexer("g($x: T, $y: S): Fn(T,S->T)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor);
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Scalar);
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor]
            )))
        );

        let tokens = lexer("g($x: T, $y: S): Fn(T,S->T,S)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, &tokens)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor);
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Scalar);
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor, IJType::Scalar]
            )))
        );

        Ok(())
    }
}
