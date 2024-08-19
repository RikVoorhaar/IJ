use crate::ast_node::Variable;
use crate::operations::Operation;
use crate::parser::lambda_variable::parse_lambda_assign_lhs;
use crate::parser::utils::{comma_separate, find_matching_parenthesis};
use crate::parser::{next_node, ASTContext, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

/// Parses the left hand side of an assignment statement.
/// Returns a Symbol node for the assignment target, a list of strings for each argument, and an optional type in case it is annotated.
fn parse_assign_lhs(
    context: &mut ASTContext,
    slice: TokenSlice,
) -> Result<(String, Vec<Rc<Node>>, Option<IJType>)> {
    let tokens = context.get_tokens_from_slice(slice);
    let symbol_name = if let [Token::Symbol(symbol_token), ..] = tokens.as_slice() {
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
    if slice.is_empty() {
        return Ok((symbol_name, args, None));
    }

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
                    let (node, rest) = parse_lambda_assign_lhs(
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
    if slice.is_empty() {
        return Ok((symbol_name, args, None));
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
        (
            context.full_slice().move_end(pos)?,
            context.full_slice().move_start(pos + 1)?,
        )
    } else {
        return Err(SyntaxError::InvalidAssignmentStatement(
            context.token_slice_to_string(context.full_slice()),
        )
        .into());
    };
    let (symbol_name, args, output_type) = parse_assign_lhs(context, lhs)?;
    let (rhs_node, rest) = next_node(rhs, context)?;
    if !rest.is_empty() {
        return Err(SyntaxError::InvalidAssignmentStatement(format!(
            "Got unparsed tokens '{}' after assignment",
            context.token_slice_to_string(rest)
        ))
        .into());
    }
    let symbol_type = if args.is_empty() {
        rhs_node.output_type.clone()
    } else {
        let args_types = args.iter().map(|arg| arg.output_type.clone()).collect();
        IJType::Function(FunctionSignature::new(
            args_types,
            vec![rhs_node.output_type.clone()],
        ))
    };
    if let Some(output_type) = output_type {
        if output_type != symbol_type {
            return Err(SyntaxError::InvalidAssignmentStatement(format!(
                "LHS has type '{}' but RHS has type '{}'",
                output_type, symbol_type
            ))
            .into());
        }
    }
    let symbol_node = Rc::new(Node::new(
        Operation::Symbol(symbol_name.clone()),
        vec![],
        symbol_type.clone(),
        vec![],
        context.get_increment_id(),
    ));
    context.insert_variable(Variable {
        name: symbol_name,
        typ: symbol_type.clone(),
    });
    let mut operands = vec![symbol_node, rhs_node.clone()];
    operands.extend(args.iter().cloned());

    Ok(Rc::new(Node::new(
        Operation::Assign,
        vec![symbol_type, rhs_node.output_type.clone()],
        IJType::Void,
        operands,
        context.get_increment_id(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ASTContext;
    use crate::tokens::lexer;

    #[test]
    fn test_parse_assign_lhs_simple() -> Result<()> {
        let tokens = lexer("g")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
        assert_eq!(symbol_name, "g");
        assert!(args.is_empty());
        assert_eq!(output_type, None);

        Ok(())
    }

    #[test]
    fn test_parse_assign_type() -> Result<()> {
        let tokens = lexer("g: T")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
        assert_eq!(symbol_name, "g");
        assert!(args.is_empty());
        assert_eq!(output_type, Some(IJType::Tensor));

        Ok(())
    }

    #[test]
    fn test_parse_assign_args() -> Result<()> {
        let tokens = lexer("g($x, $y)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
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
    fn test_assign_simple_tensor() -> Result<()> {
        let tokens = lexer("g = [1]")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(node.operands[0].output_type, IJType::Tensor);
        assert_eq!(node.operands[1].op, Operation::Array("1".to_string()));
        assert_eq!(node.operands[1].output_type, IJType::Tensor);
        println!("{:?}", node);
        Ok(())
    }

    #[test]
    fn test_assign_simple_scalar() -> Result<()> {
        let tokens = lexer("g = 1")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(node.operands[0].output_type, IJType::Scalar);
        assert_eq!(node.operands[1].output_type, IJType::Scalar);
        Ok(())
    }

    #[test]
    fn test_assign_simple_function() -> Result<()> {
        let tokens = lexer("g($x) = $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Tensor]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Tensor);
        assert_eq!(node.operands[2].output_type, IJType::Tensor);
        Ok(())
    }

    #[test]
    fn test_assign_simple_function_scalar() -> Result<()> {
        let tokens = lexer("g($x: S) = $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar],
                vec![IJType::Scalar]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Scalar);
        assert_eq!(node.operands[2].output_type, IJType::Scalar);
        Ok(())
    }

    #[test]
    fn test_assign_simple_function_tensor() -> Result<()> {
        let tokens = lexer("g($x) = /+ $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Scalar]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Scalar);
        assert_eq!(node.operands[2].output_type, IJType::Tensor);
        Ok(())
    }

    #[test]
    fn test_assign_simple_reuse_var() -> Result<()> {
        let tokens = lexer("g($x) = + $x $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Tensor]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Tensor);
        assert_eq!(node.operands[2].output_type, IJType::Tensor);
        Ok(())
    }

    #[test]
    fn test_assign_simple_reuse_var_annotated() -> Result<()> {
        let tokens = lexer("g($x:T) -> S = /+ + $x $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Scalar]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Scalar);
        assert_eq!(node.operands[2].output_type, IJType::Tensor);
        Ok(())
    }

    #[test]
    fn test_assign_multiple_args() -> Result<()> {
        let tokens = lexer("g($x:T, $y:S) -> S = /+ + $x $y")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 4);
        assert_eq!(node.operands[0].op, Operation::Symbol("g".to_string()));
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Scalar]
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Scalar);
        assert_eq!(node.operands[2].output_type, IJType::Tensor);
        assert_eq!(node.operands[3].output_type, IJType::Scalar);
        Ok(())
    }
}
