//! Parses the assignment operation `=`.
//! 
//! This operation is used to assign a value to a variable.
//! 
//! Example: `x = 1` assigns the value `1` to the variable `x`.
//! 
//! The parser supports both single variable assignments and group assignments. Group assignments assign a
//! list of variables to a list of values. For example: `(x,y) = (1,2)` assigns the values `1` to `x` and `2`
//! to `y`.
//! 
//! The parser also supports type annotations for the assigned variables. For example: `x: i32 = 1` assigns
//! the value `1` to the variable `x` with type `i32`.
//! 
//! The parser also supports lambda assignments. For
//! example: `f($x,$y) = +$x $y` assigns the lambda `(x,y) => x+y` to the variable `f`.
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
    let mut slice = slice.move_start(1)?;
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

            let args_types = args.iter().map(|arg| arg.output_type.clone()).collect();
            let (output_type, num_tokens) =
                IJType::parse_tokens(&context.get_tokens_from_slice(slice))?;
            let rest = slice.move_start(num_tokens)?;
            if !rest.is_empty() {
                return Err(SyntaxError::InvalidAssignmentStatement(format!(
                    "Got unparsed tokens '{}' after type declaration",
                    context.token_slice_to_string(rest)
                ))
                .into());
            }
            Some(IJType::Function(FunctionSignature::new(
                args_types,
                output_type,
            )))
        }
        _ => None,
    };
    Ok((symbol_name, args, output_type))
}

/// Special case for group assignment. The LHS must be a comma separated list of symbols and the RHS must return a group.
pub fn parse_group_assign(
    context: &mut ASTContext,
    lhs: TokenSlice,
    rhs: TokenSlice,
) -> Result<Rc<Node>> {
    let lhs_tokens = context.get_tokens_from_slice(lhs);
    let end_index =
        find_matching_parenthesis(context, lhs.move_start(1)?, &Token::LParen, &Token::RParen)?;
    if end_index != lhs_tokens.len() - 2 {
        return Err(SyntaxError::InvalidAssignmentStatement(format!(
            "Expected closing parenthesis but got '{}'",
            context.token_slice_to_string(lhs.move_end(end_index)?)
        ))
        .into());
    }
    let lhs_slices = comma_separate(lhs.move_start(1)?.move_end(end_index)?, context)?;
    let mut lhs_symbol_names = Vec::new();
    let mut lhs_symbol_types = Vec::new();
    for slice in &lhs_slices {
        match context.get_token_at_index(slice.start)? {
            Token::Symbol(symbol_token) => {
                lhs_symbol_names.push(symbol_token.name.clone());
                if !slice.move_start(1)?.is_empty() {
                    let slice = slice.move_start(1)?;
                    match context.get_token_at_index(slice.start)? {
                        Token::TypeDeclaration => {
                            let (output_type, num_tokens) = IJType::parse_tokens(
                                &context.get_tokens_from_slice(slice.move_start(1)?),
                            )?;
                            lhs_symbol_types.push(Some(output_type));
                            let rest = slice.move_start(num_tokens + 1)?;
                            if !rest.is_empty() {
                                return Err(SyntaxError::InvalidAssignmentStatement(format!(
                                    "Got unparsed tokens '{}' after type declaration",
                                    context.token_slice_to_string(rest)
                                ))
                                .into());
                            }
                        }
                        _ => {
                            return Err(SyntaxError::InvalidAssignmentStatement(format!(
                                "Expected type declaration after symbol but got '{}'",
                                context.token_slice_to_string(slice)
                            ))
                            .into());
                        }
                    }
                } else {
                    lhs_symbol_types.push(None);
                }
            }
            _ => {
                return Err(SyntaxError::InvalidAssignmentStatement(format!(
                    "Expected symbol but got '{}'",
                    context.token_slice_to_string(*slice)
                ))
                .into())
            }
        }
    }
    let (rhs_node, rest) = next_node(rhs, context)?;
    if !rest.is_empty() {
        return Err(SyntaxError::InvalidAssignmentStatement(format!(
            "Got unparsed tokens '{}' after assignment",
            context.token_slice_to_string(rest)
        ))
        .into());
    }
    if let IJType::Group(group_types) = &rhs_node.output_type {
        if group_types.len() != lhs_symbol_names.len() {
            return Err(SyntaxError::IncorrectNumberOfVariablesInAssignment(
                lhs_symbol_names.len(),
                group_types.len(),
            )
            .into());
        }
        let mut symbol_nodes = vec![];
        for ((lhs_name, lhs_type), rhs_type) in lhs_symbol_names
            .iter()
            .zip(lhs_symbol_types.iter())
            .zip(group_types.iter())
        {
            context.insert_variable(Variable {
                name: lhs_name.clone(),
                typ: rhs_type.clone(),
            });
            if let Some(lhs_type) = lhs_type {
                if !lhs_type.type_match(rhs_type) {
                    return Err(SyntaxError::InvalidAssignmentStatement(format!(
                        "LHS type '{}' does not match RHS type '{}'",
                        lhs_type, rhs_type
                    ))
                    .into());
                }
            }
            symbol_nodes.push(Rc::new(Node::new(
                Operation::AssignSymbol(lhs_name.clone()),
                vec![],
                rhs_type.clone(),
                vec![],
                context,
            )?));
        }
        let group_node = Rc::new(Node::new(
            Operation::Group,
            vec![],
            IJType::Group(group_types.clone()),
            symbol_nodes,
            context,
        )?);
        let assign_node = Rc::new(Node::new(
            Operation::Assign,
            vec![
                IJType::Group(group_types.clone()),
                rhs_node.output_type.clone(),
            ],
            IJType::Void,
            vec![group_node, rhs_node],
            context,
        )?);
        Ok(assign_node)
    } else {
        Err(SyntaxError::ExpectedGroupOutput(rhs_node.output_type.clone().to_string()).into())
    }
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
    if context.get_token_at_index(lhs.start)? == &Token::LParen {
        return parse_group_assign(context, lhs, rhs);
    }
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
            rhs_node.output_type.clone(),
        ))
    };
    if let Some(output_type) = output_type {
        if !output_type.type_match(&symbol_type) {
            return Err(SyntaxError::InvalidAssignmentStatement(format!(
                "LHS has type '{}' but RHS has type '{}'",
                output_type, symbol_type
            ))
            .into());
        }
    }
    let symbol_node = Rc::new(Node::new(
        Operation::AssignSymbol(symbol_name.clone()),
        vec![],
        symbol_type.clone(),
        vec![],
        context,
    )?);
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
        context,
    )?))
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
        assert_eq!(output_type, Some(IJType::Tensor(None)));

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
        let tokens = lexer("g($x: T, $y: N)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor(None));
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Number(None));
        assert_eq!(output_type, None);

        Ok(())
    }

    #[test]
    fn test_parse_assign_args_type_output_arrow() -> Result<()> {
        let tokens = lexer("g($x: T, $y: N) -> T")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor(None));
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Number(None));
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Number(None)],
                IJType::Tensor(None)
            )))
        );

        Ok(())
    }

    #[test]
    fn test_parse_assign_args_type_output_declaration() -> Result<()> {
        let tokens = lexer("g($x: T, $y: N): Fn(T,N->T)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let slice = context.full_slice();
        let (symbol_name, args, output_type) = parse_assign_lhs(&mut context, slice)?;
        assert_eq!(symbol_name, "g");
        assert_eq!(args.len(), 2);
        let arg0 = args[0].clone();
        assert_eq!(arg0.output_type, IJType::Tensor(None));
        let arg1 = args[1].clone();
        assert_eq!(arg1.output_type, IJType::Number(None));
        assert_eq!(
            output_type,
            Some(IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Number(None)],
                IJType::Tensor(None)
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
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(node.operands[0].output_type, IJType::Tensor(None));
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.operands[1].output_type, IJType::Tensor(None));
        println!("{:?}", node);
        Ok(())
    }
    #[test]
    fn test_assign_simple_tensor_type() -> Result<()> {
        let tokens = lexer("g: T = [1]")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(node.operands[0].output_type, IJType::Tensor(None));
        assert_eq!(node.operands[1].op, Operation::Array);
        assert_eq!(node.operands[1].output_type, IJType::Tensor(None));
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
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(node.operands[0].output_type, IJType::Number(None));
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        Ok(())
    }

    #[test]
    fn test_assign_simple_scalar_with_number_type() -> Result<()> {
        let tokens = lexer("g: N<i64> = 1<i64>")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(
            node.operands[0].output_type,
            IJType::Number(Some("i64".to_string()))
        );
        assert_eq!(
            node.operands[1].output_type,
            IJType::Number(Some("i64".to_string()))
        );
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
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Tensor(None));
        assert_eq!(node.operands[2].output_type, IJType::Tensor(None));
        Ok(())
    }

    #[test]
    fn test_assign_simple_function_scalar() -> Result<()> {
        let tokens = lexer("g($x: N) = $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None)],
                IJType::Number(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        assert_eq!(node.operands[2].output_type, IJType::Number(None));
        Ok(())
    }
    #[test]
    fn test_assign_simple_function_scalar_with_number_type() -> Result<()> {
        let tokens = lexer("g($x: N<i64>) = $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(Some("i64".to_string()))],
                IJType::Number(Some("i64".to_string()))
            ))
        );
        assert_eq!(
            node.operands[1].output_type,
            IJType::Number(Some("i64".to_string()))
        );
        assert_eq!(
            node.operands[2].output_type,
            IJType::Number(Some("i64".to_string()))
        );
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
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Number(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        assert_eq!(node.operands[2].output_type, IJType::Tensor(None));
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
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Tensor(None));
        assert_eq!(node.operands[2].output_type, IJType::Tensor(None));
        Ok(())
    }

    #[test]
    fn test_assign_simple_reuse_var_annotated() -> Result<()> {
        let tokens = lexer("g($x:T) -> N = /+ + $x $x")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Number(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        assert_eq!(node.operands[2].output_type, IJType::Tensor(None));
        Ok(())
    }

    #[test]
    fn test_assign_multiple_args() -> Result<()> {
        let tokens = lexer("g($x:T, $y:N) -> N = /+ + $x $y")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 4);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Number(None)],
                IJType::Number(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        assert_eq!(node.operands[2].output_type, IJType::Tensor(None));
        assert_eq!(node.operands[3].output_type, IJType::Number(None));
        Ok(())
    }

    #[test]
    fn test_assign_functional_args() -> Result<()> {
        let tokens = lexer("g($x:Fn(N,N->N)) -> N = /$x [1]")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 3);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Function(FunctionSignature::new(
                    vec![IJType::Number(None), IJType::Number(None)],
                    IJType::Number(None),
                ))],
                IJType::Number(None)
            ))
        );
        assert_eq!(node.operands[1].output_type, IJType::Number(None));
        assert_eq!(
            node.operands[2].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None), IJType::Number(None)],
                IJType::Number(None),
            ))
        );
        Ok(())
    }

    #[test]
    fn test_assign_functional_args_composition() -> Result<()> {
        let tokens = lexer("g($x:Fn(N->T), $y:Fn(T->N)) -> Fn(T->T) = ~@($x,$y)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let node = parse_assign(&mut context)?;
        println!("{:?}", node);
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.operands.len(), 4);
        assert_eq!(
            node.operands[0].op,
            Operation::AssignSymbol("g".to_string())
        );
        assert_eq!(
            node.operands[0].output_type,
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Function(FunctionSignature::new(
                        vec![IJType::Number(None)],
                        IJType::Tensor(None)
                    )),
                    IJType::Function(FunctionSignature::new(
                        vec![IJType::Tensor(None)],
                        IJType::Number(None)
                    ))
                ],
                IJType::Function(FunctionSignature::new(
                    vec![IJType::Tensor(None)],
                    IJType::Tensor(None)
                ))
            ))
        );
        assert_eq!(
            node.operands[1].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Tensor(None),
            ))
        );
        assert_eq!(
            node.operands[2].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(None)],
                IJType::Tensor(None),
            ))
        );
        assert_eq!(
            node.operands[3].output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Number(None),
            ))
        );
        Ok(())
    }

    #[test]
    fn test_assign_group() -> Result<()> {
        let tokens = lexer("(x,y) = (1,2)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let assign_node = parse_assign(&mut context)?;
        let node0 = assign_node.operands[0].clone();
        assert_eq!(
            node0.output_type,
            IJType::Group(vec![IJType::Number(None), IJType::Number(None)])
        );
        assert_eq!(node0.op, Operation::Group);
        assert_eq!(node0.operands.len(), 2);

        let tokens = lexer("(x: N<a>, y: N) = (1<a>,2<b>)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let assign_node = parse_assign(&mut context)?;
        let node0 = assign_node.operands[0].clone();
        assert_eq!(
            node0.output_type,
            IJType::Group(vec![
                IJType::Number(Some("a".to_string())),
                IJType::Number(Some("b".to_string()))
            ])
        );
        assert_eq!(node0.op, Operation::Group);
        assert_eq!(node0.operands.len(), 2);

        let tokens = lexer("(x: N<a>, y: N) = (1<a>,2)")?;
        let mut context = ASTContext::from_tokens(tokens.clone());
        let assign_node = parse_assign(&mut context)?;
        let node0 = assign_node.operands[0].clone();
        assert_eq!(
            node0.output_type,
            IJType::Group(vec![
                IJType::Number(Some("a".to_string())),
                IJType::Number(None)
            ])
        );
        assert_eq!(node0.op, Operation::Group);
        assert_eq!(node0.operands.len(), 2);
        Ok(())
    }
}
