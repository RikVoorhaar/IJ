use crate::ast_node::{ASTContext, LineHasSemicolon, Node, TokenSlice, Variable};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

pub fn parse_lines(
    tokens: Vec<Token>,
    context: &mut ASTContext,
) -> Result<Vec<(Rc<Node>, LineHasSemicolon)>> {
    let mut parsed_lines = Vec::new();
    let mut line_tokens = Vec::new();
    for token in tokens {
        if token == Token::Newline || token == Token::Semicolon {
            if line_tokens.is_empty() {
                continue;
            }
            let root = parse_line(std::mem::take(&mut line_tokens), context)?;
            let has_semicolon = match token {
                Token::Semicolon => LineHasSemicolon::Yes,
                _ => LineHasSemicolon::No,
            };
            parsed_lines.push((root, has_semicolon));
        } else {
            line_tokens.push(token);
        }
    }
    if !line_tokens.is_empty() {
        let last_token = line_tokens.last().unwrap();
        let has_semicolon = match last_token {
            Token::Semicolon => LineHasSemicolon::Yes,
            _ => LineHasSemicolon::No,
        };
        parsed_lines.push((
            parse_line(std::mem::take(&mut line_tokens), context)?,
            has_semicolon,
        ));
    }
    Ok(parsed_lines)
}

pub fn parse_line(tokens: Vec<Token>, context: &mut ASTContext) -> Result<Rc<Node>> {
    if let Some(pos) = tokens.iter().position(|t| *t == Token::Semicolon) {
        if pos != tokens.len() - 1 {
            return Err(SyntaxError::UnexpectedSemicolon.into());
        }
        context.set_tokens(tokens[..pos].to_vec());
    } else {
        context.set_tokens(tokens);
    }
    context.line_no += 1;
    parse_ast(context)
}

fn parse_ast(context: &mut ASTContext) -> Result<Rc<Node>> {
    match context.tokens.as_slice() {
        [] => Err(SyntaxError::EmptyInput.into()),
        [Token::Variable, ..] => parse_var_statement(context),
        [Token::Symbol(symbol), rest @ ..] if rest.contains(&Token::Assign) => {
            parse_assign(symbol.name.clone(), context)
        }
        _ => {
            let (node, remainder) = super::next_node(context.full_slice(), context)?;
            if !remainder.is_empty() {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::UnhandledTokens(context.tokens_to_string(remainder)).into(),
                    remainder,
                ));
            }
            Ok(node)
        }
    }
}

fn parse_var_statement(context: &mut ASTContext) -> Result<Rc<Node>> {
    match context.tokens.as_slice() {
        [Token::Variable, Token::Symbol(symbol), Token::TypeDeclaration, rest @ ..] => {
            let (typ, end_of_type) = IJType::parse_tokens(rest)?;
            let rest_slice = context.full_slice().move_start(2)?;
            let remainder_slice = rest_slice.move_start(end_of_type)?;

            if remainder_slice.is_empty() {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::UnhandledTokens(context.tokens_to_string(remainder_slice)).into(),
                    remainder_slice,
                ));
            }
            let var = Variable {
                name: symbol.name.clone(),
                typ: typ.clone(),
            };
            context.insert_variable(var);
            Ok(Rc::new(Node::new(
                Operation::Nothing,
                vec![],
                typ,
                vec![],
                context.get_increment_id(),
            )))
        }
        _ => Err(
            SyntaxError::InvalidVarStatement(context.tokens_to_string(context.full_slice())).into(),
        ),
    }
}

/// Get the nodes expressed on the right hand side of an assignment token.
fn get_variable_expression(context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
    let tokens = context.get_tokens();
    let mut split = tokens.split(|t| t == &Token::Assign);
    let assign_index: usize = split
        .next()
        .ok_or(context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()).into(),
            context.full_slice(),
        ))?
        .len();
    let mut slice = context.full_slice().move_start(assign_index + 1)?;

    let mut expressions = Vec::new();
    while !slice.is_empty() {
        let (expression, rest) = super::next_node(slice, context)?;
        expressions.push(expression);
        slice = rest;
    }
    Ok(expressions)
}

/// Recursively walk down AST. If we reach a leaf node without an operand, add it to the input types. Otherwise concatenate the input types of its operands.
pub fn compute_input_type(node: &Rc<Node>) -> Result<Vec<IJType>> {
    if node.operands.is_empty() {
        return Ok(node.input_types.clone());
    }
    let mut input_types = Vec::new();
    for operand in node.operands.iter() {
        input_types.extend(compute_input_type(operand)?);
    }
    Ok(input_types)
}

/// Parses a variable assignment statement.
pub fn parse_assign(variable_name: String, context: &mut ASTContext) -> Result<Rc<Node>> {
    let expressions = get_variable_expression(context)?;
    if expressions.len() != 1 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::InvalidAssignmentStatement(context.tokens_to_string(context.full_slice()))
                .into(),
            context.full_slice(),
        ));
    }
    let expression = expressions.into_iter().next().unwrap();
    let left_of_assign = context.tokens.split(|t| t == &Token::Assign).next().ok_or(
        context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()).into(),
            context.full_slice(),
        ),
    )?;
    let expected_type = match left_of_assign {
        [Token::Symbol(_), Token::TypeDeclaration, rest @ ..] => {
            Some(IJType::parse_tokens(rest)?.0)
        }
        [Token::Symbol(_)] => None,
        _ => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::InvalidAssignmentStatement(
                    context.tokens_to_string(context.full_slice()),
                )
                .into(),
                context.full_slice(),
            ))
        }
    };

    let input_type = compute_input_type(&expression)?;
    let output_type = expression.output_type.clone();
    let expression_type = match input_type.is_empty() {
        true => output_type.clone(),
        false => IJType::Function(FunctionSignature {
            input: input_type,
            output: vec![output_type.clone()],
        }),
    };

    if let Some(expected_type) = expected_type {
        if expression_type != expected_type {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::TypeError(expression_type.to_string(), expected_type.to_string())
                    .into(),
                context.full_slice(),
            ));
        }
    }
    context.insert_variable(Variable {
        name: variable_name.clone(),
        typ: expression_type.clone(),
    });
    let symbol_node = Rc::new(Node::new(
        Operation::Symbol(variable_name.clone()),
        vec![],
        expression_type.clone(),
        Vec::new(),
        context.get_increment_id(),
    ));
    let operand_node = expression.clone();
    let assign_node = Rc::new(Node::new(
        Operation::Assign,
        vec![expression_type, output_type],
        IJType::Void,
        vec![symbol_node, operand_node],
        context.get_increment_id(),
    ));
    Ok(assign_node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{is_specific_syntax_error, parse_str_no_context};

    #[test]
    fn test_empty_input() {
        let result = parse_str_no_context("");
        assert!(result.is_err());
        assert!(is_specific_syntax_error(
            &result.unwrap_err(),
            &SyntaxError::EmptyInput
        ));
    }

    #[test]
    fn test_variable_declaration_tensor() {
        let result = parse_str_no_context("var a: T");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Tensor,
            name: "a".to_string(),
        };
        let actual_var = context.symbols.get("a").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_variable_declaration_scalar() {
        let result = parse_str_no_context("var b: S");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Scalar,
            name: "b".to_string(),
        };
        let actual_var = context.symbols.get("b").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_function_declaration_tensor_to_tensor() {
        let result = parse_str_no_context("var add: Fn(T,T -> T)");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor, IJType::Tensor],
                output: vec![IJType::Tensor],
            }),
            name: "add".to_string(),
        };
        let actual_var = context.symbols.get("add").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_function_declaration_scalar_to_tensor() {
        let result = parse_str_no_context("var scale: Fn(S -> T)");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Scalar],
                output: vec![IJType::Tensor],
            }),
            name: "scale".to_string(),
        };
        let actual_var = context.symbols.get("scale").unwrap();
        assert_eq!(actual_var, &expected_var);
    }
}
