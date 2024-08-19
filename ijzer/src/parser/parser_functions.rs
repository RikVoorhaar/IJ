use crate::ast_node::{ASTContext, LineHasSemicolon, Node, Variable};
use crate::operations::Operation;
use crate::parser::assign::parse_assign;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
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
        [Token::Symbol(_), rest @ ..] if rest.contains(&Token::Assign) => parse_assign(context),
        _ => {
            let (node, remainder) = super::next_node(context.full_slice(), context)?;
            if !remainder.is_empty() {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::UnhandledTokens(context.token_slice_to_string(remainder)).into(),
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
                    SyntaxError::UnhandledTokens(context.token_slice_to_string(remainder_slice))
                        .into(),
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
        _ => Err(SyntaxError::InvalidVarStatement(
            context.token_slice_to_string(context.full_slice()),
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{is_specific_syntax_error, parse_str_no_context};
    use crate::types::FunctionSignature;

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
