use crate::lexer::{Number, Token};
use anyhow::{anyhow, Context, Result};
use std::fmt::Debug;
use thiserror::Error;

pub trait TokenImpl {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])>;
}

pub fn next_node(tokens: &[Token]) -> Result<(Node, &[Token])> {
    let (op, rest) = tokens.split_first().ok_or(SyntaxError::EmptyStream)?;
    match op {
        Token::Plus => BinaryOp::_next_node(op.clone(), rest),
        Token::Multiplication => BinaryOp::_next_node(op.clone(), rest),
        Token::Number(_) => Number::_next_node(op.clone(), rest),
        Token::Minus => MinusOp::_next_node(op.clone(), rest),
        Token::LParen => LParen::_next_node(op.clone(), rest),
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    }
}

#[derive(Error, Debug)]
pub enum SyntaxError {
    #[error("Token {0:?} excpected {1} arguments but got {2}.")]
    InsufficientArguments(Token, usize, usize),

    #[error("Next node called on an empty Token stream")]
    EmptyStream,

    #[error("Unmatched parenthesis")]
    UnmatchedParenthesis,

    #[error("Unresolved group {0}")]
    UnresolvedGroup(String),
}

fn _find_matching_parenthesis(tokens: &[Token]) -> Result<usize> {
    let mut depth = 1;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => (),
        }
    }
    Err(SyntaxError::UnmatchedParenthesis.into())
}

#[derive(Debug, PartialEq)]
struct LParen;
impl TokenImpl for LParen {
    fn _next_node(_op: Token, tokens: &[Token]) -> Result<(Node, &[Token])> {
        let rparen_index = _find_matching_parenthesis(tokens)?;
        let new_tokens = &tokens[..rparen_index];
        let rest = &tokens[rparen_index + 1..];

        let (node, remainder) = next_node(new_tokens)?;
        if !remainder.is_empty() {
            return Err(SyntaxError::UnresolvedGroup(format!("{:?}", new_tokens)).into());
        }
        Ok((node, rest))
    }
}

#[derive(Debug, PartialEq)]
struct BinaryOp;
impl TokenImpl for BinaryOp {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])> {
        let (lhs, rest) = next_node(tokens)
            .with_context(|| SyntaxError::InsufficientArguments(op.clone(), 2, 0))?;
        let (rhs, rest) = next_node(rest)
            .with_context(|| SyntaxError::InsufficientArguments(op.clone(), 2, 1))?;
        Ok((
            Node {
                op,
                output_arity: 1,
                operands: vec![lhs, rhs],
                functional_operands: vec![],
            },
            rest,
        ))
    }
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])> {
        let (lhs, rest) = next_node(tokens)
            .with_context(|| SyntaxError::InsufficientArguments(op.clone(), 1, 0))?;
        if rest.is_empty() {
            Ok((
                Node {
                    op,
                    output_arity: 1,
                    operands: vec![lhs],
                    functional_operands: vec![],
                },
                &[],
            ))
        } else {
            let (rhs, rest) = next_node(rest)
                .with_context(|| SyntaxError::InsufficientArguments(op.clone(), 2, 1))?;
            Ok((
                Node {
                    op,
                    output_arity: 1,
                    operands: vec![lhs, rhs],
                    functional_operands: vec![],
                },
                rest,
            ))
        }
    }
}

impl TokenImpl for Number {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])> {
        Ok((
            Node {
                op,
                output_arity: 1,
                operands: vec![],
                functional_operands: vec![],
            },
            tokens,
        ))
    }
}

impl Debug for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

pub struct Node {
    op: Token,
    output_arity: usize,
    operands: Vec<Node>,
    functional_operands: Vec<Node>,
}

impl Node {
    fn fmt_nested(&self, f: &mut std::fmt::Formatter<'_>, nesting: usize) -> std::fmt::Result {
        write!(
            f,
            "{}{:?}->({})",
            _tab_nested(nesting),
            self.op,
            self.output_arity
        )?;
        if !&self.operands.is_empty() {
            write!(f, "[")?;

            for operand in &self.operands {
                writeln!(f)?;
                operand.fmt_nested(f, nesting + 1)?;
            }
            write!(f, "\n{}]", _tab_nested(nesting))?;
        }

        if !&self.functional_operands.is_empty() {
            write!(f, "<")?;
            for operand in &self.functional_operands {
                operand.fmt_nested(f, nesting + 1)?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

fn _tab_nested(nesting: usize) -> String {
    "    ".repeat(nesting)
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_nested(f, 0)
    }
}
