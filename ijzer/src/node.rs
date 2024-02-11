use crate::lexer::{Number, Token};
use crate::operations::Operation;
use anyhow::{anyhow, Context, Result};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Clone)]
pub struct Node {
    op: Operation,
    output_arity: usize,
    operands: Vec<Node>,
    functional_operands: Vec<Node>,
}

pub trait TokenImpl {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])>;
}

pub fn next_node(tokens: &[Token]) -> Result<(Node, &[Token])> {
    let (op, rest) = tokens.split_first().ok_or(SyntaxError::EmptyStream)?;
    let out = match op {
        Token::Plus => _next_node_normal_op(Operation::Add, 2, 2, 1, rest),
        Token::Multiplication => _next_node_normal_op(Operation::Multiply, 2, 2, 1, rest),
        Token::Number(_) => Number::_next_node(op.clone(), rest),
        Token::Minus => MinusOp::_next_node(op.clone(), rest),
        Token::LParen => LParen::_next_node(op.clone(), rest),
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    };

    out
}

fn _next_node_normal_op(
    op: Operation,
    min_arity: usize,
    max_arity: usize,
    output_arity: usize,
    tokens: &[Token],
) -> Result<(Node, &[Token])> {
    let mut total_arity = 0;
    let mut overflow = 0;
    let mut operands = Vec::new();
    let mut rest = tokens;
    while !rest.is_empty() {
        let (node, new_rest) = next_node(rest)?;
        if total_arity + node.output_arity > max_arity {
            overflow = node.output_arity;
            break;
        }
        total_arity += node.output_arity;
        operands.push(node);
        rest = new_rest;
    }
    if total_arity < min_arity {
        return Err(
            SyntaxError::InsufficientArguments(op, min_arity, total_arity + overflow).into(),
        );
    }

    Ok((
        Node {
            op,
            output_arity,
            operands,
            functional_operands: Vec::new(),
        },
        rest,
    ))
}

#[derive(Error, Debug)]
pub enum SyntaxError {
    #[error("Operation {0:?} excpected {1} arguments but got {2}.")]
    InsufficientArguments(Operation, usize, usize),

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
        let mut rest = &tokens[..rparen_index];
        let remainder = &tokens[rparen_index + 1..];

        // let (node, remainder) = next_node(rest)?;
        // if !remainder.is_empty() {
        //     return Err(SyntaxError::UnresolvedGroup(format!("{:?}", rest)).into());
        // }
        let mut operands = Vec::new();
        while !rest.is_empty() {
            let (node, new_rest) = next_node(rest)?;
            rest = new_rest;
            operands.push(node);
        }

        match operands.len() {
            0 => Err(SyntaxError::EmptyStream.into()),
            1 => Ok((operands[0].clone(), remainder)),
            _ => Ok((
                Node {
                    op: Operation::Group,
                    output_arity: operands.iter().map(|n| n.output_arity).sum(),
                    operands,
                    functional_operands: Vec::new(),
                },
                remainder,
            )),
        }
    }
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node(_: Token, tokens: &[Token]) -> Result<(Node, &[Token])> {
        let (lhs, rest) = next_node(tokens)
            .with_context(|| SyntaxError::InsufficientArguments(Operation::Negate, 1, 0))?;
        if rest.is_empty() {
            Ok((
                Node {
                    op: Operation::Negate,
                    output_arity: 1,
                    operands: vec![lhs],
                    functional_operands: vec![],
                },
                &[],
            ))
        } else {
            let (rhs, rest) = next_node(rest)
                .with_context(|| SyntaxError::InsufficientArguments(Operation::Subtract, 2, 1))?;
            Ok((
                Node {
                    op: Operation::Subtract,
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
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };

        Ok((
            Node {
                op: node_op,
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
