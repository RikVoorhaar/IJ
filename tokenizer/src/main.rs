use anyhow::{anyhow, Context, Result};
use logos::Logos;
use std::{fmt::Debug, str::FromStr};
use thiserror::Error;

#[derive(PartialEq, Clone)]
struct Number {
    value: u64,
}
impl FromStr for Number {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let value = s.parse()?;
        Ok(Number { value })
    }
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ ]+")]
enum Token {
    #[token("+")]
    Plus,

    #[token("*")]
    Multiplication,

    #[token("-")]
    Minus,
    #[regex("[0-9]+", |lex| lex.slice().parse().ok())]
    Number(Number),
    // #[token("/")]
    // Reduction,

    // #[token("^")]
    // Power,

    // #[token("%")]
    // Reciprocation,

    // #[token("/.")]
    // Fold,

    // #[token("#")]
    // Shape,

    // #[token("T")]
    // Transpose,

    // #[token("&")]
    // PartialApplication,

    // #[token("@")]
    // Composition,

    // #[token(".@")]
    // Distribute,

    // #[token("~")]
    // Interchange,

    // #[token("$")]
    // Axes,

    // #[token("_")]
    // Flatten,

    // #[token(">#")]
    // Reshape,

    // #[regex("[a-zA-Z][a-zA-Z0-9]*")]
    // Symbol(&'a str),

    // #[token("=")]
    // Assign,

    // #[token("let")]
    // Let,
    #[token("(")]
    LParen,

    #[token(")")]
    RParen,
    // #[token("[")]
    // LBracket,

    // #[token("]")]
    // RBracket,
}

trait TokenImpl {
    fn _next_node(op: Token, tokens: &[Token]) -> Result<(Node, &[Token])>;
}

fn next_node(tokens: &[Token]) -> Result<(Node, &[Token])> {
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
enum SyntaxError {
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

struct Node {
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

fn main() -> Result<()> {
    let input = "+ (*2 (-1)) 1";
    let lexer = Token::lexer(input);

    let mut tokens = Vec::new();
    for token in lexer {
        match token {
            Ok(token) => tokens.push(token),
            Err(e) => return Err(anyhow!("Error: {:?}", e).context("Error in lexer")),
        }
    }
    let (root, _) = next_node(&tokens)?;
    println!("{:?}", root);
    Ok(())
}
