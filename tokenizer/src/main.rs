use anyhow::Result;
use std::{fmt::Debug, str::FromStr};

use logos::Logos;

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

    // #[token("(")]
    // LParen,

    // #[token(")")]
    // RParen,

    // #[token("[")]
    // LBracket,

    // #[token("]")]
    // RBracket,
}

trait TokenImpl {
    fn _next_node(op: Token, tokens: &[Token]) -> (Node, Option<&[Token]>);
}

fn next_node(tokens: &[Token]) -> (Node, Option<&[Token]>) {
    let (op, rest) = tokens.split_first().unwrap();
    match op {
        Token::Plus => BinaryOp::_next_node(op.clone(), rest),
        Token::Multiplication => BinaryOp::_next_node(op.clone(), rest),
        Token::Number(_) => Number::_next_node(op.clone(), rest),
        Token::Minus => MinusOp::_next_node(op.clone(), rest),
    }
}

#[derive(Debug, PartialEq)]
struct BinaryOp;
impl TokenImpl for BinaryOp {
    fn _next_node(op: Token, tokens: &[Token]) -> (Node, Option<&[Token]>) {
        let (lhs, rest) = next_node(tokens);
        let (rhs, rest) = next_node(rest.unwrap());
        (
            Node {
                op,
                output_arity: 1,
                operands: vec![lhs, rhs],
                functional_operands: vec![],
            },
            rest,
        )
    }
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node(op: Token, tokens: &[Token]) -> (Node, Option<&[Token]>) {
        let (lhs, rest) = next_node(tokens);
        match rest {
            None => (
                Node {
                    op,
                    output_arity: 1,
                    operands: vec![lhs],
                    functional_operands: vec![],
                },
                None,
            ),
            Some(rest) => {
                let (rhs, rest) = next_node(rest);
                (
                    Node {
                        op,
                        output_arity: 1,
                        operands: vec![lhs, rhs],
                        functional_operands: vec![],
                    },
                    rest,
                )
            }
        }
    }
}

impl TokenImpl for Number {
    fn _next_node(op: Token, tokens: &[Token]) -> (Node, Option<&[Token]>) {
        (
            Node {
                op,
                output_arity: 1,
                operands: vec![],
                functional_operands: vec![],
            },
            Some(tokens),
        )
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

fn main() {
    let input = "+ 1 * 2 3";
    let lexer = Token::lexer(input);

    let tokens: Result<Vec<Token>, ()> = lexer.into_iter().collect();
    let tokens = tokens.unwrap();
    let (root, rest) = next_node(&tokens);
    println!("{:?}", root);
}
