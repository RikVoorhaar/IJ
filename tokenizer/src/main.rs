use anyhow::Result;
use std::str::FromStr;

use logos::Logos;

#[derive(Debug, PartialEq)]
struct BinaryOp;

#[derive(Debug, PartialEq, Clone)]
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
    }
}

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


struct Node {
    op: Token,
    output_arity: usize,
    operands: Vec<Node>,
    functional_operands: Vec<Node>,
}

fn main() {
    let input = "+ 1 * 2 3";
    let lexer = Token::lexer(input);

    for token in lexer {
        println!("{:?}", token);
    }
}
