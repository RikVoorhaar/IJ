use anyhow::Result;
use std::str::FromStr;

use logos::Logos;

#[derive(Debug, PartialEq)]
struct Plus;

#[derive(Debug, PartialEq)]
struct Multiplication;

#[derive(Debug, PartialEq)]
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

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ ]+")]
enum Token {
    #[token("+", |_| Plus)]
    Plus(Plus),

    #[token("*", |_| Multiplication)]
    Multiplication(Multiplication),

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
