use anyhow::{anyhow, Result};
// use logos::Logos;
pub use logos::Logos;
use std::{fmt::Debug, str::FromStr};

#[derive(PartialEq, Clone)]
pub struct Number {
    pub value: u64,
}
impl FromStr for Number {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let value = s.parse()?;
        Ok(Number { value })
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct SymbolToken {
    pub name: String,
}
impl FromStr for SymbolToken {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        Ok(SymbolToken {
            name: s.to_string(),
        })
    }
}

pub fn lexer(input: &str) -> Result<Vec<Token>> {
    let lex = Token::lexer(input);
    let mut tokens = Vec::new();
    for token in lex {
        match token {
            Ok(token) => tokens.push(token),
            Err(e) => return Err(anyhow!("Error: {:?}", e).context("Error in lexer")),
        }
    }
    Ok(tokens)
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ ]+")]
pub enum Token {
    #[token("+")]
    Plus,

    #[token("*")]
    Multiplication,

    #[token("-")]
    Minus,
    #[regex("[0-9]+", |lex| lex.slice().parse().ok())]
    Number(Number),

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[regex("[a-zA-Z][a-zA-Z0-9]*", |lex| lex.slice().parse().ok())]
    Symbol(SymbolToken),

    #[token("var")]
    Variable,

    #[token("->")]
    Arrow,

    #[token("=")]
    Assign,

    #[token("I")]
    Identity,

    #[token("fn")]
    Function,

    #[token(",")]
    Comma,
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

    // #[token("let")]
    // Let,
    // #[token("[")]
    // LBracket,

    // #[token("]")]
    // RBracket,
}
