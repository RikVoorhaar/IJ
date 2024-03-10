use anyhow::{anyhow, Result};
pub use logos::Logos;
use std::{fmt::Debug, str::FromStr};

#[derive(PartialEq, Clone)]
pub struct Number {
    pub value: String,
}
impl FromStr for Number {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(Number {
            value: s.to_string(),
        })
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

#[derive(Logos, PartialEq, Clone)]
#[logos(skip r"[ ]+")]
pub enum Token {
    #[token("+")]
    Plus,

    #[token("*")]
    Multiplication,

    #[token("-")]
    Minus,

    #[regex("[0-9]+\\.?[0-9]*[a-zA-Z0-9]*", |lex| lex.slice().parse().ok())]
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

    #[token("I", priority = 3)]
    Identity,

    #[token("fn")]
    Function,

    #[token("return")]
    Return,
}

impl Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Multiplication => write!(f, "*"),
            Self::Minus => write!(f, "-"),
            Self::Number(n) => write!(f, "{:?}", n),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::Symbol(s) => write!(f, "{}", s.name),
            Self::Variable => write!(f, "var"),
            Self::Arrow => write!(f, "->"),
            Self::Assign => write!(f, "="),
            Self::Identity => write!(f, "I"),
            Self::Function => write!(f, "fn"),
            Self::Return => write!(f, "return"),
        }
    }
}
