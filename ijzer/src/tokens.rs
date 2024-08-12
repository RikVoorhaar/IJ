use anyhow::{anyhow, Result};
pub use logos::Logos;
use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

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
impl Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
impl Debug for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct SymbolToken {
    pub name: String,
}
impl FromStr for SymbolToken {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(SymbolToken {
            name: s.to_string(),
        })
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct LambdaVariableName {
    pub name: String,
}
impl FromStr for LambdaVariableName {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(LambdaVariableName {
            name: s.to_string(),
        })
    }
}

pub fn lexer(input: &str) -> Result<Vec<Token>> {
    let lex = Token::lexer(input);
    let mut tokens = Vec::new();
    let mut is_comment = false;
    for token in lex {
        match token {
            Ok(Token::Comment) => is_comment = true,
            Ok(Token::Newline) => {
                is_comment = false;
                tokens.push(Token::Newline);
            }
            Ok(token) => {
                if is_comment {
                    continue;
                }
                tokens.push(token);
            }
            Err(e) => {
                if !is_comment {
                    return Err(anyhow!("Error: {:?}", e).context("Error in lexer"));
                }
            }
        }
    }
    Ok(tokens)
}

#[derive(Logos, PartialEq, Clone)]
#[logos(skip r"[ ]+")]
pub enum Token {
    #[token("\n")]
    Newline,

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

    #[token("[")]
    LSqBracket,

    #[token("]")]
    RSqBracket,

    #[token(",")]
    Comma,

    #[regex("[a-zA-Z][a-zA-Z0-9]*", |lex| lex.slice().parse().ok())]
    Symbol(SymbolToken),

    #[regex("\\$[a-zA-Z0-9]+", |lex| lex.slice().trim_start_matches('$').parse().ok())]
    LambdaVariable(LambdaVariableName),

    #[token("var")]
    Variable,

    #[token("->")]
    Arrow,

    #[token("=")]
    Assign,

    #[token("I", priority = 3)]
    Identity,

    #[token("return")]
    Return,

    #[token("/")]
    Reduction,

    #[token("Fn")]
    FunctionType,

    #[token("S", priority = 3)]
    Scalar,

    #[token("T", priority = 3)]
    Tensor,

    #[token("N", priority = 3)]
    NumberType,

    #[token(":")]
    TypeDeclaration,

    #[token(";")]
    Semicolon,

    #[token("@")]
    FunctionComposition,

    #[token("//")]
    Comment,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Newline => write!(f, "\\n"),
            Self::Plus => write!(f, "+"),
            Self::Multiplication => write!(f, "*"),
            Self::Minus => write!(f, "-"),
            Self::Number(n) => write!(f, "{}", n),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::Symbol(s) => write!(f, "{}", s.name),
            Self::Variable => write!(f, "var"),
            Self::Arrow => write!(f, "->"),
            Self::Assign => write!(f, "="),
            Self::Identity => write!(f, "I"),
            Self::Return => write!(f, "return"),
            Self::Comma => write!(f, ","),
            Self::LSqBracket => write!(f, "["),
            Self::RSqBracket => write!(f, "]"),
            Self::Reduction => write!(f, "/"),
            Self::FunctionType => write!(f, "Fn"),
            Self::Scalar => write!(f, "S"),
            Self::Tensor => write!(f, "T"),
            Self::NumberType => write!(f, "N"),
            Self::TypeDeclaration => write!(f, ":"),
            Self::Semicolon => write!(f, ";"),
            Self::LambdaVariable(v) => write!(f, "${}", v.name),
            Self::FunctionComposition => write!(f, "@"),
            Self::Comment => write!(f, "//"),
        }
    }
}

impl Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{:?}", n),
            _ => write!(f, "{}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer() {
        let input = "1 + 1";
        let tokens = lexer(input).unwrap();
        println!("{:?}", tokens);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_comments() {
        let input = "// this is a 2309 asdfklj 19 &8 * comment b=with invalid syntax";
        let tokens = lexer(input).unwrap();
        println!("{:?}", tokens);
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_comments_multiline() {
        let input_with_comment = "x = 1; // this is a comment\n y = 2;";
        let input_with_multiline = "x = 1; // this is a comment
         y = 2;// and another comment";
        let input_without_comment = "x = 1;\n y = 2;";
        let tokens_with_comment = lexer(input_with_comment).unwrap();
        let tokens_with_multiline = lexer(input_with_multiline).unwrap();
        let tokens_without_comment = lexer(input_without_comment).unwrap();
        println!("{:?}", tokens_with_comment);
        println!("{:?}", tokens_with_multiline);
        println!("{:?}", tokens_without_comment);
        assert_eq!(tokens_with_comment.len(), tokens_without_comment.len());
        assert_eq!(tokens_with_comment, tokens_without_comment);
        assert_eq!(tokens_with_multiline.len(), tokens_with_comment.len());
        assert_eq!(tokens_with_multiline, tokens_with_comment);
    }
}
