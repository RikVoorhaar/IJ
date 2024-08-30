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
pub fn find_matching_parenthesis_on_tokens(
    tokens: &[Token],
    lparen: &Token,
    rparen: &Token,
) -> Option<usize> {
    let mut depth = 1;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            token if token == lparen => depth += 1,
            token if token == rparen => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => (),
        }
    }
    None
}

pub fn comma_separate_on_tokens(tokens: &[Token]) -> Vec<usize> {
    let mut indices = vec![];
    let mut depth = 0;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::LParen | Token::LSqBracket => depth += 1,
            Token::RParen | Token::RSqBracket => depth -= 1,
            Token::Comma => {
                if depth == 0 {
                    indices.push(i);
                }
            }
            _ => (),
        }
    }
    indices
}

pub fn split_tokens_at_indices(tokens: &[Token], indices: &[usize]) -> Vec<Vec<Token>> {
    let mut start = 0;
    let mut split_tokens: Vec<Vec<Token>> = Vec::new();
    for &index in indices {
        let (left, _) = tokens.split_at(index);
        split_tokens.push(left[start..].to_vec());
        start = index + 1;
    }
    split_tokens.push(tokens[start..].to_vec());
    split_tokens
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
    NumberToken,

    #[token(":")]
    TypeDeclaration,

    #[token(";")]
    Semicolon,

    #[token("@")]
    FunctionComposition,

    #[token("//")]
    Comment,

    #[token("<-")]
    TypeConversion,

    #[token("~")]
    AsFunction,

    #[token(".")]
    Apply,

    #[token("?")]
    GeneralizedContraction,

    #[regex(r"<\w+>", |lex| lex.slice().trim_start_matches('<').trim_end_matches('>').parse().ok())]
    NumberType(String),

    #[regex(r"(eye|randu|randn|zeros|ones)", |lex| Some(lex.slice().to_string()))]
    TensorBuilder(String),

    #[token("|")]
    Transpose,

    #[token("%")]
    Shape,

    #[token("qr")]
    QR,

    #[token("svd")]
    SVD,

    #[token(r"\")]
    Solve,
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
            Self::NumberToken => write!(f, "N"),
            Self::TypeDeclaration => write!(f, ":"),
            Self::Semicolon => write!(f, ";"),
            Self::LambdaVariable(v) => write!(f, "${}", v.name),
            Self::FunctionComposition => write!(f, "@"),
            Self::Comment => write!(f, "//"),
            Self::TypeConversion => write!(f, "<-"),
            Self::AsFunction => write!(f, "~"),
            Self::Apply => write!(f, "."),
            Self::GeneralizedContraction => write!(f, "?"),
            Self::NumberType(s) => write!(f, "<{}>", s),
            Self::TensorBuilder(s) => write!(f, "{}", s),
            Self::Transpose => write!(f, "|"),
            Self::Shape => write!(f, "%"),
            Self::QR => write!(f, "qr"),
            Self::SVD => write!(f, "svd"),
            Self::Solve => write!(f, r"\"),
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

    #[test]
    fn test_tensor_builders() {
        // This test is necessary because we neeed to make sure that the tensor builder regex
        // has higher priority than the symbol regex.
        let input = "eye randu randn zeros ones";
        let tokens = lexer(input).unwrap();
        assert!(tokens.iter().all(|t| matches!(t, Token::TensorBuilder(_))));
        println!("{:?}", tokens);
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_comma_separate_on_tokens() {
        let input = "1, (2, [3, 4]),5, 6";
        let tokens = lexer(input).unwrap();
        let indices = comma_separate_on_tokens(&tokens);
        println!("{:?}", indices);
        assert_eq!(indices, vec![1, 11, 13]);

        let split_tokens = split_tokens_at_indices(&tokens, &indices);
        assert_eq!(split_tokens.len(), 4);
        assert_eq!(split_tokens[0], lexer("1").unwrap());
        assert_eq!(split_tokens[1], lexer("(2, [3, 4])").unwrap());
        assert_eq!(split_tokens[2], lexer("5").unwrap());
        assert_eq!(split_tokens[3], lexer("6").unwrap());
    }
}
