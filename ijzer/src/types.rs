use crate::syntax_error::SyntaxError;
use crate::tokens::{find_matching_parenthesis_on_tokens, lexer, Token};
use anyhow::Result;
use std::fmt::Debug;

type NumberType = Option<String>;

#[derive(Clone, Debug, PartialEq)]
pub enum IJType {
    Void,
    Scalar(NumberType),
    Tensor(NumberType),
    Number(NumberType),
    Function(FunctionSignature),
    Group(Vec<IJType>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionSignature {
    pub input: Vec<IJType>,
    pub output: Box<IJType>,
}

fn _split_tokens(tokens: &[Token], delimiter: Token) -> Vec<&[Token]> {
    let mut parts: Vec<&[Token]> = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, t) in tokens.iter().enumerate() {
        match t {
            Token::LParen => depth += 1,
            Token::RParen => depth -= 1,
            _ if *t == delimiter => {
                if depth == 0 {
                    parts.push(&tokens[start..i]);
                    start = i + 1;
                }
            }
            _ => {}
        }
    }
    parts.push(&tokens[start..]);
    parts
}

impl FunctionSignature {
    pub fn new(input: Vec<IJType>, output: IJType) -> Self {
        FunctionSignature {
            input,
            output: Box::new(output),
        }
    }
    pub fn tensor_function(inputs: usize) -> Self {
        Self {
            input: vec![IJType::Tensor(None); inputs],
            output: Box::new(IJType::Tensor(None)),
        }
    }
    pub fn scalar_function(inputs: usize) -> Self {
        Self {
            input: vec![IJType::Scalar(None); inputs],
            output: Box::new(IJType::Scalar(None)),
        }
    }
    pub fn number_function(inputs: usize) -> Self {
        Self {
            input: vec![IJType::Number(None); inputs],
            output: Box::new(IJType::Number(None)),
        }
    }
    pub fn from_tokens(tokens: &[Token]) -> Result<Self> {
        let parts = _split_tokens(tokens, Token::Arrow);
        if parts.len() != 2 {
            return Err(SyntaxError::InvalidType(
                tokens
                    .iter()
                    .fold(String::new(), |acc, t| acc + format!("{:?}", t).as_str()),
            )
            .into());
        }
        let input = _split_tokens(parts[0], Token::Comma)
            .iter()
            .map(|part| IJType::parse_tokens(part).map(|(t, _)| t))
            .collect::<Result<_, _>>()?;
        let (output, rest) = IJType::parse_tokens(parts[1])?;
        if rest != parts[1].len() {
            return Err(SyntaxError::FunctionWithUnparsedTokens(
                tokens
                    .iter()
                    .fold(String::new(), |acc, t| acc + format!("{:?}", t).as_str()),
            )
            .into());
        }

        Ok(FunctionSignature::new(input, output))
    }
    pub fn from_string(s: &str) -> Result<Self> {
        let tokens = lexer(s).map_err(|e| SyntaxError::LexerError(e.to_string()))?;
        Self::from_tokens(&tokens)
    }
}

impl IJType {
    pub fn tensor_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor(None); inputs],
            IJType::Tensor(None),
        ))
    }
    pub fn scalar_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature::new(
            vec![IJType::Scalar(None); inputs],
            IJType::Scalar(None),
        ))
    }
    pub fn number_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature::new(
            vec![IJType::Number(None); inputs],
            IJType::Number(None),
        ))
    }
    pub fn tensor_to_scalar_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor(None); inputs],
            IJType::Scalar(None),
        ))
    }

    /// Parses tokens and returns the type and the number of tokens consumed.
    pub fn parse_tokens(tokens: &[Token]) -> Result<(Self, usize)> {
        match tokens {
            [Token::Scalar, Token::NumberType(n), ..] => {
                Ok((IJType::Scalar(Some(n.to_string())), 2))
            }
            [Token::Scalar, ..] => Ok((IJType::Scalar(None), 1)),
            [Token::Tensor, Token::NumberType(n), ..] => {
                Ok((IJType::Tensor(Some(n.to_string())), 2))
            }
            [Token::Tensor, ..] => Ok((IJType::Tensor(None), 1)),
            [Token::NumberToken, Token::NumberType(n), ..] => {
                Ok((IJType::Number(Some(n.to_string())), 2))
            }
            [Token::NumberToken, ..] => Ok((IJType::Number(None), 1)),

            [Token::FunctionType, Token::LParen, rest @ ..] => {
                let maybe_index =
                    find_matching_parenthesis_on_tokens(rest, &Token::LParen, &Token::RParen);
                match maybe_index {
                    None => Err(SyntaxError::UnmatchedParenthesis(
                        "Mismatched parentheses in function type".to_string(),
                    )
                    .into()),
                    Some(index) => {
                        let inside = &rest[..index];
                        let sig = FunctionSignature::from_tokens(inside)?;
                        Ok((IJType::Function(sig), index + 3))
                    }
                }
            }

            _ => Err(SyntaxError::InvalidType(
                tokens
                    .iter()
                    .fold(String::new(), |acc, t| acc + format!("{:?}", t).as_str()),
            )
            .into()),
        }
    }
    pub fn from_string(s: &str) -> Result<Self> {
        let tokens = lexer(s).map_err(|e| SyntaxError::LexerError(e.to_string()))?;
        Self::parse_tokens(&tokens).map(|(t, _)| t)
    }

    pub fn extract_signature(&self) -> Option<FunctionSignature> {
        match self {
            IJType::Function(sig) => Some(sig.clone()),
            _ => None,
        }
    }

    /// Extracts the number type from the type. If the type is not a scalar, tensor, or number, it returns None.
    /// Note that this thus returns an Option<Option<String>>.
    pub fn extract_number_type(&self) -> Option<NumberType> {
        match self {
            IJType::Scalar(n) => Some(n.clone()),
            IJType::Tensor(n) => Some(n.clone()),
            IJType::Number(n) => Some(n.clone()),
            _ => None,
        }
    }

    pub fn type_match(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        matches!(
            (self, other),
            (IJType::Scalar(None), IJType::Scalar(_))
                | (IJType::Scalar(_), IJType::Scalar(None))
                | (IJType::Tensor(None), IJType::Tensor(_))
                | (IJType::Tensor(_), IJType::Tensor(None))
                | (IJType::Number(None), IJType::Number(_))
                | (IJType::Number(_), IJType::Number(None))
        )
    }

    pub fn type_downcast_number_type(&self, other: &Self) -> Result<Self> {
        if !self.type_match(other) {
            return Err(SyntaxError::TypeConversionNotPossible(
                format!("{:?}", self),
                format!("{:?}", other),
            )
            .into());
        }
        match (self, other) {
            (IJType::Scalar(None), IJType::Scalar(Some(n))) => Ok(IJType::Scalar(Some(n.clone()))),
            (IJType::Scalar(Some(n)), IJType::Scalar(None)) => Ok(IJType::Scalar(Some(n.clone()))),
            (IJType::Tensor(None), IJType::Tensor(Some(n))) => Ok(IJType::Tensor(Some(n.clone()))),
            (IJType::Tensor(Some(n)), IJType::Tensor(None)) => Ok(IJType::Tensor(Some(n.clone()))),
            (IJType::Number(None), IJType::Number(Some(n))) => Ok(IJType::Number(Some(n.clone()))),
            (IJType::Number(Some(n)), IJType::Number(None)) => Ok(IJType::Number(Some(n.clone()))),
            _ => Ok(self.clone()),
        }
    }

    pub fn maybe_apply_number_type(&self, other: &Option<String>) -> Self {
        match self {
            IJType::Scalar(None) => IJType::Scalar(other.clone()),
            IJType::Tensor(None) => IJType::Tensor(other.clone()),
            IJType::Number(None) => IJType::Number(other.clone()),
            _ => self.clone(),
        }
    }
}

impl std::fmt::Display for IJType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IJType::Scalar(None) => write!(f, "S"),
            IJType::Scalar(Some(n)) => write!(f, "S<{}>", n),
            IJType::Tensor(None) => write!(f, "T"),
            IJType::Tensor(Some(n)) => write!(f, "T<{}>", n),
            IJType::Function(sig) => write!(f, "Fn({})", sig),
            IJType::Void => write!(f, "Void"),
            IJType::Number(None) => write!(f, "N"),
            IJType::Number(Some(n)) => write!(f, "N<{}>", n),
            IJType::Group(types) => write!(
                f,
                "Group({})",
                types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

impl std::fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_strings: Vec<String> = self.input.iter().map(|i| i.to_string()).collect();
        let output_string = self.output.to_string();
        write!(f, "{} -> {}", input_strings.join(", "), output_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ijtype_from_string_scalar() {
        let result = IJType::from_string("S");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Scalar(None));

        let result = IJType::from_string("S<usize>");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Scalar(Some("usize".to_string())));

        let result = IJType::from_string("S<_>");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Scalar(Some("_".to_string())));
    }

    #[test]
    fn test_ijtype_from_string_tensor() {
        let result = IJType::from_string("T");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Tensor(None));

        let result = IJType::from_string("T<usize>");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Tensor(Some("usize".to_string())));

        let result = IJType::from_string("T<_>");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Tensor(Some("_".to_string())));
    }

    #[test]
    fn test_ijtype_from_string_function() {
        let result = IJType::from_string("Fn(T,T->S)");
        let s = "Fn(T,T->S)";
        let tokens = lexer(s)
            .map_err(|e| SyntaxError::LexerError(e.to_string()))
            .unwrap();
        println!("{:?}", tokens);

        assert!(result.is_ok());
        let expected = IJType::Function(FunctionSignature::new(
            vec![IJType::Tensor(None), IJType::Tensor(None)],
            IJType::Scalar(None),
        ));
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_ijtype_from_string_invalid() {
        let result = IJType::from_string("X");
        assert!(result.is_err());
    }

    #[test]
    fn test_function_signature_from_string_tensor_to_scalar() {
        let result = FunctionSignature::from_string("T->S");
        assert!(result.is_ok());
        let expected = FunctionSignature::new(vec![IJType::Tensor(None)], IJType::Scalar(None));
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_function_signature_from_string_tensors_to_tensor() {
        let result = FunctionSignature::from_string("T,T->T");
        assert!(result.is_ok());
        let expected = FunctionSignature::new(
            vec![IJType::Tensor(None), IJType::Tensor(None)],
            IJType::Tensor(None),
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_function_signature_from_string_invalid() {
        let result = FunctionSignature::from_string("T,T");
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_function_signature() {
        let result = IJType::from_string("Fn(Fn(T,T->S),T->S)");
        println!("{:?}", result);
        assert!(result.is_ok());
        let expected = IJType::Function(FunctionSignature::new(
            vec![
                IJType::Function(FunctionSignature::new(
                    vec![IJType::Tensor(None), IJType::Tensor(None)],
                    IJType::Scalar(None),
                )),
                IJType::Tensor(None),
            ],
            IJType::Scalar(None),
        ));
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_ijtype_parse_tokens() {
        let tokens = lexer("Fn(T,T->S) a").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(
            t,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Tensor(None)],
                IJType::Scalar(None),
            ))
        );
        assert_eq!(tokens[i..].len(), 1);

        let tokens = lexer("T a").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(t, IJType::Tensor(None));
        assert_eq!(tokens[i..].len(), 1);

        let tokens = lexer("Fn(T,T->Fn(T->T)) S").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(
            t,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None), IJType::Tensor(None)],
                IJType::Function(FunctionSignature::new(
                    vec![IJType::Tensor(None)],
                    IJType::Tensor(None),
                )),
            ))
        );
        assert_eq!(tokens[i..].len(), 1);
    }
}
