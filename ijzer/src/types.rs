use crate::syntax_error::SyntaxError;
use crate::tokens::{lexer, Token};
use anyhow::Result;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub enum IJType {
    Void,
    Scalar,
    Tensor,
    Function(FunctionSignature),
    Group(Vec<IJType>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionSignature {
    pub input: Vec<IJType>,
    pub output: Vec<IJType>,
}

impl FunctionSignature {
    pub fn new(input: Vec<IJType>, output: Vec<IJType>) -> Self {
        FunctionSignature { input, output }
    }
    pub fn tensor_function(inputs: usize, outputs: usize) -> Self {
        Self {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Tensor; outputs],
        }
    }
    pub fn scalar_function(inputs: usize, outputs: usize) -> Self {
        Self {
            input: vec![IJType::Scalar; inputs],
            output: vec![IJType::Scalar; outputs],
        }
    }
    pub fn from_tokens(tokens: &[Token]) -> Result<Self> {
        let parts: Vec<&[Token]> = tokens.split(|t| *t == Token::Arrow).collect();

        if parts.len() != 2 {
            return Err(SyntaxError::InvalidType(
                tokens
                    .iter()
                    .fold(String::new(), |acc, t| acc + format!("{:?}", t).as_str()),
            )
            .into());
        }
        let input = parts[0]
            .split(|t| *t == Token::Comma)
            .map(IJType::from_tokens)
            .collect::<Result<_, _>>()?;
        let output = parts[1]
            .split(|t| *t == Token::Comma)
            .map(IJType::from_tokens)
            .collect::<Result<_, _>>()?;

        Ok(FunctionSignature { input, output })
    }
    pub fn from_string(s: &str) -> Result<Self> {
        let tokens = lexer(s).map_err(|e| SyntaxError::LexerError(e.to_string()))?;
        Self::from_tokens(&tokens)
    }
}

impl IJType {
    pub fn tensor_function(inputs: usize, outputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Tensor; outputs],
        })
    }
    pub fn scalar_function(inputs: usize, outputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Scalar; inputs],
            output: vec![IJType::Scalar; outputs],
        })
    }
    pub fn tensor_to_scalar_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Scalar; 1],
        })
    }
    pub fn from_tokens(tokens: &[Token]) -> Result<Self> {
        match tokens {
            [Token::Scalar] => Ok(IJType::Scalar),
            [Token::Tensor] => Ok(IJType::Tensor),
            [Token::FunctionType, Token::LParen, rest @ ..] => {
                let mut depth = 1;
                let mut inside = Vec::new();
                for token in rest {
                    match token {
                        Token::LParen => depth += 1,
                        Token::RParen => {
                            if depth == 1 {
                                break;
                            } else {
                                depth -= 1;
                            }
                        }
                        _ => {
                            if depth > 0 {
                                inside.push(token.clone());
                            }
                        }
                    }
                }
                if depth != 1 {
                    return Err(SyntaxError::UnmatchedParenthesis(
                        "Mismatched parentheses in function type".to_string(),
                    )
                    .into());
                }
                let sig = FunctionSignature::from_tokens(&inside)?;
                Ok(IJType::Function(sig))
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
        Self::from_tokens(&tokens)
    }
}

impl std::fmt::Display for IJType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IJType::Scalar => write!(f, "S"),
            IJType::Tensor => write!(f, "T"),
            IJType::Function(sig) => write!(f, "Fn({})", sig),
            IJType::Void => write!(f, "Void"),
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
        let output_strings: Vec<String> = self.output.iter().map(|o| o.to_string()).collect();
        write!(
            f,
            "{} -> {}",
            input_strings.join(", "),
            output_strings.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ijtype_from_string_scalar() {
        let result = IJType::from_string("S");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Scalar);
    }

    #[test]
    fn test_ijtype_from_string_tensor() {
        let result = IJType::from_string("T");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), IJType::Tensor);
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
        let expected = IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor, IJType::Tensor],
            output: vec![IJType::Scalar],
        });
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
        let expected = FunctionSignature {
            input: vec![IJType::Tensor],
            output: vec![IJType::Scalar],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_function_signature_from_string_tensors_to_tensor() {
        let result = FunctionSignature::from_string("T,T->T");
        assert!(result.is_ok());
        let expected = FunctionSignature {
            input: vec![IJType::Tensor, IJType::Tensor],
            output: vec![IJType::Tensor],
        };
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_function_signature_from_string_invalid() {
        let result = FunctionSignature::from_string("T,T");
        assert!(result.is_err());
    }
}
