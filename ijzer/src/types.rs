use crate::syntax_error::SyntaxError;
use crate::tokens::{find_matching_parenthesis_on_tokens, lexer, Token};
use anyhow::Result;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub enum IJType {
    Void,
    Scalar,
    Tensor,
    Number,
    Function(FunctionSignature),
    Group(Vec<IJType>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionSignature {
    pub input: Vec<IJType>,
    pub output: Vec<IJType>,
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
    pub fn number_function(inputs: usize, outputs: usize) -> Self {
        Self {
            input: vec![IJType::Number; inputs],
            output: vec![IJType::Number; outputs],
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
        let output = _split_tokens(parts[1], Token::Comma)
            .iter()
            .map(|part| IJType::parse_tokens(part).map(|(t, _)| t))
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
    pub fn number_function(inputs: usize, outputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Number; inputs],
            output: vec![IJType::Number; outputs],
        })
    }
    pub fn tensor_to_scalar_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Scalar; 1],
        })
    }
    // pub fn from_tokens(tokens: &[Token]) -> Result<Self> {
    //     match tokens {
    //         [Token::Scalar] => Ok(IJType::Scalar),
    //         [Token::Tensor] => Ok(IJType::Tensor),
    //         [Token::NumberType] => Ok(IJType::Number),
    //         [Token::FunctionType, Token::LParen, rest @ ..] => {
    //             let maybe_index =
    //                 find_matching_parenthesis_on_tokens(rest, &Token::LParen, &Token::RParen);
    //             match maybe_index {
    //                 None => Err(SyntaxError::UnmatchedParenthesis(
    //                     "Mismatched parentheses in function type".to_string(),
    //                 )
    //                 .into()),
    //                 Some(index) => {
    //                     let inside = &rest[..index];
    //                     let sig = FunctionSignature::from_tokens(inside)?;
    //                     Ok(IJType::Function(sig))
    //                 }
    //             }
    //         }

    //         _ => Err(SyntaxError::InvalidType(
    //             tokens
    //                 .iter()
    //                 .fold(String::new(), |acc, t| acc + format!("{:?}", t).as_str()),
    //         )
    //         .into()),
    //     }
    // }
    /// Parses tokens and returns the type and the number of tokens consumed.
    pub fn parse_tokens(tokens: &[Token]) -> Result<(Self, usize)> {
        match tokens {
            [Token::Scalar, ..] => Ok((IJType::Scalar, 1)),
            [Token::Tensor, ..] => Ok((IJType::Tensor, 1)),
            [Token::NumberType, ..] => Ok((IJType::Number, 1)),
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
}

impl std::fmt::Display for IJType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IJType::Scalar => write!(f, "S"),
            IJType::Tensor => write!(f, "T"),
            IJType::Function(sig) => write!(f, "Fn({})", sig),
            IJType::Void => write!(f, "Void"),
            IJType::Number => write!(f, "N"),
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

    #[test]
    fn test_nested_function_signature() {
        let result = IJType::from_string("Fn(Fn(T,T->S),T->S)");
        println!("{:?}", result);
        assert!(result.is_ok());
        let expected = IJType::Function(FunctionSignature {
            input: vec![
                IJType::Function(FunctionSignature {
                    input: vec![IJType::Tensor, IJType::Tensor],
                    output: vec![IJType::Scalar],
                }),
                IJType::Tensor,
            ],
            output: vec![IJType::Scalar],
        });
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_ijtype_parse_tokens() {
        let tokens = lexer("Fn(T,T->S) a").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(
            t,
            IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor, IJType::Tensor],
                output: vec![IJType::Scalar],
            })
        );
        assert_eq!(tokens[i..].len(), 1);

        let tokens = lexer("T a").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(t, IJType::Tensor);
        assert_eq!(tokens[i..].len(), 1);

        let tokens = lexer("Fn(T,T->Fn(T->T)) S").unwrap();
        let (t, i) = IJType::parse_tokens(&tokens).unwrap();
        assert_eq!(
            t,
            IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor, IJType::Tensor],
                output: vec![IJType::Function(FunctionSignature {
                    input: vec![IJType::Tensor],
                    output: vec![IJType::Tensor],
                })],
            })
        );
        assert_eq!(tokens[i..].len(), 1);
    }
}
