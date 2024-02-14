use crate::tokens::Token;

use thiserror::Error;
#[derive(Error, Debug)]
pub enum SyntaxError {
    #[error("Excpected {0} arguments but got {1}.")]
    InsufficientArguments(usize, usize),

    #[error("Next node called on an empty Token stream")]
    EmptyStream,

    #[error("Unmatched parenthesis")]
    UnmatchedParenthesis,

    #[error("Unresolved group {0}")]
    UnresolvedGroup(String),

    #[error("Unhandled tokens {0:?}")]
    UnhandledTokens(Vec<Token>),

    #[error("Unexpected token {0:?}")]
    UnexpectedToken(Token),

    #[error("Unknown symbol {0}")]
    UnknownSymbol(String),
}
