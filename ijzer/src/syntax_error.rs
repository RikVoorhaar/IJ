use crate::tokens::Token;


use thiserror::Error;
#[derive(Error, Debug, PartialEq)]
pub enum SyntaxError {
    #[error("Excpected {0} arguments but got {1}.")]
    InsufficientArguments(usize, usize),

    #[error("Next node called on an empty Token stream")]
    EmptyStream,

    #[error("Unmatched parenthesis: '{0}'")]
    UnmatchedParenthesis(String),

    #[error("Unresolved group {0}")]
    UnresolvedGroup(String),

    #[error("Unhandled tokens {0}")]
    UnhandledTokens(String),

    #[error("Unexpected token {0:?}")]
    UnexpectedToken(Token),

    #[error("Unexpected semicolon")]
    UnexpectedSemicolon,

    #[error("Unknown symbol {0}")]
    UnknownSymbol(String),

    #[error("Too many inputs")]
    TooManyInputs,

    #[error("Function without inputs")]
    FunctionWithoutInputs,

    #[error("Variable is not allowed to have input arguments")]
    VariableWithInput,

    #[error("Number of variables does not match output arity; got {0} variables but output arity {1}")]
    IncorrectNumVariables(usize, usize),

    #[error("Invalid number format: {0}")]
    InvalidNumberFormat(String),

    #[error("Expected function but got {0}")]
    ExpectedFunction(String),

    #[error("Expected binary operator but got {0}")]
    ExpectedBinaryOperator(String),

    #[error("Expected operator with output arity {0} but got {1}")]
    ExpectedOperatorWithOutputArity(usize, String),

    #[error("Expected type {0} but got {1}")]
    TypeError(String, String),

    #[error("Operation with multiple outputs is not supported here.")]
    MultipleOutputs,

    #[error("Input arity {0} of node {1} exceeds summed output arity {2} of children: {3}")]
    InputArityExceedsOutputArity(usize, String, usize, String),

    #[error("Function signature {0} does not match expected signature {1}")]
    FunctionSignatureMismatch(String, String),

    #[error("Function with multiple outputs is not supported yet. Use a group as output instead.")]
    FunctionWithMultipleOutputs(usize),

    #[error("Tried to match multiple types {0} but none of them matched. Longest string has type {1}")]
    GatherMismatch(String, String),

    #[error("Created invalid slice")]
    InvalidSlice,

    #[error("Invalid type '{0}'")]
    InvalidType(String),

    #[error("Error inside of lexer: {0}")]
    LexerError(String),

    #[error("Invalid var statement: {0}")]
    InvalidVarStatement(String),

    #[error("Invalid assignment statement: {0}")]
    InvalidAssignmentStatement(String),

    #[error("Empty input")]
    EmptyInput,

    #[error("Expected token {0} but got {1}")]
    ExpectedToken(String, String),

    #[error("Empty slice")]
    EmptySlice,
}
