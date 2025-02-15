//! Syntax errors for the parser
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

    #[error("Functions must have a single output, but found unparsed tokens: '{0}'")]
    FunctionWithUnparsedTokens(String),

    #[error("Variable is not allowed to have input arguments")]
    VariableWithInput,

    #[error(
        "Number of variables does not match output arity; got {0} variables but output arity {1}"
    )]
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

    #[error(
        "Tried to match multiple types {0} but none of them matched. Longest string has type {1}"
    )]
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

    #[error("Not a pure function: {0}")]
    NotPureFunction(String),

    #[error("Empty input")]
    EmptyInput,

    #[error("Expected token {0} but got {1}")]
    ExpectedToken(String, String),

    #[error("Empty slice")]
    EmptySlice,

    #[error("Function chain is not consistent: {0}")]
    FunctionChainInconsistency(String),

    #[error(
        "Function chain has ambiguous type. It should match exactly one type of the following: {0}"
    )]
    FunctionChainAmbiguousType(String),

    #[error("Function type ambiguous. The slice {0} can resolve to types {1}")]
    FunctionTypeAmbiguous(String, String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Type conversion not possible from {0} to {1}")]
    TypeConversionNotPossible(String, String),

    #[error("Groups can only return a single function, remaining tokens: {0}")]
    GroupReturnsMultipleFunctions(String),

    #[error("Empty group")]
    EmptyGroup,

    #[error("The slice {0} cannot be parsed as a function because {1} is a token without a functional variant")]
    SliceCannotBeParsedAsFunction(String, String),

    #[error("At least one argument is required")]
    EmptyFunctionComposition,

    #[error("Tried to retrieve out of bounds token. Index: {0}, length: {1}")]
    OutOfBoundsToken(usize, usize),

    #[error("None of the function variants {0} match the types of the gathered operands {1}")]
    FunctionVariantMismatch(String, String),

    #[error("None of the required outputs {0} match the function outputs {1}")]
    RequiredOutputsDoNotMatchFunctionOutputs(String, String),

    #[error("Cannot infer output number type from operands. Got both {0} and {1}")]
    CannotInferOutputNumberType(String, String),

    #[error("Number type mismatch {0} and {1}")]
    NumberTypeMismatch(String, String),

    #[error("Number of variables {0} does not match number of outputs {1} in assignment")]
    IncorrectNumberOfVariablesInAssignment(usize, usize),

    #[error("Expected group output, but got type {0}")]
    ExpectedGroupOutput(String),

    #[error("Expected [ token after index operator")]
    ExpectedLSqBracketAfterIndex,

    #[error("Indexing must either be a tensor or a list of numbers.")]
    IndexingMustBeTensorOrNumbers,

    #[error("Cannot mix tensor and colon in indexing")]
    CannotMixTensorAndColon,

    #[error("Invalid enum variant: {0}")]
    InvalidEnumVariant(String),
}
