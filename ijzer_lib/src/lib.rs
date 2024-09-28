//! This crate provides the implementation of the IJzer language.
//! It defines a parser for the IJzer language, a syntax tree for the IJzer language, and a transpiler for the IJzer language to Rust code.
//! For documentation of the language itself and how you can use this see the documentation of the `ijzer` crate.

use anyhow::Result;
use proc_macro2::TokenStream;

pub mod ast_node;
pub mod comparison_funcs;
pub mod compiler;
pub mod function_enums;
pub mod operations;
pub mod parser;
pub mod syntax_error;
pub mod tensor;
pub mod tokens;
pub mod types;

/// Compiles a string of IJzer code into a Rust `TokenStream`.
pub fn compile(input: &str) -> Result<TokenStream> {
    let mut ast_context = ast_node::ASTContext::new();
    let tokens = tokens::lexer(input)?;
    let parsed_lines = parser::parse_lines(tokens, &mut ast_context)?;

    parsed_lines
        .into_iter()
        .map(|(root, has_semicolon)| compiler::compile_line_from_node(root, has_semicolon))
        .collect::<Result<TokenStream>>()
}
