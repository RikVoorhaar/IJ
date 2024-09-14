use anyhow::Result;
use proc_macro2::TokenStream;

pub mod ast_node;
pub mod compiler;
pub mod operations;
pub mod parser;
pub mod syntax_error;
pub mod tensor;
pub mod tokens;
pub mod types;


pub fn compile(input: &str) -> Result<TokenStream> {
    let mut ast_context = ast_node::ASTContext::new();
    let tokens = tokens::lexer(input)?;
    let parsed_lines = parser::parse_lines(tokens, &mut ast_context)?;

    parsed_lines
        .into_iter()
        .map(|(root, has_semicolon)| {
            compiler::compile_line_from_node(root, has_semicolon)
        })
        .collect::<Result<TokenStream>>()
}
