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

pub fn compile(input: &str, number_type: &str) -> Result<TokenStream> {
    let number_type = compiler::number_type_from_string(number_type)?;
    let mut ast_context = ast_node::ASTContext::new();
    let tokens = tokens::lexer(input)?;
    let parsed_lines = parser::parse_lines(tokens, &mut ast_context)?;

    parsed_lines
        .into_iter()
        .map(|(root, has_semicolon)| {
            compiler::compile_line_from_node(root, has_semicolon, number_type.clone())
        })
        .collect::<Result<TokenStream>>()
}
