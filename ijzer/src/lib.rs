use anyhow::Result;
use proc_macro2::TokenStream;
use std::rc::Rc;

pub mod arrays;
pub mod compiler;
pub mod parser;
pub mod operations;
pub mod pretty_print;
pub mod syntax_error;
pub mod tokens;
pub mod tensor;

pub use arrays::*;
pub use arrays::ArrayFunc;
pub use ndarray::array;

pub fn compile(input: &str) -> Result<TokenStream> {
    let mut symbol_table = parser::ASTContext::new();
    let mut parsed_lines: Vec<Rc<parser::Node>> = Vec::new();

    for line in input.lines() {
        let tokens = tokens::lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        let root = parser::parse_line(tokens, &mut symbol_table)?;
        parsed_lines.push(root);
    }

    let mut stream = TokenStream::new();
    for root in parsed_lines.iter() {
        let mut compiler = compiler::CompilerContext::new(root.clone());
        let mut line_stream: TokenStream = TokenStream::new();

        while let Some(node_id) = compiler.pop_next_parseable() {
            line_stream = compiler.compile_node(node_id)?;

            compiler.submit_as_parsed(node_id, line_stream.clone())
        }
        stream.extend(line_stream);
    }

    Ok(stream)
}
