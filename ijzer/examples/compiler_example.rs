use std::rc::Rc;

use anyhow::Result;
use ijzer::{
    compiler::CompilerContext,
    node::{parse_line, ASTContext, Node},
    tokens::lexer,
};

use proc_macro2::TokenStream;

fn main() -> Result<()> {
    let input: &str = "
    var x -> 1
    y = (+ (x 1))
    z = - x y
    fn z = * y I
    x y = (1 2)
    * (z 1 -2)
    ";
    let mut symbol_table = ASTContext::new();
    let mut parsed_lines: Vec<Rc<Node>> = Vec::new();

    for line in input.lines() {
        let tokens = lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        let root = parse_line(tokens, &mut symbol_table)?;
        parsed_lines.push(root);
    }

    for root in parsed_lines.iter() {
        let mut compiler = CompilerContext::new(root.clone());
        let mut stream: TokenStream = TokenStream::new();

        while let Some(node_id) = compiler.pop_next_parseable() {
            stream = compiler.compile_node(node_id)?;

            compiler.submit_as_parsed(node_id, stream.clone())
        }
        println!("{:?}", stream.to_string());
    }
    Ok(())
}
