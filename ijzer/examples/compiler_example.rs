use std::process::{Command, Stdio};


use anyhow::Result;
use ijzer::{
    compiler::CompilerContext,
    node::{parse_line, ASTContext, Node},
    tokens::lexer,
    pretty_print::pretty_print
};
use proc_macro2::TokenStream;

fn main() -> Result<()> {
    let input: &str = "
    x = 1
    y = (+ (x 1))
    fn z = * y I
    * (z 1 2)
    x y = (1 2)
    ";
    let mut symbol_table = ASTContext::new();
    let mut parsed_lines: Vec<Node> = Vec::new();

    for line in input.lines() {
        let tokens = lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        println!("\ninput: {:?}", line);
        let root = parse_line(tokens, &mut symbol_table)?;
        println!("{:?}", root);
        parsed_lines.push(root);
    }
    println!("-----------------------");

    let root = parsed_lines.first().unwrap();
    let mut compiler = CompilerContext::new(root.clone());
    println!("{:?}", compiler.node_map.keys().collect::<Vec<_>>());
    while let Some(node_id) = compiler.pop_next_parseable() {
        println!("-> {:?}", compiler.node_map[&node_id]);
        let stream = compiler.compile_node(node_id)?;
        println!("{:?}", stream.to_string());

        compiler.submit_as_parsed(node_id, stream)
    }

    let _2 = (1, 2);
    let (x, y) = _2;

    Ok(())
}
