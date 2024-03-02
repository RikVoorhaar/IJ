use anyhow::Result;
use ijzer::{
    compiler::CompilerContext,
    node::{parse_line, ASTContext, Node},
    tokens::lexer,
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
        // println!("\ninput: {:?}", line);
        let root = parse_line(tokens, &mut symbol_table)?;
        // println!("{:?}", root);
        parsed_lines.push(root);
    }
    // println!("-----------------------");

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
