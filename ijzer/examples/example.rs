use anyhow::Result;
use ijzer::node::{parse_ast, SymbolTable};
use ijzer::tokens::lexer;

fn main() -> Result<()> {
    let mut symbol_table = SymbolTable::new();

    let input = "var x 1 -> 2";
    println!("input: {:?}", input);
    let tokens = lexer(input)?;
    let root = parse_ast(&tokens, &mut symbol_table)?;
    println!("{:?}", root);

    let input = "y = (+ (x 1) 2)";
    println!("input: {:?}", input);
    let tokens = lexer(input)?;
    let root = parse_ast(&tokens, &mut symbol_table)?;
    println!("{:?}", root);

    let input = "* y";
    println!("input: {:?}", input);
    let tokens = lexer(input)?;
    let root = parse_ast(&tokens, &mut symbol_table)?;
    println!("{:?}", root);

    Ok(())
}
