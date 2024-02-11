use anyhow::Result;
use ijzer::lexer::lexer;
use ijzer::node::next_node;

fn main() -> Result<()> {
    let input = "+ (2 (-1) 3) ";
    println!("input: {:?}", input);
    let tokens = lexer(input)?;
    let (root, _) = next_node(&tokens)?;
    println!("{:?}", root);
    Ok(())
}
