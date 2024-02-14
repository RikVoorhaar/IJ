use anyhow::Result;
use ijzer::node::{parse_ast, SymbolTable};
use ijzer::tokens::lexer;

static INPUT: &str = r#"
var x 1 -> 2
y = (+ (x (1)) 2)
z = * y 
return (1 (1 2))
"#;

fn main() -> Result<()> {
    let mut symbol_table = SymbolTable::new();

    for line in INPUT.lines() {
        let tokens = lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        let root = parse_ast(&tokens, &mut symbol_table)?;
        println!("\ninput: {:?}", line);
        println!("{:?}", root);
    }

    Ok(())
}
