use anyhow::Result;
use ijzer::node::{parse_line, ASTContext};
use ijzer::tokens::lexer;

static INPUT: &str = r#"
var x 1 -> 2
y = (+ (x (1)))
fn z = * y I
* (z 1 2)
x y z= (1 2)
"#;

fn main() -> Result<()> {
    let mut symbol_table = ASTContext::new();

    for line in INPUT.lines() {
        let tokens = lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        let root = parse_line(tokens, &mut symbol_table)?;
        println!("\ninput: {:?}", line);
        println!("{:?}", root);
    }

    Ok(())
} 
