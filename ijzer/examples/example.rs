use anyhow::Result;
use ijzer::parser::{parse_line, ASTContext};
use ijzer::tokens::lexer;

static INPUT: &str = r#"
var x -> 1
y = (+ (x 1))
z = - x y
fn z = * y I
x y = (1 2)
u = [1,2,3]
* (z 1 -2)
"#;

fn main() -> Result<()> {
    let mut symbol_table = ASTContext::new();

    for line in INPUT.lines() {
        let tokens = lexer(line)?;
        if tokens.is_empty() {
            continue;
        }
        println!("\ninput: {:?}", line);
        let root = parse_line(tokens, &mut symbol_table)?;
        println!("{:?}", root);
    }

    Ok(())
} 
