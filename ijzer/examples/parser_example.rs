use anyhow::Result;
use ijzer::ast_node::ASTContext;
use ijzer::parser::parse_line;
use ijzer::tokens::lexer;

static INPUT: &str = r#"
var x: T
(+ x [1])
y: T = (+ x [1])
z: T = - x y
z: Fn(T->T) = * y I
u = [1,2,3]
v = /+ u
(* z [1] [-2])
(- (+ (1) 2))
/+ [1,2]
var f: Fn(S,S->S)
@(-, f) 1 2
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
