use anyhow::Result;
use ijzer::{
    ast_node::ASTContext, compile, compiler::compile_line_from_node, parser::parse_lines,
    tokens::lexer,
};

fn main() -> Result<()> {
    let input: &str = "
    var x: T
    (+ x [1.0]);
    y = (+ x [1.0])
    z = - x y
    z = * y I
    u = [1.0,2.0,3.0]
    f: Fn(T->S) = /+ $x
    g: Fn(T,T->T) = * $x $y;
    @(+,+ ) u;
    v = /+ u
    (* z [1.0] [-2.0])
    ";
    let tokens = lexer(input)?;
    println!("{:?}", tokens);
    let stream = compile(input)?;
    println!("{:?}", stream.to_string());
    println!("------------------------");

    let mut symbol_table = ASTContext::new();
    let parsed_lines = parse_lines(tokens, &mut symbol_table)?;

    for (root, has_semicolon) in parsed_lines.into_iter() {
        let line_stream = compile_line_from_node(root, has_semicolon)?;
        println!("{:?}", line_stream.to_string());
    }
    Ok(())
}

fn test() -> Result<()> {
    let closure = (|x1: ijzer::tensor::Tensor<_>, x2: ijzer::tensor::Tensor<_>| {
        (|x: ijzer::tensor::Tensor<i64>| x.map(|a| -a))(
            (|x1: ijzer::tensor::Tensor<_>, x2: ijzer::tensor::Tensor<_>| {
                x1.apply_binary_op(&x2, |a, b| a + b).unwrap()
            })(x1, x2),
        )
    })(
        ijzer::tensor::Tensor::scalar(1),
        ijzer::tensor::Tensor::scalar(2),
    );
    let x = ijzer::tensor::Tensor::<i64>::from_vec(vec![1], None);
    Ok(())
}
