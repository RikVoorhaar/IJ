use anyhow::Result;
use ijzer::tensor::Tensor;
use ijzer_macro::ijzer;

#[ijzer]
fn test_function(x: Tensor<f64>) -> Tensor<f64> {
    r#"
    var x: T
    y = (+ x [1.0])
    z = - x y
    z = * y I
    u = [1.0,2.0,3.0]
    v = /+ u
    (* z [1.0] [-2.0])
    "#
}

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2]));
    let y = test_function(x);
    println!("{}", y);
    Ok(())
}
