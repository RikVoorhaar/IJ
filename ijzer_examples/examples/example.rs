use anyhow::Result;
use ijzer::tensor::Tensor;
use ijzer_macro::ijzer;


#[ijzer(f64)]
fn test_function(x: Tensor<f64>) -> Tensor<f64> {
    r#"
    var x: T
    y = (+ x [1.0])
    z = - x y
    z($x) = * y $x
    u = [1.0,2.0,3.0]
    v = @(-,+) [1.0] [2.0]
    g($x, $y) = + $x $y
    v
    "#
}

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2]));
    let y = test_function(x);
    // let y2 = test_function2(x);
    println!("{}", y);
    Ok(())
}
