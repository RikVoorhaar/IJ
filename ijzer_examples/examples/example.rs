use anyhow::Result;
use ijzer_macro::ijzer;
use ijzer::tensor::Tensor;

#[ijzer]
fn test_function(x: Tensor<f64>) -> Tensor<f64> {
    r#"
    var x->1
    u = [3.0,4.0]
    y = (+ (x u))
    fn z = * y I
    * (z 1.0 -2.4)
    "#
}


fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
    let y = test_function(x);
    println!("{}", y);
    Ok(())
}
