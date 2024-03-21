use anyhow::Result;
use ijzer_macro::ijzer;
use ijzer::tensor::Tensor;

#[ijzer]
fn test_function(x: Tensor<f64>) -> Tensor<f64> {
    r#"
    var x->1
    /* x
    "#
}


fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2]));
    let y = test_function(x);
    println!("{}", y);
    Ok(())
}
