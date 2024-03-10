use anyhow::Result;
use ijzer::ArrayFunc;
use ijzer_macro::ijzer;
use ndarray::{array, Array1};

#[ijzer]
fn test_function(x: Array1<f64>) -> Array1<f64> {
    r#"
    var x->1
    y = (+ (x 1.0))
    fn z = * y I
    * (z 1.0 -2.4)
    "#
}


fn main() -> Result<()> {
    println!("{:?}", test_function(array![1.0, 2.0]));
    Ok(())
}
