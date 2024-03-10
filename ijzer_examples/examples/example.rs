use anyhow::Result;
use ijzer::ArrayFunc;
use ijzer_macro::ijzer;
use ndarray::{array, Array1};

#[ijzer]
fn test_function(x: Array1<u64>) -> Array1<u64> {
    "var x->1
    y = (+ (x 1))
    fn z = * y I
    * (z 1 1)"
}

fn main() -> Result<()> {
    println!("{:?}", test_function(array![1u64]));
    Ok(())
}
