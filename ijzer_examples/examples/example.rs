use anyhow::Result;
use ijzer::ArrayFunc;
use ijzer_macro::ijzer;
use ndarray::Array1;

#[ijzer]
fn test_function() -> Array1<u64> {
    "x = 1
    y = (+ (x 1))
    z = - x y
    fn z = * y I
    x y = (1 2)
    * (1 -2)"
}

// fn test_function2() -> Array1<u64> {
//     let x = ndarray::array![1u64];
//     let y = ijzer::arrays::Add::call(&[&x, &ndarray::array![1u64]]);
//     let z = ijzer::arrays::Subtract::call(&[&x, &y]);
//     let z = { |_21| ijzer::arrays::Multiply::call(&[&y, &_21]) };
//     let (x, y) = (ndarray::array![1u64], ndarray::array![2u64]);
//     ijzer::arrays::Multiply::call(&[
//         &ndarray::array![1u64],
//         &ijzer::arrays::Negate::call(&[&ndarray::array![2u64]]),
//     ])
// }

fn main() -> Result<()> {
    Ok(())
}
