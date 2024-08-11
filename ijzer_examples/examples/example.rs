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
    v = @(-,+) [1] [2]
    (* z [1.0] [-2.0])
    "#
}

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2]));
    let y = test_function(x);
    println!("{}", y);
    Ok(())
}

fn test_function2(x: Tensor<f64>) -> Tensor<f64> {
    let y = x
        .apply_binary_op(&ijzer::tensor::Tensor::from_vec(vec![1.0], None), |a, b| {
            a + b
        })
        .unwrap();
    let z = x.apply_binary_op(&y, |a, b| a - b).unwrap();
    let z = { |_12| y.apply_binary_op(&_12, |a, b| a * b).unwrap() };
    let u = ijzer::tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0], None);
    let v = (|x1, x2| (|a| -a)((|a, b| a + b)(x1, x2)))(
        ijzer::tensor::Tensor::from_vec(vec![1], None),
        ijzer::tensor::Tensor::from_vec(vec![2], None),
    );
    z(ijzer::tensor::Tensor::from_vec(vec![1.0], None))
        .apply_binary_op(
            &ijzer::tensor::Tensor::from_vec(vec![-2.0], None),
            |a, b| a * b,
        )
        .unwrap()
}
