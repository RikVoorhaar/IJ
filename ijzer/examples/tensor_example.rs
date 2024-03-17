use anyhow::Result;
use ijzer::tensor::Tensor;

fn main() -> Result<()> {
    let t1 = Tensor::<f64>::zeros(&[1, 3, 4]);
    let t2 = Tensor::<f64>::ones(&[2, 1, 4]);
    // let t3 = t1.apply_binary_op(&t2, |a, b| a + b).unwrap();
    let t4 = Tensor::<_>::scalar(2.5);
    let t5 = t2.apply_binary_op(&t4, |a, b| a * b).unwrap();

    //Next we need some way to create scalars, and to load tensors from arbitrary data
    // such as [1.0, 2.0, 3.0] or something
    // Then we need to convert to and from ndarray
    //finally we need some unit tests.
    // println!("{}", t3);
    println!("{}", t5);
    Ok(())
}
