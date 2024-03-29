
use anyhow::Result;
use ijzer::arrays::{self, ArrayFunc};
use ndarray::array;

fn main() -> Result<()> {
    let array1 = array![1, 2, 3];
    let array2 = array![0, 2, 4];
    let new_array = {
        let mut _temp = arrays::Multiply::call(&[&array1, &array2]);
        _temp.remove(0)
    };
    println!("{:?}", new_array);

    Ok(())
}
