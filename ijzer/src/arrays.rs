use anyhow::Result;
use ndarray::{Array, Dimension};
use num_traits::Num;

pub trait ArrayFunc<T, D>
where
    T: Clone + Num,
    D: Dimension,
{
    fn get_n_inputs() -> usize;
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D>;
}

pub fn call_safe<T, D, F>(inputs: &[&Array<T, D>]) -> Result<Array<T, D>, anyhow::Error>
where
    T: Clone + Num,
    D: Dimension,
    F: ArrayFunc<T, D>,
{
    if inputs.len() != F::get_n_inputs() {
        return Err(anyhow::anyhow!("Wrong number of inputs"));
    }
    let res = F::call(inputs);
    Ok(res)
}

pub struct Identity<T: Clone + Num, D: Dimension> {
    _marker: std::marker::PhantomData<(T, D)>,
}
impl<T: Clone + Num, D: Dimension> ArrayFunc<T, D> for Identity<T, D> {
    fn get_n_inputs() -> usize {
        1
    }
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D> {
        inputs[0].clone()
    }
}

pub struct Add<T: Clone + Num, D: Dimension> {
    _marker: std::marker::PhantomData<(T, D)>,
}
impl<T: Clone + Num, D: Dimension> ArrayFunc<T, D> for Add<T, D> {
    fn get_n_inputs() -> usize {
        2
    }
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D> {
        inputs[0] + inputs[1]
    }
}

pub struct Multiply<T: Clone + Num, D: Dimension> {
    _marker: std::marker::PhantomData<(T, D)>,
}
impl<T: Clone + Num, D: Dimension> ArrayFunc<T, D> for Multiply<T, D> {
    fn get_n_inputs() -> usize {
        2
    }
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D> {
        inputs[0] * inputs[1]
    }
}

pub struct Negate<T: Clone + Num, D: Dimension> {
    _marker: std::marker::PhantomData<(T, D)>,
}
impl<T: Clone + Num, D: Dimension> ArrayFunc<T, D> for Negate<T, D> {
    fn get_n_inputs() -> usize {
        1
    }
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D> {
        inputs[0].mapv(|x| T::zero() - x)
    }
}

pub struct Subtract<T: Clone + Num, D: Dimension> {
    _marker: std::marker::PhantomData<(T, D)>,
}
impl<T: Clone + Num, D: Dimension> ArrayFunc<T, D> for Subtract<T, D> {
    fn get_n_inputs() -> usize {
        2
    }
    fn call(inputs: &[&Array<T, D>]) -> Array<T, D> {
        inputs[0] - inputs[1]
    }
}
