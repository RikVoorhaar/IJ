use num_traits::Num;
use std::{ops::Index, process::Output};

#[derive(Clone)]
struct Tensor<T: Clone + Num> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i]
    }
    strides
}

impl<T: Clone + Num> Tensor<T> {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn zeros(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::zero(); shape.iter().product()].into_boxed_slice();
        return Tensor {
            shape,
            strides,
            data,
        };
    }
    fn ones(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::one(); shape.iter().product()].into_boxed_slice();
        return Tensor {
            shape,
            strides,
            data,
        };
    }
}

impl<T: Clone + Num> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> Self::Output {
        let position: usize = index
            .iter()
            .enumerate()
            .map(|(i, j)| j * self.strides[i])
            .sum();
        self.data[position].clone()
    }
}
