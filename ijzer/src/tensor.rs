use anyhow::{anyhow, Result};
use itertools::Itertools;
use num_traits::Num;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Tensor<T: Clone + Num> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

pub fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i]
    }
    strides
}

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result = vec![];
    for (s1, s2) in shape1.iter().rev().zip(shape2.iter().rev()) {
        match (*s1, *s2) {
            (s1, s2) if s1 == s2 => result.push(s1),
            (1, s2) => result.push(s2),
            (s1, 1) => result.push(s1),
            _ => return None,
        }
    }
    result.reverse();
    Some(result)
}

impl<T: Clone + Num> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn zeros(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::zero(); shape.iter().product()].into_boxed_slice();
        return Tensor {
            shape,
            strides,
            data,
        };
    }
    pub fn ones(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::one(); shape.iter().product()].into_boxed_slice();
        return Tensor {
            shape,
            strides,
            data,
        };
    }
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        if new_shape.iter().product::<usize>() != self.size() {
            return Err(anyhow!("New shape size must be equal to the original size"));
        }
        self.shape = new_shape.to_vec();
        self.strides = strides_from_shape(&self.shape);
        Ok(())
    }
    pub fn map(&mut self, f: impl Fn(T) -> T) {
        for i in 0..self.size() {
            self.data[i] = f(self.data[i].clone());
        }
    }
    pub fn apply_binary_op(&self, other: &Tensor<T>, f: impl Fn(T, T) -> T) -> Option<Tensor<T>> {
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let mut result = Tensor::zeros(&out_shape);

        let shape1 = [
            vec![1; out_shape.len() - self.shape.len()],
            self.shape.to_vec(),
        ]
        .concat();
        let shape2 = [
            vec![1; out_shape.len() - other.shape.len()],
            other.shape.to_vec(),
        ]
        .concat();

        for idx in out_shape.iter().map(|&s| 0..s).multi_cartesian_product() {
            let idx1 = idx
                .iter()
                .zip(shape1.iter())
                .map(|(&i, &s)| i % s)
                .collect::<Vec<_>>();
            let idx2 = idx
                .iter()
                .zip(shape2.iter())
                .map(|(&i, &s)| i % s)
                .collect::<Vec<_>>();

            let val1 = self.index(&idx1);
            let val2 = other.index(&idx2);
            *result.index_mut(&idx) = f(val1.clone(), val2.clone());
        }

        Some(result)
    }
}

impl<T: Clone + Num> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        let position: usize = index
            .iter()
            .enumerate()
            .map(|(i, j)| j * self.strides[i])
            .sum();
        &self.data[position]
    }
}

impl<T: Clone + Num> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let position: usize = index
            .iter()
            .enumerate()
            .map(|(i, j)| j * self.strides[i])
            .sum();
        &mut self.data[position]
    }
}

impl<T: fmt::Display + Clone + Num> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self.shape.clone();
        if dims.is_empty() {
            return write!(f, "[]");
        }

        let mut output = String::from("[");

        let mut indices = vec![0; dims.len()];
        loop {
            let val = self.index(&indices);
            output.push_str(&format!("{}", val));

            for dim_index in (0..dims.len()).rev() {
                if indices[dim_index] + 1 < dims[dim_index] {
                    indices[dim_index] += 1;
                    if dim_index < dims.len() - 1 {
                        output.push_str(", ");
                        for _ in dim_index + 1..dims.len() {
                            output.push_str("[");
                        }
                    }
                    break;
                } else {
                    indices[dim_index] = 0;
                    if dim_index == 0 {
                        output.push_str("]");
                        return write!(f, "{}", output);
                    } else {
                        output.push_str("], ");
                    }
                }
            }
        }
    }
}
