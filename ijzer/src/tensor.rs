use anyhow::{anyhow, Result};
use itertools::Itertools;
use num_traits::{Float, Num};
use rand::Rng;
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
    for i in (1..shape.len()).rev() {
        strides[i - 1] = strides[i] * shape[i]
    }
    strides
}

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result = vec![];
    let max_len = std::cmp::max(shape1.len(), shape2.len());
    let shape1_padded = shape1
        .iter()
        .rev()
        .chain(std::iter::repeat(&1))
        .take(max_len);
    let shape2_padded = shape2
        .iter()
        .rev()
        .chain(std::iter::repeat(&1))
        .take(max_len);
    for (s1, s2) in shape1_padded.zip(shape2_padded) {
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

    pub fn flat_to_index(&self, index: usize) -> Vec<usize> {
        let mut result = vec![0; self.shape.len()];
        for (i, &s) in self.strides.iter().enumerate() {
            result[i] = (index / s) % self.shape[i];
        }
        result
    }

    pub fn zeros(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::zero(); shape.iter().product()].into_boxed_slice();
        Tensor {
            shape,
            strides,
            data,
        }
    }
    pub fn ones(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        let data = vec![T::one(); shape.iter().product()].into_boxed_slice();
        Tensor {
            shape,
            strides,
            data,
        }
    }
    pub fn scalar(value: T) -> Tensor<T> {
        let shape = vec![1];
        let strides = vec![1];
        let data = vec![value].into_boxed_slice();
        Tensor {
            shape,
            strides,
            data,
        }
    }
    pub fn from_vec(data: Vec<T>, shape: Option<Vec<usize>>) -> Tensor<T> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let strides = strides_from_shape(&shape);
        Tensor {
            shape,
            strides,
            data: data.into_boxed_slice(),
        }
    }
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone().into_vec()
    }
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        if new_shape.iter().product::<usize>() != self.size() {
            return Err(anyhow!("New shape size must be equal to the original size"));
        }
        self.shape = new_shape.to_vec();
        self.strides = strides_from_shape(&self.shape);
        Ok(())
    }
    pub fn map(&self, f: impl Fn(T) -> T) -> Tensor<T> {
        let mut new_data = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            new_data.push(f(self.data[i].clone()));
        }
        Tensor {
            data: new_data.into_boxed_slice(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
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
                .skip(out_shape.len() - self.shape.len())
                .map(|(&i, &s)| i % s)
                .collect::<Vec<_>>();
            let idx2 = idx
                .iter()
                .zip(shape2.iter())
                .skip(out_shape.len() - other.shape.len())
                .map(|(&i, &s)| i % s)
                .collect::<Vec<_>>();

            let val1 = self.index(&idx1);
            let val2 = other.index(&idx2);
            *result.index_mut(&idx) = f(val1.clone(), val2.clone());
        }

        Some(result)
    }
    pub fn reduce(&self, mut f: impl FnMut(&T, &T) -> T) -> Tensor<T> {
        let result = self.data.iter().cloned().reduce(|x, y| f(&x, &y)).unwrap();
        Tensor::scalar(result)
    }
    pub fn sub_tensor(&self, indices: Vec<Option<usize>>) -> Result<Tensor<T>, &'static str> {
        if indices.len() != self.shape.len() {
            return Err("Indices length must match tensor shape length");
        }

        let mut sub_shape = Vec::new();
        let mut sub_locations = Vec::new();

        for (i, &index) in indices.iter().enumerate() {
            match index {
                Some(idx) => {
                    if idx >= self.shape[i] {
                        return Err("Index out of bounds for tensor shape");
                    }
                }
                None => {
                    sub_shape.push(self.shape[i]);
                    sub_locations.push(i);
                }
            }
        }

        let iter_dims = sub_shape.iter().map(|&s| 0..s).multi_cartesian_product();
        let mut final_idx = vec![0; self.shape.len()];
        for (i, &val) in indices.iter().enumerate() {
            if let Some(idx) = val {
                final_idx[i] = idx;
            }
        }

        let mut output: Tensor<T> = Tensor::zeros(&sub_shape);
        for idx in iter_dims {
            for (i, &val) in idx.iter().enumerate() {
                final_idx[sub_locations[i]] = val;
            }
            output[&idx] = self[&final_idx].clone();
        }
        Ok(output)
    }
}
impl<T: Clone + Float> Tensor<T> {
    pub fn randn(shape: &[usize]) -> Tensor<T> {
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let data: Vec<T> = shape
            .iter()
            .map(|_| T::from(rng.sample(normal)).unwrap())
            .collect();
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        Tensor {
            shape,
            strides,
            data: data.into_boxed_slice(),
        }
    }

    pub fn randu(shape: &[usize]) -> Tensor<T> {
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(0.0, 1.0);
        let data: Vec<T> = shape
            .iter()
            .map(|_| T::from(rng.sample(uniform)).unwrap())
            .collect();
        let shape = shape.to_vec();
        let strides = strides_from_shape(&shape);
        Tensor {
            shape,
            strides,
            data: data.into_boxed_slice(),
        }
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

        let mut output = String::new();
        for i in 0..self.size() {
            let indices = self.flat_to_index(i);
            let num_zeros = indices.iter().rev().take_while(|&&x| x == 0).count();
            let num_non_zeros = indices.len() - num_zeros;
            if num_zeros > 0 {
                output.push_str(&" ".repeat(num_non_zeros));
                output.push_str(&"[".repeat(num_zeros));
            }
            output.push_str(&format!("{}", self.index(&indices)));
            let num_full = indices
                .iter()
                .zip(self.shape.iter())
                .rev()
                .take_while(|(&i, &s)| i + 1 == s)
                .count();
            output.push_str(&"]".repeat(num_full));
            if num_full < self.shape.len() {
                output.push_str(", ");
            }
            if num_full > 0 && num_full < self.shape.len() {
                output.push('\n');
            }
        }
        write!(f, "{}", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::<f64>::zeros(&[1, 3, 4]);
        assert_eq!(tensor.shape, vec![1, 3, 4]);
        assert_eq!(tensor.strides, vec![12, 4, 1]);
        assert_eq!(tensor.data.len(), 12);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::<f64>::ones(&[1, 3, 4]);
        assert_eq!(tensor.shape, vec![1, 3, 4]);
        assert_eq!(tensor.strides, vec![12, 4, 1]);
        assert_eq!(tensor.data.len(), 12);
        assert!(tensor.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[1, 3, 4], &[3, 4]), Some(vec![1, 3, 4]));
        assert_eq!(
            broadcast_shapes(&[1, 3, 4], &[1, 3, 4]),
            Some(vec![1, 3, 4])
        );
        assert_eq!(
            broadcast_shapes(&[1, 3, 4], &[1, 1, 4]),
            Some(vec![1, 3, 4])
        );
        assert_eq!(broadcast_shapes(&[1, 3, 4], &[1, 1]), Some(vec![1, 3, 4]));
        assert_eq!(
            broadcast_shapes(&[1, 3, 4], &[1, 1, 1, 1]),
            Some(vec![1, 1, 3, 4])
        );
        assert_eq!(
            broadcast_shapes(&[1, 3, 4], &[1, 1, 1]),
            Some(vec![1, 3, 4])
        );
    }

    #[test]
    fn test_tensor_map() {
        let tensor = Tensor::<f64>::ones(&[1, 3, 4]);
        let tensor = tensor.map(|x| x + 1.0);
        assert!(tensor.data.iter().all(|&x| x == 2.0));
        let tensor = tensor.map(|x| x * 2.0);
        assert!(tensor.data.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn test_tensor_apply_binary_op() {
        let tensor1 = Tensor::<f64>::ones(&[1, 3, 4]);
        let tensor2 = Tensor::<f64>::ones(&[2, 1, 4]);
        let tensor3 = tensor1.apply_binary_op(&tensor2, |x, y| x + y).unwrap();
        assert!(tensor3.data.iter().all(|&x| x == 2.0));
        assert_eq!(tensor3.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_tensor_multiply_scalar() {
        let tensor = Tensor::<f64>::ones(&[2, 2]);
        let scalar = Tensor::<f64>::scalar(3.0);
        let result = tensor.apply_binary_op(&scalar, |x, y| x * y).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert!(result.data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_scalar_construction() {
        let scalar_value = 5.0;
        let tensor = Tensor::<f64>::scalar(scalar_value);
        assert_eq!(tensor.shape, vec![1]);
        assert_eq!(tensor.strides, vec![1]);
        assert_eq!(tensor.data.len(), 1);
        assert_eq!(tensor.data[0], scalar_value);
    }

    #[test]
    fn test_from_vec() {
        let data_u64 = vec![1u64, 2, 3, 4];
        let tensor_u64 = Tensor::from_vec(data_u64, Some(vec![2, 2]));
        assert_eq!(tensor_u64.shape, vec![2, 2]);
        assert_eq!(tensor_u64.data.to_vec(), vec![1u64, 2, 3, 4]);

        let data_f64 = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor_f64 = Tensor::from_vec(data_f64, Some(vec![2, 2]));
        assert_eq!(tensor_f64.shape, vec![2, 2]);
        assert_eq!(tensor_f64.data.to_vec(), vec![1.0f64, 2.0, 3.0, 4.0]);

        let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor_f32 = Tensor::from_vec(data_f32, Some(vec![2, 2]));
        assert_eq!(tensor_f32.shape, vec![2, 2]);
        assert_eq!(tensor_f32.data.to_vec(), vec![1.0f32, 2.0, 3.0, 4.0]);

        let data_i64 = vec![1i64, -2, 3, -4];
        let tensor_i64 = Tensor::from_vec(data_i64, Some(vec![2, 2]));
        assert_eq!(tensor_i64.shape, vec![2, 2]);
        assert_eq!(tensor_i64.data.to_vec(), vec![1i64, -2, 3, -4]);

        let data_usize = vec![1usize, 2, 3, 4];
        let tensor_usize = Tensor::from_vec(data_usize, Some(vec![2, 2]));
        assert_eq!(tensor_usize.shape, vec![2, 2]);
        assert_eq!(tensor_usize.data.to_vec(), vec![1usize, 2, 3, 4]);
    }

    #[test]
    fn test_reshape_success() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut tensor = Tensor::from_vec(data.clone(), Some(vec![2, 3]));
        assert!(tensor.reshape(&[3, 2]).is_ok());
        assert_eq!(tensor.shape, vec![3, 2]);
        assert_eq!(tensor.index(&[0, 1]), &1.0);
        assert_eq!(tensor.index(&[2, 1]), &5.0);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_reshape_failure_due_to_incompatible_size() {
        let mut tensor = Tensor::<f64>::ones(&[2, 3]);
        assert!(tensor.reshape(&[4, 2]).is_err());
    }

    #[test]
    fn test_reshape_failure_due_to_zero_dimension() {
        let mut tensor = Tensor::<f64>::ones(&[2, 3]);
        assert!(tensor.reshape(&[0, 6]).is_err());
    }

    #[test]
    fn test_map_division() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let tensor = Tensor::from_vec(data, Some(vec![2, 2]));
        let result = tensor.map(|x| x / 2.0);
        assert_eq!(result.to_vec(), vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_map_overflow() {
        let data = vec![u8::MAX, 0];
        let tensor = Tensor::from_vec(data, Some(vec![2]));
        let result = tensor.map(|x| x.wrapping_add(1));
        assert_eq!(result.to_vec(), vec![0, 1]);
    }

    #[test]
    fn test_map_underflow() {
        let data = vec![0u8, 1];
        let tensor = Tensor::from_vec(data, Some(vec![2]));
        let result = tensor.map(|x| x.wrapping_sub(1));
        assert_eq!(result.to_vec(), vec![u8::MAX, 0]);
    }

    #[test]
    fn test_apply_binary_op_subtraction() {
        let tensor1 = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
        let result = tensor1.apply_binary_op(&tensor2, |a, b| a - b).unwrap();
        assert_eq!(result.to_vec(), vec![9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_apply_binary_op_division() {
        let tensor1 = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![2.0, 4.0, 5.0, 10.0], Some(vec![2, 2]));
        let result = tensor1.apply_binary_op(&tensor2, |a, b| a / b).unwrap();
        assert_eq!(result.to_vec(), vec![5.0, 5.0, 6.0, 4.0]);
    }

    #[test]
    fn test_apply_binary_op_non_commutative() {
        let tensor1 = Tensor::from_vec(vec![2, 4, 6, 8], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
        let result_forward = tensor1.apply_binary_op(&tensor2, |a, b| a - b).unwrap();
        let result_backward = tensor2.apply_binary_op(&tensor1, |a, b| a - b).unwrap();
        assert_eq!(result_forward.to_vec(), vec![1, 2, 3, 4]);
        assert_ne!(result_forward.to_vec(), result_backward.to_vec());
    }

    #[test]
    fn test_randn_constructor() {
        let shape = vec![2, 3];
        let tensor = Tensor::<f64>::randn(&shape);
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_randu_constructor() {
        let shape = vec![4, 2];
        let tensor = Tensor::<f64>::randu(&shape);
        assert!(tensor.to_vec().iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 8);
    }

    #[test]
    fn test_reduce_multiplication() {
        let tensor = Tensor::from_vec(vec![2, 3, 4], Some(vec![3]));
        let result = tensor.reduce(|a, b| a * b);
        assert_eq!(result.to_vec(), vec![24]);
    }

    #[test]
    fn test_reduce_summation() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![4]));
        let result = tensor.reduce(|a, b| a + b);
        assert_eq!(result.to_vec(), vec![10]);
    }

    #[test]
    fn test_sub_tensor() {
        let tensor = Tensor::from_vec((0..24).collect(), Some(vec![2, 3, 4]));

        // Sub-tensor to get a 2x4 tensor
        let sub_tensor_2x4 = tensor.sub_tensor(vec![None, Some(1), None]).unwrap();
        assert_eq!(sub_tensor_2x4.shape(), &vec![2, 4]);
        assert_eq!(sub_tensor_2x4.to_vec(), vec![4, 5, 6, 7, 16, 17, 18, 19]);

        // Sub-tensor to get a 2 tensor
        let sub_tensor_2 = tensor.sub_tensor(vec![None, Some(1), Some(2)]).unwrap();
        assert_eq!(sub_tensor_2.shape(), &vec![2]);
        assert_eq!(sub_tensor_2.to_vec(), vec![6, 18]);

        // Sub-tensor to get a 3 tensor
        let sub_tensor_3 = tensor.sub_tensor(vec![Some(0), None, Some(1)]).unwrap();
        assert_eq!(sub_tensor_3.shape(), &vec![3]);
        assert_eq!(sub_tensor_3.to_vec(), vec![1, 5, 9]);
    }
}
