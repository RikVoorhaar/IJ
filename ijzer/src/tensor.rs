use anyhow::{anyhow, Result};
use itertools::Itertools;
use ndarray::{s, Array, Array2, ArrayBase, Dimension, IxDyn};

use ndarray_linalg::diagonal::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::solve::*;
use ndarray_linalg::svd::*;
use ndarray_linalg::triangular::*;

use ndarray_linalg::{Lapack, Scalar};
use num_traits::{Float, Num, Zero};
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
    pub fn new(data: Box<[T]>, shape: Vec<usize>) -> Tensor<T> {
        let strides = strides_from_shape(&shape);
        let size = shape.iter().product();
        if data.len() != size {
            panic!("Data length must match the product of the shape dimensions");
        }
        Tensor {
            data,
            shape,
            strides,
        }
    }
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
        let data = vec![T::zero(); shape.iter().product()].into_boxed_slice();
        Tensor::new(data, shape)
    }
    pub fn ones(shape: &[usize]) -> Tensor<T> {
        let shape = shape.to_vec();
        let data = vec![T::one(); shape.iter().product()].into_boxed_slice();
        Tensor::new(data, shape)
    }
    pub fn eye(shape: &[usize]) -> Tensor<T> {
        let mut tensor = Self::zeros(shape);
        for i in 0..*shape.iter().min().unwrap() {
            tensor[&vec![i; shape.len()]] = T::one();
        }
        tensor
    }
    pub fn diag(tensor: &Tensor<T>) -> Tensor<T> {
        let shape_diag = [tensor.shape.clone(), tensor.shape.clone()].concat();
        let mut result = Tensor::zeros(&shape_diag);
        for i in 0..tensor.size() {
            let index = tensor.flat_to_index(i);
            let diag_index = [index.clone(), index.clone()].concat();
            result[&diag_index] = tensor[&index].clone();
        }
        result
    }
    pub fn scalar(value: T) -> Tensor<T> {
        let shape = vec![1];
        let data = vec![value].into_boxed_slice();
        Tensor::new(data, shape)
    }
    pub fn is_scalar(&self) -> bool {
        self.shape == vec![1]
    }
    pub fn extract_scalar(&self) -> Result<T> {
        if !self.is_scalar() {
            return Err(anyhow!("Tensor is not a scalar"));
        }
        Ok(self.data[0].clone())
    }

    pub fn from_vec(data: Vec<T>, shape: Option<Vec<usize>>) -> Tensor<T> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        Tensor::new(data.into_boxed_slice(), shape)
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
    pub fn transpose(&self) -> Tensor<T> {
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let mut new_tensor = Tensor::zeros(&new_shape);
        for i in 0..self.size() {
            let indices = self.flat_to_index(i);
            let mut indices_rev = indices.clone();
            indices_rev.reverse();
            new_tensor[&indices_rev] = self[&indices].clone();
        }
        new_tensor
    }
    pub fn map(&self, f: impl Fn(T) -> T) -> Tensor<T> {
        let mut new_data = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            new_data.push(f(self.data[i].clone()));
        }
        Tensor::new(new_data.into_boxed_slice(), self.shape.clone())
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
    pub fn reduce(&self, f: impl Fn(T, T) -> T) -> Tensor<T> {
        let result = self.data.iter().cloned().reduce(f).unwrap();
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
    pub fn generalized_contraction(
        &self,
        other: &Tensor<T>,
        f: impl Fn(&Tensor<T>) -> T,
        g: impl Fn(T, T) -> T,
    ) -> Result<Tensor<T>> {
        if self.shape.last() != other.shape.first() {
            return Err(anyhow!("Incompatible shapes for dot product"));
        }
        let left_shape = self.shape[..self.shape.len() - 1].to_vec();
        let right_shape = other.shape[1..].to_vec();
        let output_shape = [left_shape.clone(), right_shape.clone()].concat();
        let middle_len = *self.shape.last().unwrap();
        let mut reduc_tensor = Tensor::zeros(&[middle_len]);
        let result = match (left_shape.len(), right_shape.len()) {
            (0, 0) => {
                for idx_middle in 0..middle_len {
                    let val1 = self.index(&[idx_middle]);
                    let val2 = other.index(&[idx_middle]);
                    reduc_tensor[&[idx_middle]] = g(val1.clone(), val2.clone());
                }
                let val_reduc = f(&reduc_tensor);
                Tensor::scalar(val_reduc)
            }
            (0, _) => {
                let mut result = Tensor::zeros(&output_shape);
                for idx_right in right_shape.iter().map(|&s| 0..s).multi_cartesian_product() {
                    for idx_middle in 0..middle_len {
                        let val1 = self.index(&[idx_middle]);
                        let val2 = other.index(&[vec![idx_middle], idx_right.clone()].concat());
                        reduc_tensor[&[idx_middle]] = g(val1.clone(), val2.clone());
                    }
                    let val_reduc = f(&reduc_tensor);
                    *result.index_mut(&idx_right.clone()) = val_reduc;
                }
                result
            }
            (_, 0) => {
                let mut result = Tensor::zeros(&output_shape);
                for idx_left in left_shape.iter().map(|&s| 0..s).multi_cartesian_product() {
                    for idx_middle in 0..middle_len {
                        let val1 = self.index(&[idx_left.clone(), vec![idx_middle]].concat());
                        let val2 = other.index(&[idx_middle]);
                        reduc_tensor[&[idx_middle]] = g(val1.clone(), val2.clone());
                    }
                    let val_reduc = f(&reduc_tensor);
                    *result.index_mut(&idx_left.clone()) = val_reduc;
                }
                result
            }
            (_, _) => {
                let mut result = Tensor::zeros(&output_shape);
                for idx_left in left_shape.iter().map(|&s| 0..s).multi_cartesian_product() {
                    for idx_right in right_shape.iter().map(|&s| 0..s).multi_cartesian_product() {
                        for idx_middle in 0..middle_len {
                            let val1 = self.index(&[idx_left.clone(), vec![idx_middle]].concat());
                            let val2 = other.index(&[vec![idx_middle], idx_right.clone()].concat());
                            reduc_tensor[&[idx_middle]] = g(val1.clone(), val2.clone());
                        }
                        let val_reduc = f(&reduc_tensor);
                        *result.index_mut(&[idx_left.clone(), idx_right.clone()].concat()) =
                            val_reduc;
                    }
                }
                result
            }
        };

        Ok(result)
    }
    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        self.generalized_contraction(
            other,
            |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
            |a, b| a * b,
        )
    }
    pub fn to_ndarray2(&self) -> Array2<T> {
        let shape = self.shape.clone();
        let shape2d = match shape.len() {
            1 => (shape[0], 1),
            2 => (shape[0], shape[1]),
            _ => (shape[0], shape.iter().skip(1).product()),
        };
        let data = self.data.clone().into_vec();
        Array2::from_shape_vec(shape2d, data).unwrap()
    }
    pub fn from_ndarray<D: Dimension>(array: ArrayBase<ndarray::OwnedRepr<T>, D>) -> Tensor<T> {
        let shape = array.shape().to_vec();
        let data = array.into_iter().collect::<Vec<T>>();
        Tensor::new(data.into_boxed_slice(), shape)
    }
    pub fn to_ndarray(&self) -> Array<T, IxDyn> {
        let shape = IxDyn(&self.shape);
        let data = self.data.clone().into_vec();
        Array::from_shape_vec(shape, data).unwrap()
    }
}
impl<T: Num + Clone + Scalar + Lapack> Tensor<T> {
    pub fn qr(&self) -> Result<(Tensor<T>, Tensor<T>)> {
        let (q, r) = self.to_ndarray2().qr()?;

        Ok((Tensor::from_ndarray(q), Tensor::from_ndarray(r)))
    }
    pub fn svd(&self) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)> {
        let mat = self.to_ndarray2();
        let (m, n) = mat.dim();
        let min_dim = m.min(n);
        let (u, s, vt) = mat.svd(true, true)?;
        let s_converted = s.mapv(|x| T::from_real(x));
        let u = u.unwrap().slice(s![..m, ..min_dim]).to_owned();
        let vt = vt.unwrap().slice(s![..min_dim, ..n]).to_owned();

        Ok((
            Tensor::from_ndarray(u),
            Tensor::from_ndarray(s_converted),
            Tensor::from_ndarray(vt),
        ))
    }

    pub fn solve(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        let l_shape = self.shape[..self.shape.len() - 1].to_vec();
        let l = l_shape.iter().product();
        let m = *self.shape.last().unwrap();
        let n = other.shape[l_shape.len()..].iter().product();
        let mut a = self.clone();
        a.reshape(&[l, m])?;
        let mut b = other.clone();
        b.reshape(&[l, n])?;
        let x = match (l, m) {
            (l, m) if l == m => matrix_solve_square(&a.to_ndarray2(), &b.to_ndarray2())?,
            (l, m) if l > m => matrix_solve_overdetermined(&a.to_ndarray2(), &b.to_ndarray2())?,
            (l, m) if l < m => matrix_solve_least_squares(&a.to_ndarray2(), &b.to_ndarray2())?,
            _ => unreachable!(),
        };
        Ok(Tensor::from_ndarray(x))
    }
}

// Solve a matrix equation of form A*X = B, where A is (l,m), B is (l,n), X is (m,n). Uses LU factorization, and assumes l=m (A is square)
fn matrix_solve_square<T: Clone + Num + Scalar + Lapack>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>> {
    let (l, m) = a.dim();
    let (ll, n) = b.dim();
    if l != m {
        return Err(anyhow!("Matrix A must be square"));
    }
    if l != ll {
        return Err(anyhow!("Matrix A and B must have same number of rows"));
    }
    let lu = a.factorize()?;
let mut x = Array2::zeros((m, n));
    for i in 0..n {
        let b_col = b.column(i);
        x.column_mut(i).assign(&lu.solve(&b_col)?);
    }
    Ok(x)
}

// Solve A*X = B, where A is (l,m), B is (l,n), X is (m,n). Uses QR factorization, and assumes l>m (A is tall / overdetermined)
fn matrix_solve_overdetermined<T: Clone + Num + Scalar + Lapack>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>> {
    let (l, m) = a.dim();
    let (ll, n) = b.dim();
    if l < m {
        return Err(anyhow!("Matrix A must have more rows than columns"));
    }
    if l != ll {
        return Err(anyhow!("Matrix A and B must have same number of rows"));
    }
    let (q, r) = a.qr()?;
    let b_orth = q.t().dot(b);
    let mut x = Array2::zeros((m, n));
    for i in 0..n {
        let b_col = b_orth.column(i).to_owned();
        x.column_mut(i)
            .assign(&r.solve_triangular(UPLO::Upper, Diag::NonUnit, &b_col)?);
    }
    Ok(x)
}

// Solve A*X = B, where A is (l,m), B is (l,n), X is (m,n). Gives least square solution by solving normal equations using SVD to multiply by pseudoinverse
fn matrix_solve_least_squares<T: Clone + Num + Scalar + Lapack>(
    a: &Array2<T>,
    b: &Array2<T>,
) -> Result<Array2<T>> {
    let (l, m) = a.dim();
    let (ll, _) = b.dim();
    if l > m {
        return Err(anyhow!("Matrix A must have more columns than rows"));
    }
    if l != ll {
        return Err(anyhow!("Matrix A and B must have same number of rows"));
    }
    let (u, s, vt) = a.svd(true, true)?;
    let u = u.unwrap();
    let vt = vt.unwrap();
    // Take the first l rows of vt
    let vt_truncated = vt.slice(s![..l, ..]);

    let s_inv = s.mapv(|x| {
        if x.is_zero() {
            T::zero()
        } else {
            T::one() / T::from_real(x)
        }
    });
    let s_inv_mat = Array2::from_diag(&s_inv);
    Ok(vt_truncated.t().dot(&s_inv_mat.dot(&u.t().dot(b))))
}

impl<T: Clone + Float> Tensor<T> {
    pub fn randn(shape: &[usize]) -> Tensor<T> {
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.sample(normal)).unwrap())
            .collect();
        let shape = shape.to_vec();
        Tensor::new(data.into_boxed_slice(), shape)
    }

    pub fn randu(shape: &[usize]) -> Tensor<T> {
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(0.0, 1.0);
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.sample(uniform)).unwrap())
            .collect();
        let shape = shape.to_vec();
        Tensor::new(data.into_boxed_slice(), shape)
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
    use approx::assert_abs_diff_eq;

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
    fn test_tensor_eye() {
        let tensor = Tensor::<f64>::eye(&[3, 3]);
        assert_eq!(tensor.shape, vec![3, 3]);
        assert_eq!(tensor.strides, vec![3, 1]);
        assert_eq!(tensor.data.len(), 9);
        assert!(tensor.data.iter().enumerate().all(|(i, &x)| {
            if i % 4 == 0 {
                x == 1.0
            } else {
                x == 0.0
            }
        }));

        let tensor = Tensor::<f64>::eye(&[3, 1]);
        assert_eq!(tensor.shape, vec![3, 1]);
        assert_eq!(tensor.strides, vec![1, 1]);
        assert_eq!(tensor.data.len(), 3);
        println!("{:?}", tensor.data);
        assert!(tensor.data.iter().enumerate().all(|(i, &x)| {
            if i == 0 {
                x == 1.0
            } else {
                x == 0.0
            }
        }));

        let tensor = Tensor::<f64>::eye(&[3]);
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.strides, vec![1]);
        assert_eq!(tensor.data.len(), 3);
        println!("{:?}", tensor.data);
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
    fn test_tensor_multiply_tensor_scalar() {
        let tensor = Tensor::<f64>::ones(&[2, 2]);
        let scalar = Tensor::<f64>::scalar(3.0);
        let result = tensor.apply_binary_op(&scalar, |x, y| x * y).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert!(result.data.iter().all(|&x| x == 3.0));
    }
    #[test]
    fn test_tensor_multiply_scalar_tensor() {
        let tensor = Tensor::<f64>::ones(&[2, 2]);
        let scalar = Tensor::<f64>::scalar(3.0);
        let result = scalar.apply_binary_op(&tensor, |x, y| x * y).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert!(result.data.iter().all(|&x| x == 3.0));
    }
    #[test]
    fn test_tensor_multiply_scalar_scalar() {
        let scalar1 = Tensor::<f64>::scalar(2.0);
        let scalar2 = Tensor::<f64>::scalar(3.0);
        let result = scalar1.apply_binary_op(&scalar2, |x, y| x * y).unwrap();
        assert!(result.is_scalar());
        assert_eq!(result.data.to_vec(), vec![6.0]);
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
        assert_eq!(tensor.data.len(), 6);
        assert_eq!(tensor.size(), 6);

        let shape = vec![2, 3];
        let tensor = Tensor::<f32>::randn(&shape);
        assert_eq!(tensor.data.len(), 6);
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
        assert_eq!(tensor.data.len(), 8);
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

    #[test]
    fn test_extract_scalar() {
        // Test with a scalar tensor
        let scalar_tensor = Tensor::scalar(42.0);
        assert_eq!(scalar_tensor.extract_scalar().unwrap(), 42.0);

        // Test with a non-scalar tensor
        let non_scalar_tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], Some(vec![3]));
        assert!(non_scalar_tensor.extract_scalar().is_err());

        // Test with a tensor that has been reshaped to a scalar
        let mut reshaped_tensor = Tensor::from_vec(vec![5.0], Some(vec![1]));
        reshaped_tensor.reshape(&[1]).unwrap();
        assert_eq!(reshaped_tensor.extract_scalar().unwrap(), 5.0);
    }

    #[test]
    fn test_dot_matrix_multiplication_diagonal() {
        let tensor1 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![2.0, 0.0, 0.0, 3.0], Some(vec![2, 2]));
        let result = tensor1
            .generalized_contraction(
                &tensor2,
                |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
                |a, b| a * b,
            )
            .unwrap();
        assert_eq!(result.to_vec(), vec![2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dot_matrix_multiplication_non_diagonal() {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Some(vec![2, 2]));
        let result = tensor1
            .generalized_contraction(
                &tensor2,
                |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
                |a, b| a * b,
            )
            .unwrap();
        assert_eq!(result.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dot_matrix_multiplication_vector() {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
        let tensor2 = Tensor::from_vec(vec![3.0, 4.0], Some(vec![2]));
        let result = tensor1
            .generalized_contraction(
                &tensor2,
                |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
                |a, b| a * b,
            )
            .unwrap();
        assert_eq!(result.to_vec(), vec![11.0]);
    }

    #[test]
    fn test_dot_matrix_multiplication_vector_matrix() {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
        let tensor2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], Some(vec![2, 2]));
        let result = tensor1
            .generalized_contraction(
                &tensor2,
                |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
                |a, b| a * b,
            )
            .unwrap();
        assert_eq!(result.to_vec(), vec![13.0, 16.0]);
    }

    #[test]
    fn test_dot_matrix_multiplication_matrix_vector() {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
        let tensor2 = Tensor::from_vec(vec![5.0, 6.0], Some(vec![2]));
        let result = tensor1
            .generalized_contraction(
                &tensor2,
                |x| x.reduce(|a, b| a + b).extract_scalar().unwrap(),
                |a, b| a * b,
            )
            .unwrap();
        assert_eq!(result.to_vec(), vec![17.0, 39.0]);
    }

    #[test]
    fn test_transpose() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
        let transposed_tensor = tensor.transpose();
        assert_eq!(transposed_tensor.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![4]));
        let transposed_tensor = tensor.transpose();
        assert_eq!(transposed_tensor.to_vec(), tensor.to_vec());

        let tensor = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Some(vec![2, 2, 2]),
        );
        let transposed_tensor = tensor.transpose();
        assert_eq!(
            transposed_tensor.to_vec(),
            vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]
        );
    }

    #[test]
    fn test_to_ndarray2() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), Some(vec![2, 2]));
        let ndarray2 = tensor.to_ndarray2();
        assert_eq!(ndarray2.shape(), &[2, 2]);
        assert_eq!(
            ndarray2.clone().into_iter().collect::<Vec<_>>(),
            data.clone()
        );
        let tensor2 = Tensor::from_ndarray(ndarray2);
        assert_eq!(tensor2.to_vec(), data);
    }

    #[test]
    fn test_diag() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
        let diag = Tensor::diag(&tensor);
        assert_eq!(diag.shape(), &vec![2, 2]);
        assert_eq!(diag.to_vec(), vec![1.0, 0.0, 0.0, 2.0]);

        let tensor = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2, 1]));
        let diag = Tensor::diag(&tensor);
        assert_eq!(diag.shape(), &vec![2, 1, 2, 1]);
        assert_eq!(diag.to_vec(), vec![1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_qr() -> Result<()> {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_vec(input.clone(), Some(vec![3, 3]));
        let (q, r) = tensor.qr().unwrap();
        println!("{:?}", q.to_vec());
        println!("{:?}", r.to_vec());
        let result = q.dot(&r)?;
        for (a, b) in result.to_vec().iter().zip(input.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_svd() -> Result<()> {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = Tensor::from_vec(input.clone(), Some(vec![3, 3]));
        let (u, s, vt) = tensor.svd()?;
        let s_diag = Tensor::diag(&s);
        let result = u.dot(&s_diag)?.dot(&vt)?;
        for (a, b) in result.to_vec().iter().zip(input.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::from_vec(input.clone(), Some(vec![3, 4]));
        let (u, s, vt) = tensor.svd()?;
        let s_diag = Tensor::diag(&s);
        let result = u.dot(&s_diag)?.dot(&vt)?;
        for (a, b) in result.to_vec().iter().zip(input.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::from_vec(input.clone(), Some(vec![4, 3]));
        let (u, s, vt) = tensor.svd()?;
        let s_diag = Tensor::diag(&s);
        let result = u.dot(&s_diag)?.dot(&vt)?;
        for (a, b) in result.to_vec().iter().zip(input.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_solve() -> Result<()> {
        let a = Tensor::<f64>::randn(&[3, 3]);
        let x = Tensor::<f64>::randn(&[3, 4]);
        let b = a.clone().dot(&x)?;
        let x_sol = a.solve(&b)?;
        let result = a.dot(&x_sol)?;
        for (x_sol_, x_) in x_sol.to_vec().iter().zip(x.to_vec().iter()) {
            assert_abs_diff_eq!(x_sol_, x_, epsilon = 1e-6);
        }
        for (r, b_) in result.to_vec().iter().zip(b.to_vec().iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }

        let a = Tensor::<f64>::randn(&[4, 3]);
        let x = Tensor::<f64>::randn(&[3, 5]);
        let b = a.clone().dot(&x)?;
        let x_sol = a.solve(&b)?;
        let result = a.dot(&x_sol)?;
        for (x_sol_, x_) in x_sol.to_vec().iter().zip(x.to_vec().iter()) {
            assert_abs_diff_eq!(x_sol_, x_, epsilon = 1e-6);
        }
        for (r, b_) in result.to_vec().iter().zip(b.to_vec().iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }

        let a = Tensor::<f64>::randn(&[3, 4]);
        let x = Tensor::<f64>::randn(&[4, 5]);
        let b = a.clone().dot(&x)?;
        let x_sol = a.solve(&b)?;
        let result = a.dot(&x_sol)?;
        for (r, b_) in result.to_vec().iter().zip(b.to_vec().iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_solve_square() -> Result<()> {
        let a = Tensor::<f64>::randn(&[3, 3]).to_ndarray2();
        let x = Tensor::<f64>::randn(&[3, 4]).to_ndarray2();
        let b = a.dot(&x);
        let x_sol = matrix_solve_square(&a, &b)?;
        let result = a.dot(&x_sol);
        for (x_sol_, x_) in x_sol.iter().zip(x.iter()) {
            assert_abs_diff_eq!(x_sol_, x_, epsilon = 1e-6);
        }
        for (r, b_) in result.iter().zip(b.iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_solve_overdetermined() -> Result<()> {
        let a = Tensor::<f64>::randn(&[4, 3]).to_ndarray2();
        let x = Tensor::<f64>::randn(&[3, 5]).to_ndarray2();
        let b = a.dot(&x);
        let x_sol = matrix_solve_overdetermined(&a, &b)?;
        let result = a.dot(&x_sol);
        for (x_sol_, x_) in x_sol.iter().zip(x.iter()) {
            assert_abs_diff_eq!(x_sol_, x_, epsilon = 1e-6);
        }
        for (r, b_) in result.iter().zip(b.iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_solve_least_squares() -> Result<()> {
        let a = Tensor::<f64>::randn(&[3, 5]).to_ndarray2();
        let x = Tensor::<f64>::randn(&[5, 2]).to_ndarray2();
        let b = a.dot(&x);
        let x_sol = matrix_solve_least_squares(&a, &b)?;
        let result = a.dot(&x_sol);
        for (r, b_) in result.iter().zip(b.iter()) {
            assert_abs_diff_eq!(r, b_, epsilon = 1e-6);
        }
        Ok(())
    }
}
