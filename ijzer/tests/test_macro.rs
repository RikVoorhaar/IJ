use anyhow::Result;
use approx::assert_abs_diff_eq;
use ijzer::prelude::*;

#[test]
fn test_identity() {
    #[ijzer]
    fn _test_identity(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T
        x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let y = _test_identity(x.clone());
    assert_eq!(y.to_vec(), x.to_vec());
}

#[test]
fn test_assign() {
    #[ijzer]
    fn _test_assign_tensor() -> Tensor<f64> {
        r#"
        x: T = [1.0]
        x
        "#
    }

    let x = Tensor::from_vec(vec![1.0], Some(vec![1]));
    let y = _test_assign_tensor();
    assert_eq!(y.to_vec(), x.to_vec());

    #[ijzer]
    fn _test_assign_scalar() -> i64 {
        r#"
        x: N = 1
        x
        "#
    }

    let x = 1;
    let y = _test_assign_scalar();
    assert_eq!(y, x);
}

#[test]
fn test_simple_arithmetic() {
    #[ijzer]
    fn _test_add(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        + x x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2], Some(vec![2]));
    let expected = Tensor::from_vec(vec![2, 4], Some(vec![2]));
    let y = _test_add(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_sub(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        - x x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2], Some(vec![2]));
    let expected = Tensor::from_vec(vec![0, 0], Some(vec![2]));
    let y = _test_sub(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_neg(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        - x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2], Some(vec![2]));
    let expected = Tensor::from_vec(vec![-1, -2], Some(vec![2]));
    let y = _test_neg(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_mul(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        * x x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2], Some(vec![2]));
    let expected = Tensor::from_vec(vec![1, 4], Some(vec![2]));
    let y = _test_mul(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_external_function() {
    #[ijzer]
    fn f(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        + x x
        "#
    }

    #[ijzer]
    fn _test_call_f() -> Tensor<i64> {
        r#"
        var f: Fn(T->T)
        f([1, 2])
        "#
    }

    let expected = Tensor::from_vec(vec![2, 4], Some(vec![2]));
    let y = _test_call_f();
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_return_function() {
    #[ijzer]
    fn _return_function() -> fn(Tensor<i64>) -> Tensor<i64> {
        r#"
        square($x: T) -> T = * $x $x
        ~square
        "#
    }

    let x = Tensor::from_vec(vec![1, 2], Some(vec![2]));
    let square = _return_function();
    let expected = Tensor::from_vec(vec![1, 4], Some(vec![2]));
    let y = square(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_reduce() {
    #[ijzer]
    fn _test_reduce(x: Tensor<f64>) -> f64 {
        r#"
        var x: T
        /+ x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
    let expected = 3.0;
    let y = _test_reduce(x.clone());
    assert_eq!(y, expected);

    #[ijzer]
    fn _test_reduce_2(x: Tensor<f64>, f: fn(f64, f64) -> f64) -> f64 {
        r#"
        var x: T
        var f: Fn(N,N->N)
        /f x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], Some(vec![3]));
    let expected = 6.0;
    let y = _test_reduce_2(x.clone(), |a, b| a * b);
    assert_eq!(y, expected);
}

#[test]
fn test_function_composition() {
    #[ijzer]
    fn _test_function_composition() -> i64 {
        r#"
        @(/+,+) [1,2]<i64> [3,4]<i64>
        "#
    }

    let y = _test_function_composition();
    assert_eq!(y, 10);

    #[ijzer]
    fn _test_external_function_composition(f: fn(Tensor<i64>, Tensor<i64>) -> Tensor<i64>) -> i64 {
        r#"
        var f: Fn(T,T->T)
        @(/+, f) [1,2] [3,4] 
        "#
    }

    let y =
        _test_external_function_composition(|a, b| a.apply_binary_op(&b, |x, y| x * y).unwrap());
    assert_eq!(y, 11);
}

#[test]
fn test_type_conversion() {
    #[ijzer]
    fn _test_type_conversion(x: i64) -> Tensor<i64> {
        r#"
        var x: N
        <-T x
        "#
    }

    let x = 1;
    let expected = Tensor::scalar(1);
    let y = _test_type_conversion(x);
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_type_conversion_func(f: fn(Tensor<i64>, Tensor<i64>) -> i64) -> i64 {
        r#"
        var f: Fn(T,T->N)
        /<-Fn(N,N->N) f [1,2,3,4,5]
        "#
    }

    let f = |a: Tensor<i64>, b: Tensor<i64>| {
        a.apply_binary_op(&b, |x, y| x * y)
            .unwrap()
            .extract_scalar()
            .unwrap()
    };
    let expected = 120;
    let y = _test_type_conversion_func(f);
    assert_eq!(y, expected);
}

#[test]
fn test_apply_higher_function() {
    type ScalarFun = Box<dyn Fn(i64) -> i64>;
    #[ijzer]
    fn _test_apply(f: fn(i64) -> ScalarFun) -> i64 {
        r#"
        var f: Fn(N-> Fn(N->N))
        .(f 2) 4
        "#
    }

    /// Turns `x` into a function that multiplies by `x`
    fn f(x: i64) -> ScalarFun {
        Box::new(move |y: i64| x * y)
    }

    let expected = 8;
    let y = _test_apply(f);
    assert_eq!(y, expected);
}

#[test]
fn test_matrix_multiplication() {
    #[ijzer]
    fn _test_matrix_multiplication(x: Tensor<i64>, y: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        var y: T
        ?/+* x y
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let expected = Tensor::from_vec(vec![7, 10, 15, 22], Some(vec![2, 2]));
    let z = _test_matrix_multiplication(x.clone(), y.clone());
    assert_eq!(z.to_vec(), expected.to_vec());
}

#[test]
fn test_tensor_builder() {
    #[ijzer]
    fn _test_tensor_builder_ones() -> Tensor<i64> {
        r#"
        ones [2,3]<usize>
        "#
    }
    let expected = Tensor::ones(&[2, 3]);
    let y = _test_tensor_builder_ones();
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_tensor_builder_zeros(shape: Tensor<usize>) -> Tensor<i64> {
        r#"
        var shape: T<usize>
        zeros shape
        "#
    }
    let shape = Tensor::from_vec(vec![2, 2, 1], None);
    let expected = Tensor::zeros(&[2, 2, 1]);
    let y = _test_tensor_builder_zeros(shape);
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_tensor_builder_eye() -> Tensor<f64> {
        r#"
        x = eye<f64> [3,3]
        y = ones<f64> [3,3]
        ?/+* x y
        "#
    }
    let expected = Tensor::ones(&[3, 3]);
    let y = _test_tensor_builder_eye();
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_tensor_builder_randu_randn() -> Tensor<f64> {
        r#"
        A = randu<f64> [3,4]
        B = randn<f64> [4,5]
        ?/+* A B
        "#
    }
    let y = _test_tensor_builder_randu_randn();
    assert_eq!(y.shape(), &[3, 5]);
}

#[test]
fn test_apply_unary() {
    #[ijzer]
    fn _test_apply(x: Tensor<i64>) -> i64 {
        r#"
        var x: T
        .~/+ x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = _test_apply(x.clone());
    assert_eq!(y, 10);
}

#[test]
fn test_transpose() {
    #[ijzer]
    fn _test_transpose_functional(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        .~| x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let expected = Tensor::from_vec(vec![1, 3, 2, 4], Some(vec![2, 2]));
    let y = _test_transpose_functional(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_transpose_method(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T
        |x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let expected = Tensor::from_vec(vec![1, 3, 2, 4], Some(vec![2, 2]));
    let y = _test_transpose_method(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_shape() {
    #[ijzer]
    fn _test_shape(x: Tensor<i64>) -> Tensor<usize> {
        r#"
        var x: T<i64>
        %x
        "#
    }
    #[ijzer]
    fn _test_functional(x: Tensor<i64>) -> Tensor<usize> {
        r#"
        var x: T<i64>
        .~%x
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let expected = x.shape();
    let y = _test_shape(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());
    let y = _test_functional(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_group_return() {
    #[ijzer]
    fn _test_group_return() -> Tensor<i64> {
        r#"
        (x,y) = ( [1,2,3,4]<i64>, [5,6,7,8]<i64>)
        * x y
        "#
    }

    let expected = Tensor::from_vec(vec![5, 12, 21, 32], Some(vec![2, 2]));
    let y = _test_group_return();
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_qr() -> Result<()> {
    #[ijzer]
    fn _test_qr(x: Tensor<f64>) -> (Tensor<f64>, Tensor<f64>) {
        r#"
        var x: T<f64>
        (q,r) = qr x
        (q,r)
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let (q, r) = _test_qr(x.clone());
    let (q_real, r_real) = x.qr()?;
    assert_eq!(q.to_vec(), q_real.to_vec());
    assert_eq!(r.to_vec(), r_real.to_vec());

    Ok(())
}

#[test]
fn test_svd() -> Result<()> {
    #[ijzer]
    fn _test_svd(x: Tensor<f64>) -> (Tensor<f64>, Tensor<f64>, Tensor<f64>) {
        r#"
        var x: T<f64>
        (u,s,vt) = svd x
        (u,s,vt)
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let (u, s, vt) = _test_svd(x.clone());
    let (u_real, s_real, vt_real) = x.svd()?;
    assert_eq!(u.to_vec(), u_real.to_vec());
    assert_eq!(s.to_vec(), s_real.to_vec());
    assert_eq!(vt.to_vec(), vt_real.to_vec());

    Ok(())
}

#[test]
fn test_solve() -> Result<()> {
    #[ijzer]
    fn _test_solve(a: Tensor<f64>, b: Tensor<f64>) -> Tensor<f64> {
        r#"
        var a: T<f64>
        var b: T<f64>
        \ a b
        "#
    }

    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let b = Tensor::from_vec(vec![5.0, 6.0], Some(vec![2]));
    let x = _test_solve(a.clone(), b.clone());
    let x_real = a.solve(&b)?;
    assert_eq!(x.to_vec(), x_real.to_vec());

    Ok(())
}

#[test]
fn test_diag() -> Result<()> {
    #[ijzer]
    fn _test_diag(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        diag x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let d = _test_diag(x.clone());
    let d_real = Tensor::diag(&x);
    assert_eq!(d.to_vec(), d_real.to_vec());

    Ok(())
}

#[test]
fn test_svd_diag_mul() -> Result<()> {
    #[ijzer]
    fn _test_svd_diag_mul(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        (u,s,vt) = svd x
        sd = diag s
        ?/+* ?/+* u sd vt
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let y = _test_svd_diag_mul(x.clone());
    for (a, b) in y.to_vec().iter().zip(x.to_vec().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }

    Ok(())
}

#[test]
fn test_indexing() {
    #[ijzer]
    fn _test_indexing_scalar(x: Tensor<i64>) -> i64 {
        r#"
        var x: T<i64>
        <|x[0,0]
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = _test_indexing_scalar(x.clone());
    assert_eq!(y, 1);

    #[ijzer]
    fn _test_indexing_tensor(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T<i64>
        y1 = [0]
        y2 = [0,1]
        <|x[y1,y2]
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = _test_indexing_tensor(x.clone());
    assert_eq!(y.to_vec(), vec![1, 2]);

    #[ijzer]
    fn _test_indexing_sub_tensor(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T<i64>
        <|x[0,:]
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = _test_indexing_sub_tensor(x.clone());
    assert_eq!(y.to_vec(), vec![1, 2]);
}

#[test]
fn test_array_nested() {
    #[ijzer]
    fn _test_array_nested(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T<i64>
        y = [x,x,x]
        y
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = _test_array_nested(x.clone());
    assert_eq!(y.shape(), &[3, 2, 2]);
    assert_eq!(y.to_vec(), vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
}

#[test]
fn test_add_array_scalar() {
    #[ijzer]
    fn _test_add_array_scalar(x: Tensor<i64>, y: i64) -> Tensor<i64> {
        r#"
        var x: T<i64>
        var y: N<i64>
        + x y
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4], Some(vec![2, 2]));
    let y = 1;
    let z = _test_add_array_scalar(x, y);
    assert_eq!(z.to_vec(), vec![2, 3, 4, 5]);
}

#[test]
fn test_reshape() {
    #[ijzer]
    fn _test_reshape(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T<i64>
        >% x [2,3]
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], None);
    let y = _test_reshape(x.clone());
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    #[ijzer]
    fn _test_reshape_functional(x: Tensor<i64>) -> Tensor<i64> {
        r#"
        var x: T<i64>
        .~>% x [2,3]
        "#
    }

    let x = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], None);
    let y = _test_reshape_functional(x.clone());
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_unary_function() {
    #[ijzer]
    fn _test_unary_function_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        sin x 
        "#
    }

    let pi = std::f64::consts::PI;
    let x = Tensor::from_vec(
        vec![pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi],
        Some(vec![2, 2]),
    );
    let y = _test_unary_function_on_tensor(x);
    let expected = [1.0, 0.0, -1.0, 0.0];
    let actual = y.to_vec();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-6);
    }

    #[ijzer]
    fn _test_unary_function_on_number(x: f64) -> f64 {
        r#"
        var x: N<f64>
        sin x 
        "#
    }

    let x = 1.0 * pi;
    let y = _test_unary_function_on_number(x);
    assert_abs_diff_eq!(y, 0.0, epsilon = 1e-6);

    #[ijzer]
    fn _test_unary_function_on_tensor_functional(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        .~ sin: Fn(T<f64>->T<f64>) x
        "#
    }

    let pi = std::f64::consts::PI;
    let x = Tensor::from_vec(
        vec![pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi],
        Some(vec![2, 2]),
    );
    let y = _test_unary_function_on_tensor_functional(x);
    let expected = [1.0, 0.0, -1.0, 0.0];
    let actual = y.to_vec();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-6);
    }

    #[ijzer]
    fn _test_unary_function_on_number_functional(x: f64) -> f64 {
        r#"
        var x: N<f64>
        .~ sin: Fn(N<f64>->N<f64>) x
        "#
    }

    let x = 1.0 * pi;
    let y = _test_unary_function_on_number_functional(x);
    assert_abs_diff_eq!(y, 0.0, epsilon = 1e-6);
}

#[test]
fn test_division() {
    #[ijzer]
    fn _test_division_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: T<f64>
        /: x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let z = _test_division_tensor_tensor(x, y);
    assert_eq!(z.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);

    #[ijzer]
    fn _test_division_tensor_scalar(x: Tensor<f64>, y: f64) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: N<f64>
        /: x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let y = 2.0;
    let z = _test_division_tensor_scalar(x, y);
    assert_eq!(z.to_vec(), vec![0.5, 1.0, 1.5, 2.0]);

    #[ijzer]
    fn _test_division_scalar_tensor(x: f64, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: N<f64>
        var y: T<f64>
        /: x y
        "#
    }

    let x = 2.0;
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let z = _test_division_scalar_tensor(x, y);
    assert_eq!(z.to_vec(), vec![2.0, 1.0, 2.0 / 3.0, 0.5]);

    #[ijzer]
    fn _test_division_scalar_scalar(x: f64, y: f64) -> f64 {
        r#"
        var x: N<f64>
        var y: N<f64>
        /: x y
        "#
    }

    let x = 2.0;
    let y = 1.0;
    let z = _test_division_scalar_scalar(x, y);
    assert_eq!(z, 2.0);
}

#[test]
fn test_pow() {
    #[ijzer]
    fn _test_pow_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: T<f64>
        ^ x y
        "#
    }

    let x = Tensor::from_vec(vec![2.0, 3.0, 4.0], Some(vec![3]));
    let y = Tensor::from_vec(vec![2.0, 3.0, 4.0], Some(vec![3]));
    let z = _test_pow_tensor_tensor(x, y);
    assert_eq!(z.to_vec(), vec![4.0, 27.0, 256.0]);

    #[ijzer]
    fn _test_pow_tensor_scalar(x: Tensor<f64>, y: f64) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: N<f64>
        ^ x y
        "#
    }

    let x = Tensor::from_vec(vec![2.0, 3.0, 4.0], Some(vec![3]));
    let y = 2.0;
    let z = _test_pow_tensor_scalar(x, y);
    assert_eq!(z.to_vec(), vec![4.0, 9.0, 16.0]);

    #[ijzer]
    fn _test_pow_scalar_tensor(x: f64, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: N<f64>
        var y: T<f64>
        ^ x y
        "#
    }

    let x = 2.0;
    let y = Tensor::from_vec(vec![2.0, 3.0, 4.0], Some(vec![3]));
    let z = _test_pow_scalar_tensor(x, y);
    assert_eq!(z.to_vec(), vec![4.0, 8.0, 16.0]);

    #[ijzer]
    fn _test_pow_scalar_scalar(x: f64, y: f64) -> f64 {
        r#"
        var x: N<f64>
        var y: N<f64>
        ^ x y
        "#
    }

    let x = 2.0;
    let y = 2.0;
    let z = _test_pow_scalar_scalar(x, y);
    assert_eq!(z, 4.0);
}

#[test]
fn test_max() {
    #[ijzer]
    fn _test_max_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: T<f64>
        max x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
    let z = _test_max_tensor_tensor(x, y);
    assert_eq!(z.to_vec(), vec![1.0, 4.0, 9.0, 16.0]);

    #[ijzer]
    fn _test_max_tensor_scalar(x: Tensor<f64>, y: f64) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: N<f64>
        max x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let y = 4.0;
    let z = _test_max_tensor_scalar(x, y);
    assert_eq!(z.to_vec(), vec![4.0, 4.0, 9.0, 16.0]);

    #[ijzer]
    fn _test_max_scalar_tensor(x: f64, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: N<f64>
        var y: T<f64>
        max x y
        "#
    }

    let x = 4.0;
    let y = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let z = _test_max_scalar_tensor(x, y);
    assert_eq!(z.to_vec(), vec![4.0, 4.0, 9.0, 16.0]);

    #[ijzer]
    fn _test_max_scalar_scalar(x: f64, y: f64) -> f64 {
        r#"
        var x: N<f64>
        var y: N<f64>
        max x y
        "#
    }

    let x = 4.0;
    let y = 2.0;
    let z = _test_max_scalar_scalar(x, y);
    assert_eq!(z, 4.0);
}

#[test]
fn test_min() {
    #[ijzer]
    fn _test_min_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: T<f64>
        min x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
    let z = _test_min_tensor_tensor(x, y);
    assert_eq!(z.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_equal() {
    #[ijzer]
    fn _test_equal_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: T<f64>
        == x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
    let z = _test_equal_tensor_tensor(x, y);
    assert_eq!(z.to_vec(), vec![1.0, 1.0, 0.0, 0.0]);

    #[ijzer]
    fn _test_equal_tensor_scalar(x: Tensor<f64>, y: f64) -> Tensor<f64> {
        r#"
        var x: T<f64>
        var y: N<f64>
        == x y
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let y = 2.0;
    let z = _test_equal_tensor_scalar(x, y);
    assert_eq!(z.to_vec(), vec![0.0, 1.0, 0.0, 0.0]);

    #[ijzer]
    fn _test_equal_scalar_tensor(x: f64, y: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: N<f64>
        var y: T<f64>
        == x y
        "#
    }

    let x = 2.0;
    let y = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
    let z = _test_equal_scalar_tensor(x, y);
    assert_eq!(z.to_vec(), vec![0.0, 1.0, 0.0, 0.0]);

    #[ijzer]
    fn _test_equal_scalar_scalar(x: f64, y: f64) -> f64 {
        r#"
        var x: N<f64>
        var y: N<f64>
        == x y
        "#
    }

    let x = 2.0;
    let y = 2.0;
    let z = _test_equal_scalar_scalar(x, y);
    assert_eq!(z, 1.0);
}
