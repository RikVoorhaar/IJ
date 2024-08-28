// extern crate ijzer_macro;
use ijzer::tensor::Tensor;
use ijzer_macro::ijzer;

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
    fn _test_assign_scalar() -> Tensor<i64> {
        r#"
        x: S = 1
        x
        "#
    }

    let x = Tensor::scalar(1);
    let y = _test_assign_scalar();
    assert_eq!(y.to_vec(), x.to_vec());
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
    fn _test_reduce(x: Tensor<f64>) -> Tensor<f64> {
        r#"
        var x: T
        /+ x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2]));
    let expected = Tensor::from_vec(vec![3.0], Some(vec![1]));
    let y = _test_reduce(x.clone());
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_reduce_2(x: Tensor<f64>, f: fn(f64, f64) -> f64) -> Tensor<f64> {
        r#"
        var x: T
        var f: Fn(N,N->N)
        /f x
        "#
    }

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], Some(vec![3]));
    let expected = Tensor::from_vec(vec![6.0], Some(vec![1]));
    let y = _test_reduce_2(x.clone(), |a, b| a * b);
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_function_composition() {
    #[ijzer]
    fn _test_function_composition() -> Tensor<i64> {
        r#"
        @(/+,+) [1,2]<i64> [3,4]<i64>
        "#
    }

    let expected = Tensor::from_vec(vec![10], Some(vec![1]));
    let y = _test_function_composition();
    assert_eq!(y.to_vec(), expected.to_vec());

    #[ijzer]
    fn _test_external_function_composition(
        f: fn(Tensor<i64>, Tensor<i64>) -> Tensor<i64>,
    ) -> Tensor<i64> {
        r#"
        var f: Fn(T,T->T)
        @(/+, f) [1,2] [3,4] 
        "#
    }

    let expected = Tensor::from_vec(vec![11], Some(vec![1]));
    let y =
        _test_external_function_composition(|a, b| a.apply_binary_op(&b, |x, y| x * y).unwrap());
    assert_eq!(y.to_vec(), expected.to_vec());
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
    fn _test_type_conversion_func(f: fn(Tensor<i64>, Tensor<i64>) -> Tensor<i64>) -> Tensor<i64> {
        r#"
        var f: Fn(S,S->S)
        /<-Fn(N,N->N) f [1,2,3,4,5]
        "#
    }

    let f = |a: Tensor<i64>, b: Tensor<i64>| a.apply_binary_op(&b, |x, y| x * y).unwrap();
    let expected = Tensor::from_vec(vec![120], Some(vec![1]));
    let y = _test_type_conversion_func(f);
    assert_eq!(y.to_vec(), expected.to_vec());
}

#[test]
fn test_apply() {
    type TensorFun = Box<dyn Fn(Tensor<i64>) -> Tensor<i64>>;
    #[ijzer]
    fn _test_apply(f: fn(Tensor<i64>) -> TensorFun) -> Tensor<i64> {
        r#"
        var f: Fn(S-> Fn(S->S))
        .(f 2) 4
        "#
    }

    /// Turns `x` into a function that multiplies by `x`
    fn f(x: Tensor<i64>) -> TensorFun {
        Box::new(move |y: Tensor<i64>| x.apply_binary_op(&y, |a, b| a * b).unwrap())
    }

    let expected = Tensor::scalar(8);
    let y = _test_apply(f);
    assert_eq!(y.to_vec(), expected.to_vec());
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