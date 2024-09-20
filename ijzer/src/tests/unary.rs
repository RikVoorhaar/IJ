// abs|acos|asin|atan|ceil|cos|cosh|exp|floor|ln|log2|log10|round|sin|sinh|sqrt|tan|tanh
#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use approx::assert_abs_diff_eq;

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
    fn test_abs() {
        #[ijzer]
        fn _test_abs_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        abs x 
        "#
        }

        let x = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], Some(vec![2, 2]));
        let y = _test_abs_on_tensor(x);
        let expected = [1.0, 2.0, 3.0, 4.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_acos() {
        #[ijzer]
        fn _test_acos_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        acos x 
        "#
        }
        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![1.0, 0.0, -1.0], Some(vec![3]));
        let y = _test_acos_on_tensor(x);
        let expected = [0.0, pi / 2.0, pi];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_asin() {
        #[ijzer]
        fn _test_asin_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        asin x 
        "#
        }

        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![1.0, 0.0, -1.0], Some(vec![3]));
        let y = _test_asin_on_tensor(x);
        let expected = [pi / 2.0, 0.0, -pi / 2.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_atan() {
        #[ijzer]
        fn _test_atan_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        atan x 
        "#
        }

        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![1.0, 0.0, -1.0], Some(vec![3]));
        let y = _test_atan_on_tensor(x);
        let expected = [pi / 4.0, 0.0, -pi / 4.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ceil() {
        #[ijzer]
        fn _test_ceil_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        ceil x 
        "#
        }

        let x = Tensor::from_vec(vec![-0.5, 0.5, -1.5], Some(vec![3]));
        let y = _test_ceil_on_tensor(x);
        let expected = [-0.0, 1.0, -1.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cos() {
        #[ijzer]
        fn _test_cos_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        cos x 
        "#
        }

        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![0.0, pi / 2.0, pi], Some(vec![3]));
        let y = _test_cos_on_tensor(x);
        let expected = [1.0, 0.0, -1.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cosh() {
        #[ijzer]
        fn _test_cosh_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        cosh x 
        "#
        }


        let x = Tensor::from_vec(vec![0.0, f64::ln(2.0)], Some(vec![2]));
        let y = _test_cosh_on_tensor(x);
        let expected = [1.0, 5.0 / 4.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_exp() {
        #[ijzer]
        fn _test_exp_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        exp x 
        "#
        }

        let e = std::f64::consts::E;
        let x = Tensor::from_vec(vec![0.0, 1.0, 2.0], Some(vec![3]));
        let y = _test_exp_on_tensor(x);
        let expected = [1.0, e, e * e];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_floor() {
        #[ijzer]
        fn _test_floor_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        floor x 
        "#
        }

        let x = Tensor::from_vec(vec![-0.5, 0.5, -1.5], Some(vec![3]));
        let y = _test_floor_on_tensor(x);
        let expected = [-1.0, 0.0, -2.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_ln() {
        #[ijzer]
        fn _test_ln_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        ln x 
        "#
        }

        let e = std::f64::consts::E;
        let x = Tensor::from_vec(vec![1.0, e, e * e], Some(vec![3]));
        let y = _test_ln_on_tensor(x);
        let expected = [0.0, 1.0, 2.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_log2() {
        #[ijzer]
        fn _test_log2_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        log2 x 
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 4.0], Some(vec![3]));
        let y = _test_log2_on_tensor(x);
        let expected = [0.0, 1.0, 2.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_log10() {
        #[ijzer]
        fn _test_log10_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        log10 x 
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 10.0, 100.0], Some(vec![3]));
        let y = _test_log10_on_tensor(x);
        let expected = [0.0, 1.0, 2.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_round() {
        #[ijzer]
        fn _test_round_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        round x 
        "#
        }

        let x = Tensor::from_vec(vec![-0.9, 0.49, 1.5, 2.49, 3.5], Some(vec![5]));
        let y = _test_round_on_tensor(x);
        let expected = [-1.0, 0.0, 2.0, 2.0, 4.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sin() {
        #[ijzer]
        fn _test_sin_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        sin x 
        "#
        }

        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![0.0, pi / 2.0, pi], Some(vec![3]));
        let y = _test_sin_on_tensor(x);
        let expected = [0.0, 1.0, 0.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sinh() {
        #[ijzer]
        fn _test_sinh_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        sinh x 
        "#
        }

        let x = Tensor::from_vec(vec![0.0, f64::ln(2.0)], Some(vec![2]));
        let y = _test_sinh_on_tensor(x);
        let expected = [0.0, 3.0 / 4.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sqrt() {
        #[ijzer]
        fn _test_sqrt_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        sqrt x 
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 4.0, 9.0], Some(vec![3]));
        let y = _test_sqrt_on_tensor(x);
        let expected = [1.0, 2.0, 3.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_tan() {
        #[ijzer]
        fn _test_tan_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        tan x 
        "#
        }

        let pi = std::f64::consts::PI;

        let x = Tensor::from_vec(vec![0.0, pi / 4.0], Some(vec![2]));
        let y = _test_tan_on_tensor(x);
        let expected = [0.0, 1.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_tanh() {
        #[ijzer]
        fn _test_tanh_on_tensor(x: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        tanh x 
        "#
        }

        let x = Tensor::from_vec(vec![0.0, f64::ln(2.0)], Some(vec![2]));
        let y = _test_tanh_on_tensor(x);
        let expected = [0.0, 3.0 / 5.0];
        let actual = y.to_vec();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-6);
        }
    }   
}
