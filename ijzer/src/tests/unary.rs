#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use anyhow::Result;
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
}
