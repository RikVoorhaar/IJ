#[cfg(test)]
mod tests {
    use crate::prelude::*;

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
    #[test]
    fn test_not_equal() {
        #[ijzer]
        fn _test_not_equal_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        != x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2, 2]));
        let z = _test_not_equal_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![0.0, 0.0, 1.0, 1.0]);

        #[ijzer]
        fn _test_not_equal_tensor_scalar(x: Tensor<f64>, y: f64) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: N<f64>
        != x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = 2.0;
        let z = _test_not_equal_tensor_scalar(x, y);
        assert_eq!(z.to_vec(), vec![1.0, 0.0, 1.0, 1.0]);

        #[ijzer]
        fn _test_not_equal_scalar_tensor(x: f64, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: N<f64>
        var y: T<f64>
        != x y
        "#
        }

        let x = 2.0;
        let y = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let z = _test_not_equal_scalar_tensor(x, y);
        assert_eq!(z.to_vec(), vec![1.0, 0.0, 1.0, 1.0]);

        #[ijzer]
        fn _test_not_equal_scalar_scalar(x: f64, y: f64) -> f64 {
            r#"
        var x: N<f64>
        var y: N<f64>
        != x y
        "#
        }

        let x = 2.0;
        let y = 2.0;
        let z = _test_not_equal_scalar_scalar(x, y);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn test_greater() {
        #[ijzer]
        fn _test_greater_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        >. x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
        let z = _test_greater_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_greater_equal() {
        #[ijzer]
        fn _test_greater_equal_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        >= x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
        let z = _test_greater_equal_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_less() {
        #[ijzer]
        fn _test_less_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        <. x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
        let z = _test_less_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_less_equal() {
        #[ijzer]
        fn _test_less_equal_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        <= x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 2.0, 9.0, 16.0], Some(vec![2, 2]));
        let y = Tensor::from_vec(vec![1.0, 4.0, 3.0, 4.0], Some(vec![2, 2]));
        let z = _test_less_equal_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_and() {
        #[ijzer]
        fn _test_and_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        && x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0, 1.0], Some(vec![5]));
        let y = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 2.0], Some(vec![5]));
        let z = _test_and_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_or() {
        #[ijzer]
        fn _test_or_tensor_tensor(x: Tensor<f64>, y: Tensor<f64>) -> Tensor<f64> {
            r#"
        var x: T<f64>
        var y: T<f64>
        || x y
        "#
        }

        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0, 1.0], Some(vec![5]));
        let y = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 2.0], Some(vec![5]));
        let z = _test_or_tensor_tensor(x, y);
        assert_eq!(z.to_vec(), vec![1.0,0.0,1.0, 1.0, 1.0]);
    }
}
