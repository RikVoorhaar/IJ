//! Comparison functions, treating non-boolean types as either 0 or 1.
use num_traits::{One, Zero};

pub fn equals<T: One + Zero + PartialEq>(a: T, b: T) -> T {
    if a == b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn not_equals<T: One + Zero + PartialEq>(a: T, b: T) -> T {
    if a != b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn greater_than<T: One + Zero + PartialOrd>(a: T, b: T) -> T {
    if a > b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn less_than<T: One + Zero + PartialOrd>(a: T, b: T) -> T {
    if a < b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn greater_than_or_equal<T: One + Zero + PartialOrd>(a: T, b: T) -> T {
    if a >= b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn less_than_or_equal<T: One + Zero + PartialOrd>(a: T, b: T) -> T {
    if a <= b {
        T::one()
    } else {
        T::zero()
    }
}

pub fn or<T: One + Zero + PartialEq>(a: T, b: T) -> T {
    if a != T::zero() || b != T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

pub fn and<T: One + Zero + PartialEq>(a: T, b: T) -> T {
    if a != T::zero() && b != T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equals() {
        assert_eq!(equals(1.0_f64, 1.0_f64), 1.0);
        assert_eq!(equals(1.0_f32, 2.0_f32), 0.0);
        assert_eq!(equals(1_i64, 1_i64), 1);
        assert_eq!(equals(1_usize, 2_usize), 0);
    }

    #[test]
    fn test_not_equals() {
        assert_eq!(not_equals(1.0_f64, 1.0_f64), 0.0);
        assert_eq!(not_equals(1.0_f32, 2.0_f32), 1.0);
        assert_eq!(not_equals(1_i64, 1_i64), 0);
        assert_eq!(not_equals(1_usize, 2_usize), 1);
    }

    #[test]
    fn test_greater_than() {
        assert_eq!(greater_than(2.0_f64, 1.0_f64), 1.0);
        assert_eq!(greater_than(1.0_f32, 2.0_f32), 0.0);
        assert_eq!(greater_than(2_i64, 1_i64), 1);
        assert_eq!(greater_than(1_usize, 2_usize), 0);
    }

    #[test]
    fn test_less_than() {
        assert_eq!(less_than(1.0_f64, 2.0_f64), 1.0);
        assert_eq!(less_than(2.0_f32, 1.0_f32), 0.0);
        assert_eq!(less_than(1_i64, 2_i64), 1);
        assert_eq!(less_than(2_usize, 1_usize), 0);
    }

    #[test]
    fn test_greater_than_or_equal() {
        assert_eq!(greater_than_or_equal(2.0_f64, 1.0_f64), 1.0);
        assert_eq!(greater_than_or_equal(1.0_f32, 2.0_f32), 0.0);
        assert_eq!(greater_than_or_equal(2_i64, 1_i64), 1);
        assert_eq!(greater_than_or_equal(1_usize, 2_usize), 0);
        assert_eq!(greater_than_or_equal(2.0_f64, 2.0_f64), 1.0);
        assert_eq!(greater_than_or_equal(2_i64, 2_i64), 1);
    }

    #[test]
    fn test_less_than_or_equal() {
        assert_eq!(less_than_or_equal(1.0_f64, 2.0_f64), 1.0);
        assert_eq!(less_than_or_equal(2.0_f32, 1.0_f32), 0.0);
        assert_eq!(less_than_or_equal(1_i64, 2_i64), 1);
        assert_eq!(less_than_or_equal(2_usize, 1_usize), 0);
        assert_eq!(less_than_or_equal(2.0_f64, 2.0_f64), 1.0);
        assert_eq!(less_than_or_equal(2_i64, 2_i64), 1);
    }

    #[test]
    fn test_or() {
        assert_eq!(or(1.0_f64, 0.0_f64), 1.0);
        assert_eq!(or(0.0_f64, 1.0_f64), 1.0);
        assert_eq!(or(0.0_f64, 0.0_f64), 0.0);
        assert_eq!(or(2.0_f64, 1.0_f64), 1.0);
        assert_eq!(or(0.0_f64, 2.0_f64), 1.0);
        assert_eq!(or(1_i64, 0_i64), 1);
        assert_eq!(or(0_i64, 1_i64), 1);
        assert_eq!(or(0_i64, 0_i64), 0);

    }

    #[test]
    fn test_and() {
        assert_eq!(and(1.0_f64, 1.0_f64), 1.0);
        assert_eq!(and(1.0_f64, 0.0_f64), 0.0);
        assert_eq!(and(0.0_f64, 1.0_f64), 0.0);
        assert_eq!(and(0.0_f64, 0.0_f64), 0.0);
        assert_eq!(and(1_i64, 1_i64), 1);
        assert_eq!(and(1_i64, 0_i64), 0);
        assert_eq!(and(0_i64, 1_i64), 0);
        assert_eq!(and(0_i64, 0_i64), 0);
        assert_eq!(and(2_i64, 1_i64), 1);
        assert_eq!(and(1_i64, 2_i64), 1);
        assert_eq!(and(0_i64, 2_i64), 0);
    }
}
