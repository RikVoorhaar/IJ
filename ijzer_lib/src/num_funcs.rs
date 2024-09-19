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

