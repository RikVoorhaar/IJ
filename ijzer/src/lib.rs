pub mod prelude {
    pub use ijzer_lib::num_funcs::*;
    pub use ijzer_lib::tensor::Tensor;
    pub use ijzer_macro::ijzer;
    pub use num_traits::{pow::Pow, Float};
}

#[cfg(test)]
mod tests;