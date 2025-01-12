//! Macro wrapper for the IJzer compiler.
//! 
//! This allows using IJzer in a rust project.
//! 
//! Example:
//! 
//! ```rust
//! use ijzer_macro::ijzer;
//! use ijzer_lib::tensor::Tensor;
//! 
//! #[ijzer]
//! fn square_tensor(x: Tensor<f32>) -> Tensor<f32> {
//!     r#"
//!     * x x 
//!     "#
//! }
//! let x: Tensor<f32> = Tensor::new(vec![1.0, 2.0, 3.0]);
//! let y: Tensor<f32> = square_tensor(x);
//! ```
//! Because the `ijzer` language is not valid rust, the body must be wrapped in a (raw) string. The signature of the function is not parsed, and only the body is parsed to the IJzer compiler.
extern crate proc_macro;
use ijzer_lib::compile;
use proc_macro::TokenStream;
use quote::quote;
use regex::Regex;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn ijzer(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let sig = &input_fn.sig;

    let body = &input_fn.block;

    let quoted_body = quote! {#body}.to_string();
    println!("{}", quoted_body);
    let body_tokens_regex = Regex::new(r#"(?s)\"(.*?)\""#).unwrap();
    let mut ijzer_code = String::new();
    for cap in body_tokens_regex.captures_iter(&quoted_body) {
        if let Some(matched) = cap.get(1) {
            ijzer_code.push_str(matched.as_str());
        }
    }

    let compiled_ijzer = match compile(ijzer_code.as_str()) {
        Ok(code) => code,
        Err(e) => {
            let error_message = e.to_string();
            return TokenStream::from(quote! { compile_error!(#error_message); });
        }
    };

    let new_body = quote! {
        {
            #compiled_ijzer
        }
    };

    // Generate the new function item
    let output = quote! {
        #sig
        #new_body
    };
    println!("{}", output);

    TokenStream::from(output)
}
