![IJzer Logo](logo.svg)

# IJzer

IJzer (/ˈɛi̯zər/) is a proof-of-concept implementation of the an array programming language (IJ) in Rust. The syntax of the IJ language is inspired by J in that it is ASCII-only, but is otherwise mostly made up on the spot. Tensors are the primary object of this language, like `numpy` or `NDArray`. The language uses infix notation.

## Why?

Array programming languages like APL, J, and BQN are really cool and very powerful for solving _some_ problems, but they are not general purpose programming languages and are not easily integrated into projects using other languages.

IJzer brings the power of an array programming language **inside your Rust projects** using proc macros. This way you can implement algorithms in IJ, but use them inside of large Rust projects.


## Usage

Just import `ijzer` and use the `ijzer` macro:

```rust
use ijzer::prelude::*;
use ijzer::Tensor;


#[ijzer]
/// Add a random normal tensor to the input array
fn example(x: Tensor<f64>) -> Tensor<f64> {
    r"#
    var x: T<f64> // declare that x is a Tensor<f64>
    y = randn<f64> %x // create a random normal tensor with same shape
    + x y // add the two tensors and return the result implicitly
    "#
}

// We can now use the function like any rust function
let x = Tensor::range(0,20);
println!("{:?}", example(x));
```

For an explanation of the language see the docs or the blog post. See also the examples included in the `ijzer` directory.