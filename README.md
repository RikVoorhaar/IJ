![IJzer Logo](https://raw.githubusercontent.com/RikVoorhaar/IJ/main/logo.svg)

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

## Tensors

Tensors are the main object of this language. There are already great implementations of tensors in Rust, such as `ndarray`, `nalgebra`. For flexibility (and for fun) this language has it's own `Tensor` object defined in [`ijzer_lib/tensor`](https://docs.rs/ijzer_lib/latest/ijzer_lib/tensor/index.html).

Briefly, it can be used like this:

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0], Some(vec![2])); // Create a tensor of shape `[2]` with entries `[1.0, 2.0]`. The entries are type f32, inferred from the input.
let diag = Tensor::diag(&tensor); // Create a diagonal matrix of shape `[2,2]`

let ones = Tensor::<i32>::ones(&[3, 3, 1]); // Create a tensor with entries of type i32 of ones with shape `[3, 3, 1]`

let X = Tensor::<f64>::randn(&[2, 3]); // Random matrix of shape `[2,3]`
let (U, S, Vt) = X.svd()?; // Compute the SVD of X

let A = Tensor::<f64>::randu(&[2,2]);
let Y = A.solve(Vt)?; // Solve linear system A Y = X. Solution has shape [2,3].
```

There are a bunch of useful features like shape broadcasting (like in `numpy` and `matlab`), some matrix decompositions, and convenience functions (`map`, `reduce`). See [the documentation](https://docs.rs/ijzer_lib/latest/ijzer_lib/tensor/index.html) for a complete overview.

## Overview of the `IJ` language

The `IJ` language is completely infix. It is inspired by `J` (in that it is an array language using ASCII only, and some of the syntax has been taken from there). The functionality has also been inspired by `numpy` and the authors background in numerical linear algebra.

Consider the `IJ` snippet below:

```ijzer
var x: T<f64> // x is an external variable of type T<f64>
+ x x // compute x + x
```

This code first declares that `x` is an external variable. This means that in the transpiled rust code the symbol is assumed to be in scope. Its type is annotated as `T<f64>`, which is short-hand for `ijzer_lib::tensor::Tensor::<f64>`. Since this language uses infix notation, brackets are _entirely optional_ when passing arguments to a function. But you can still do it if you want. The following are equivalent:

-   `+ x x`
-   `+ (x x)`
-   `+ (x, x)`

Since the last line of the snippet above does not end with a semicolon, it is implicitly returned. Note that semicolons are optional, and mainly useful for chaining multiple statemtns on the same line.

### Creating tensors

There are multiple methods for defining tensors, we list them below with example usage:

```ijzer
// Externally defined
var x: T<f64>

// Array notation
a = [[1,2],[3,4]] // A [2,2] matrix. The type is inferred by rust.
b = [a, a, a] // A [3, 2, 2] tensor
c: T<f64> = [1.0, 2.0]

// creation operators
a = eye [2,2] // A [2,2] identity matrix
b: T<i32> = zeros [3,1,2] // A [3,1,2] matrix of zeros with type i32
s: T<usize> = [2,2] // A usize vector; can be used for shapes
c = ones s // A tensor of ones with shape given by `s`
d: T<f32> = randu s // Random uniform tensor of type f32
e = randn %d // Random normal tensor with the same shape as `d`

// Slicing other arrays
a = <| c [:,0] // a vector with the first column of `c`
b = <| c [[0,0],[0,1]] // A vector with entries of `c` corresponding to elements [0,0] and [0,1] (i.e. the frist column)
```

### The type system

`IJ` contains two primitive types, numbers (`N`) and tensors (`T`). Primitive types can optionally be annotated with a _number type_, corresponding to a rust type. For example `N<f32>` is a number with type `f32`, or `T<usize>` is a tensor whose entries are all `usize`. As mentioned the number type annotation is completely optional, and if ommited is equivalent to a type of `_`.

Composite types consist of _functions_ and _groups_. For example `Fn(T,T->N)` represents a function that takes two tensors as input and outputs a number. If a function returns multiple arguments it returns a group, e.g., `Fn(T->(T,T))` takes a tensor and returns two tensors. Composite types can be nested and contain number type annotatoins for any primitive, go wild: `(T, Fn(T->(N<f64>, Fn(T<usize>,N<usize>->T<_>))), N<_>)`.

### Basic binary operations

Most binary operations work the same way in `IJ`. We've already seen the example `+ x x` which adds `x` to `x`. Some more interesting examples:

```ijzer
x: T<f64> = randn [2,3]
y: T<f64> = randn [2,3]
z: N<f64> = randn [1,3]

a = max x y // Pairwise maximum, tensor of shape [2,3]
b = * x 12.0 // Multiple each element of `x` by `12.0`. Returns tensor of shape [2,3]
c = /: x z // Divide element of `x` by an element of `z`. The elements in each column of `x`
// are all divided by the same element of `z`. Returns tensor of shape [2,3]
```

A complete list of all binary operations:

-   `max`
-   `min`
-   `^` (power)
-   `+`
-   `*`
-   `/:` (division)
-   `==` (equal)
-   `!=` (not equal)
-   `>.` (greater than)
-   `<.` (less than)
-   `>=` (greater than or equal to)
-   `<=` (less than or equal to)
-   `&&` (logical and)
-   `||` (logical or)

The binary operations do not change number type and instead use `1` and `0` of the corresponding number type. For example `>= 1.0<f32> 2.0<f32>` returns `0.0<f32>`.

In general the final shape of a binary operator applied to two tensors is determined using shape broadcasting, [just like in numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html). Broadcasting is an intuitive and powerful heuristic to allow binary operations to apply to as many tensors as possible without producing an unexpected result.

### Unary operations

The langauage also supports a range of unary math operations. For example `abs x` takes the absolute value of each entry of `x` and returns a tensor with the same shape. Here's a complete list:

-   `abs`: absolute value
-   `acos`: inverse cosine
-   `asin`: inverse sine
-   `atan`: inverse tangent
-   `ceil`: ceiling function
-   `cos`: cosine
-   `cosh`: hyperbolic cosine
-   `exp`: exponential
-   `floor`: floor function
-   `ln`: natural logarithm
-   `log2`: logarithm base 2
-   `log10`: logarithm base 10
-   `round`: round to nearest integer
-   `sin`: sine
-   `sinh`: hyperbolic sine
-   `sqrt`: square root
-   `tan`: tangent
-   `tanh`: hyperbolic tangent

### Solve

You can solve linear systems and least squares problem using the `\` operator (inspired by Matlab). If `A` is a matrix and `B` a vector or matrix, then `\ A B` returns the solution to the linear system `AX=B`. The matri `A` is `[l,m]` and `B` is `[l,n]`. Depending on the shape `[l,m]` of `A` we distinguish 3 cases, where `A` is always assumed to be of maximum rank (`/min [l,m]`):
- `l==m`: `AX=B` is solved using the LU decomposition and a triangular solver.
- `l>m`: overdetermined system. This is solved using QR factorization: we compute `Q,R=qr(A)` and then $R^\top X = Q^\top B$ is solved with a triangular solver. 
- `l<m`: undertermined system; least squares solution is determined using pseudoinverse. The SVD $USV^\top=A$ is computed after which the solution is obtained using $X = VS^\dagger U^\top$, where $S^\dagger$ is the pseudoinverse of $S$ (since $S$ is diagonal, we just compute the reciprical of all non-zero elements). 

If either `A` or `B` are not matrices, they are first converted to matrices. For `A` we compute the matriziation with respect to the last dimension; if it has shape $(a_0,\dots,a_n)$ then it is turned into a matrix of shape $(\ell,a_n)$ with $\ell =\prod_{i=0}^{n-1}a_i$. Then tensor $B$ must have shape $(a_0,\dots, a_{n-1}, b_0,\dots,b_m)$, i.e. the first $n$ dimensions of $A$ and $B$ must match. It is then matricized to shape $(\ell, \prod_{i=0}^m b_i)$. 

### Reduction operator

The `/` operator denotes reduction and has singature `Fn(N,N->N),T->N`. It takes a biunary operator or function and applies it to a tensor and reduces the result to a number. For example `/+ x` computes the sum of all elements of `x`, while `/* x` computes the product and `/max x` computes the largest element.

### Externally defined variables and functions

### Functions

### Composition

### Function conversion

### Matrix multiplication

### Shape operations

### Matrix decompositions

The SVD and QR decompositions are supported as language primitives. For both operations, the input tensor is first converted to a matrix. The first dimension is left alone, and the second dimension is determined by the product of remaining dimensions. The result is always returned as matrices. For example, a `[3,2,4]` tensor is reshaped to a `[3,8]` matrix, or a `[3]` tensor/vector is reshaped to a `[3,1]` matrix. If you don't like this, you will have to reshape manually.

The [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) has signature `T->T,T` and computes the reduced QR decomposition. That is, if `X` has shape `[m,n]` then `(Q,R) = qr X` creates matrices of shape `[m,/min[m,n]]` and `[/min[m,n],n]`. The first matrix has orthonormal columns, while the second is upper-triangular.

The [Singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) has signature `T->T,T,T` and computes the reduced SVD. If `X` has shape `[m,n]` (for simplicity assume `m>=n`) then `(U, S, Vt) = svd X` gives tensors of shapes `[m,n], [n], [n,n]`. To reconstruct the matrix `X` for this we have to do `?/+* U (?/+* diag(S) Vt)` (the `diag` is optional due to broadcasting).
