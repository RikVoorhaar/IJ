![IJzer Logo](https://raw.githubusercontent.com/RikVoorhaar/IJ/main/logo.svg)

# IJzer

IJzer (/ˈɛi̯zər/) is a proof-of-concept implementation of an array programming language (IJ) in Rust. The syntax of the IJ language is inspired by J in that it is ASCII-only, but is otherwise mostly ad hoc. Tensors are the primary object of this language, like `numpy` or `NDArray`. The language uses infix notation.

## Why?

Array programming languages like APL, J, and BQN are really cool and very powerful for solving _some_ problems, but they are not general-purpose programming languages and are not easily integrated into projects using other languages.

IJzer brings the power of an array programming language **inside your Rust projects** using proc macros. This way, you can implement algorithms in IJ but use them within large Rust projects.

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
    y = randn<f64> %x // create a random normal tensor with the same shape
    + x y // add the two tensors and return the result implicitly
    "#
}

// We can now use the function like any Rust function
let x = Tensor::range(0,20);
println!("{:?}", example(x));
```

For an explanation of the language, see the docs or the blog post. See also the examples included in the `ijzer` directory.

## Tensors

Tensors are the main object of this language. There are already great implementations of tensors in Rust, such as `ndarray` and `nalgebra`. For flexibility (and for fun), this language has its own `Tensor` object defined in [`ijzer_lib/tensor`](https://docs.rs/ijzer_lib/latest/ijzer_lib/tensor/index.html).

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

There are a bunch of useful features like shape broadcasting (like in `numpy` and `MATLAB`), some matrix decompositions, and convenience functions (`map`, `reduce`). See [the documentation](https://docs.rs/ijzer_lib/latest/ijzer_lib/tensor/index.html) for a complete overview.

## Overview of the `IJ` Language

The `IJ` language is completely infix. It is inspired by `J` (in that it is an array language using ASCII only, and some of the syntax has been taken from there). The functionality has also been inspired by `numpy` and the author's background in numerical linear algebra.

Consider the `IJ` snippet below:

```ijzer
var x: T<f64> // x is an external variable of type T<f64>
+ x x // compute x + x
```

This code first declares that `x` is an external variable. This means that in the transpiled Rust code, the symbol is assumed to be in scope. Its type is annotated as `T<f64>`, which is shorthand for `ijzer_lib::tensor::Tensor::<f64>`. Since this language uses infix notation, brackets are _entirely optional_ when passing arguments to a function. But you can still use them if you want. The following are equivalent:

-   `+ x x`
-   `+ (x x)`
-   `+ (x, x)`

Since the last line of the snippet above does not end with a semicolon, it is implicitly returned. Note that semicolons are optional and mainly useful for chaining multiple statements on the same line.

### Creating Tensors

There are multiple methods for defining tensors; we list them below with example usage:

```ijzer
// Externally defined
var x: T<f64>

// Array notation
a = [[1,2],[3,4]] // A [2,2] matrix. The type is inferred by Rust.
b = [a, a, a] // A [3, 2, 2] tensor
c: T<f64> = [1.0, 2.0]

// Creation operators
a = eye [2,2] // A [2,2] identity matrix
b: T<i32> = zeros [3,1,2] // A [3,1,2] matrix of zeros with type i32
s: T<usize> = [2,2] // A usize vector; can be used for shapes
c = ones s // A tensor of ones with shape given by `s`
d: T<f32> = randu s // Random uniform tensor of type f32
e = randn %d // Random normal tensor with the same shape as `d`

// Slicing other arrays
a = <| c [:,0] // a vector with the first column of `c`
b = <| c [[0,0],[0,1]] // A vector with entries of `c` corresponding to elements [0,0] and [0,1] (i.e., the first column)
```

### The Type System

`IJ` contains two primitive types, numbers (`N`) and tensors (`T`). Primitive types can optionally be annotated with a _number type_, corresponding to a Rust type. For example, `N<f32>` is a number with type `f32`, or `T<usize>` is a tensor whose entries are all `usize`. As mentioned, the number type annotation is completely optional, and if omitted, it is equivalent to a type of `_`.

Composite types consist of _functions_ and _groups_. For example, `Fn(T,T->N)` represents a function that takes two tensors as input and outputs a number. If a function returns multiple arguments, it returns a group, e.g., `Fn(T->(T,T))` takes a tensor and returns two tensors. Composite types can be nested and contain number type annotations for any primitive. Go wild: `(T, Fn(T->(N<f64>, Fn(T<usize>,N<usize>->T<_>))), N<_>)`.

### Basic Binary Operations

Most binary operations work the same way in `IJ`. We've already seen the example `+ x x`, which adds `x` to `x`. Here are some more interesting examples:

```ijzer
x: T<f64> = randn [2,3]
y: T<f64> = randn [2,3]
z: N<f64> = randn [1,3]

a = max x y // Pairwise maximum, tensor of shape [2,3]
b = * x 12.0 // Multiply each element of `x` by `12.0`. Returns tensor of shape [2,3]
c = /: x z // Divide elements of `x` by elements of `z`. The elements in each column of `x`
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

The binary operations do not change the number type and instead use `1` and `0` of the corresponding number type. For example, `>= 1.0<f32> 2.0<f32>` returns `0.0<f32>`.

In general, the final shape of a binary operator applied to two tensors is determined using shape broadcasting, [just like in NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html). Broadcasting is an intuitive and powerful heuristic to allow binary operations to apply to as many tensors as possible without producing an unexpected result.

### Unary Operations

The language also supports a range of unary math operations. For example, `abs x` takes the absolute value of each entry of `x` and returns a tensor with the same shape. Here's a complete list:

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

You can solve linear systems and least squares problems using the `\` operator (inspired by MATLAB). If `A` is a matrix and `B` a vector or matrix, then `\ A B` returns the solution to the linear system `AX=B`. The matrix `A` is `[l,m]` and `B` is `[l,n]`. Depending on the shape `[l,m]` of `A`, we distinguish three cases, where `A` is always assumed to be of maximum rank (`/min [l,m]`):

-   `l == m`: `AX=B` is solved using the LU decomposition and a triangular solver.
-   `l > m`: Overdetermined system. This is solved using QR factorization: we compute `Q, R = qr(A)` and then $R^\top X = Q^\top B$ is solved with a triangular solver.
-   `l < m`: Underdetermined system; the least squares solution is determined using pseudoinverse. The SVD $USV^\top = A$ is computed, after which the solution is obtained using $X = VS^\dagger U^\top$, where $S^\dagger$ is the pseudoinverse of $S$ (since $S$ is diagonal, we just compute the reciprocal of all non-zero elements).

If either `A` or `B` are not matrices, they are first converted to matrices. For `A`, we compute the matricization with respect to the last dimension; if it has shape $(a_0,\dots,a_n)$, then it is turned into a matrix of shape $(\ell, a_n)$ with $\ell = \prod_{i=0}^{n-1}a_i$. Then tensor `B` must have shape $(a_0,\dots, a_{n-1}, b_0,\dots,b_m)$, i.e., the first $n$ dimensions of `A` and `B` must match. It is then matricized to shape $(\ell, \prod_{i=0}^m b_i)$.

### Reduction Operator

The `/` operator denotes reduction and has the signature `Fn(N, N -> N), T -> N`. It takes a binary operator or function and applies it to a tensor and reduces the result to a number. For example, `/+ x` computes the sum of all elements of `x`, while `/* x` computes the product and `/max x` computes the largest element.

### Externally Defined Variables and Functions

Externally defined variables and functions are declared using the `var` keyword. For example, `var x: T<f64>` declares that `x` is an external variable of type `T<f64>`, or `var f: Fn(N -> N)` declares that `f` is an `N -> N` function. The variable is then available in the scope of the IJzer macro. The IJzer compiler does not check if the variable is actually available within its scope; this is up to the user.

The `var` keyword can be used to tell IJzer that _any_ variable is available inside the scope, whether that's because it's a function argument, or it's just a function or variable defined in an outer scope.

For example:

```rust
use ijzer::prelude::*;
use ijzer::Tensor;

fn square(x: f64) -> f64 {
    x * x
}

#[ijzer]
fn example(x: f64) -> f64 {
    r"#
    var x: N<f64>
    var square: Fn(N<f64> -> N<f64>)
    square(x)
    "#
}

let x = 2.0f64;
println!("{:?}", example(x)); // Returns 4.0
```

This allows IJzer to interface with arbitrary Rust code. Since `ijzer` is just a macro, any Rust code can also use any `ijzer` code. It is thus truly an extension of the Rust language.

### Functions

Functions can be created using an assignment operator, just like variables. The functions' arguments are denoted with `$` signs.

```ij
plus($x: T, $y: T) -> T = + $x $y
plus [1,2] [3,4]  // returns [4,6]
```

The type declaration here is optional. Undeclared inputs are assumed to be tensors, and the output is inferred; `plus($x, $y) = + $x $y` is equivalent.

### Composition

Functions can be composed using the `@` operator. For example, `@(-,+)` is the composition of addition followed by negation, and `@(f,g,h) x` is equivalent to `f g h x`.

Some functions like `+` have multiple variants (e.g., scalar + scalar, or scalar + tensor). The composition operator takes all possible variants into account and only keeps the variants which are consistent.

### Type Conversion

Any scalar can be interpreted as a tensor of shape `[1]`. This can be done using the `<-` operator. For example, `<-T 1.0` returns the tensor `[1]`. Tensors can, of course, also be converted to tensors `<-T [1.0]` results in `[1.0]`.

More interestingly, type conversion can also be applied to functions. The arguments to a function can be converted from tensor to number (equivalent to pre-composing the function with a type conversion), whereas the output of a function can be converted from number to tensor. For example, the following operations are allowed:

```ij
var f: Fn(T -> N)
<-Fn(N -> N) f 1 // first convert 1 to [1] and then apply f
<-Fn(T -> T) f [1] // first apply f then convert f([1]) to [f([1])]
<-Fn(N -> T) f 1 // first convert 1 to [1], then apply f, then convert to tensor
```

Sometimes you want to evaluate an expression as a function and not evaluate the expression on any argument. For example:

```ij
var f: Fn(T, T -> T)
g: Fn(T, T -> T) = @(-, f)  // Gives an error
```

The reason this fails is that it tries to parse the arguments of this function, but no arguments are given. To fix this, you can use the `as_function` operator `~`:

```ij
var f: Fn(T, T -> T)
g: Fn(T, T -> T) = ~@(-, +)  // This works
```

This is also a second way to define functions.

In some cases, the input to the `~` operator faces an ambiguous function, for example, `~+` could be a function of type `Fn(N, N -> N)`, `Fn(T, N -> T)`, `Fn(N, T -> T)`, or `Fn(T, T -> T)`. In this case, you can give a type annotation to the `~` operator, for example, `~+: Fn(N, N -> N)` is a function of type `Fn(N, N -> N)`.

### Matrix Multiplication

There is no matrix multiplication primitive in `IJ`. Instead, it can be constructed using the generalized contraction operator `?`. Matrix multiplication is just `?/+*`, which may be all you have to remember:

```ij
?/+* A B // Matrix multiply A and B
```

The generalized contraction operator takes in two functions `f: Fn(T -> T)` and `g: Fn(T, T -> T)` (for matrix multiplication `f = /+` is summation and `g = *` is a pairwise product). A tensor `A` of shape $[a_1, a_2, \dots, a_m, k]$ is taken together with a tensor `B` of shape $[k, b_1, \dots, b_n]$. The final result is then a tensor of shape $[a_1, \dots, a_m, b_1, \dots, b_n]$ with entries:

$$
\mathtt{?fg(A,\, B)}[i_1, \dots, i_m, j_1, \dots, j_n] = f(g(A[i_1, \dots, i_m, \colon], B[\colon, j_1, \dots, j_n]))
$$

> I personally don't know of any other uses of the generalized contraction operator other than matrix multiplication. This is inspired by the J language, where the same operation exists (albeit in a slightly different form).

### Shape Operations

The shape of tensors can be requested using the `%` operator, which always returns a `T<usize>`. Tensors can also be reshaped using `>%`:

```ij
x: T = [1.0, 2.0]
y: T = >% x [1, 2]
z: T<usize> = %y  // [1, 2]
```

### Matrix Decompositions

The SVD and QR decompositions are supported as language primitives. For both operations, the input tensor is first converted to a matrix. The first dimension is left alone, and the second dimension is determined by the product of the remaining dimensions. The result is always returned as matrices. For example, a `[3,2,4]` tensor is reshaped to a `[3,8]` matrix, or a `[3]` tensor/vector is reshaped to a `[3,1]` matrix. If you don't like this, you will have to reshape manually.

The [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) has the signature `T -> T, T` and computes the reduced QR decomposition. That is, if `X` has shape `[m,n]`, then `(Q, R) = qr X` creates matrices of shape `[m, /min[m,n]]` and `[/min[m,n], n]`. The first matrix has orthonormal columns, while the second is upper-triangular.

The [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) has the signature `T -> T, T, T` and computes the reduced SVD. If `X` has shape `[m,n]` (for simplicity, assume `m ≥ n`), then `(U, S, Vt) = svd X` gives tensors of shapes `[m,n]`, `[n]`, `[n,n]`. To reconstruct the matrix `X`, we have to do `?/+* U (?/+* diag(S) Vt)` (the `diag` is optional due to broadcasting).

## Control Flow

Why would you need that?