use quote::quote;

fn main() {
    let output = quote! {
        "var x -> 1
        y = (+ (x 1))
        z = - x y
        fn z = * y I
        x y = (1 2)
        * (z 1 -2)"
    };
    println!("{:?}", output);
}
