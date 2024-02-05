use anyhow::Result;
use logos::Logos;

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ ]+")]
enum Token {
    #[token("-")]
    Minus,

    #[token("+")]
    Plus,

    #[token("*")]
    Multiplication,

    #[token("^")]
    Power,

    #[token("%")]
    Reciprocation,

    #[token("/")]
    Reduction,

    #[token("/.")]
    Fold,

    #[token("#")]
    Shape,

    #[token("T")]
    Transpose,

    #[token("&")]
    PartialApplication,

    #[token("@")]
    Composition,

    #[token(".@")]
    Distribute,

    #[token("~")]
    Interchange,

    #[token("$")]
    Axes,

    #[token("_")]
    Flatten,

    #[token(">#")]
    Reshape,

    #[regex("[0-9]+", |lex| lex.slice().parse().ok())]
    Number(u64),

    #[regex("[a-zA-Z][a-zA-Z0-9]*")]
    Symbol,

    #[token("=")]
    Assign,

    #[token("let")]
    Let,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,
}

fn main() {
    let input = "let x = 1 + 2 * 3";
    let lexer = Token::lexer(input);

    for token in lexer {
        println!("{:?}", token);
    }
}
