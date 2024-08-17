use std::fmt::Debug;

use crate::tokens::Number;

#[derive(Clone, PartialEq)]
pub enum Operation {
    Add,
    Multiply,
    Subtract,
    Negate,
    Group,
    Number(Number),
    Symbol(String),
    Assign,
    Identity,
    Nothing,
    Function(String),
    Array(String),
    Reduce,
    Scalar,
    FunctionComposition(usize, usize),
    TypeConversion,
    LambdaVariable(String),
    Apply,
    AsFunction,
}

impl Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "Add"),
            Self::Multiply => write!(f, "Multiply"),
            Self::Subtract => write!(f, "Subtract"),
            Self::Negate => write!(f, "Negate"),
            Self::Group => write!(f, "Group"),
            Self::Number(n) => write!(f, "Number({:?})", n),
            Self::Symbol(s) => write!(f, "Symbol({})", s),
            Self::Assign => write!(f, "Assign"),
            Self::Identity => write!(f, "Identity"),
            Self::Nothing => write!(f, "Nothing"),
            Self::Function(s) => write!(f, "Function({})", s),
            Self::Array(s) => write!(f, "Array[{}]", s),
            Self::Reduce => write!(f, "Reduce"),
            Self::Scalar => write!(f, "Scalar"),
            Self::FunctionComposition(n, m) => write!(f, "FunctionComposition(functions: {}, operands: {})", n, m),
            Self::TypeConversion => write!(f, "TypeConversion"),
            Self::LambdaVariable(s) => write!(f, "LambdaVariable(name: {})", s),
            Self::Apply => write!(f, "Apply"),
            Self::AsFunction => write!(f, "AsFunction"),
        }
    }
}
