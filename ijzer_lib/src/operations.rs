use std::fmt::Debug;
use crate::function_enums::{TensorBuilderEnum, UnaryMathFunctionEnum};

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
    Array,
    Reduce,
    FunctionComposition(usize),
    TypeConversion,
    LambdaVariable(String),
    Apply,
    GeneralizedContraction,
    TensorBuilder(TensorBuilderEnum),
    Transpose,
    Shape,
    QR,
    Svd,
    Solve,
    Diag,
    Index,
    AssignSymbol(String),
    Reshape,
    UnaryFunction(UnaryMathFunctionEnum),
    BinaryFunction(String),
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
            Self::Array => write!(f, "Array"),
            Self::Reduce => write!(f, "Reduce"),
            Self::FunctionComposition(n) => write!(f, "FunctionComposition(functions: {})", n),
            Self::TypeConversion => write!(f, "TypeConversion"),
            Self::LambdaVariable(s) => write!(f, "LambdaVariable(name: {})", s),
            Self::Apply => write!(f, "Apply"),
            Self::GeneralizedContraction => write!(f, "GeneralizedContraction"),
            Self::TensorBuilder(s) => write!(f, "TensorBuilder({})", s),
            Self::Transpose => write!(f, "Transpose"),
            Self::Shape => write!(f, "Shape"),
            Self::QR => write!(f, "qr"),
            Self::Svd => write!(f, "svd"),
            Self::Solve => write!(f, r"\"),
            Self::Diag => write!(f, "diag"),
            Self::Index => write!(f, "<|"),
            Self::AssignSymbol(s) => write!(f, "AssignSymbol({})", s),
            Self::Reshape => write!(f, "Reshape"),
            Self::UnaryFunction(s) => write!(f, "UnaryFunction({})", s),
            Self::BinaryFunction(s) => write!(f, "BinaryFunction({})", s),
        }
    }
}
