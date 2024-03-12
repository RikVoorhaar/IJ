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
    FunctionDeclaration,
    Identity,
    Nothing,
    Function(String),
    Array(String),
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
            Self::FunctionDeclaration => write!(f, "fn"),
            Self::Identity => write!(f, "Identity"),
            Self::Nothing => write!(f, "Nothing"),
            Self::Function(s) => write!(f, "Function({})", s),
            Self::Array(s) => write!(f, "Array[{}]", s),
        }
    }
}
