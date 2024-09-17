use crate::syntax_error;
use anyhow::Result;
use std::convert::TryFrom;
use std::fmt;

#[derive(Debug, PartialEq, Eq,  Clone)]
pub enum TensorBuilderEnum {
    Eye,
    Randu,
    Randn,
    Zeros,
    Ones,
}

impl TryFrom<String> for TensorBuilderEnum {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        let variant = match value.as_str() {
            "eye" => TensorBuilderEnum::Eye,
            "randu" => TensorBuilderEnum::Randu,
            "randn" => TensorBuilderEnum::Randn,
            "zeros" => TensorBuilderEnum::Zeros,
            "ones" => TensorBuilderEnum::Ones,
            _ => return Err(syntax_error::SyntaxError::InvalidEnumVariant(value).into()),
        };
        Ok(variant)
    }
}

impl fmt::Display for TensorBuilderEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorBuilderEnum::Eye => write!(f, "eye"),
            TensorBuilderEnum::Randu => write!(f, "randu"),
            TensorBuilderEnum::Randn => write!(f, "randn"),
            TensorBuilderEnum::Zeros => write!(f, "zeros"),
            TensorBuilderEnum::Ones => write!(f, "ones"),
        }
    }
}




#[derive(Debug, PartialEq, Eq,  Clone)]
pub enum UnaryMathFunctionEnum {
    Abs,
    Acos,
    Asin,
    Atan,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Floor,
    Ln,
    Log2,
    Log10,
    Round,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
}

impl TryFrom<String> for UnaryMathFunctionEnum {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        let variant = match value.as_str() {
            "abs" => UnaryMathFunctionEnum::Abs,
            "acos" => UnaryMathFunctionEnum::Acos,
            "asin" => UnaryMathFunctionEnum::Asin,
            "atan" => UnaryMathFunctionEnum::Atan,
            "ceil" => UnaryMathFunctionEnum::Ceil,
            "cos" => UnaryMathFunctionEnum::Cos,
            "cosh" => UnaryMathFunctionEnum::Cosh,
            "exp" => UnaryMathFunctionEnum::Exp,
            "floor" => UnaryMathFunctionEnum::Floor,
            "ln" => UnaryMathFunctionEnum::Ln,
            "log2" => UnaryMathFunctionEnum::Log2,
            "log10" => UnaryMathFunctionEnum::Log10,
            "round" => UnaryMathFunctionEnum::Round,
            "sin" => UnaryMathFunctionEnum::Sin,
            "sinh" => UnaryMathFunctionEnum::Sinh,
            "sqrt" => UnaryMathFunctionEnum::Sqrt,
            "tan" => UnaryMathFunctionEnum::Tan,
            "tanh" => UnaryMathFunctionEnum::Tanh,
            _ => return Err(syntax_error::SyntaxError::InvalidEnumVariant(value).into()),
        };
        Ok(variant)
    }
}

impl fmt::Display for UnaryMathFunctionEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryMathFunctionEnum::Abs => write!(f, "abs"),
            UnaryMathFunctionEnum::Acos => write!(f, "acos"),
            UnaryMathFunctionEnum::Asin => write!(f, "asin"),
            UnaryMathFunctionEnum::Atan => write!(f, "atan"),
            UnaryMathFunctionEnum::Ceil => write!(f, "ceil"),
            UnaryMathFunctionEnum::Cos => write!(f, "cos"),
            UnaryMathFunctionEnum::Cosh => write!(f, "cosh"),
            UnaryMathFunctionEnum::Exp => write!(f, "exp"),
            UnaryMathFunctionEnum::Floor => write!(f, "floor"),
            UnaryMathFunctionEnum::Ln => write!(f, "ln"),
            UnaryMathFunctionEnum::Log2 => write!(f, "log2"),
            UnaryMathFunctionEnum::Log10 => write!(f, "log10"),
            UnaryMathFunctionEnum::Round => write!(f, "round"),
            UnaryMathFunctionEnum::Sin => write!(f, "sin"),
            UnaryMathFunctionEnum::Sinh => write!(f, "sinh"),
            UnaryMathFunctionEnum::Sqrt => write!(f, "sqrt"),
            UnaryMathFunctionEnum::Tan => write!(f, "tan"),
            UnaryMathFunctionEnum::Tanh => write!(f, "tanh"),
        }
    }
}

#[derive(Debug, PartialEq, Eq,  Clone)]
pub enum BinaryMathFunctionEnum {
    Max,
    Min,
    Power,
    Add,
    Multiply,
    Div,
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

impl TryFrom<String> for BinaryMathFunctionEnum {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        let variant = match value.as_str() {
            "max" => BinaryMathFunctionEnum::Max,
            "min" => BinaryMathFunctionEnum::Min,
            "^" => BinaryMathFunctionEnum::Power,
            "+" => BinaryMathFunctionEnum::Add,
            "*" => BinaryMathFunctionEnum::Multiply,
            "//" => BinaryMathFunctionEnum::Div,
            "==" => BinaryMathFunctionEnum::Equals,
            "!=" => BinaryMathFunctionEnum::NotEquals,
            ">." => BinaryMathFunctionEnum::GreaterThan,
            "<." => BinaryMathFunctionEnum::LessThan,
            ">=" => BinaryMathFunctionEnum::GreaterThanOrEqual,
            "<=" => BinaryMathFunctionEnum::LessThanOrEqual,
            _ => return Err(syntax_error::SyntaxError::InvalidEnumVariant(value).into()),
        };
        Ok(variant)
    }
}

impl fmt::Display for BinaryMathFunctionEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryMathFunctionEnum::Max => write!(f, "max"),
            BinaryMathFunctionEnum::Min => write!(f, "min"),
            BinaryMathFunctionEnum::Power => write!(f, "^"),
            BinaryMathFunctionEnum::Add => write!(f, "+"),
            BinaryMathFunctionEnum::Multiply => write!(f, "*"),
            BinaryMathFunctionEnum::Div => write!(f, "//"),
            BinaryMathFunctionEnum::Equals => write!(f, "=="),
            BinaryMathFunctionEnum::NotEquals => write!(f, "!="),
            BinaryMathFunctionEnum::GreaterThan => write!(f, ">."),
            BinaryMathFunctionEnum::LessThan => write!(f, "<."),
            BinaryMathFunctionEnum::GreaterThanOrEqual => write!(f, ">="),
            BinaryMathFunctionEnum::LessThanOrEqual => write!(f, "<="),
        }
    }
}