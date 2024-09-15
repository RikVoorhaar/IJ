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


