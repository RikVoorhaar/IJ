use crate::ast::{ASTContext, Node, ParseNode, SyntaxError};
use crate::lexer::{Token, TokenSlice};
use crate::parser::{comma_separate, find_matching_parenthesis};
use crate::types::{FunctionSignature, IJType};
use std::rc::Rc;

struct FunctionChain {
    functions: Vec<Rc<Node>>,
}

fn _get_output_type_from_function(function: &Node) -> Result<IJType, SyntaxError> {
    match function.output_type {
        IJType::Function(function_signature) => Ok(function_signature.output_type),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function.output_type)).into()),
    }
}

fn _get_input_types_from_function(function: &Node) -> Result<Vec<IJType>, SyntaxError> {
    match function.input_types {
        IJType::Function(function_signature) => Ok(function_signature.input_types),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function.input_types)).into()),
    }
}

impl FunctionChain {
    fn new(functions: Vec<Rc<Node>>) -> Result<Self, SyntaxError> {
        let mut chain = Self { functions };
        chain.verify_consistency()?;
        Ok(chain)
    }

    fn empty() -> Self {
        Self { functions: vec![] }
    }

    fn get_output_type(&self) -> Result<IJType, SyntaxError> {
        let last_function = self.functions.last().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_output_type_from_function(last_function)
    }

    fn get_input_types(&self) -> Result<Vec<IJType>, SyntaxError> {
        let first_function = self.functions.first().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_input_types_from_function(first_function)
    }

    fn verify_consistency(&self) -> Result<(), SyntaxError> {
        for function in self.functions.iter() {
            if !matches!(function.output_type, IJType::Function(_)) {
                return Err(SyntaxError::TypeError(
                    "Function".to_string(),
                    format!("{:?}", function.output_type),
                )
                .into());
            }
        }
        for &[ref prev_function, ref next_function] in self.functions.iter().windows(2) {
            let prev_input_types = _get_input_types_from_function(prev_function)?;
            if prev_input_types.len() != 1 {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function is not last in chain but does not have exactly one input type. Instead it has {:?}",
                    prev_input_types
                ))
                .into());
            }
            let prev_input_type = prev_input_types[0];
            let next_output_type = _get_output_type_from_function(next_function)?;
            if prev_input_type != next_output_type {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function chain is not consistent: {:?} -> {:?}",
                    prev_input_type, next_output_type
                ))
                .into());
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.functions.len()
    }

    fn extend(self, node: Rc<Node>) -> Result<Option<Self>, SyntaxError> {
        let last_input_types = self.get_input_types()?;
        if last_input_types.len() != 1 {
            return Ok(None);
        }
        let last_input_type = last_input_types[0];

        let node_output_type = _get_output_type_from_function(&node)?;
        if last_input_type != node_output_type {
            return Ok(None);
        }
        let mut new_functions = self.functions.clone();
        new_functions.push(node);
        Ok(Some(Self::new(new_functions)?))
    }
}

struct FunctionComposition;
impl ParseNode for FunctionComposition {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let next_token = context.get_token_at_index(slice.start)?;
        if next_token != &Token::LParen {
            return Err(SyntaxError::ExpectedToken(
                "(".to_string(),
                format!("found {:?}", next_token),
            )
            .into());
        }
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;
        let slices = comma_separate(slice, context)?;
        // let (nodes, rest) = gather_operands(vec![vec![IJType::Tensor]], slices, context)?;
    }
}
