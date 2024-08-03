use crate::ast_node::{ASTContext, FunctionSignature, IJType, Node, TokenSlice};
use crate::parser::{comma_separate, find_matching_parenthesis, ParseNode, ParseNodeFunctional};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use anyhow::Result;
use itertools::Itertools;
use std::rc::Rc;

struct FunctionChain {
    functions: Vec<Rc<Node>>,
}

fn _get_output_type_from_function(function: Rc<Node>) -> Result<Vec<IJType>, SyntaxError> {
    match &function.output_type {
        IJType::Function(function_signature) => Ok(function_signature.output.clone()),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function)).into()),
    }
}

fn _get_input_types_from_function(function: Rc<Node>) -> Result<Vec<IJType>, SyntaxError> {
    match &function.output_type {
        IJType::Function(function_signature) => Ok(function_signature.input.clone()),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function)).into()),
    }
}

fn _is_function(node: Rc<Node>) -> bool {
    matches!(node.output_type, IJType::Function(_))
}

impl FunctionChain {
    fn new(functions: Vec<Rc<Node>>) -> Result<Self, SyntaxError> {
        let chain = Self { functions };
        chain.verify_consistency()?;
        Ok(chain)
    }

    fn empty() -> Self {
        Self { functions: vec![] }
    }

    fn get_output_type(&self) -> Result<Vec<IJType>, SyntaxError> {
        let last_function = self.functions.last().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_output_type_from_function(last_function.clone())
    }

    fn get_input_types(&self) -> Result<Vec<IJType>, SyntaxError> {
        let first_function = self.functions.first().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_input_types_from_function(first_function.clone())
    }

    fn verify_consistency(&self) -> Result<(), SyntaxError> {
        for function in self.functions.iter() {
            if !_is_function(function.clone()) {
                return Err(SyntaxError::TypeError(
                    "Function".to_string(),
                    format!("{:?}", function.output_type),
                )
                .into());
            }
        }
        for (prev_function, next_function) in self.functions.iter().tuple_windows() {
            let prev_input_types = _get_input_types_from_function(prev_function.clone())?;
            if prev_input_types.len() != 1 {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function is not last in chain but does not have exactly one input type. Instead it has {:?}",
                    prev_input_types
                ))
                .into());
            }
            let prev_input_type = prev_input_types[0].clone();
            let next_output_types = _get_output_type_from_function(next_function.clone())?;
            if next_output_types.len() != 1 {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function is not first in chain but does not have exactly one output type. Instead it has {:?}",
                    next_output_types
                ))
                .into());
            }
            let next_output_type = next_output_types[0].clone();
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
        if !_is_function(node.clone()) {
            return Err(SyntaxError::NotPureFunction(format!("{:?}", node)));
        }
        if self.len() == 0 {
            return Ok(Some(Self::new(vec![node])?));
        }

        let last_input_types = self.get_input_types()?;
        if last_input_types.len() != 1 {
            return Ok(None);
        }
        let last_input_type = last_input_types[0].clone();

        let node_output_types = _get_output_type_from_function(node.clone())?;
        if node_output_types.len() != 1 {
            return Ok(None);
        }
        let node_output_type = node_output_types[0].clone();
        if last_input_type != node_output_type {
            return Ok(None);
        }
        let mut new_functions = self.functions.clone();
        new_functions.push(node);
        Ok(Some(Self::new(new_functions)?))
    }
}

pub struct FunctionComposition;
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

        Err(SyntaxError::NotImplemented("Function composition".to_string()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::Node;
    use crate::operations::Operation;

    fn _create_function_node(
        input_types: Vec<IJType>,
        output_types: Vec<IJType>,
        id: usize,
    ) -> Rc<Node> {
        let function_signature = FunctionSignature::new(input_types, output_types);
        Rc::new(Node::new(
            Operation::Function("test_functoin".to_string()),
            vec![],
            IJType::Function(function_signature),
            vec![],
            id,
        ))
    }

    fn _create_non_function_node(id: usize) -> Rc<Node> {
        Rc::new(Node::new(
            Operation::Symbol("test_symbol".to_string()),
            vec![],
            IJType::Tensor,
            vec![],
            id,
        ))
    }

    #[test]
    fn test_function_chain_empty() {
        let chain = FunctionChain::empty();
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_function_chain_extend1() {
        let chain = FunctionChain::empty();
        let node = _create_function_node(vec![IJType::Tensor], vec![IJType::Tensor], 0);
        let result = chain.extend(node);
        assert!(result.is_ok());
        let new_chain = result.unwrap().unwrap();
        assert_eq!(new_chain.len(), 1);
    }

    #[test]
    fn test_function_chain_extend2() {
        let chain = FunctionChain::empty();
        let node1 = _create_function_node(vec![IJType::Tensor], vec![IJType::Tensor], 0);
        let node2 = _create_function_node(vec![IJType::Tensor], vec![IJType::Tensor], 1);
        let result = chain.extend(node1);
        assert!(result.is_ok());
        let new_chain = result.unwrap().unwrap();
        let result = new_chain.extend(node2);
        assert!(result.is_ok());
        let new_chain = result.unwrap().unwrap();
        assert_eq!(new_chain.len(), 2);
    }

    #[test]
    fn test_function_chain_extend_incompatible() {
        let chain = FunctionChain::empty();
        let node1 = _create_function_node(vec![IJType::Tensor], vec![IJType::Tensor], 0);
        let node2 = _create_function_node(vec![IJType::Tensor], vec![IJType::Scalar], 1);
        let result = chain.extend(node1);
        assert!(result.is_ok());
        let new_chain = result.unwrap().unwrap();
        let result = new_chain.extend(node2);
        assert!(result.is_ok());
        let new_chain = result.unwrap();
        assert!(new_chain.is_none());
    }

    #[test]
    fn test_function_chain_extend_not_function() {
        let chain = FunctionChain::empty();
        let node1 = _create_non_function_node(0);
        let result = chain.extend(node1);
        assert!(result.is_err());
    }
}
