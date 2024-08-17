use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::parser::{comma_separate, find_matching_parenthesis, ParseNode, ParseNodeFunctional};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::{Error, Result};
use itertools::Itertools;
use std::rc::Rc;

use super::next_node_functional;
use super::utils::gather_operands;

struct FunctionChain {
    functions: Vec<Rc<Node>>,
}

fn _get_output_type_from_function(function: Rc<Node>) -> Result<Vec<IJType>, SyntaxError> {
    match &function.output_type {
        IJType::Function(function_signature) => Ok(function_signature.output.clone()),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function))),
    }
}

fn _get_input_types_from_function(function: Rc<Node>) -> Result<Vec<IJType>, SyntaxError> {
    match &function.output_type {
        IJType::Function(function_signature) => Ok(function_signature.input.clone()),
        _ => Err(SyntaxError::NotPureFunction(format!("{:?}", function))),
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
        let first_function = self.functions.first().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_output_type_from_function(first_function.clone())
    }

    fn get_input_types(&self) -> Result<Vec<IJType>, SyntaxError> {
        let last_function = self.functions.last().ok_or_else(|| {
            SyntaxError::FunctionChainInconsistency("Function chain is empty".to_string())
        })?;
        _get_input_types_from_function(last_function.clone())
    }

    fn get_node_types(&self) -> Vec<IJType> {
        self.functions
            .iter()
            .map(|node| node.output_type.clone())
            .collect()
    }

    fn get_chain_type(&self) -> Result<IJType, SyntaxError> {
        let input_types = self.get_input_types()?;
        let output_types = self.get_output_type()?;
        Ok(IJType::Function(FunctionSignature::new(
            input_types,
            output_types,
        )))
    }

    fn verify_consistency(&self) -> Result<(), SyntaxError> {
        for function in self.functions.iter() {
            if !_is_function(function.clone()) {
                return Err(SyntaxError::TypeError(
                    "Function".to_string(),
                    format!("{:?}", function.output_type),
                ));
            }
        }
        for (prev_function, next_function) in self.functions.iter().tuple_windows() {
            let prev_input_types = _get_input_types_from_function(prev_function.clone())?;
            if prev_input_types.len() != 1 {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function is not last in chain but does not have exactly one input type. Instead it has {:?}",
                    prev_input_types
                ))
                );
            }
            let prev_input_type = prev_input_types[0].clone();
            let next_output_types = _get_output_type_from_function(next_function.clone())?;
            if next_output_types.len() != 1 {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function is not first in chain but does not have exactly one output type. Instead it has {:?}",
                    next_output_types
                ))
                );
            }
            let next_output_type = next_output_types[0].clone();
            if prev_input_type != next_output_type {
                return Err(SyntaxError::FunctionChainInconsistency(format!(
                    "Function chain is not consistent: {:?} -> {:?}",
                    prev_input_type, next_output_type
                )));
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.functions.len()
    }

    fn extend(&self, node: Rc<Node>) -> Result<Option<Self>, SyntaxError> {
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

    fn match_input_type(&self, input_types: Vec<IJType>) -> Result<bool, SyntaxError> {
        let chain_input_types = self.get_input_types()?;
        Ok(chain_input_types == input_types)
    }
}

fn extend_chains(
    chains: Vec<FunctionChain>,
    nodes: Vec<Rc<Node>>,
) -> Result<Vec<FunctionChain>, SyntaxError> {
    let mut new_chains = vec![];
    new_chains.extend(chains.iter().flat_map(|chain| {
        nodes
            .iter()
            .filter_map(move |node| chain.extend(node.clone()).ok().flatten())
    }));
    Ok(new_chains)
}

pub struct FunctionComposition;
impl FunctionComposition {
    fn get_chains_from_slice(
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Vec<FunctionChain>, TokenSlice)> {
        let next_token = context.get_token_at_index(slice.start)?;
        if next_token != &Token::LParen {
            return Err(SyntaxError::ExpectedToken(
                "(".to_string(),
                format!("found {:?}", next_token),
            )
            .into());
        }
        let slice = slice.move_start(1)?;
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice
            .move_end(rparen_index)
            .map_err(|e| context.add_context_to_syntax_error(e.into(), slice))?;
        if slice.is_empty() {
            return Err(context
                .add_context_to_syntax_error(SyntaxError::EmptyFunctionComposition.into(), slice));
        }

        let slices = comma_separate(slice, context)?;
        let function_operands = slices
            .into_iter()
            .map(|s| {
                let (nodes, rest) = next_node_functional(s, context, None)
                    .map_err(|e| context.add_context_to_syntax_error(e, s))?;
                if !rest.is_empty() {
                    return Err(context.add_context_to_syntax_error(
                        SyntaxError::UnhandledTokens(format!("found {:?}", rest)).into(),
                        rest,
                    ));
                }
                Ok(nodes)
            })
            .collect::<Result<Vec<Vec<Rc<Node>>>, Error>>()?;
        let chains = function_operands
            .into_iter()
            .try_fold(vec![FunctionChain::empty()], extend_chains)?;
        Ok((chains, remainder))
    }
}

impl ParseNode for FunctionComposition {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (chains, remainder) = Self::get_chains_from_slice(slice, context)?;
        let types = chains
            .iter()
            .map(|chain| chain.get_input_types())
            .collect::<Result<Vec<Vec<IJType>>, SyntaxError>>()?;
        if types.is_empty() {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::FunctionChainInconsistency(
                    "Types of function are not composible".to_string(),
                )
                .into(),
                slice,
            ));
        }
        let (value_operands, rest) = gather_operands(types, remainder, context)?;

        let operand_types = value_operands
            .iter()
            .map(|operand| operand.output_type.clone())
            .collect::<Vec<IJType>>();

        let mut matching_chain: Option<FunctionChain> = None;
        for chain in chains {
            if chain.match_input_type(operand_types.clone())? {
                if matching_chain.is_some() {
                    return Err(SyntaxError::FunctionChainAmbiguousType(format!(
                        "{:?}",
                        operand_types
                    ))
                    .into());
                }
                matching_chain = Some(chain);
            }
        }
        if matching_chain.is_none() {
            return Err(
                SyntaxError::FunctionChainAmbiguousType(format!("{:?}", operand_types)).into(),
            );
        }
        let matching_chain = matching_chain.unwrap();
        let mut full_input_type: Vec<IJType> = vec![];
        let mut all_nodes: Vec<Rc<Node>> = vec![];
        for node in matching_chain.functions.iter() {
            full_input_type.push(node.output_type.clone());
            all_nodes.push(node.clone());
        }
        full_input_type.extend(operand_types);
        let num_value_operands = value_operands.len();
        all_nodes.extend(value_operands);

        let output_type = matching_chain.get_output_type()?;
        if output_type.len() != 1 {
            return Err(SyntaxError::MultipleOutputs.into());
        }
        let output_type = output_type[0].clone();

        Ok((
            Rc::new(Node::new(
                Operation::FunctionComposition(matching_chain.len(), num_value_operands),
                full_input_type,
                output_type,
                all_nodes,
                context.get_increment_id(),
            )),
            rest,
        ))
    }
}

impl ParseNodeFunctional for FunctionComposition {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice), Error> {
        let slice = slice.move_start(1)?;
        let (chains, remainder) = Self::get_chains_from_slice(slice, context)?;

        let matching_chains = match needed_outputs {
            Some(needed_outputs) => chains
                .into_iter()
                .filter(|chain| {
                    needed_outputs
                        .iter()
                        .any(|output_type| match chain.get_output_type() {
                            Ok(chain_output_type) => chain_output_type == output_type.clone(),
                            Err(_) => false,
                        })
                })
                .collect(),
            None => chains,
        };

        let nodes: Vec<Rc<Node>> = matching_chains
            .into_iter()
            .map(|chain| {
                let chain_type = chain.get_chain_type()?;
                if let IJType::Function(function_signature) = chain_type.clone() {
                    let num_value_operands = function_signature.input.len();

                    Ok(Rc::new(Node::new(
                        Operation::FunctionComposition(chain.len(), num_value_operands),
                        chain.get_node_types(),
                        chain_type,
                        chain.functions.clone(),
                        context.get_increment_id(),
                    )))
                } else {
                    Err(SyntaxError::TypeError(
                        "Function".to_string(),
                        format!("{:?}", chain_type),
                    ))
                }
            })
            .collect::<Result<Vec<Rc<Node>>, SyntaxError>>()?;

        Ok((nodes, remainder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::Node;
    use crate::operations::Operation;
    use crate::parser::{parse_str, parse_str_no_context};
    use crate::types::FunctionSignature;

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

    #[test]
    fn test_function_composition_parser_minus_plus_scalar_scalar() {
        let result = parse_str_no_context("@(-,+) 1 2");
        assert!(result.is_ok());
        let (composition_node, _) = result.unwrap();
        let operands = composition_node.operands.clone();
        assert_eq!(operands.len(), 4);
        let node1 = operands[0].clone();
        assert_eq!(
            node1.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar],
                vec![IJType::Scalar],
            ))
        );
        let node2 = operands[1].clone();
        assert_eq!(
            node2.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar, IJType::Scalar],
                vec![IJType::Scalar],
            ))
        );
        let node3 = operands[2].clone();
        assert_eq!(node3.output_type, IJType::Scalar);
        let node4 = operands[3].clone();
        assert_eq!(node4.output_type, IJType::Scalar);
    }

    #[test]
    fn test_function_composition_parser_minus_plus_scalar_tensor() {
        let result = parse_str_no_context("@(-,+) 1 [2]");
        assert!(result.is_ok());
        let (composition_node, _) = result.unwrap();
        let operands = composition_node.operands.clone();
        assert_eq!(operands.len(), 4);
        let node1 = operands[0].clone();
        assert_eq!(
            node1.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Tensor],
            ))
        );
        let node2 = operands[1].clone();
        assert_eq!(
            node2.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar, IJType::Tensor],
                vec![IJType::Tensor],
            ))
        );
        let node3 = operands[2].clone();
        assert_eq!(node3.output_type, IJType::Scalar);
        let node4 = operands[3].clone();
        assert_eq!(node4.output_type, IJType::Tensor);
    }
    #[test]
    fn test_function_composition_parser_minus_plus_tensor_tensor() {
        let result = parse_str_no_context("@(-,+) [1] [2]");
        assert!(result.is_ok());
        let (composition_node, _) = result.unwrap();
        let operands = composition_node.operands.clone();
        assert_eq!(operands.len(), 4);
        let node1 = operands[0].clone();
        assert_eq!(
            node1.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor],
                vec![IJType::Tensor],
            ))
        );
        let node2 = operands[1].clone();
        assert_eq!(
            node2.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Tensor],
                vec![IJType::Tensor],
            ))
        );
        let node3 = operands[2].clone();
        assert_eq!(node3.output_type, IJType::Tensor);
        let node4 = operands[3].clone();
        assert_eq!(node4.output_type, IJType::Tensor);
    }

    #[test]
    fn test_function_composition_parser_minus_tensor_scalar() {
        let result = parse_str_no_context("@(-) [1] 2");
        assert!(result.is_ok());
        let (composition_node, _) = result.unwrap();
        let operands = composition_node.operands.clone();
        assert_eq!(operands.len(), 3);
        let node1 = operands[0].clone();
        assert_eq!(
            node1.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor, IJType::Scalar],
                vec![IJType::Tensor],
            ))
        );
        let node2 = operands[1].clone();
        assert_eq!(node2.output_type, IJType::Tensor);
        let node3 = operands[2].clone();
        assert_eq!(node3.output_type, IJType::Scalar);
    }

    #[test]
    fn test_function_composition_parser_custom_function() {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(S,S->S)", &mut context).unwrap();
        let result = parse_str("@(-, f) 1 2", &mut context);
        assert!(result.is_ok());
        let composition_node = result.unwrap();
        let operands = composition_node.operands.clone();
        assert_eq!(operands.len(), 4);
        let node1 = operands[0].clone();
        assert_eq!(
            node1.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar],
                vec![IJType::Scalar],
            ))
        );
        let node2 = operands[1].clone();
        assert_eq!(
            node2.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Scalar, IJType::Scalar],
                vec![IJType::Scalar],
            ))
        );
        let node3 = operands[2].clone();
        assert_eq!(node3.output_type, IJType::Scalar);
        let node4 = operands[3].clone();
        assert_eq!(node4.output_type, IJType::Scalar);
    }

    #[test]
    fn test_function_composition_empty() {
        assert!(parse_str_no_context("@()").is_err());
    }

    #[test]
    fn test_function_composition_functional() -> Result<()> {
        let (node, _) = parse_str_no_context("~@(+): Fn(S,S->S)")?;
        assert_eq!(node.op, Operation::FunctionComposition(1, 2));

        let (node, _) = parse_str_no_context("~@(-,+): Fn(S,S->S)")?;
        assert_eq!(node.op, Operation::FunctionComposition(2, 2));
        Ok(())
    }
}
