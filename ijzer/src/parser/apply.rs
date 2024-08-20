use super::{gather_operands, next_node_functional, Operation, ParseNode};
use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct Apply;

impl ParseNode for Apply {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (function_nodes, rest) = next_node_functional(slice, context, None)?;
        let wanted_types: Vec<Vec<IJType>> = function_nodes
            .iter()
            .map(|n| n.output_type.extract_signature().unwrap().input)
            .collect();
        let (operands, rest) = gather_operands(wanted_types, rest, context)?;
        let operands_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();
        let function_node = function_nodes
            .iter()
            .find(|n| n.output_type.extract_signature().unwrap().input == operands_types)
            .ok_or(
                context.add_context_to_syntax_error(
                    SyntaxError::FunctionVariantMismatch(
                        function_nodes
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<String>>()
                            .join(", "),
                        operands_types
                            .iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<String>>()
                            .join(", "),
                    )
                    .into(),
                    slice,
                ),
            )?;

        let function_signature = function_node.output_type.extract_signature().unwrap();
        let node_input_types = [vec![function_node.output_type.clone()], operands_types].concat();
        let output_type = function_signature
            .output
            .first()
            .ok_or(SyntaxError::InvalidType(
                "Function has no output type".to_string(),
            ))?;
        let operands = [vec![function_node.clone()], operands].concat();

        Ok((
            Rc::new(Node::new(
                Operation::Apply,
                node_input_types,
                output_type.clone(),
                operands,
                context.id_counter,
            )),
            rest,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::ASTContext;
    use crate::parser::test_utils::{parse_str, parse_str_no_context};
    use crate::tokens::lexer;
    use crate::types::FunctionSignature;

    use anyhow::Result;

    #[test]
    fn test_apply() -> Result<()> {
        let (node, _) = parse_str_no_context(".+ 1 2")?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_with_var() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(S,S->S)", &mut context)?;

        let node = parse_str(".f 1 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_with_composition() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(S,S->S)", &mut context)?;

        let node = parse_str(".@(-,f) 1 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 3);
        Ok(())
    }

    #[test]
    fn test_apply_function() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(S->Fn(S->S))", &mut context)?;

        let node = parse_str("(f 1) 2", &mut context)?;
        assert_eq!(node.op, Operation::Apply);
        assert_eq!(node.operands.len(), 2);
        Ok(())
    }
}
