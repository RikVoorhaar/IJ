//! Parses the `as_function` operation `~`.
//! 
//! This operation is used to convert a value to a function.
//! For example if `f: N->N` then `f` without arguments would be an error, since `f` tries to consume an argument. To avoid this you can use `~f` to convert `f` to a function, returning a value of type `Fn(N->N)`. For example `f = ~@(-,+)` creates a function `f: Fn(N,N->N)` that adds and negates its arguments.
use super::{next_node_functional, ParseNode, ParseNodeFunctional};
use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::Result;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct AsFunction;

impl AsFunction {
    fn next_node_functional_part(
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let (nodes, rest) = next_node_functional(slice, context, needed_outputs)?;

        if !rest.is_empty() && context.get_token_at_index(rest.start)? == &Token::TypeDeclaration {
            let rest = rest.move_start(1)?;
            let (fn_type, type_end) = IJType::parse_tokens(&context.get_tokens_from_slice(rest))?;
            let rest = rest.move_start(type_end)?;
            if let IJType::Function(_) = fn_type {
            } else {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::ExpectedFunction(fn_type.to_string()).into(),
                    slice,
                ));
            }

            let node = nodes
                .iter()
                .find(|n| n.output_type.type_match_function(&fn_type).unwrap_or(false))
                .ok_or_else(|| {
                    SyntaxError::TypeError(
                        fn_type.to_string(),
                        nodes
                            .iter()
                            .map(|n| n.output_type.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    )
                })?;

            return Ok((
                vec![Rc::new(Node::new_skip_number_inference(
                    node.op.clone(),
                    node.input_types.clone(),
                    fn_type,
                    node.operands.clone(),
                    context,
                )?)],
                rest,
            ));
        }

        Ok((nodes, rest))
    }
}

impl ParseNode for AsFunction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (nodes, rest) = AsFunction::next_node_functional_part(slice, context, None)?;
        if nodes.len() != 1 {
            Err(context.add_context_to_syntax_error(
                SyntaxError::FunctionTypeAmbiguous(
                    context.token_slice_to_string(slice),
                    nodes
                        .iter()
                        .map(|n| n.output_type.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
                .into(),
                slice,
            ))
        } else {
            Ok((nodes[0].clone(), rest))
        }
    }
}

impl ParseNodeFunctional for AsFunction {
    fn next_node_functional_impl(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let slice = slice.move_start(1)?;
        AsFunction::next_node_functional_part(slice, context, needed_outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_enums::BinaryMathFunctionEnum;
    use crate::operations::Operation;
    use crate::parser::{parse_str, parse_str_no_context};
    use crate::types::FunctionSignature;

    #[test]
    fn test_as_function_simple() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N,N->N)", &mut context)?;

        let node = parse_str("~f", &mut context)?;
        assert_eq!(node.op, Operation::Function("f".to_string()));
        assert_eq!(node.output_type, IJType::number_function(2));
        Ok(())
    }

    #[test]
    fn test_as_function_with_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(N<a>,N<b>->N<c>)", &mut context)?;

        let node = parse_str("~f", &mut context)?;
        assert_eq!(node.op, Operation::Function("f".to_string()));
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![
                    IJType::Number(Some("a".to_string())),
                    IJType::Number(Some("b".to_string()))
                ],
                IJType::Number(Some("c".to_string()))
            ))
        );
        Ok(())
    }

    #[test]
    fn test_as_function_declaration() -> Result<()> {
        let (node, _) = parse_str_no_context("~+:Fn(N,N->N)")?;

        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Add)
        );
        assert_eq!(node.output_type, IJType::number_function(2));
        Ok(())
    }

    #[test]
    fn test_as_function_ambiguous() -> Result<()> {
        assert!(parse_str_no_context("~+").is_err());

        Ok(())
    }

    #[test]
    fn test_as_function_function() -> Result<()> {
        let (node, _) = parse_str_no_context("~(/ ~+)")?;
        assert_eq!(node.op, Operation::Reduce);

        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Tensor(None)],
                IJType::Number(None)
            ))
        );
        Ok(())
    }

    #[test]
    fn test_as_function_double() -> Result<()> {
        let (node, _) = parse_str_no_context("~~+:Fn(N,N->N)")?;

        assert_eq!(
            node.op,
            Operation::BinaryFunction(BinaryMathFunctionEnum::Add)
        );
        assert_eq!(node.output_type, IJType::number_function(2));
        Ok(())
    }
}
