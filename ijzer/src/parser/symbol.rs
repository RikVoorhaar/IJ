use super::{check_ok_needed_outputs, gather_operands, ParseNode, ParseNodeFunctional};

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{SymbolToken, Token};
use crate::types::IJType;
use anyhow::Result;
use std::rc::Rc;
pub struct Symbol;
impl ParseNode for Symbol {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        if let Token::Symbol(SymbolToken { name }) = op {
            let variable = context
                .symbols
                .get(&name)
                .ok_or(SyntaxError::UnknownSymbol(name.clone()))?;

            let (node, rest) = match variable.typ.clone() {
                IJType::Tensor(n) => (
                    Node::new(
                        Operation::Symbol(name.clone()),
                        vec![],
                        IJType::Tensor(n),
                        vec![],
                        context,
                    )?,
                    slice,
                ),
                IJType::Function(signature) => {
                    let (operands, rest) =
                        gather_operands(vec![signature.input.clone()], slice, context)?;
                    let node = Node::new(
                        Operation::Function(name.clone()),
                        signature.input.clone(),
                        *signature.output.clone(),
                        operands,
                        context,
                    )?;
                    (node, rest)
                }
                IJType::Number(n) => (
                    Node::new(
                        Operation::Symbol(name.clone()),
                        vec![],
                        IJType::Number(n),
                        vec![],
                        context,
                    )?,
                    slice,
                ),
                IJType::Number(n) => (
                    Node::new(
                        Operation::Symbol(name.clone()),
                        vec![],
                        IJType::Number(n),
                        vec![],
                        context,
                    )?,
                    slice,
                ),
                _ => unreachable!(),
            };
            Ok((Rc::new(node), rest))
        } else {
            unreachable!()
        }
    }
}
impl ParseNodeFunctional for Symbol {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
        let name = match op {
            Token::Symbol(SymbolToken { name }) => name,
            _ => unreachable!(),
        };
        let variable = context
            .symbols
            .get(name.as_str())
            .ok_or(SyntaxError::UnknownSymbol(name.clone()))?;
        if let IJType::Function(ref func_signature) = variable.typ {
            if !check_ok_needed_outputs(needed_outputs, &func_signature.output) {
                let err_string = format!(
                    "{:?}, which has outputs: {:?}, but we expected output types to be one of {:?}",
                    context.token_slice_to_string(slice),
                    func_signature.output,
                    needed_outputs.unwrap()
                );
                return Err(SyntaxError::ExpectedFunction(err_string).into());
            }

            Ok((
                vec![Rc::new(Node::new(
                    Operation::Function(name.clone()),
                    vec![],
                    variable.typ.clone(),
                    vec![],
                    context,
                )?)],
                slice.move_start(1)?,
            ))
        } else {
            Err(SyntaxError::ExpectedFunction(context.token_slice_to_string(slice)).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;
    use crate::types::FunctionSignature;

    #[test]
    fn test_symbol_number_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var x: N<a>", &mut context)?;
        let node = parse_str("x", &mut context)?;
        assert_eq!(node.output_type, IJType::Number(Some("a".to_string())));

        let mut context = ASTContext::new();
        parse_str("var x: T<a>", &mut context)?;
        let node = parse_str("x", &mut context)?;
        assert_eq!(node.output_type, IJType::Tensor(Some("a".to_string())));

        let mut context = ASTContext::new();
        parse_str("var x: N<a>", &mut context)?;
        let node = parse_str("x", &mut context)?;
        assert_eq!(node.output_type, IJType::Number(Some("a".to_string())));

        let mut context = ASTContext::new();
        parse_str("var x: Fn(N<a>->T<a>)", &mut context)?;
        let node = parse_str("~x", &mut context)?;
        assert_eq!(
            node.output_type,
            IJType::Function(FunctionSignature::new(
                vec![IJType::Number(Some("a".to_string()))],
                IJType::Tensor(Some("a".to_string())),
            ))
        );

        Ok(())
    }
}
