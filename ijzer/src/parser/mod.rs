#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub use test_utils::{
    create_context_with_tokens_from_str, is_specific_syntax_error, parse_str, parse_str_no_context,
};

mod parser_functions;
pub use parser_functions::{parse_line, parse_lines};

mod utils;
use utils::{
    check_ok_needed_outputs, comma_separate, find_matching_parenthesis, gather_all, gather_operands,
};

mod binary_op;
use binary_op::{next_node_simple_binary_op, Add, Multiply};

mod parenthesis;
use parenthesis::LParen;

mod array;
use array::LSqBracket;

mod symbol;
use symbol::Symbol;

mod reduction;
use reduction::Reduction;

mod minus;
use minus::MinusOp;

mod number;
use number::NumberNode;

mod identity;
use identity::IdentityNode;

mod lambda_variable;
use lambda_variable::LambdaVariable;

mod function_composition;
use function_composition::FunctionComposition;

mod type_conversion;
use type_conversion::TypeConversion;

mod as_function;
use as_function::AsFunction;

mod assign;

mod apply;
use apply::Apply;

mod generalized_contraction;
use generalized_contraction::GeneralizedContraction;

mod tensor_builder;
use tensor_builder::TensorBuilder;

mod transpose;
use transpose::Transpose;

mod shape;
use shape::Shape;

mod qr;
use qr::QR;

mod svd;
use svd::Svd;

mod solve;
use solve::Solve;

use crate::ast_node::{ASTContext, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::{FunctionSignature, IJType};
use anyhow::Result;
use std::rc::Rc;

trait ParseNode {
    fn next_node(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)>;
}

trait ParseNodeFunctional {
    fn next_node_functional_impl(
        op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[IJType]>,
    ) -> Result<(Vec<Rc<Node>>, TokenSlice)>;
}

/// Main logic for parsing a node.
pub fn next_node(slice: TokenSlice, context: &mut ASTContext) -> Result<(Rc<Node>, TokenSlice)> {
    let op = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    match op {
        Token::Plus => next_node_simple_binary_op(Operation::Add, rest, context),
        Token::Multiplication => next_node_simple_binary_op(Operation::Multiply, rest, context),
        Token::Number(_) => NumberNode::next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::next_node(op.clone(), rest, context),
        Token::LParen => LParen::next_node(op.clone(), rest, context),
        Token::LSqBracket => LSqBracket::next_node(op.clone(), rest, context),
        Token::Symbol(_) => Symbol::next_node(op.clone(), rest, context),
        Token::Identity => IdentityNode::next_node(op.clone(), rest, context),
        Token::Reduction => Reduction::next_node(op.clone(), rest, context),
        Token::LambdaVariable(_) => LambdaVariable::next_node(op.clone(), rest, context),
        Token::FunctionComposition => FunctionComposition::next_node(op.clone(), rest, context),
        Token::TypeConversion => TypeConversion::next_node(op.clone(), rest, context),
        Token::AsFunction => AsFunction::next_node(op.clone(), rest, context),
        Token::Apply => Apply::next_node(op.clone(), rest, context),
        Token::GeneralizedContraction => {
            GeneralizedContraction::next_node(op.clone(), rest, context)
        }
        Token::TensorBuilder(_) => TensorBuilder::next_node(op.clone(), rest, context),
        Token::Transpose => Transpose::next_node(op.clone(), rest, context),
        Token::Shape => Shape::next_node(op.clone(), rest, context),
        Token::Solve => Solve::next_node(op.clone(), rest, context),
        Token::SVD => Svd::next_node(op.clone(), rest, context),
        Token::QR => QR::next_node(op.clone(), rest, context),
        _ => Err(SyntaxError::UnexpectedToken(op.clone()).into()),
    }
}

/// Parses a functional operand. This must be a function or a symbol (encoding a function).
/// No arguments are parsed for the operand, since this function is passed on to something modifying the function's behavior.
fn next_node_functional(
    slice: TokenSlice,
    context: &mut ASTContext,
    needed_outputs: Option<&[IJType]>,
) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
    let token = context.get_token_at_index(slice.start)?;
    let (nodes, rest) = match token {
        Token::Reduction => {
            Reduction::next_node_functional_impl(Token::Reduction, slice, context, needed_outputs)?
        }
        Token::Plus => Add::next_node_functional_impl(Token::Plus, slice, context, needed_outputs)?,
        Token::Multiplication => Multiply::next_node_functional_impl(
            Token::Multiplication,
            slice,
            context,
            needed_outputs,
        )?,
        Token::Symbol(_) => {
            Symbol::next_node_functional_impl(token.clone(), slice, context, needed_outputs)?
        }
        Token::Minus => {
            MinusOp::next_node_functional_impl(Token::Minus, slice, context, needed_outputs)?
        }
        Token::TypeConversion => TypeConversion::next_node_functional_impl(
            Token::TypeConversion,
            slice,
            context,
            needed_outputs,
        )?,
        Token::LParen => {
            LParen::next_node_functional_impl(Token::LParen, slice, context, needed_outputs)?
        }
        Token::FunctionComposition => FunctionComposition::next_node_functional_impl(
            Token::FunctionComposition,
            slice,
            context,
            needed_outputs,
        )?,
        Token::AsFunction => AsFunction::next_node_functional_impl(
            Token::AsFunction,
            slice,
            context,
            needed_outputs,
        )?,
        Token::Identity => IdentityNode::next_node_functional_impl(
            Token::Identity,
            slice,
            context,
            needed_outputs,
        )?,
        Token::LambdaVariable(name) => LambdaVariable::next_node_functional_impl(
            Token::LambdaVariable(name.clone()),
            slice,
            context,
            needed_outputs,
        )?,
        Token::GeneralizedContraction => GeneralizedContraction::next_node_functional_impl(
            Token::GeneralizedContraction,
            slice,
            context,
            needed_outputs,
        )?,
        Token::TensorBuilder(name) => TensorBuilder::next_node_functional_impl(
            Token::TensorBuilder(name.clone()),
            slice,
            context,
            needed_outputs,
        )?,
        Token::Transpose => {
            Transpose::next_node_functional_impl(Token::Transpose, slice, context, needed_outputs)?
        }
        Token::Shape => {
            Shape::next_node_functional_impl(Token::Shape, slice, context, needed_outputs)?
        }
        Token::Solve => {
            Solve::next_node_functional_impl(Token::Solve, slice, context, needed_outputs)?
        }
        Token::SVD => Svd::next_node_functional_impl(Token::SVD, slice, context, needed_outputs)?,
        Token::QR => QR::next_node_functional_impl(Token::QR, slice, context, needed_outputs)?,
        _ => {
            return Err(SyntaxError::SliceCannotBeParsedAsFunction(
                context.token_slice_to_string(slice),
                token.to_string(),
            )
            .into())
        }
    };
    Ok((nodes, rest))
}

/// Like next_node_functional, but will only match a specific function signature.
pub fn next_node_specific_function(
    slice: TokenSlice,
    context: &mut ASTContext,
    signature: FunctionSignature,
) -> Result<(Rc<Node>, TokenSlice)> {
    let function_type = IJType::Function(signature.clone());
    let (function_options, rest) =
        next_node_functional(slice, context, Some(&[*signature.output]))?;
    let function = function_options.into_iter().find(|n| {
        n.output_type
            .type_match_function(&function_type)
            .unwrap_or(false)
    });

    if function.is_none() {
        return Err(SyntaxError::TypeError(
            function_type.to_string(),
            "Function type not found".to_string(),
        )
        .into());
    }
    Ok((function.unwrap(), rest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::Variable;
    use crate::tokens;
    use crate::types::FunctionSignature;

    #[test]
    fn test_tensor_assignment() {
        let result = parse_str_no_context("x: T = [1]");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.input_types, vec![IJType::Tensor(None); 2]);
        assert_eq!(node.output_type, IJType::Void);
        let first_operand = &node.operands[0];
        assert_eq!(first_operand.op, Operation::Symbol("x".to_string()));
        assert_eq!(first_operand.output_type, IJType::Tensor(None));
        let second_operand = &node.operands[1];
        assert_eq!(second_operand.output_type, IJType::Tensor(None));
        assert_eq!(second_operand.op, Operation::Array("1".to_string()));

        let expected_var = Variable {
            typ: IJType::Tensor(None),
            name: "x".to_string(),
        };
        let actual_var = context.symbols.get("x").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_scalar_assignment() {
        let result = parse_str_no_context("x: S = 1");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.input_types, vec![IJType::Scalar(None); 2]);
        assert_eq!(node.output_type, IJType::Void);
        let first_operand = &node.operands[0];
        assert_eq!(first_operand.op, Operation::Symbol("x".to_string()));
        assert_eq!(first_operand.output_type, IJType::Scalar(None));
        let second_operand = &node.operands[1];
        assert_eq!(second_operand.output_type, IJType::Scalar(None));
        assert_eq!(
            second_operand.op,
            Operation::Number(tokens::Number {
                value: "1".to_string()
            })
        );

        let expected_var = Variable {
            typ: IJType::Scalar(None),
            name: "x".to_string(),
        };
        let actual_var = context.symbols.get("x").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_wrongly_annotated_assignment() {
        let result = parse_str_no_context("x: T = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_function_assignment() {
        let result = parse_str_no_context("x($x): Fn(T->T) = + [1] $x");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Assign);
        let expected_signature = FunctionSignature::tensor_function(1);
        assert_eq!(
            node.input_types,
            vec![
                IJType::Function(expected_signature.clone()),
                IJType::Tensor(None)
            ]
        );
        assert_eq!(node.output_type, IJType::Void);
        let first_operand = &node.operands[0];
        assert_eq!(first_operand.op, Operation::Symbol("x".to_string()));

        assert_eq!(
            first_operand.output_type,
            IJType::Function(expected_signature.clone())
        );
        let second_operand = &node.operands[1];
        assert_eq!(second_operand.op, Operation::Add);
        assert_eq!(second_operand.output_type, IJType::Tensor(None));

        let expected_var = Variable {
            typ: IJType::Function(expected_signature),
            name: "x".to_string(),
        };
        let actual_var = context.symbols.get("x").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_undefined_variable() {
        let result = parse_str_no_context("+ x 1");
        assert!(result.is_err());
        let err = result.unwrap_err();
        println!("{:?}", err);
        assert!(is_specific_syntax_error(
            &err,
            &SyntaxError::UnknownSymbol("x".to_string())
        ));
    }

    #[test]
    fn test_var_definition_and_use() {
        let mut context = ASTContext::new();
        parse_str("var x: Fn(S->T)", &mut context).unwrap();
        let result = parse_str("x 1", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Function("x".to_string()));
        assert_eq!(node.input_types, vec![IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_assign_definition_and_use() {
        let mut context = ASTContext::new();
        parse_str("add2($x) -> T = + $x [2]", &mut context).unwrap();
        let result = parse_str("add2 [1]", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Function("add2".to_string()));
        assert_eq!(node.input_types, vec![IJType::Tensor(None)]);
        assert_eq!(node.output_type, IJType::Tensor(None));
    }

    #[test]
    fn test_nested_parentheses() {
        let result = parse_str_no_context("(- (+ (1) 2))");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Negate);
        assert_eq!(node.input_types, vec![IJType::Scalar(None)]);
        assert_eq!(node.output_type, IJType::Scalar(None));
    }

    #[test]
    fn test_semicolon_handling() {
        // Test with semicolon at the end - should pass
        let result_with_semicolon = parse_str_no_context("var x: S;");
        assert!(result_with_semicolon.is_ok());

        // Test without semicolon at the end - should also pass and be equivalent
        let result_without_semicolon = parse_str_no_context("var x: S");
        assert!(result_without_semicolon.is_ok());

        // Compare the results to ensure they are equivalent
        let (_, context_with_semicolon) = result_with_semicolon.unwrap();
        let (_, context_without_semicolon) = result_without_semicolon.unwrap();
        assert_eq!(
            context_with_semicolon.symbols.get("x"),
            context_without_semicolon.symbols.get("x")
        );

        // Test with semicolon not at the end - should fail
        let result_error = parse_str_no_context("var x: S; var y: T");
        assert!(result_error.is_err());
        assert!(is_specific_syntax_error(
            &result_error.unwrap_err(),
            &SyntaxError::UnexpectedSemicolon
        ));
    }

    #[test]
    fn test_group_return_type() -> Result<()> {
        let mut context = ASTContext::new();
        parse_str("var f: Fn(T->(S,T))", &mut context)?;
        let node = parse_str("f [1]", &mut context)?;
        assert_eq!(
            node.output_type,
            IJType::Group(vec![IJType::Scalar(None), IJType::Tensor(None)])
        );

        let mut context = ASTContext::new();
        parse_str("var f: Fn(T->(S,T))", &mut context)?;
        let node = parse_str("x = f[1]", &mut context)?;
        assert_eq!(node.op, Operation::Assign);
        let node0 = &node.operands[0];
        assert_eq!(
            node0.output_type,
            IJType::Group(vec![IJType::Scalar(None), IJType::Tensor(None)])
        );
        let node1 = &node.operands[1];
        assert_eq!(
            node1.output_type,
            IJType::Group(vec![IJType::Scalar(None), IJType::Tensor(None)])
        );

        Ok(())
    }
}
