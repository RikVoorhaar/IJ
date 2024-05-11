#[cfg(test)]
mod test_utils;

mod parser_functions;
pub use parser_functions::{parse_line, parse_lines};

mod utils;
use utils::{check_ok_needed_outputs, find_matching_parenthesis, gather_all, gather_operands};

mod binary_op;
use binary_op::{next_node_simple_binary_op, Add, Multiply};

mod parenthesis;
use parenthesis::LParen;

mod square_brackets;
use square_brackets::LSqBracket;

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

use crate::ast_node::{ASTContext, IJType, Node, TokenSlice};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

trait ParseNode {
    fn next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)>;
}

trait ParseNodeFunctional {
    fn next_node_functional_impl(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
        needed_outputs: Option<&[Vec<IJType>]>,
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
        _ => Err(SyntaxError::UnexpectedToken(op.clone()).into()),
    }
}

/// Parses a functional operand. This must be a function or a symbol (encoding a function).
/// No arguments are parsed for the operand, since this function is passed on to something modifying the function's behavior.
fn next_node_functional(
    slice: TokenSlice,
    context: &mut ASTContext,
    needed_outputs: Option<&[Vec<IJType>]>,
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
        _ => return Err(SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into()),
    };
    Ok((nodes, rest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_node::{FunctionSignature, Variable};
    use crate::tokens::{self, Number};
    use test_utils::{is_specific_syntax_error, parse_str, parse_str_no_context};

    #[test]
    fn test_empty_input() {
        let result = parse_str_no_context("");
        assert!(result.is_err());
        assert!(is_specific_syntax_error(
            &result.unwrap_err(),
            &SyntaxError::EmptyInput
        ));
    }

    #[test]
    fn test_variable_declaration_tensor() {
        let result = parse_str_no_context("var a: T");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Tensor,
            name: "a".to_string(),
        };
        let actual_var = context.symbols.get("a").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_variable_declaration_scalar() {
        let result = parse_str_no_context("var b: S");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Scalar,
            name: "b".to_string(),
        };
        let actual_var = context.symbols.get("b").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_function_declaration_tensor_to_tensor() {
        let result = parse_str_no_context("var add: Fn(T,T -> T)");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor, IJType::Tensor],
                output: vec![IJType::Tensor],
            }),
            name: "add".to_string(),
        };
        let actual_var = context.symbols.get("add").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_function_declaration_scalar_to_tensor() {
        let result = parse_str_no_context("var scale: Fn(S -> T)");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Nothing);
        let expected_var = Variable {
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Scalar],
                output: vec![IJType::Tensor],
            }),
            name: "scale".to_string(),
        };
        let actual_var = context.symbols.get("scale").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_tensor_assignment() {
        let result = parse_str_no_context("x: T = [1]");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Assign);
        assert_eq!(node.input_types, vec![IJType::Tensor; 2]);
        assert_eq!(node.output_type, IJType::Void);
        let first_operand = &node.operands[0];
        assert_eq!(first_operand.op, Operation::Symbol("x".to_string()));
        assert_eq!(first_operand.output_type, IJType::Tensor);
        let second_operand = &node.operands[1];
        assert_eq!(second_operand.output_type, IJType::Tensor);
        assert_eq!(second_operand.op, Operation::Array("1".to_string()));

        let expected_var = Variable {
            typ: IJType::Tensor,
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
        assert_eq!(node.input_types, vec![IJType::Scalar; 2]);
        assert_eq!(node.output_type, IJType::Void);
        let first_operand = &node.operands[0];
        assert_eq!(first_operand.op, Operation::Symbol("x".to_string()));
        assert_eq!(first_operand.output_type, IJType::Scalar);
        let second_operand = &node.operands[1];
        assert_eq!(second_operand.output_type, IJType::Scalar);
        assert_eq!(
            second_operand.op,
            Operation::Number(tokens::Number {
                value: "1".to_string()
            })
        );

        let expected_var = Variable {
            typ: IJType::Scalar,
            name: "x".to_string(),
        };
        let actual_var = context.symbols.get("x").unwrap();
        assert_eq!(actual_var, &expected_var);
    }

    #[test]
    fn test_wrongly_annotated_assignment() {
        let result = parse_str_no_context("x: T = 1");
        assert!(result.is_err());
        assert!(is_specific_syntax_error(
            &result.unwrap_err(),
            &SyntaxError::TypeError(IJType::Scalar.to_string(), IJType::Tensor.to_string())
        ));
    }

    #[test]
    fn test_function_assignment() {
        let result = parse_str_no_context("x: Fn(T->T) = + [1] I");
        assert!(result.is_ok());
        let (node, context) = result.unwrap();
        assert_eq!(node.op, Operation::Assign);
        let expected_signature = FunctionSignature {
            input: vec![IJType::Tensor],
            output: vec![IJType::Tensor],
        };
        assert_eq!(
            node.input_types,
            vec![IJType::Function(expected_signature.clone()), IJType::Tensor]
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
        assert_eq!(second_operand.output_type, IJType::Tensor);

        assert_eq!(
            parser_functions::compute_input_type(second_operand).unwrap(),
            expected_signature.input.clone()
        );

        let expected_var = Variable {
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor],
                output: vec![IJType::Tensor],
            }),
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
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_assign_definition_and_use() {
        let mut context = ASTContext::new();
        parse_str("add2: Fn(T->T) = + (I) [2]", &mut context).unwrap();
        let result = parse_str("add2 [1]", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Function("add2".to_string()));
        assert_eq!(node.input_types, vec![IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_add_with_tensors() {
        let result = parse_str_no_context("+ [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_add_with_tensors_and_scalar() {
        let result = parse_str_no_context("+ [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_add_with_scalars() {
        let result = parse_str_no_context("+ 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Scalar, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_multiply_with_tensors() {
        let result = parse_str_no_context("* [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_multiply_with_tensors_and_scalar() {
        let result = parse_str_no_context("* [1] 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_multiply_with_scalars() {
        let result = parse_str_no_context("* 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Multiply);
        assert_eq!(node.input_types, vec![IJType::Scalar, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_negate_with_scalar() {
        let result = parse_str_no_context("- 1.0");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "-1.0".to_string()
            })
        );
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }
    #[test]
    fn test_double_negate_with_scalar() {
        let result = parse_str_no_context("-- 1.0");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(
            node.op,
            Operation::Number(Number {
                value: "1.0".to_string()
            })
        );
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_negate_with_tensor() {
        let result = parse_str_no_context("- [1]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Negate);
        assert_eq!(node.input_types, vec![IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_tensors() {
        let result = parse_str_no_context("- [1] [2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_tensor_and_scalar() {
        let result = parse_str_no_context("- [1] 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_subtract_with_scalars() {
        let result = parse_str_no_context("- 1 2");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Subtract);
        assert_eq!(node.input_types, vec![IJType::Scalar, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_nested_parentheses() {
        let result = parse_str_no_context("(- (+ (1) 2))");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Negate);
        assert_eq!(node.input_types, vec![IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_reduce_plus_with_tensor() {
        let result = parse_str_no_context("/+ [1,2]");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Reduce);

        assert_eq!(
            node.input_types,
            vec![IJType::scalar_function(2, 1), IJType::Tensor]
        );
        assert_eq!(node.output_type, IJType::Scalar);
    }

    #[test]
    fn test_lambda_variable() {
        let result = parse_str_no_context("$x");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::LambdaVariable("x".to_string()));

        let result = parse_str_no_context("+ $x 1");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Scalar]);
        assert_eq!(node.output_type, IJType::Tensor);

        let result = parse_str_no_context("+ $x $x");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);

        let result = parse_str_no_context("+ $x $y");
        assert!(result.is_ok());
        let (node, _) = result.unwrap();
        assert_eq!(node.op, Operation::Add);
        assert_eq!(node.input_types, vec![IJType::Tensor, IJType::Tensor]);
        assert_eq!(node.output_type, IJType::Tensor);
    }

    #[test]
    fn test_reduce_with_function_and_tensor() {
        let mut context = ASTContext::new();
        let var_declaration = parse_str("var f: Fn(S,S->S)", &mut context);
        assert!(var_declaration.is_ok());
        let result = parse_str("/f [1,2]", &mut context);
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.op, Operation::Reduce);
        assert_eq!(
            node.input_types,
            vec![IJType::scalar_function(2, 1), IJType::Tensor]
        );
        assert_eq!(node.output_type, IJType::Scalar);
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
}
