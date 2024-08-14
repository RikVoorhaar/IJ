use super::{parser_functions, ASTContext, Node, SyntaxError};

use crate::{ast_node::TokenSlice, tokens::lexer};
use anyhow::{anyhow, Result};
use std::rc::Rc;

pub fn tokenize_str_no_context(input: &str) -> Result<(ASTContext, TokenSlice), anyhow::Error> {
    let tokens = match lexer(input) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    let mut context = ASTContext::new();
    let num_tokens = tokens.len();
    context.set_tokens(tokens);
    let slice = TokenSlice::new(0, num_tokens, num_tokens);
    Ok((context, slice))
}

pub fn parse_str(input: &str, context: &mut ASTContext) -> Result<Rc<Node>, anyhow::Error> {
    let tokens = match lexer(input) {
        Ok(t) => t,
        Err(e) => {
            println!("Error in lexer:\n {:?}", e);
            return Err(e);
        }
    };
    let result = match parser_functions::parse_line(tokens, context) {
        Ok(n) => n,
        Err(e) => {
            println!("Error in parser:\n {:?}", e);
            return Err(e);
        }
    };
    let tree_ok = verify_types_tree(&result);
    if let Err(e) = tree_ok {
        println!("Error veryfing tree:\n {:?}", e);
    }

    Ok(result)
}

/// Parses a string and returns a node and the context
/// Useful for testing and debugging
pub fn parse_str_no_context(input: &str) -> Result<(Rc<Node>, ASTContext)> {
    let mut context = ASTContext::new();
    let node = match parse_str(input, &mut context) {
        Ok(node) => {
            println!("Node: {:?}", node);
            node
        }
        Err(e) => {
            println!("Error: {:?}", e);
            return Err(e);
        }
    };
    Ok((node, context))
}

pub fn is_specific_syntax_error(error: &anyhow::Error, expected_error: &SyntaxError) -> bool {
    if let Some(actual_error) = error.downcast_ref::<SyntaxError>() {
        actual_error == expected_error
    } else {
        false
    }
}

pub fn verify_types_tree(node: &Node) -> Result<(), anyhow::Error> {
    if node.operands.is_empty() {
        return Ok(());
    }
    if node.operands.len() != node.input_types.len() {
        return Err(anyhow!("Operands and input types length mismatch"));
    }
    for (operand, input_type) in node.operands.iter().zip(node.input_types.iter()) {
        if operand.output_type != *input_type {
            return Err(anyhow!("Operand and input type mismatch"));
        }
    }

    node.operands.iter().try_for_each(|n| verify_types_tree(n))
}

pub fn create_context_with_tokens_from_str(input: &str) -> Result<ASTContext, anyhow::Error> {
    let tokens = match lexer(input) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    let mut context = ASTContext::new();
    context.set_tokens(tokens);
    Ok(context)
}
