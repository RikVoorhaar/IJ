use super::next_node;

use crate::ast_node::{ASTContext, IJType, Node, TokenSlice};
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use anyhow::Result;
use std::rc::Rc;

/// Given a list of list of types, gather operands whose type together matches any of the variants.
/// Returns the nodes matching the longest variant.
pub fn gather_operands(
    types: Vec<Vec<IJType>>,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
    let mut operands = Vec::new();
    let longest_variant_length = types.iter().map(|t| t.len()).max().unwrap();
    let shortest_variant_length = types.iter().map(|t| t.len()).min().unwrap();
    let mut rest = slice;
    let mut longest_match_length = 0;
    for _ in 0..longest_variant_length {
        let result = next_node(rest, context);
        if let Err(e) = result {
            if longest_match_length < shortest_variant_length {
                return Err(e);
            }
            break;
        }

        let (node, new_rest) = result.unwrap();
        rest = new_rest;
        operands.push(node);
        let operands_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();
        if types.iter().any(|t| t == &operands_types) {
            longest_match_length = operands.len();
        }
    }
    if longest_match_length < shortest_variant_length {
        let input_types_str = types
            .iter()
            .map(|t| {
                format!(
                    "[{}]",
                    t.iter()
                        .map(|inner_t| inner_t.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            })
            .collect::<Vec<String>>()
            .join(", ");
        let found_types_str = operands
            .iter()
            .map(|n| n.output_type.to_string())
            .collect::<String>();
        let syntax_error = SyntaxError::GatherMismatch(input_types_str, found_types_str);
        return Err(context.add_context_to_syntax_error(syntax_error.into(), slice));
    }
    let operands = operands[..longest_match_length].to_vec();

    Ok((operands, rest))
}

/// Match as many tokens of any kind as possible.
/// Always consumes the entire token slice
pub fn gather_all(slice: TokenSlice, context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
    let mut operands = Vec::new();
    let mut rest = slice;
    while !rest.is_empty() {
        let (node, new_rest) =
            next_node(rest, context).map_err(|e| context.add_context_to_syntax_error(e, rest))?;
        operands.push(node);
        rest = new_rest;
    }
    Ok(operands)
}

/// Finds the index of the matching parenthesis in the token slice.
/// This function assumes that the opening parenthesis is not part of the slice.
pub fn find_matching_parenthesis(
    context: &mut ASTContext,
    slice: TokenSlice,
    lparen: &Token,
    rparen: &Token,
) -> Result<usize> {
    let tokens = &context.get_tokens()[slice.start..slice.end];
    let mut depth = 1;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            token if token == lparen => depth += 1,
            token if token == rparen => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => (),
        }
    }
    Err(SyntaxError::UnmatchedParenthesis(context.tokens_to_string(slice)).into())
}

pub fn check_ok_needed_outputs(
    needed_outputs: Option<&[Vec<IJType>]>,
    actual_outputs: &[IJType],
) -> bool {
    let num_outputs = actual_outputs.len();
    if let Some(needed_outputs) = needed_outputs {
        if needed_outputs
            .iter()
            .filter(|outputs| outputs.len() <= num_outputs)
            .all(|outputs| actual_outputs != &outputs[..num_outputs])
        {
            return false;
        }
    }
    true
}

/// Splits out a token slice into a vector of token slices, separated by commas.
pub fn comma_separate(slice: TokenSlice, context: &mut ASTContext) -> Result<Vec<TokenSlice>> {
    let mut slices: Vec<TokenSlice> = vec![];
    let mut last_endpoint = slice.start;
    let tokens = &context.get_tokens()[slice.start..slice.end];
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        match token {
            Token::Comma => {
                let new_slice = slice.move_end(index)?.move_start(last_endpoint)?;
                slices.push(new_slice);
                last_endpoint = index + 1;
            }

            Token::LParen => {
                let group_len = find_matching_parenthesis(
                    context,
                    slice.move_start(index + 1)?,
                    &Token::LParen,
                    &Token::RParen,
                )?;
                index += group_len;
            }
            Token::LSqBracket => {
                let group_len = find_matching_parenthesis(
                    context,
                    slice.move_start(index + 1)?,
                    &Token::LSqBracket,
                    &Token::RSqBracket,
                )?;
                index += group_len;
            }

            _ => (),
        }

        index += 1;
    }
    let last_slice = slice.move_end(tokens.len())?.move_start(last_endpoint)?;
    slices.push(last_slice);

    if slices.iter().any(|slice| slice.is_empty()) {
        return Err(SyntaxError::EmptySlice.into());
    }

    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::next_node_functional;
    use crate::parser::test_utils::*;

    #[test]
    fn test_comma_separate() {
        let maybe_context = create_context_with_tokens_from_str("+,/-,f");
        if let Err(e) = maybe_context {
            panic!("Failed to create context: {:?}", e);
        }
        let mut context = maybe_context.unwrap();

        let maybe_slices = comma_separate(context.full_slice(), &mut context);
        if let Err(e) = maybe_slices {
            panic!("Failed to comma separate: {:?}", e);
        }
        let slices = maybe_slices.unwrap();
        for slice in &slices {
            let tokens = &context.get_tokens()[slice.start..slice.end];
            println!("{:?}: '{:?}'", slice, tokens);
        }
        let correct_slices = vec![
            TokenSlice::new(0, 1, 6),
            TokenSlice::new(2, 4, 6),
            TokenSlice::new(5, 6, 6),
        ];
        assert_eq!(slices, correct_slices);
    }

    #[test]
    fn test_comma_separate_empty() {
        let maybe_context = create_context_with_tokens_from_str("");
        if let Err(e) = maybe_context {
            panic!("Failed to create context: {:?}", e);
        }
        let mut context = maybe_context.unwrap();

        let maybe_slices = comma_separate(context.full_slice(), &mut context);
        assert!(maybe_slices.is_err());

        let error = maybe_slices.unwrap_err();
        let actual_error = error.downcast_ref::<SyntaxError>().unwrap();
        assert_eq!(actual_error, &SyntaxError::EmptySlice);
    }

    #[test]
    fn test_comma_separate_with_nesting() {
        let maybe_context = create_context_with_tokens_from_str("+,(x,y),f");
        if let Err(e) = maybe_context {
            panic!("Failed to create context: {:?}", e);
        }
        let mut context = maybe_context.unwrap();

        let maybe_slices = comma_separate(context.full_slice(), &mut context);
        let slices = maybe_slices.unwrap();
        for slice in &slices {
            let tokens = &context.get_tokens()[slice.start..slice.end];
            println!("{:?}: '{:?}'", slice, tokens);
        }
        let correct_slices = vec![
            TokenSlice::new(0, 1, 9),
            TokenSlice::new(2, 7, 9),
            TokenSlice::new(8, 9, 9),
        ];
        assert_eq!(slices, correct_slices);
    }

    #[test]
    fn test_comma_separate_with_array() {
        let maybe_context = create_context_with_tokens_from_str("+,[0,1],f");
        if let Err(e) = maybe_context {
            panic!("Failed to create context: {:?}", e);
        }
        let mut context = maybe_context.unwrap();

        let maybe_slices = comma_separate(context.full_slice(), &mut context);
        let slices = maybe_slices.unwrap();
        for slice in &slices {
            let tokens = &context.get_tokens()[slice.start..slice.end];
            println!("{:?}: '{:?}'", slice, tokens);
        }
        let correct_slices = vec![
            TokenSlice::new(0, 1, 9),
            TokenSlice::new(2, 7, 9),
            TokenSlice::new(8, 9, 9),
        ];
        assert_eq!(slices, correct_slices);
    }

    #[test]
    fn test_comma_separate_next_node_functional() {
        let maybe_context = create_context_with_tokens_from_str("/+,+");
        if let Err(e) = maybe_context {
            panic!("Failed to create context: {:?}", e);
        }
        let mut context = maybe_context.unwrap();

        let maybe_slices = comma_separate(context.full_slice(), &mut context);
        let slices = maybe_slices.unwrap();
        for slice in &slices {
            let tokens = &context.get_tokens()[slice.start..slice.end];
            println!("{:?}: '{:?}'", slice, tokens);
            let (node, rest) = next_node_functional(*slice, &mut context, None).unwrap();
            println!("{:?}", node);
            println!("{:?}", rest.is_empty());
            println!("---");
        }
    }
}
