use crate::ast_node::{
    ASTContext, FunctionSignature, IJType, LineHasSemicolon, Node, TokenSlice, Variable,
};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, SymbolToken, Token};
use anyhow::{anyhow, Context, Result};
use std::fmt::Debug;
use std::rc::Rc;

trait ParseNode {
    fn next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)>;
}

pub fn parse_lines(
    tokens: Vec<Token>,
    context: &mut ASTContext,
) -> Result<Vec<(Rc<Node>, LineHasSemicolon)>> {
    let mut parsed_lines = Vec::new();
    let mut line_tokens = Vec::new();
    for token in tokens {
        if token == Token::Newline || token == Token::Semicolon {
            if line_tokens.is_empty() {
                continue;
            }
            let root = parse_line(std::mem::take(&mut line_tokens), context)?;
            let has_semicolon = match token {
                Token::Semicolon => LineHasSemicolon::Yes,
                _ => LineHasSemicolon::No,
            };
            parsed_lines.push((root, has_semicolon));
        } else {
            line_tokens.push(token);
        }
    }
    if !line_tokens.is_empty() {
        let last_token = line_tokens.last().unwrap();
        let has_semicolon = match last_token {
            Token::Semicolon => LineHasSemicolon::Yes,
            _ => LineHasSemicolon::No,
        };
        parsed_lines.push((
            parse_line(std::mem::take(&mut line_tokens), context)?,
            has_semicolon,
        ));
    }
    Ok(parsed_lines)
}

pub fn parse_line(tokens: Vec<Token>, context: &mut ASTContext) -> Result<Rc<Node>> {
    if let Some(pos) = tokens.iter().position(|t| *t == Token::Semicolon) {
        if pos != tokens.len() - 1 {
            return Err(SyntaxError::UnexpectedSemicolon.into());
        }
        context.set_tokens(tokens[..pos].to_vec());
    } else {
        context.set_tokens(tokens);
    }
    context.line_no += 1;
    parse_ast(context)
}

fn parse_ast(context: &mut ASTContext) -> Result<Rc<Node>> {
    match context.tokens.as_slice() {
        [] => Err(SyntaxError::EmptyInput.into()),
        [Token::Variable, ..] => parse_var_statement(context),
        [Token::Symbol(symbol), rest @ ..] if rest.contains(&Token::Assign) => {
            parse_assign(symbol.name.clone(), context)
        }
        _ => {
            let (node, remainder) = next_node(context.full_slice(), context)?;
            if !remainder.is_empty() {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::UnhandledTokens(context.tokens_to_string(remainder)).into(),
                    remainder,
                ));
            }
            Ok(node)
        }
    }
}

fn parse_var_statement(context: &mut ASTContext) -> Result<Rc<Node>> {
    match context.tokens.as_slice() {
        [Token::Variable, Token::Symbol(symbol), Token::TypeDeclaration, rest @ ..] => {
            let typ = IJType::from_tokens(rest)?;
            let var = Variable {
                name: symbol.name.clone(),
                typ: typ.clone(),
            };
            context.insert_variable(var);
            Ok(Rc::new(Node::new(
                Operation::Nothing,
                vec![],
                typ,
                vec![],
                context.get_increment_id(),
            )))
        }
        _ => Err(
            SyntaxError::InvalidVarStatement(context.tokens_to_string(context.full_slice())).into(),
        ),
    }
}

/// Get the nodes expressed on the right hand side of an assignment token.
fn get_variable_expression(context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
    let tokens = context.get_tokens();
    let mut split = tokens.split(|t| t == &Token::Assign);
    let assign_index: usize = split
        .next()
        .ok_or(context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()).into(),
            context.full_slice(),
        ))?
        .len();
    let mut slice = context.full_slice().move_start(assign_index + 1)?;

    let mut expressions = Vec::new();
    while !slice.is_empty() {
        let (expression, rest) = next_node(slice, context)?;
        expressions.push(expression);
        slice = rest;
    }
    Ok(expressions)
}

/// Recursively walk down AST. If we reach a leaf node without an operand, add it to the input types. Otherwise concatenate the input types of its operands.
pub fn compute_input_type(node: &Rc<Node>) -> Result<Vec<IJType>> {
    if node.operands.is_empty() {
        return Ok(node.input_types.clone());
    }
    let mut input_types = Vec::new();
    for operand in node.operands.iter() {
        input_types.extend(compute_input_type(operand)?);
    }
    Ok(input_types)
}

/// Parses a variable assignment statement.
pub fn parse_assign(variable_name: String, context: &mut ASTContext) -> Result<Rc<Node>> {
    let expressions = get_variable_expression(context)?;
    if expressions.len() != 1 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::InvalidAssignmentStatement(context.tokens_to_string(context.full_slice()))
                .into(),
            context.full_slice(),
        ));
    }
    let expression = expressions.into_iter().next().unwrap();
    let left_of_assign = context.tokens.split(|t| t == &Token::Assign).next().ok_or(
        context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()).into(),
            context.full_slice(),
        ),
    )?;
    let expected_type = match left_of_assign {
        [Token::Symbol(_), Token::TypeDeclaration, rest @ ..] => Some(IJType::from_tokens(rest)?),
        [Token::Symbol(_)] => None,
        _ => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::InvalidAssignmentStatement(
                    context.tokens_to_string(context.full_slice()),
                )
                .into(),
                context.full_slice(),
            ))
        }
    };

    let input_type = compute_input_type(&expression)?;
    let output_type = expression.output_type.clone();
    let expression_type = match input_type.is_empty() {
        true => output_type.clone(),
        false => IJType::Function(FunctionSignature {
            input: input_type,
            output: vec![output_type.clone()],
        }),
    };

    if let Some(expected_type) = expected_type {
        if expression_type != expected_type {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::TypeError(expression_type.to_string(), expected_type.to_string())
                    .into(),
                context.full_slice(),
            ));
        }
    }
    context.insert_variable(Variable {
        name: variable_name.clone(),
        typ: expression_type.clone(),
    });
    let symbol_node = Rc::new(Node::new(
        Operation::Symbol(variable_name.clone()),
        vec![],
        expression_type.clone(),
        Vec::new(),
        context.get_increment_id(),
    ));
    let operand_node = expression.clone();
    let assign_node = Rc::new(Node::new(
        Operation::Assign,
        vec![expression_type, output_type],
        IJType::Void,
        vec![symbol_node, operand_node],
        context.get_increment_id(),
    ));
    Ok(assign_node)
}

/// Main logic for parsing a node.
fn next_node(slice: TokenSlice, context: &mut ASTContext) -> Result<(Rc<Node>, TokenSlice)> {
    let op = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    match op {
        Token::Plus => next_node_simple_binary_op(Operation::Add, rest, context),
        Token::Multiplication => next_node_simple_binary_op(Operation::Multiply, rest, context),
        Token::Number(_) => Number::next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::next_node(op.clone(), rest, context),
        Token::LParen => LParen::next_node(op.clone(), rest, context),
        Token::LSqBracket => LSqBracket::next_node(op.clone(), rest, context),
        Token::Symbol(_) => Symbol::next_node(op.clone(), rest, context),
        Token::Identity => IdentityOp::next_node(op.clone(), rest, context),
        Token::Reduction => Reduction::next_node(op.clone(), rest, context),
        _ => Err(SyntaxError::UnexpectedToken(op.clone()).into()),
    }
}

/// Given a list of list of types, gather operands whose type together matches any of the variants.
/// Returns the nodes matching the longest variant.
fn gather_operands(
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
fn gather_all(slice: TokenSlice, context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
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

/// Parses a simple tensor function, such as addition or multiplication. These all have
/// a range of input aritys. All inputs tensors or scalars.
fn next_node_simple_binary_op(
    op: Operation,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let types = vec![
        vec![IJType::Tensor, IJType::Tensor],
        vec![IJType::Scalar, IJType::Tensor],
        vec![IJType::Tensor, IJType::Scalar],
        vec![IJType::Scalar, IJType::Scalar],
    ];
    let (operands, rest) = gather_operands(types, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    let input_types = operands
        .iter()
        .map(|n| n.output_type.clone())
        .collect::<Vec<IJType>>();
    let output_type = match input_types.as_slice() {
        [IJType::Tensor, IJType::Tensor] => IJType::Tensor,
        [IJType::Scalar, IJType::Tensor] => IJType::Tensor,
        [IJType::Tensor, IJType::Scalar] => IJType::Tensor,
        [IJType::Scalar, IJType::Scalar] => IJType::Scalar,
        _ => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::TypeError(
                    input_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<String>(),
                    "Tensor -> Tensor or Scalar -> Scalar".to_string(),
                )
                .into(),
                slice,
            ))
        }
    };
    Ok((
        Rc::new(Node::new(
            op,
            input_types,
            output_type,
            operands,
            context.get_increment_id(),
        )),
        rest,
    ))
}

/// Finds the index of the matching parenthesis in the token slice.
/// This function assumes that the opening parenthesis is not part of the slice.
fn find_matching_parenthesis(
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

/// Left parenthesis denotes the start of a group.
/// This node is therefore a function with operands as inputs and a group as output.
/// A group with a single operand is however just the identity function.
#[derive(Debug, PartialEq)]
struct LParen;
impl ParseNode for LParen {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;

        let operands = gather_all(slice, context)
            .with_context(|| anyhow!("Error created by bracket parsing"))?;

        match operands.len() {
            0 => unreachable!(),
            1 => Ok((operands[0].clone(), remainder)),
            _ => {
                let operands_types = operands
                    .iter()
                    .map(|n| n.output_type.clone())
                    .collect::<Vec<IJType>>();
                Ok((
                    Rc::new(Node::new(
                        Operation::Group,
                        operands_types.clone(),
                        IJType::Group(operands_types),
                        operands,
                        context.get_increment_id(),
                    )),
                    remainder,
                ))
            }
        }
    }
}

/// Square brackets denote the start of an array.
#[derive(Debug, PartialEq)]
struct LSqBracket;
impl ParseNode for LSqBracket {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LSqBracket, &Token::RSqBracket)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;
        let contents = context.tokens_to_string(slice);

        Ok((
            Rc::new(Node::new(
                Operation::Array(contents),
                vec![],
                IJType::Tensor,
                vec![],
                context.get_increment_id(),
            )),
            remainder,
        ))
    }
}

struct Symbol;
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
                IJType::Tensor => (
                    Node::new(
                        Operation::Symbol(name.clone()),
                        vec![],
                        IJType::Tensor,
                        vec![],
                        context.get_increment_id(),
                    ),
                    slice,
                ),
                IJType::Function(signature) => {
                    let (operands, rest) =
                        gather_operands(vec![signature.input.clone()], slice, context)?;
                    let output_type = match signature.output.len() {
                        1 => signature.clone().output.into_iter().next().unwrap(),
                        _ => IJType::Group(signature.clone().output),
                    };
                    let node = Node::new(
                        Operation::Function(name.clone()),
                        signature.clone().input,
                        output_type,
                        operands,
                        context.get_increment_id(),
                    );
                    (node, rest)
                }
                IJType::Scalar => (
                    Node::new(
                        Operation::Symbol(name.clone()),
                        vec![],
                        IJType::Scalar,
                        vec![],
                        context.get_increment_id(),
                    ),
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

/// Parses a functional operand. This must be a function or a symbol (encoding a function).
/// No arguments are parsed for the operand, since this function is passed on to something modifying the function's behavior.
fn next_node_functional(
    slice: TokenSlice,
    context: &mut ASTContext,
    signature: &FunctionSignature,
) -> Result<(Rc<Node>, TokenSlice)> {
    let token = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    let op = match token {
        Token::Plus => {
            if signature != &FunctionSignature::tensor_function(2, 1)
                && signature != &FunctionSignature::scalar_function(2, 1)
            {
                return Err(SyntaxError::FunctionSignatureMismatch(
                    signature.to_string(),
                    "Tensor -> Tensor -> Tensor or Scalar -> Scalar -> Scalar".to_string(),
                )
                .into());
            }
            Operation::Add
        }
        Token::Multiplication => {
            if signature != &FunctionSignature::tensor_function(2, 1)
                && signature != &FunctionSignature::scalar_function(2, 1)
            {
                return Err(SyntaxError::FunctionSignatureMismatch(
                    signature.to_string(),
                    "Tensor -> Tensor -> Tensor or Scalar -> Scalar -> Scalar".to_string(),
                )
                .into());
            }
            Operation::Multiply
        }

        Token::Symbol(SymbolToken { name }) => {
            let variable = context
                .symbols
                .get(name.as_str())
                .ok_or(SyntaxError::UnknownSymbol(name.clone()))?;
            if let IJType::Function(ref func_signature) = variable.typ {
                if func_signature != signature {
                    return Err(
                        SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into(),
                    );
                }
                Operation::Function(name.clone())
            } else {
                return Err(SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into());
            }
        }

        _ => return Err(SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into()),
    };
    Ok((
        Rc::new(Node::new(
            op,
            vec![],
            IJType::Function(signature.clone()),
            vec![],
            context.get_increment_id(),
        )),
        rest,
    ))
}

struct Reduction;
impl ParseNode for Reduction {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let function_signature =
            FunctionSignature::new(vec![IJType::Scalar; 2], vec![IJType::Scalar]);
        let function_type = IJType::Function(function_signature.clone());
        let (function, rest) = next_node_functional(slice, context, &function_signature)?;
        if function.output_type != function_type {
            return Err(SyntaxError::TypeError(
                function_type.to_string(),
                function.output_type.to_string(),
            )
            .into());
        }

        let (operands, rest) = gather_operands(vec![vec![IJType::Tensor]], rest, context)?;
        let operand = operands.into_iter().next().unwrap();
        Ok((
            Rc::new(Node::new(
                Operation::Reduce,
                vec![function_type, IJType::Tensor],
                IJType::Scalar,
                vec![function, operand],
                context.get_increment_id(),
            )),
            rest,
        ))
    }
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl ParseNode for MinusOp {
    fn next_node(
        _: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        // let single_tensor = vec![IJType::Tensor];
        // let two_tensors = vec![IJType::Tensor; 2];
        let allowed_types = vec![
            vec![IJType::Tensor],
            vec![IJType::Scalar],
            vec![IJType::Tensor, IJType::Tensor],
            vec![IJType::Scalar, IJType::Scalar],
            vec![IJType::Tensor, IJType::Scalar],
            vec![IJType::Scalar, IJType::Tensor],
        ];
        let (operands, rest) = gather_operands(allowed_types, slice, context)?;
        let input_types = operands
            .iter()
            .map(|n| n.output_type.clone())
            .collect::<Vec<IJType>>();
        let output_type = if input_types.iter().all(|t| t == &IJType::Scalar) {
            IJType::Scalar
        } else {
            IJType::Tensor
        };
        if operands.len() == 1 {
            if let Operation::Number(x) = operands[0].op.clone() {
                let x_val = x.value;
                let modified_x_val = if x_val.starts_with('-') {
                    x_val.trim_start_matches('-').to_string()
                } else {
                    format!("-{}", x_val)
                };

                let node = Node::new(
                    Operation::Number(Number {
                        value: modified_x_val,
                    }),
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                );
                return Ok((Rc::new(node), rest));
            }
        }
        match operands.len() {
            1 => Ok((
                Rc::new(Node::new(
                    Operation::Negate,
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    input_types,
                    output_type,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            _ => unreachable!(),
        }
    }
}

struct IdentityOp;
impl ParseNode for IdentityOp {
    fn next_node(
        _: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let (mut operands, rest) =
            gather_operands(vec![vec![], vec![IJType::Tensor]], tokens, context)?;
        if !operands.is_empty() {
            Ok((operands.remove(0), rest))
        } else {
            let node = Node::new(
                Operation::Identity,
                vec![IJType::Tensor],
                IJType::Tensor,
                operands,
                context.get_increment_id(),
            );
            Ok((Rc::new(node), rest))
        }
    }
}

impl ParseNode for Number {
    fn next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };

        Ok((
            Rc::new(Node::new(
                node_op,
                vec![],
                IJType::Scalar,
                vec![],
                context.get_increment_id(),
            )),
            tokens,
        ))
    }
}

impl Debug for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{self, lexer};

    fn parse_str(input: &str, context: &mut ASTContext) -> Result<Rc<Node>, anyhow::Error> {
        let tokens = lexer(input).map_err(|e| SyntaxError::LexerError(e.to_string()))?;
        let result = parse_line(tokens, context)?;
        verify_types_tree(&result)?;
        Ok(result)
    }

    fn parse_str_no_context(input: &str) -> Result<(Rc<Node>, ASTContext)> {
        let mut context = ASTContext::new();
        let node = parse_str(input, &mut context)?;
        Ok((node, context))
    }

    fn is_specific_syntax_error(error: &anyhow::Error, expected_error: &SyntaxError) -> bool {
        if let Some(actual_error) = error.downcast_ref::<SyntaxError>() {
            actual_error == expected_error
        } else {
            false
        }
    }

    fn verify_types_tree(node: &Node) -> Result<(), anyhow::Error> {
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
            compute_input_type(second_operand).unwrap(),
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
