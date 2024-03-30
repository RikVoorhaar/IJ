use crate::ast_node::{ASTContext, FunctionSignature, IJType, Node, TokenSlice, Variable};
use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, SymbolToken, Token};
use anyhow::{anyhow, Context, Result};
use std::fmt::Debug;
use std::rc::Rc;

trait TokenImpl {
    fn _next_node(
        op: Token,
        tokens: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)>;
}

pub fn parse_line(tokens: Vec<Token>, context: &mut ASTContext) -> Result<Rc<Node>> {
    context.set_tokens(tokens);
    context.line_no += 1;
    parse_ast(context)
}

fn parse_ast(context: &mut ASTContext) -> Result<Rc<Node>> {
    let (op, _) = &context
        .tokens
        .split_first()
        .ok_or(SyntaxError::EmptyStream)?;
    match op {
        Token::Variable => parse_var_statement(context),
        Token::Function => parse_fn_statement(context),
        Token::Symbol(symbol) => parse_assign(symbol.name.clone(), context, false),
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

pub fn parse_fn_statement(context: &mut ASTContext) -> Result<Rc<Node>> {
    let mut token_iter = context.tokens.iter().skip(1);
    let name = match token_iter.next().as_ref() {
        Some(Token::Symbol(symbol)) => symbol.name.clone(),
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()).into(),
                context.full_slice(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    parse_assign(name, context, true)
}

pub fn parse_var_statement(context: &mut ASTContext) -> Result<Rc<Node>> {
    match context.tokens.as_slice() {
        [Token::Variable, Token::Symbol(symbol), Token::TypeDeclaration, rest @ ..] => {
            let typ = IJType::from_tokens(rest)?;
            let var = Variable {
                name: symbol.name.clone(),
                typ: typ.clone(),
            };
            context.insert_variable(var, None);
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
fn _get_variable_expression(context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
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

/// Revusrively walk down AST. If we reach a leaf node without an operand, add it to the input types. Otherwise concatenate the input types of its operands.
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

pub fn parse_assign(
    variable_name: String,
    context: &mut ASTContext,
    is_function: bool,
) -> Result<Rc<Node>> {
    let expressions = _get_variable_expression(context)?;
    let input_types = expressions
        .iter()
        .map(|expr| compute_input_type(expr))
        .try_fold(vec![], |mut acc, x| {
            x.map(|y| {
                acc.extend(y);
                acc
            })
        })
        .map_err(|e| context.add_context_to_syntax_error(e, context.full_slice()))?;
    if is_function && input_types.is_empty() {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::FunctionWithoutInputs.into(),
            context.full_slice(),
        ));
    }
    if !is_function && input_types.len() != 0 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::VariableWithInput.into(),
            context.full_slice(),
        ));
    }
    if !is_function {
        let left_of_assign = context.tokens.split(|t| t == &Token::Assign).next().ok_or(
            context.add_context_to_syntax_error(
                SyntaxError::UnresolvedGroup("=".to_string()).into(),
                context.full_slice(),
            ),
        )?;
        let symbol_names: Vec<String> = left_of_assign
            .iter()
            .filter(|&t| matches!(t, Token::Symbol(_)))
            .map(|t| match t {
                Token::Symbol(s) => s.name.clone(),
                _ => unreachable!(),
            })
            .collect();
        if symbol_names.len() != expressions.len() {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::IncorrectNumVariables(symbol_names.len(), expressions.len()).into(),
                context.full_slice(),
            ));
        }
        let symbol_nodes: Vec<_> = symbol_names
            .into_iter()
            .map(|s| {
                Rc::new(Node::new(
                    Operation::Symbol(s),
                    Vec::new(),
                    IJType::Tensor,
                    Vec::new(),
                    context.get_increment_id(),
                ))
            })
            .collect();
        for symbol_node in symbol_nodes.iter() {
            context.insert_variable(
                Variable {
                    name: variable_name.clone(),
                    typ: IJType::Tensor,
                },
                Some(symbol_node.clone()),
            );
        }
        Ok(Rc::new(Node::new(
            Operation::Assign,
            vec![IJType::Group(vec![IJType::Tensor; symbol_nodes.len()]); 2],
            IJType::Void,
            vec![
                Rc::new(Node::new(
                    Operation::Group,
                    vec![],
                    IJType::Group(vec![IJType::Tensor; symbol_nodes.len()]),
                    symbol_nodes,
                    context.get_increment_id(),
                )),
                Rc::new(Node::new(
                    Operation::Group,
                    vec![],
                    IJType::Group(vec![IJType::Tensor; expressions.len()]),
                    expressions,
                    context.get_increment_id(),
                )),
            ],
            context.get_increment_id(),
        )))
    } else {
        // is_function
        if expressions.len() != 1 {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::FunctionWithMultipleOutputs(expressions.len()).into(),
                context.full_slice(),
            ));
        }
        let expression = expressions.into_iter().next().unwrap();
        let typ = IJType::Function(FunctionSignature {
            input: input_types,
            output: vec![IJType::Tensor],
        });
        let symbol_node = Node::new(
            Operation::Function(variable_name.clone()),
            vec![],
            typ.clone(),
            Vec::new(),
            context.get_increment_id(),
        );
        context.insert_variable(
            Variable {
                name: variable_name.clone(),
                typ: typ.clone(),
            },
            Some(expression.clone()),
        );

        Ok(Rc::new(Node::new(
            Operation::FunctionDeclaration,
            vec![typ; 2],
            IJType::Void,
            vec![Rc::new(symbol_node), expression],
            context.get_increment_id(),
        )))
    }
}

fn next_node(slice: TokenSlice, context: &mut ASTContext) -> Result<(Rc<Node>, TokenSlice)> {
    let op = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    match op {
        Token::Plus => _next_node_simple_tensor_function(Operation::Add, 2, 2, rest, context),
        Token::Multiplication => {
            _next_node_simple_tensor_function(Operation::Multiply, 2, 2, rest, context)
        }
        Token::Number(_) => Number::_next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::_next_node(op.clone(), rest, context),
        Token::LParen => LParen::_next_node(op.clone(), rest, context),
        Token::LSqBracket => LSqBracket::_next_node(op.clone(), rest, context),
        Token::Symbol(op) => _next_node_symbol(op.clone(), rest, context),
        Token::Identity => IdentityOp::_next_node(op.clone(), rest, context),
        Token::Reduction => Reduction::_next_node(op.clone(), rest, context),
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
    let operands = operands[..longest_match_length]
        .into_iter()
        .map(|n| n.clone())
        .collect();

    Ok((operands, rest))
}

/// Match as many tokens of any kind as possible.
/// Always consumes the entire token slice
fn gather_all(slice: TokenSlice, context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
    let mut operands = Vec::new();
    let mut rest = slice;
    while !rest.is_empty() {
        let (node, new_rest) = next_node(rest, context)
            .map_err(|e| context.add_context_to_syntax_error(e.into(), rest))?;
        operands.push(node);
        rest = new_rest;
    }
    Ok(operands)
}

/// Parses a simple tensor function, such as addition or multiplication. These all have
/// a range of input aritys. All inputs are tensors, the output is a single tensor.
fn _next_node_simple_tensor_function(
    op: Operation,
    min_arity: usize,
    max_arity: usize,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let types = (min_arity..=max_arity)
        .map(|arity| vec![IJType::Tensor; arity])
        .collect::<Vec<_>>();
    let (operands, rest) = gather_operands(types, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    Ok((
        Rc::new(Node::new(
            op,
            vec![IJType::Tensor; operands.len()],
            IJType::Tensor,
            operands,
            context.get_increment_id(),
        )),
        rest,
    ))
}

fn _find_matching_parenthesis(
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
#[derive(Debug, PartialEq)]
struct LParen;
impl TokenImpl for LParen {
    fn _next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            _find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        // let slice = TokenSlice::new(slice.start, rparen_index, slice.max);
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

#[derive(Debug, PartialEq)]
struct LSqBracket;
impl TokenImpl for LSqBracket {
    fn _next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let rparen_index =
            _find_matching_parenthesis(context, slice, &Token::LSqBracket, &Token::RSqBracket)
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

fn _next_node_symbol(
    op: SymbolToken,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let (variable, _) = context
        .symbols
        .get(&op.name)
        .ok_or(SyntaxError::UnknownSymbol(op.name.clone()))?;

    let (node, rest) = match variable.typ.clone() {
        IJType::Tensor => (
            Node::new(
                Operation::Symbol(op.name),
                vec![],
                IJType::Tensor,
                vec![],
                context.get_increment_id(),
            ),
            slice,
        ),
        IJType::Function(signature) => {
            let (operands, rest) = gather_operands(vec![signature.input.clone()], slice, context)?;
            let output_type = match signature.output.len() {
                1 => signature.clone().output.into_iter().next().unwrap(),
                _ => IJType::Group(signature.clone().output),
            };
            let node = Node::new(
                Operation::Function(op.name),
                signature.clone().input,
                output_type,
                operands,
                context.get_increment_id(),
            );
            (node, rest)
        }
        IJType::Scalar => (
            Node::new(
                Operation::Symbol(op.name),
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
            let (variable, _) = context
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
impl TokenImpl for Reduction {
    fn _next_node(
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

        let operand_type = IJType::Tensor;
        let (operands, rest) = gather_operands(vec![vec![operand_type.clone()]], rest, context)?;
        let operand = operands.into_iter().next().unwrap();
        Ok((
            Rc::new(Node::new(
                Operation::Reduce,
                vec![function_type, operand_type],
                IJType::Tensor,
                vec![function, operand],
                context.get_increment_id(),
            )),
            rest,
        ))
    }
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node(
        _: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let single_tensor = vec![IJType::Tensor];
        let two_tensors = vec![IJType::Tensor; 2];
        let (operands, rest) = gather_operands(
            vec![single_tensor.clone(), two_tensors.clone()],
            slice,
            context,
        )?;
        match operands.len() {
            1 => Ok((
                Rc::new(Node::new(
                    Operation::Negate,
                    single_tensor,
                    IJType::Tensor,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    two_tensors,
                    IJType::Tensor,
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
impl TokenImpl for IdentityOp {
    fn _next_node(
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

impl TokenImpl for Number {
    fn _next_node(
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
