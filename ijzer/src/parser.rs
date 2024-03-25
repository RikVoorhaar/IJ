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
                    SyntaxError::UnhandledTokens(context.tokens_to_string(remainder)),
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
                SyntaxError::UnexpectedToken(token.clone()),
                context.full_slice(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    parse_assign(name, context, true)
}

pub fn parse_var_statement(context: &mut ASTContext) -> Result<Rc<Node>> {
    let mut token_iter = context.tokens.iter().skip(1);
    let name = match token_iter.next().as_ref() {
        Some(Token::Symbol(symbol)) => symbol.name.clone(),
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.full_slice(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    let input_arity = match token_iter.next().as_ref() {
        Some(Token::Number(number)) => {
            match token_iter.next().as_ref() {
                Some(&Token::Arrow) => {}
                Some(&token) => {
                    return Err(context.add_context_to_syntax_error(
                        SyntaxError::UnexpectedToken(token.clone()),
                        context.full_slice(),
                    ))
                }
                None => return Err(SyntaxError::EmptyStream.into()),
            }
            number.value.parse::<usize>().map_err(|_| {
                context.add_context_to_syntax_error(
                    SyntaxError::InvalidNumberFormat(number.value.clone()),
                    context.full_slice(),
                )
            })?
        }
        Some(&Token::Arrow) => 0,
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.full_slice(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    let output_arity = match token_iter.next().as_ref() {
        Some(Token::Number(number)) => number.value.parse::<usize>().map_err(|_| {
            context.add_context_to_syntax_error(
                SyntaxError::InvalidNumberFormat(number.value.clone()),
                context.full_slice(),
            )
        })?,
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.full_slice(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    context.insert_variable(
        Variable {
            name: name.clone(),
            typ: IJType::Function(FunctionSignature {
                input: vec![IJType::Tensor; input_arity],
                output: vec![IJType::Tensor; output_arity],
            }),
        },
        None,
    );

    Ok(Rc::new(Node::new(
        Operation::Nothing,
        IJType::Function(FunctionSignature {
            input: vec![],
            output: vec![],
        }),
        Vec::new(),
        context.get_increment_id(),
    )))
}

/// Get the nodes expressed on the right hand side of an assignment token.
fn _get_variable_expression(context: &mut ASTContext) -> Result<Vec<Rc<Node>>> {
    let tokens = context.get_tokens();
    let mut split = tokens.split(|t| t == &Token::Assign);
    let assign_index: usize = split
        .next()
        .ok_or(context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()),
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

fn _compute_input_arity(node: &Rc<Node>) -> Result<usize, SyntaxError> {
    let input_arity = node.typ.input_arity();
    let output_arity_sum: usize = node.operands.iter().map(|n| n.typ.output_arity()).sum();
    if input_arity < output_arity_sum {
        return Err(SyntaxError::InputArityExceedsOutputArity(
            input_arity,
            node.to_string(),
            output_arity_sum,
            node.operands
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(", "),
        )
        .into());
    }
    let this_node_arity: usize = input_arity - output_arity_sum;
    let children: usize = node
        .operands
        .iter()
        .map(_compute_input_arity)
        .try_fold(0, |acc, x| x.map(|y| acc + y))?;
    Ok(this_node_arity + children)
}

/// Recursively walk down the AST. For each node compare the signature to the output of its operands. Any non-machtching operand is considered an input. A functoin with no inputs is equal to its output type. (()-> Tensor = Tensor)
pub fn compute_input_type(node: &Rc<Node>) -> Result<IJType, SyntaxError> {
}

pub fn parse_assign(
    variable_name: String,
    context: &mut ASTContext,
    is_function: bool,
) -> Result<Rc<Node>> {
    let expressions = _get_variable_expression(context)?;
    let input_arity = expressions
        .iter()
        .map(|expr| _compute_input_arity(expr))
        .try_fold(0, |acc, x| x.map(|y| acc + y))
        .map_err(|e| context.add_context_to_syntax_error(e, context.full_slice()))?;
    if is_function && input_arity == 0 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::FunctionWithoutInputs,
            context.full_slice(),
        ));
    }
    if !is_function && input_arity != 0 {
        return Err(context
            .add_context_to_syntax_error(SyntaxError::VariableWithInput, context.full_slice()));
    }
    if !is_function {
        let left_of_assign = context.tokens.split(|t| t == &Token::Assign).next().ok_or(
            context.add_context_to_syntax_error(
                SyntaxError::UnresolvedGroup("=".to_string()),
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
                SyntaxError::IncorrectNumVariables(symbol_names.len(), expressions.len()),
                context.full_slice(),
            ));
        }
        let symbol_nodes: Vec<_> = symbol_names
            .into_iter()
            .map(|s| {
                Rc::new(Node::new(
                    Operation::Symbol(s),
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
            IJType::Function(FunctionSignature {
                input: vec![IJType::Group(vec![IJType::Tensor; symbol_nodes.len()]); 2],
                output: vec![],
            }),
            vec![
                Rc::new(Node::new(
                    Operation::Group,
                    IJType::Group(vec![IJType::Tensor; symbol_nodes.len()]),
                    symbol_nodes,
                    context.get_increment_id(),
                )),
                Rc::new(Node::new(
                    Operation::Group,
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
                SyntaxError::FunctionWithMultipleOutputs(expressions.len()),
                context.full_slice(),
            ));
        }
        let expression = expressions.into_iter().next().unwrap();
        let typ = IJType::tensor_function(input_arity, 1);
        let symbol_node = Node::new(
            Operation::Function(variable_name.clone()),
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
            IJType::Function(FunctionSignature {
                input: vec![typ],
                output: vec![],
            }),
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
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    }
}

/// TODO: This should not just take min_arity and max_arity; but also type.
/// I think it would be ok to restrict to either only tensors or only scalars
fn gather_operands(
    min_arity: usize,
    max_arity: usize,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Vec<Rc<Node>>, TokenSlice)> {
    let mut total_arity = 0;
    let mut overflow = 0;
    let mut operands = Vec::new();
    let mut rest = slice;
    while !rest.is_empty() {
        let (node, new_rest) = next_node(rest, context)?;
        if total_arity + node.typ.output_arity() > max_arity {
            overflow = node.typ.output_arity();
            break;
        }
        total_arity += node.typ.output_arity();
        match node.op {
            Operation::Group => {
                operands.extend(node.operands.clone());
            }
            _ => operands.push(node),
        }
        rest = new_rest;
    }
    if total_arity < min_arity {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::InsufficientArguments(min_arity, total_arity + overflow),
            slice,
        ));
    }

    Ok((operands, rest))
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
    let (operands, rest) = gather_operands(min_arity, max_arity, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    Ok((
        Rc::new(Node::new(
            op,
            IJType::tensor_function(operands.len(), 1),
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
) -> Result<usize, SyntaxError> {
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
    Err(SyntaxError::UnmatchedParenthesis(
        context.tokens_to_string(slice),
    ))
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

        let (operands, rest) = gather_operands(1, usize::MAX, slice, context)
            .with_context(|| anyhow!("Error created by bracket parsing"))?;
        if !rest.is_empty() {
            return Err(SyntaxError::UnhandledTokens(context.tokens_to_string(rest)).into());
        }

        match operands.len() {
            0 => unreachable!(),
            1 => Ok((operands[0].clone(), remainder)),
            _ => {
                let operands_types = operands
                    .iter()
                    .map(|n| n.typ.clone())
                    .collect::<Vec<IJType>>();
                Ok((
                    Rc::new(Node::new(
                        Operation::Group,
                        IJType::Function(FunctionSignature::new(
                            operands_types.clone(),
                            vec![IJType::Group(operands_types)],
                        )),
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
                IJType::Tensor,
                Vec::new(),
                context.get_increment_id(),
            )),
            remainder,
        ))
    }
}

// TODO: not sure this is correct in general.
// I think that there should be a `_next_node_function` method that takes a function signature as input. Then in this case when we have a function that's what we output. In the case of a Tensor or Scalar, we directly create the node. (Which is a symbol node). The group variant should be impossible by construction and throw a syntax error.
fn _next_node_symbol(
    op: SymbolToken,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let (variable, _) = context
        .symbols
        .get(&op.name)
        .ok_or(SyntaxError::UnknownSymbol(op.name.clone()))?;

    let node_op = match variable.typ {
        IJType::Tensor => Operation::Symbol(op.name),
        IJType::Function(_) => Operation::Function(op.name),
        IJType::Group(_) => Operation::Group,
        IJType::Scalar => Operation::Scalar,
    };
    if variable.typ.output_arity() > 1 {
        return Err(SyntaxError::MultipleOutputs.into());
    }

    _next_node_simple_tensor_function(
        node_op,
        variable.typ.input_arity(),
        variable.typ.input_arity(),
        slice,
        context,
    )
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
            match variable.typ {
                IJType::Function(ref func_signature) if func_signature == signature => {
                    Operation::Function(name.clone())
                }
                _ => {
                    return Err(
                        SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into(),
                    )
                }
            }
        }

        _ => return Err(SyntaxError::ExpectedFunction(context.tokens_to_string(slice)).into()),
    };
    Ok((
        Rc::new(Node::new(
            op,
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
        if function.typ != function_type {
            return Err(SyntaxError::TypeError(
                function_type.to_string(),
                function.typ.to_string(),
            )
            .into());
        }

        let (operands, rest) = gather_operands(1, 1, rest, context)?;
        Ok((
            Rc::new(Node::new(
                Operation::Reduce,
                IJType::Function(FunctionSignature::new(
                    vec![function_type, IJType::Tensor],
                    vec![IJType::Scalar],
                )),
                [function].into_iter().chain(operands.into_iter()).collect(),
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
        let (operands, rest) = gather_operands(1, 2, slice, context)?;
        match operands.iter().map(|node| node.typ.output_arity()).sum() {
            1 => Ok((
                Rc::new(Node::new(
                    Operation::Negate,
                    IJType::tensor_function(1, 1),
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    IJType::tensor_function(2, 1),
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
        let (mut operands, rest) = gather_operands(0, 1, tokens, context)?;
        if !operands.is_empty() {
            Ok((operands.remove(0), rest))
        } else {
            let node = Node::new(
                Operation::Identity,
                IJType::tensor_function(1, 1),
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
