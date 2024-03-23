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

fn _get_variable_expression(context: &mut ASTContext) -> Result<Rc<Node>> {
    let tokens = context.get_tokens();
    let mut split = tokens.split(|t| t == &Token::Assign);
    let assign_index: usize = split
        .next()
        .ok_or(context.add_context_to_syntax_error(
            SyntaxError::UnresolvedGroup("=".to_string()),
            context.full_slice(),
        ))?
        .len();
    let slice = context.full_slice().move_start(assign_index + 1)?;
    let (expression, rest) = next_node(slice, context)?;
    if !rest.is_empty() {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::UnhandledTokens(context.tokens_to_string(rest)),
            rest,
        ));
    }
    Ok(expression)
}

fn _compute_input_arity(node: &Rc<Node>) -> usize {
    let this_node: usize = node.typ.input_arity()
        - node
            .operands
            .iter()
            .map(|n| n.typ.output_arity())
            .sum::<usize>();
    let children: usize = node
        .operands
        .iter()
        .map(_compute_input_arity)
        .sum::<usize>();
    this_node + children
}

pub fn parse_assign(
    variable_name: String,
    context: &mut ASTContext,
    is_function: bool,
) -> Result<Rc<Node>> {
    let expression = _get_variable_expression(context)?;
    let input_arity = _compute_input_arity(&expression);
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
        if symbol_names.len() != expression.typ.output_arity() {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::IncorrectNumVariables(
                    symbol_names.len(),
                    expression.typ.output_arity(),
                ),
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
            IJType::Tensor,
            vec![
                Rc::new(Node::new(
                    Operation::Group,
                    IJType::tensor_function(0, expression.typ.output_arity()),
                    symbol_nodes,
                    context.get_increment_id(),
                )),
                expression,
            ],
            context.get_increment_id(),
        )))
    } else {
        // is_function
        let output_arity = expression.typ.output_arity();
        let typ = IJType::tensor_function(input_arity, output_arity);
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
        Token::Plus => _next_node_normal_op(Operation::Add, 2, 2, 1, rest, context),
        Token::Multiplication => _next_node_normal_op(Operation::Multiply, 2, 2, 1, rest, context),
        Token::Number(_) => Number::_next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::_next_node(op.clone(), rest, context),
        Token::LParen => LParen::_next_node(op.clone(), rest, context),
        Token::LSqBracket => LSqBracket::_next_node(op.clone(), rest, context),
        Token::Symbol(op) => _next_node_symbol(op.clone(), rest, context),
        Token::Identity => IdentityOp::_next_node(op.clone(), rest, context),
        Token::Reduction => _next_node_reduction(op.clone(), rest, context),
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    }
}

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

fn _next_node_normal_op(
    op: Operation,
    min_arity: usize,
    max_arity: usize,
    output_arity: usize,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let (operands, rest) = gather_operands(min_arity, max_arity, slice, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    let input_arity = operands.iter().map(|n| n.typ.output_arity()).sum();
    let typ = IJType::tensor_function(input_arity, output_arity);
    Ok((
        Rc::new(Node::new(
            op,
            typ,
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
            _ => Ok((
                Rc::new(Node::new(
                    Operation::Group,
                    operands.iter().map(|n| n.output_arity).sum(),
                    operands,
                    context.get_increment_id(),
                )),
                remainder,
            )),
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
                1,
                Vec::new(),
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

    let node_op = match variable.input_arity {
        0 => Operation::Symbol(op.name),
        _ => Operation::Function(op.name),
    };

    _next_node_normal_op(
        node_op,
        variable.input_arity,
        variable.input_arity,
        variable.output_arity,
        slice,
        context,
    )
}

fn next_node_functional(
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let op = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    let functional_operand = match op {
        Token::Plus => {
            Node::new_with_input_arity(Operation::Add, 1, vec![], context.get_increment_id(), 2)?
        }
        Token::Multiplication => Node::new_with_input_arity(
            Operation::Multiply,
            1,
            vec![],
            context.get_increment_id(),
            2,
        )?,

        Token::Symbol(SymbolToken { name }) => {
            let (variable, _) = context
                .symbols
                .get(name.as_str())
                .ok_or(SyntaxError::UnknownSymbol(name.clone()))?;
            if variable.input_arity == 0 {
                return Err(SyntaxError::ExpectedFunctionalOperator(
                    context.tokens_to_string(slice),
                )
                .into());
            }
            Node::new(
                Operation::Function(name.clone()),
                variable.output_arity,
                vec![],
                context.get_increment_id(),
            )
        }

        _ => {
            let (node, rest) = next_node(slice, context)?;
            let input_arity_difference = node.input_arity.saturating_sub(node.operands.len());
            if input_arity_difference == 0 {
                return Err(SyntaxError::InsufficientArguments(
                    node.input_arity,
                    node.operands.len(),
                )
                .into());
            }
            return Ok((node, rest));
        }
    };
    Ok((Rc::new(functional_operand), rest))
}

fn _next_node_reduction(
    _op: Token,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let (functional_operand, rest) = next_node_functional(slice, context)?;
    if functional_operand.output_arity != 1 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::ExpectedOperatorWithOutputArity(1, context.tokens_to_string(slice)),
            slice,
        ));
    }
    if functional_operand.input_arity != 2 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::ExpectedBinaryOperator(context.tokens_to_string(slice)),
            slice,
        ));
    }

    let (operands, rest) = gather_operands(1, 1, rest, context)?;
    Ok((
        Rc::new(Node {
            op: Operation::Reduce,
            output_arity: 1,
            operands,
            functional_operands: vec![functional_operand],
            input_arity: 1,
            id: context.get_increment_id(),
        }),
        rest,
    ))
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
        match operands.iter().map(|node| node.output_arity).sum() {
            1 => Ok((
                Rc::new(Node::new(
                    Operation::Negate,
                    1,
                    operands,
                    context.get_increment_id(),
                )),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(
                    Operation::Subtract,
                    1,
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
            let node = Node::new_with_input_arity(
                Operation::Identity,
                1,
                operands,
                context.get_increment_id(),
                1,
            )
            .map_err(|e| context.add_context_to_syntax_error(e, tokens))?;
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
            Rc::new(Node::new(node_op, 1, vec![], context.get_increment_id())),
            tokens,
        ))
    }
}

impl Debug for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

impl Node {
    fn fmt_nested(&self, f: &mut std::fmt::Formatter<'_>, nesting: usize) -> std::fmt::Result {
        write!(
            f,
            "{}{:?} | ({}->{})",
            _tab_nested(nesting),
            self.op,
            self.input_arity,
            self.output_arity
        )?;
        if !&self.operands.is_empty() {
            write!(f, "[")?;

            for operand in &self.operands {
                writeln!(f)?;
                operand.fmt_nested(f, nesting + 1)?;
            }
            write!(f, "\n{}]", _tab_nested(nesting))?;
        }

        if !&self.functional_operands.is_empty() {
            write!(f, "<")?;
            for operand in &self.functional_operands {
                operand.fmt_nested(f, nesting + 1)?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

fn _tab_nested(nesting: usize) -> String {
    "    ".repeat(nesting)
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_nested(f, 0)
    }
}
