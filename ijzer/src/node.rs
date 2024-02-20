use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, SymbolToken, Token};
use anyhow::{anyhow, Context, Error, Result};
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone)]
pub struct Node {
    op: Operation,
    output_arity: usize,
    operands: Vec<Node>,
    functional_operands: Vec<Node>,
    input_arity: usize,
}

impl Node {
    pub fn new(op: Operation, output_arity: usize, operands: Vec<Node>) -> Self {
        Node {
            op,
            output_arity,
            input_arity: operands.iter().map(|n| n.output_arity).sum(),
            operands,
            functional_operands: Vec::new(),
        }
    }

    pub fn new_with_input_arity(
        op: Operation,
        output_arity: usize,
        operands: Vec<Node>,
        input_arity: usize,
    ) -> Result<Self, SyntaxError> {
        let operand_arity: usize = operands.iter().map(|n| n.output_arity).sum();
        if input_arity < operand_arity {
            Err(SyntaxError::TooManyInputs)
        } else {
            Ok(Node {
                op,
                output_arity,
                input_arity,
                operands,
                functional_operands: Vec::new(),
            })
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Variable {
    name: String,
    output_arity: usize,
    input_arity: usize,
}

#[derive(Debug, Default)]
pub struct ASTContext {
    symbols: HashMap<String, (Variable, Option<Node>)>,
    tokens: Vec<Token>,
    pub line_no: usize,
}

impl ASTContext {
    pub fn new() -> Self {
        ASTContext {
            symbols: HashMap::new(),
            tokens: Vec::new(),
            line_no: 0,
        }
    }

    pub fn insert_variable(&mut self, var: Variable, expression: Option<Node>) {
        self.symbols.insert(var.name.clone(), (var, expression));
    }

    fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = tokens;
    }

    fn get_tokens(&self) -> &[Token] {
        &self.tokens
    }

    fn get_position(&self, tokens: &[Token]) -> (usize, usize) {
        let vec_ptr = self.tokens.as_ptr();
        let slice_ptr = tokens.as_ptr();
        let offset = unsafe { slice_ptr.offset_from(vec_ptr) as usize };
        (offset, offset + tokens.len())
    }

    fn add_context_to_syntax_error(&self, error: SyntaxError, tokens: &[Token]) -> Error {
        let (start, end) = self.get_position(tokens);
        anyhow!(error).context(format!(
            "Error at line {}:{}-{} \n {:?}",
            self.line_no, start, end, tokens
        ))
    }
}

pub trait TokenImpl {
    fn _next_node<'a>(
        op: Token,
        tokens: &'a [Token],
        context: &ASTContext,
    ) -> Result<(Node, &'a [Token])>;
}

pub fn parse_line(tokens: Vec<Token>, context: &mut ASTContext) -> Result<Node> {
    context.set_tokens(tokens);
    context.line_no += 1;
    parse_ast(context)
}

fn parse_ast(context: &mut ASTContext) -> Result<Node> {
    let (op, _) = &context
        .tokens
        .split_first()
        .ok_or(SyntaxError::EmptyStream)?;
    match op {
        Token::Variable => parse_var_statement(context),
        Token::Function => parse_fn_statement(context),
        Token::Symbol(symbol) => parse_assign(symbol.name.clone(), context, false),
        _ => {
            let (node, remainder) = next_node(&context.tokens, context)?;
            if !remainder.is_empty() {
                return Err(context.add_context_to_syntax_error(
                    SyntaxError::UnhandledTokens(remainder.to_vec()),
                    remainder,
                ));
            }
            Ok(node)
        }
    }
}

pub fn parse_fn_statement(context: &mut ASTContext) -> Result<Node> {
    let mut token_iter = context.tokens.iter().skip(1);
    let name = match token_iter.next().as_ref() {
        Some(Token::Symbol(symbol)) => symbol.name.clone(),
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.get_tokens(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    parse_assign(name, context, true)
}

pub fn parse_var_statement(context: &mut ASTContext) -> Result<Node> {
    let mut token_iter = context.tokens.iter().skip(1);
    let name = match token_iter.next().as_ref() {
        Some(Token::Symbol(symbol)) => symbol.name.clone(),
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.get_tokens(),
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
                        context.get_tokens(),
                    ))
                }
                None => return Err(SyntaxError::EmptyStream.into()),
            }
            number.value as usize
        }
        Some(&Token::Arrow) => 0,
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.get_tokens(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    let output_arity = match token_iter.next().as_ref() {
        Some(Token::Number(number)) => number.value as usize,
        Some(&token) => {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::UnexpectedToken(token.clone()),
                context.get_tokens(),
            ))
        }
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    context.insert_variable(
        Variable {
            name: name.clone(),
            output_arity,
            input_arity,
        },
        None,
    );

    Ok(Node::new(Operation::Symbol(name), output_arity, Vec::new()))
}

fn _get_variable_expression(context: &ASTContext, tokens: &[Token]) -> Result<Node> {
    let mut split = tokens.split(|t| t == &Token::Assign);
    if split.next().is_none() {
        return Err(context
            .add_context_to_syntax_error(SyntaxError::UnresolvedGroup("=".to_string()), tokens));
    }
    let rest = split
        .next()
        .ok_or(context.add_context_to_syntax_error(SyntaxError::EmptyStream, tokens))?;

    let (expression, rest) = next_node(rest, context)?;
    if !rest.is_empty() {
        return Err(
            context.add_context_to_syntax_error(SyntaxError::UnhandledTokens(rest.to_vec()), rest)
        );
    }
    Ok(expression)
}

fn _compute_input_arity(node: &Node) -> usize {
    let this_node: usize =
        node.input_arity - node.operands.iter().map(|n| n.output_arity).sum::<usize>();
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
) -> Result<Node> {
    let expression = _get_variable_expression(context, context.get_tokens())?;
    let input_arity = _compute_input_arity(&expression);
    if is_function && input_arity == 0 {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::FunctionWithoutInputs,
            context.get_tokens(),
        ));
    }
    if !is_function && input_arity != 0 {
        return Err(context
            .add_context_to_syntax_error(SyntaxError::VariableWithInput, context.get_tokens()));
    }
    if !is_function {
        let left_of_assign = context.tokens.split(|t| t == &Token::Assign).next().ok_or(
            context.add_context_to_syntax_error(
                SyntaxError::UnresolvedGroup("=".to_string()),
                context.get_tokens(),
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
        if symbol_names.len() != expression.output_arity {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::IncorrectNumVariables(symbol_names.len(), expression.output_arity),
                context.get_tokens(),
            ));
        }
        let symbol_nodes: Vec<_> = symbol_names
            .into_iter()
            .map(|s| Node::new(Operation::Symbol(s), 1, Vec::new()))
            .collect();
        for symbol_node in symbol_nodes.iter() {
            context.insert_variable(
                Variable {
                    name: variable_name.clone(),
                    output_arity: 1,
                    input_arity: 0,
                },
                Some(symbol_node.clone()),
            );
        }
        Ok(Node::new(
            Operation::Assign,
            0,
            vec![
                Node::new(Operation::Group, expression.output_arity, symbol_nodes),
                expression,
            ],
        ))
    } else {
        let symbol_node = Node::new_with_input_arity(
            Operation::Symbol(variable_name.clone()),
            expression.output_arity,
            Vec::new(),
            input_arity,
        )?;
        context.insert_variable(
            Variable {
                name: variable_name.clone(),
                output_arity: expression.output_arity,
                input_arity,
            },
            Some(expression.clone()),
        );

        Ok(Node::new(
            Operation::Assign,
            0,
            vec![symbol_node, expression],
        ))
    }
}

pub fn next_node<'a>(tokens: &'a [Token], context: &ASTContext) -> Result<(Node, &'a [Token])> {
    let (op, rest) = tokens.split_first().ok_or(SyntaxError::EmptyStream)?;
    let out = match op {
        Token::Plus => _next_node_normal_op(Operation::Add, 2, 2, 1, rest, context),
        Token::Multiplication => _next_node_normal_op(Operation::Multiply, 2, 2, 1, rest, context),
        Token::Number(_) => Number::_next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::_next_node(op.clone(), rest, context),
        Token::LParen => LParen::_next_node(op.clone(), rest, context),
        Token::Symbol(op) => _next_node_symbol(op.clone(), rest, context),
        Token::Identity => IdentityOp::_next_node(op.clone(), rest, context),
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    };

    out
}

fn gather_operands<'a>(
    min_arity: usize,
    max_arity: usize,
    tokens: &'a [Token],
    context: &ASTContext,
) -> Result<(Vec<Node>, &'a [Token])> {
    let mut total_arity = 0;
    let mut overflow = 0;
    let mut operands = Vec::new();
    let mut rest = tokens;
    while !rest.is_empty() {
        let (node, new_rest) = next_node(rest, context)?;
        if total_arity + node.output_arity > max_arity {
            overflow = node.output_arity;
            break;
        }
        total_arity += node.output_arity;
        match node.op {
            Operation::Group => {
                operands.extend(node.operands);
            }
            _ => operands.push(node),
        }
        rest = new_rest;
    }
    if total_arity < min_arity {
        return Err(context.add_context_to_syntax_error(
            SyntaxError::InsufficientArguments(min_arity, total_arity + overflow),
            tokens,
        ));
    }

    Ok((operands, rest))
}

fn _next_node_normal_op<'a>(
    op: Operation,
    min_arity: usize,
    max_arity: usize,
    output_arity: usize,
    tokens: &'a [Token],
    context: &ASTContext,
) -> Result<(Node, &'a [Token])> {
    let (operands, rest) = gather_operands(min_arity, max_arity, tokens, context)
        .with_context(|| anyhow!("Error caused by {:?}", op))?;
    Ok((Node::new(op, output_arity, operands), rest))
}

fn _find_matching_parenthesis(tokens: &[Token]) -> Result<usize, SyntaxError> {
    let mut depth = 1;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => (),
        }
    }
    Err(SyntaxError::UnmatchedParenthesis)
}

#[derive(Debug, PartialEq)]
struct LParen;
impl TokenImpl for LParen {
    fn _next_node<'a>(
        _op: Token,
        tokens: &'a [Token],
        context: &ASTContext,
    ) -> Result<(Node, &'a [Token])> {
        let rparen_index = _find_matching_parenthesis(tokens)
            .map_err(|e| context.add_context_to_syntax_error(e, tokens))?;
        let remainder = &tokens[rparen_index + 1..];
        let tokens = &tokens[..rparen_index];

        let (operands, rest) = gather_operands(1, usize::MAX, tokens, context)
            .with_context(|| anyhow!("Error created by bracket parsing"))?;
        if !rest.is_empty() {
            return Err(SyntaxError::UnhandledTokens(rest.to_vec()).into());
        }

        match operands.len() {
            0 => unreachable!(),
            1 => Ok((operands[0].clone(), remainder)),
            _ => Ok((
                Node::new(
                    Operation::Group,
                    operands.iter().map(|n| n.output_arity).sum(),
                    operands,
                ),
                remainder,
            )),
        }
    }
}

fn _next_node_symbol<'a>(
    op: SymbolToken,
    tokens: &'a [Token],
    context: &ASTContext,
) -> Result<(Node, &'a [Token])> {
    let (symbol, _) = context
        .symbols
        .get(&op.name)
        .ok_or(SyntaxError::UnknownSymbol(op.name.clone()))?;

    return _next_node_normal_op(
        Operation::Symbol(op.name),
        symbol.input_arity,
        symbol.input_arity,
        symbol.output_arity,
        tokens,
        context,
    );
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node<'a>(
        _: Token,
        tokens: &'a [Token],
        context: &ASTContext,
    ) -> Result<(Node, &'a [Token])> {
        let (operands, rest) = gather_operands(1, 2, tokens, context)?;
        match operands.iter().map(|node| node.output_arity).sum() {
            1 => Ok((Node::new(Operation::Negate, 1, operands), rest)),
            2 => Ok((Node::new(Operation::Subtract, 1, operands), rest)),
            _ => unreachable!(),
        }
    }
}

struct IdentityOp;
impl TokenImpl for IdentityOp {
    fn _next_node<'a>(
        _: Token,
        tokens: &'a [Token],
        context: &ASTContext,
    ) -> Result<(Node, &'a [Token])> {
        let (mut operands, rest) = gather_operands(0, 1, tokens, context)?;
        if !operands.is_empty() {
            Ok((operands.remove(0), rest))
        } else {
            let node = Node::new_with_input_arity(Operation::Identity, 1, operands, 1)
                .map_err(|e| context.add_context_to_syntax_error(e, tokens))?;
            Ok((node, rest))
        }
    }
}

impl TokenImpl for Number {
    fn _next_node<'a>(
        op: Token,
        tokens: &'a [Token],
        _: &ASTContext,
    ) -> Result<(Node, &'a [Token])> {
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };

        Ok((Node::new(node_op, 1, vec![]), tokens))
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
