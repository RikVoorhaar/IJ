use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, SymbolToken, Token};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone)]
pub struct Node {
    op: Operation,
    output_arity: usize,
    operands: Vec<Node>,
    functional_operands: Vec<Node>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Variable {
    name: String,
    output_arity: usize,
    input_arity: usize,
}

impl Variable {
    fn into_node(self, operands: Vec<Node>) -> Node {
        Node {
            op: Operation::Symbol(self.name),
            output_arity: self.output_arity,
            operands,
            functional_operands: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct SymbolTable {
    symbols: HashMap<String, (Variable, Option<Node>)>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            symbols: HashMap::new(),
        }
    }

    pub fn insert(&mut self, var: Variable, expression: Option<Node>) {
        self.symbols.insert(var.name.clone(), (var, expression));
    }
}

pub trait TokenImpl {
    fn _next_node<'a>(
        op: Token,
        tokens: &'a [Token],
        symbol_table: &SymbolTable,
    ) -> Result<(Node, &'a [Token])>;
}

pub fn parse_ast(tokens: &[Token], symbol_table: &mut SymbolTable) -> Result<Node> {
    let (op, rest) = tokens.split_first().ok_or(SyntaxError::EmptyStream)?;
    match op {
        Token::Variable => parse_var_statement(rest, symbol_table),
        _ => {
            let (node, remainer) = next_node(tokens, symbol_table)?;
            if !remainer.is_empty() {
                return Err(SyntaxError::UnhandledTokens(remainer.to_vec()).into());
            }
            Ok(node)
        }
    }
}

pub fn parse_var_statement(tokens: &[Token], symbol_table: &mut SymbolTable) -> Result<Node> {
    let mut token_iter = tokens.iter();
    let name = match token_iter.next().as_ref() {
        Some(Token::Symbol(symbol)) => symbol.value.clone(),
        Some(&token) => return Err(SyntaxError::UnexpectedToken(token.clone()).into()),
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    let input_arity = match token_iter.next().as_ref() {
        Some(Token::Number(number)) => {
            match token_iter.next().as_ref() {
                Some(&Token::Arrow) => {}
                Some(&token) => return Err(SyntaxError::UnexpectedToken(token.clone()).into()),
                None => return Err(SyntaxError::EmptyStream.into()),
            }
            number.value as usize
        }
        Some(&Token::Arrow) => 0,
        Some(&token) => return Err(SyntaxError::UnexpectedToken(token.clone()).into()),
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    let output_arity = match token_iter.next().as_ref() {
        Some(Token::Number(number)) => number.value as usize,
        Some(&token) => return Err(SyntaxError::UnexpectedToken(token.clone()).into()),
        None => return Err(SyntaxError::EmptyStream.into()),
    };
    symbol_table.insert(
        Variable {
            name: name.clone(),
            output_arity,
            input_arity,
        },
        None,
    );

    Ok(Node {
        op: Operation::Symbol(name),
        output_arity,
        operands: Vec::new(),
        functional_operands: Vec::new(),
    })
}

pub fn next_node<'a>(
    tokens: &'a [Token],
    symbol_table: &SymbolTable,
) -> Result<(Node, &'a [Token])> {
    let (op, rest) = tokens.split_first().ok_or(SyntaxError::EmptyStream)?;
    let out = match op {
        Token::Plus => _next_node_normal_op(Operation::Add, 2, 2, 1, rest, symbol_table),
        Token::Multiplication => {
            _next_node_normal_op(Operation::Multiply, 2, 2, 1, rest, symbol_table)
        }
        Token::Number(_) => Number::_next_node(op.clone(), rest, symbol_table),
        Token::Minus => MinusOp::_next_node(op.clone(), rest, symbol_table),
        Token::LParen => LParen::_next_node(op.clone(), rest, symbol_table),
        Token::Symbol(op) => _next_node_symbol(op.clone(), rest, symbol_table),
        _ => Err(anyhow!("Unexpected token {:?}", op)),
    };

    out
}

fn _next_node_normal_op<'a>(
    op: Operation,
    min_arity: usize,
    max_arity: usize,
    output_arity: usize,
    tokens: &'a [Token],
    symbol_table: &SymbolTable,
) -> Result<(Node, &'a [Token])> {
    let mut total_arity = 0;
    let mut overflow = 0;
    let mut operands = Vec::new();
    let mut rest = tokens;
    while !rest.is_empty() {
        let (node, new_rest) = next_node(rest, symbol_table)?;
        if total_arity + node.output_arity > max_arity {
            overflow = node.output_arity;
            break;
        }
        total_arity += node.output_arity;
        operands.push(node);
        rest = new_rest;
    }
    if total_arity < min_arity {
        return Err(
            SyntaxError::InsufficientArguments(op, min_arity, total_arity + overflow).into(),
        );
    }

    Ok((
        Node {
            op,
            output_arity,
            operands,
            functional_operands: Vec::new(),
        },
        rest,
    ))
}

fn _find_matching_parenthesis(tokens: &[Token]) -> Result<usize> {
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
    Err(SyntaxError::UnmatchedParenthesis.into())
}

#[derive(Debug, PartialEq)]
struct LParen;
impl TokenImpl for LParen {
    fn _next_node<'a>(
        _op: Token,
        tokens: &'a [Token],
        symbol_table: &SymbolTable,
    ) -> Result<(Node, &'a [Token])> {
        let rparen_index = _find_matching_parenthesis(tokens)?;
        let mut rest = &tokens[..rparen_index];
        let remainder = &tokens[rparen_index + 1..];

        let mut operands = Vec::new();
        while !rest.is_empty() {
            let (node, new_rest) = next_node(rest, symbol_table)?;
            rest = new_rest;
            operands.push(node);
        }

        match operands.len() {
            0 => Err(SyntaxError::EmptyStream.into()),
            1 => Ok((operands[0].clone(), remainder)),
            _ => Ok((
                Node {
                    op: Operation::Group,
                    output_arity: operands.iter().map(|n| n.output_arity).sum(),
                    operands,
                    functional_operands: Vec::new(),
                },
                remainder,
            )),
        }
    }
}

fn _next_node_symbol<'a>(
    op: SymbolToken,
    tokens: &'a [Token],
    symbol_table: &SymbolTable,
) -> Result<(Node, &'a [Token])> {
    let (symbol, _) = symbol_table
        .symbols
        .get(&op.value)
        .ok_or(SyntaxError::UnknownSymbol(op.value.clone()))?;

    return _next_node_normal_op(
        Operation::Symbol(op.value),
        symbol.input_arity,
        symbol.input_arity,
        symbol.output_arity,
        tokens,
        symbol_table,
    );
}

#[derive(Debug, PartialEq)]
struct MinusOp;
impl TokenImpl for MinusOp {
    fn _next_node<'a>(
        _: Token,
        tokens: &'a [Token],
        symbol_table: &SymbolTable,
    ) -> Result<(Node, &'a [Token])> {
        let (lhs, rest) = next_node(tokens, symbol_table)
            .with_context(|| SyntaxError::InsufficientArguments(Operation::Negate, 1, 0))?;
        if rest.is_empty() {
            Ok((
                Node {
                    op: Operation::Negate,
                    output_arity: 1,
                    operands: vec![lhs],
                    functional_operands: vec![],
                },
                &[],
            ))
        } else {
            let (rhs, rest) = next_node(rest, symbol_table)
                .with_context(|| SyntaxError::InsufficientArguments(Operation::Subtract, 2, 1))?;
            Ok((
                Node {
                    op: Operation::Subtract,
                    output_arity: 1,
                    operands: vec![lhs, rhs],
                    functional_operands: vec![],
                },
                rest,
            ))
        }
    }
}

impl TokenImpl for Number {
    fn _next_node<'a>(
        op: Token,
        tokens: &'a [Token],
        _: &SymbolTable,
    ) -> Result<(Node, &'a [Token])> {
        let node_op = match op {
            Token::Number(n) => Operation::Number(n),
            _ => unreachable!(),
        };

        Ok((
            Node {
                op: node_op,
                output_arity: 1,
                operands: vec![],
                functional_operands: vec![],
            },
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
            "{}{:?}->({})",
            _tab_nested(nesting),
            self.op,
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
