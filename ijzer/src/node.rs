use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::{Number, SymbolToken, Token};
use anyhow::{anyhow, Context, Error, Result};
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Clone)]
pub struct Node {
    pub op: Operation,
    pub output_arity: usize,
    pub operands: Vec<Rc<Node>>,
    pub functional_operands: Vec<Rc<Node>>,
    pub input_arity: usize,
    pub id: usize,
}

#[derive(Copy, Clone)]
struct TokenSlice {
    start: usize,
    end: usize,
    max: usize,
}

impl TokenSlice {
    fn new(start: usize, end: usize, max: usize) -> Self {
        TokenSlice { start, end, max }
    }

    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    fn check_valid(&self) -> Result<()> {
        if self.start > self.end {
            return Err(anyhow!("Invalid slice"));
        }
        if self.end > self.max {
            return Err(anyhow!("Invalid slice"));
        }
        Ok(())
    }

    fn move_start(&self, amount: usize) -> Result<Self> {
        let out = TokenSlice::new(self.start + amount, self.end, self.max);
        out.check_valid()?;
        Ok(out)
    }

    fn move_end(&self, new_length: usize) -> Result<Self> {
        if self.start + new_length > self.max {
            return Err(anyhow!("Invalid slice"));
        }
        let out = TokenSlice::new(self.start, self.start + new_length, self.max);
        out.check_valid()?;
        Ok(out)
    }
}

impl Node {
    pub fn new(op: Operation, output_arity: usize, operands: Vec<Rc<Node>>, id: usize) -> Self {
        Node {
            op,
            output_arity,
            input_arity: operands.iter().map(|n| n.output_arity).sum(),
            operands,
            functional_operands: Vec::new(),
            id,
        }
    }

    pub fn new_with_input_arity(
        op: Operation,
        output_arity: usize,
        operands: Vec<Rc<Node>>,
        id: usize,
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
                id,
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
    symbols: HashMap<String, (Variable, Option<Rc<Node>>)>,
    tokens: Rc<Vec<Token>>,
    pub line_no: usize,
    id_counter: usize,
}

impl ASTContext {
    pub fn new() -> Self {
        ASTContext {
            symbols: HashMap::new(),
            tokens: Rc::new(Vec::new()),
            line_no: 0,
            id_counter: 0,
        }
    }

    pub fn insert_variable(&mut self, var: Variable, expression: Option<Rc<Node>>) {
        self.symbols.insert(var.name.clone(), (var, expression));
    }

    fn get_increment_id(&mut self) -> usize {
        let id = self.id_counter;
        self.id_counter += 1;
        id
    }

    fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = Rc::new(tokens);
    }

    fn get_tokens(&self) -> Rc<Vec<Token>> {
        Rc::clone(&self.tokens)
    }

    fn tokens_to_string(&self, slice: TokenSlice) -> String {
        self.get_tokens()[slice.start..slice.end]
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<String>>()
            .join(" ")
    }

    fn get_token_at_index(&self, index: usize) -> Result<&Token> {
        self.tokens
            .get(index)
            .ok_or(SyntaxError::EmptyStream.into())
    }

    fn add_context_to_syntax_error(&self, error: SyntaxError, slice: TokenSlice) -> Error {
        let token_str = self.get_tokens()[slice.start..slice.end]
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<String>>()
            .join(" ");
        anyhow!(error).context(format!(
            "Error at line {}:{}-{} \n {}",
            self.line_no, slice.start, slice.end, token_str
        ))
    }

    fn full_slice(&self) -> TokenSlice {
        TokenSlice::new(0, self.tokens.len(), self.tokens.len())
    }
}

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
            number.value as usize
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
        Some(Token::Number(number)) => number.value as usize,
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
            output_arity,
            input_arity,
        },
        None,
    );

    Ok(Rc::new(Node::new(
        Operation::Symbol(name),
        output_arity,
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
        if symbol_names.len() != expression.output_arity {
            return Err(context.add_context_to_syntax_error(
                SyntaxError::IncorrectNumVariables(symbol_names.len(), expression.output_arity),
                context.full_slice(),
            ));
        }
        let symbol_nodes: Vec<_> = symbol_names
            .into_iter()
            .map(|s| {
                Rc::new(Node::new(
                    Operation::Symbol(s),
                    1,
                    Vec::new(),
                    context.get_increment_id(),
                ))
            })
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
        Ok(Rc::new(Node::new(
            Operation::Assign,
            0,
            vec![
                Rc::new(Node::new(
                    Operation::Group,
                    expression.output_arity,
                    symbol_nodes,
                    context.get_increment_id(),
                )),
                expression,
            ],
            context.get_increment_id(),
        )))
    } else {
        // is_function
        let symbol_node = Node::new_with_input_arity(
            Operation::Symbol(variable_name.clone()),
            expression.output_arity,
            Vec::new(),
            context.get_increment_id(),
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

        Ok(Rc::new(Node::new(
            Operation::Function,
            0,
            vec![Rc::new(symbol_node), expression],
            context.get_increment_id(),
        )))
    }
}

// fn _next_node_get_all_tokens(context: &mut ASTContext) -> Result<(Node, TokenSlice)> {
//     next_node(context.get_tokens(), context)
// }

fn next_node(slice: TokenSlice, context: &mut ASTContext) -> Result<(Rc<Node>, TokenSlice)> {
    let op = context.get_token_at_index(slice.start)?;
    let rest = slice.move_start(1)?;
    match op {
        Token::Plus => _next_node_normal_op(Operation::Add, 2, 2, 1, rest, context),
        Token::Multiplication => _next_node_normal_op(Operation::Multiply, 2, 2, 1, rest, context),
        Token::Number(_) => Number::_next_node(op.clone(), rest, context),
        Token::Minus => MinusOp::_next_node(op.clone(), rest, context),
        Token::LParen => LParen::_next_node(op.clone(), rest, context),
        Token::Symbol(op) => _next_node_symbol(op.clone(), rest, context),
        Token::Identity => IdentityOp::_next_node(op.clone(), rest, context),
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
        if total_arity + node.output_arity > max_arity {
            overflow = node.output_arity;
            break;
        }
        total_arity += node.output_arity;
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
    Ok((
        Rc::new(Node::new(
            op,
            output_arity,
            operands,
            context.get_increment_id(),
        )),
        rest,
    ))
}

fn _find_matching_parenthesis(
    context: &mut ASTContext,
    slice: TokenSlice,
) -> Result<usize, SyntaxError> {
    let tokens = &context.get_tokens()[slice.start..slice.end];
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
        let rparen_index = _find_matching_parenthesis(context, slice)
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

fn _next_node_symbol(
    op: SymbolToken,
    slice: TokenSlice,
    context: &mut ASTContext,
) -> Result<(Rc<Node>, TokenSlice)> {
    let (symbol, _) = context
        .symbols
        .get(&op.name)
        .ok_or(SyntaxError::UnknownSymbol(op.name.clone()))?;

    _next_node_normal_op(
        Operation::Symbol(op.name),
        symbol.input_arity,
        symbol.input_arity,
        symbol.output_arity,
        slice,
        context,
    )
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
                Rc::new(Node::new(Operation::Negate, 1, operands, context.get_increment_id())),
                rest,
            )),
            2 => Ok((
                Rc::new(Node::new(Operation::Subtract, 1, operands, context.get_increment_id())),
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
