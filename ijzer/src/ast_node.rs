use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use crate::types::IJType;
use anyhow::{anyhow, Error, Result};
use std::collections::HashMap;
use std::fmt::{format, Debug};
use std::rc::Rc;

#[derive(Clone)]
pub struct Node {
    pub op: Operation,
    pub operands: Vec<Rc<Node>>,
    pub output_type: IJType,
    pub input_types: Vec<IJType>,
    pub id: usize,
}

#[derive(Copy, Clone, PartialEq)]
pub enum LineHasSemicolon {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq)]
pub struct TokenSlice {
    pub start: usize,
    pub end: usize,
    pub max: usize,
}
#[derive(Clone, PartialEq, Debug)]
pub struct Variable {
    pub name: String,
    pub typ: IJType,
}

#[derive(Debug, Default)]
pub struct ASTContext {
    pub symbols: HashMap<String, Variable>,
    pub tokens: Rc<Vec<Token>>,
    pub line_no: usize,
    pub id_counter: usize,
}

impl TokenSlice {
    pub fn new(start: usize, end: usize, max: usize) -> Self {
        TokenSlice { start, end, max }
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn check_valid(&self) -> Result<(), SyntaxError> {
        if self.start > self.end {
            return Err(SyntaxError::InvalidSlice);
        }
        if self.end > self.max {
            return Err(SyntaxError::InvalidSlice);
        }
        Ok(())
    }

    /// Moves the start of the slice to a new length, relative to the start of the slice.
    pub fn move_start(&self, amount: usize) -> Result<Self, SyntaxError> {
        let out = TokenSlice::new(self.start + amount, self.end, self.max);
        out.check_valid()?;
        Ok(out)
    }

    /// Moves the start of the slice to a new length, relative to the start of the slice.
    /// If the start moves past the end, return an empty slice.
    pub fn move_start_saturating(&self, amount: usize) -> Self {
        let new_start = (self.start + amount).min(self.end);
        let out = TokenSlice::new(new_start, self.end, self.max);
        out
    }

    /// Moves the end of the slice to a new length, relative to the start of the slice.
    pub fn move_end(&self, new_length: usize) -> Result<Self, SyntaxError> {
        if self.start + new_length > self.max {
            return Err(SyntaxError::InvalidSlice);
        }
        let out = TokenSlice::new(self.start, self.start + new_length, self.max);
        out.check_valid()?;
        Ok(out)
    }

    pub fn move_end_saturating(&self, new_length: usize) -> Self {
        let new_end = (self.start + new_length).min(self.max);
        let out = TokenSlice::new(self.start, new_end, self.max);
        out
    }

    /// Returns the length of the slice.
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

impl Node {
    pub fn new(
        op: Operation,
        input_types: Vec<IJType>,
        output_type: IJType,
        operands: Vec<Rc<Node>>,
        id: usize,
    ) -> Self {
        Node {
            op,
            output_type,
            input_types,
            operands,
            id,
        }
    }
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

    pub fn from_tokens(tokens: Vec<Token>) -> Self {
        let mut context = ASTContext::new();
        context.set_tokens(tokens);
        context
    }

    pub fn insert_variable(&mut self, var: Variable) {
        self.symbols.insert(var.name.clone(), var);
    }

    pub fn insert_lambda_variable(&mut self, name: String, var_type: IJType) {
        self.symbols.insert(
            format!("_arg_{}_{}", self.line_no, name),
            Variable {
                name,
                typ: var_type,
            },
        );
    }

    pub fn get_lambda_var_type(&self, name: String) -> Option<IJType> {
        match self.symbols.get(&format!("_arg_{}_{}", self.line_no, name)) {
            Some(var) => Some(var.typ.clone()),
            None => None,
        }
    }

    pub fn get_increment_id(&mut self) -> usize {
        let id = self.id_counter;
        self.id_counter += 1;
        id
    }

    pub fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = Rc::new(tokens);
    }

    pub fn get_tokens(&self) -> Rc<Vec<Token>> {
        Rc::clone(&self.tokens)
    }

    pub fn token_slice_to_string(&self, slice: TokenSlice) -> String {
        self.get_tokens()[slice.start..slice.end]
            .iter()
            .map(|t| format!("{}", t))
            .collect::<Vec<String>>()
            .join(" ")
    }

    pub fn get_token_at_index(&self, index: usize) -> Result<&Token, SyntaxError> {
        self.tokens
            .get(index)
            .ok_or(SyntaxError::OutOfBoundsToken(index, self.tokens.len()))
    }

    pub fn get_tokens_from_slice(&self, slice: TokenSlice) -> Vec<Token> {
        self.tokens[slice.start..slice.end].to_vec()
    }

    pub fn add_context_to_syntax_error(&self, error: Error, slice: TokenSlice) -> Error {
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

    pub fn full_slice(&self) -> TokenSlice {
        TokenSlice::new(0, self.tokens.len(), self.tokens.len())
    }
}

impl Node {
    fn fmt_nested(&self, f: &mut std::fmt::Formatter<'_>, nesting: usize) -> std::fmt::Result {
        write!(
            f,
            "{}{:?}<{}>",
            _tab_nested(nesting),
            self.op,
            self.output_type
        )?;
        if !&self.operands.is_empty() {
            write!(f, "[")?;

            for operand in &self.operands {
                writeln!(f)?;
                operand.fmt_nested(f, nesting + 1)?;
            }
            write!(f, "\n{}]", _tab_nested(nesting))?;
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

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}<{}>", self.op, self.output_type)?;
        if !self.operands.is_empty() {
            let operand_types: Vec<String> = self
                .operands
                .iter()
                .map(|operand| format!("{}", operand.output_type))
                .collect();
            write!(f, "({})", operand_types.join(", "))?;
        }
        Ok(())
    }
}

impl Debug for TokenSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TokenSlice<{:?}:{:?}/max={:?}>",
            self.start, self.end, self.max
        )
    }
}
