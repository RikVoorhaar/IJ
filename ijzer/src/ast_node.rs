use crate::operations::Operation;
use crate::syntax_error::SyntaxError;
use crate::tokens::Token;
use anyhow::{anyhow, Error, Result};
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
pub struct TokenSlice {
    pub start: usize,
    pub end: usize,
    pub max: usize,
}
#[derive(Clone, PartialEq, Debug)]
pub struct Variable {
    pub name: String,
    pub output_arity: usize,
    pub input_arity: usize,
}

#[derive(Debug, Default)]
pub struct ASTContext {
    pub symbols: HashMap<String, (Variable, Option<Rc<Node>>)>,
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

    pub fn check_valid(&self) -> Result<()> {
        if self.start > self.end {
            return Err(anyhow!("Invalid slice"));
        }
        if self.end > self.max {
            return Err(anyhow!("Invalid slice"));
        }
        Ok(())
    }

    pub fn move_start(&self, amount: usize) -> Result<Self> {
        let out = TokenSlice::new(self.start + amount, self.end, self.max);
        out.check_valid()?;
        Ok(out)
    }

    pub fn move_end(&self, new_length: usize) -> Result<Self> {
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

    pub fn all_operands(&self) -> impl Iterator<Item = Rc<Node>> + '_ {
        self.functional_operands
            .iter()
            .chain(&self.operands)
            .cloned()
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

    pub fn insert_variable(&mut self, var: Variable, expression: Option<Rc<Node>>) {
        self.symbols.insert(var.name.clone(), (var, expression));
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

    pub fn tokens_to_string(&self, slice: TokenSlice) -> String {
        self.get_tokens()[slice.start..slice.end]
            .iter()
            .map(|t| format!("{}", t))
            .collect::<Vec<String>>()
            .join(" ")
    }

    pub fn get_token_at_index(&self, index: usize) -> Result<&Token> {
        self.tokens
            .get(index)
            .ok_or(SyntaxError::EmptyStream.into())
    }

    pub fn add_context_to_syntax_error(&self, error: SyntaxError, slice: TokenSlice) -> Error {
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
