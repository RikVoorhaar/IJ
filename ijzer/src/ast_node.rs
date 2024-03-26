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
    pub operands: Vec<Rc<Node>>,
    pub output_type: IJType,
    pub input_types: Vec<IJType>,
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
    pub typ: IJType,
}

#[derive(Debug, Default)]
pub struct ASTContext {
    pub symbols: HashMap<String, (Variable, Option<Rc<Node>>)>,
    pub tokens: Rc<Vec<Token>>,
    pub line_no: usize,
    pub id_counter: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IJType {
    Void,
    Scalar,
    Tensor,
    Function(FunctionSignature),
    Group(Vec<IJType>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionSignature {
    pub input: Vec<IJType>,
    pub output: Vec<IJType>,
}

impl FunctionSignature {
    pub fn new(input: Vec<IJType>, output: Vec<IJType>) -> Self {
        FunctionSignature { input, output }
    }
    pub fn tensor_function(inputs: usize, outputs: usize) -> Self {
        Self {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Tensor; outputs],
        }
    }
    pub fn scalar_function(inputs: usize, outputs: usize) -> Self {
        Self {
            input: vec![IJType::Scalar; inputs],
            output: vec![IJType::Scalar; outputs],
        }
    }
}

impl IJType {
    // pub fn output_arity(&self) -> usize {
    //     match self {
    //         IJType::Scalar => 1,
    //         IJType::Tensor => 1,
    //         IJType::Function(sig) => sig.output.len(),
    //         IJType::Void => 0,
    //     }
    // }
    // pub fn input_arity(&self) -> usize {
    //     match self {
    //         IJType::Scalar => 0,
    //         IJType::Tensor => 0,
    //         IJType::Function(sig) => sig.input.len(),
    //         IJType::Void => 0,
    //     }
    // }
    // pub fn output_types(&self) -> Vec<IJType> {
    //     match self {
    //         IJType::Scalar => vec![IJType::Scalar],
    //         IJType::Tensor => vec![IJType::Tensor],
    //         IJType::Function(sig) => sig.output.clone(),
    //         IJType::Void => vec![],
    //     }
    // }
    pub fn tensor_function(inputs: usize, outputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Tensor; outputs],
        })
    }
    pub fn scalar_function(inputs: usize, outputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Scalar; inputs],
            output: vec![IJType::Scalar; outputs],
        })
    }
    pub fn tensor_to_scalar_function(inputs: usize) -> IJType {
        IJType::Function(FunctionSignature {
            input: vec![IJType::Tensor; inputs],
            output: vec![IJType::Scalar; 1],
        })
    }
}

impl std::fmt::Display for IJType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IJType::Scalar => write!(f, "Scalar"),
            IJType::Tensor => write!(f, "Tensor"),
            IJType::Function(sig) => write!(f, "Function({})", sig),
            IJType::Void => write!(f, "Void"),
            IJType::Group(types) => write!(f, "Group({})", types.iter().map(|t| t.to_string()).collect::<Vec<String>>().join(", ")),
        }
    }
}

impl std::fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_strings: Vec<String> = self.input.iter().map(|i| i.to_string()).collect();
        let output_strings: Vec<String> = self.output.iter().map(|o| o.to_string()).collect();
        write!(
            f,
            "({}) -> ({})",
            input_strings.join(", "),
            output_strings.join(", ")
        )
    }
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

    pub fn move_start(&self, amount: usize) -> Result<Self, SyntaxError> {
        let out = TokenSlice::new(self.start + amount, self.end, self.max);
        out.check_valid()?;
        Ok(out)
    }

    pub fn move_end(&self, new_length: usize) -> Result<Self, SyntaxError> {
        if self.start + new_length > self.max {
            return Err(SyntaxError::InvalidSlice);
        }
        let out = TokenSlice::new(self.start, self.start + new_length, self.max);
        out.check_valid()?;
        Ok(out)
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

    pub fn get_token_at_index(&self, index: usize) -> Result<&Token, SyntaxError> {
        self.tokens
            .get(index)
            .ok_or(SyntaxError::EmptyStream.into())
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
        write!(f, "{}{:?}<{}>", _tab_nested(nesting), self.op, self.output_type)?;
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
