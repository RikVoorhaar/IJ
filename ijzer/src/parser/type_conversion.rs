use crate::{
    ast_node::{ASTContext, Node, TokenSlice},
    operations::Operation,
    parser::ParseNode,
    syntax_error::SyntaxError,
    types::{FunctionSignature, IJType},
};
use anyhow::{Error, Result};

use super::next_node_functional;



// struct TypeConversion;
// impl ParseNode for TypeConversion {
//    fn next_node(
//            _op: crate::tokens::Token,
//            slice: TokenSlice,
//            context: &mut ASTContext,
//        ) -> Result<(Vec<std::rc::Rc<Node>>, TokenSlice)> {
//         // get the type from the slice

//         // find a function that matches the type

//         // return the correct node

//         Err(())
//    }
// }