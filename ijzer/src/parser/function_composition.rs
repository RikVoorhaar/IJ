struct FunctionComposition;
impl ParseNode for FunctionComposition {
    fn next_node(
        _op: Token,
        slice: TokenSlice,
        context: &mut ASTContext,
    ) -> Result<(Rc<Node>, TokenSlice)> {
        let next_token = context.get_token_at_index(slice.start)?;
        if next_token != &Token::LParen {
            return Err(SyntaxError::ExpectedToken(
                "(".to_string(),
                format!("found {:?}", next_token),
            )
            .into());
        }
        let rparen_index =
            find_matching_parenthesis(context, slice, &Token::LParen, &Token::RParen)
                .map_err(|e| context.add_context_to_syntax_error(e, slice))?;
        let remainder = slice.move_start(rparen_index + 1)?;
        let slice = slice.move_end(rparen_index)?;
        let slices = comma_separate(slice, context)?;
        // let (nodes, rest) = gather_operands(vec![vec![IJType::Tensor]], slices, context)?;
    }
}
